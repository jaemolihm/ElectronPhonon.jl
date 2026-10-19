# Utilities that calculators call from their `run_calculator!` methods, on any backend. Not used inside ElectronPhonon.jl
# itself and not exported; downstream calculators reach them as `ElectronPhonon.<name>`. The
# backend/device primitives (`to_device`, `batched_gemm!`, …) live in `common/gpu_utils.jl`; this
# file holds the higher-level, calculator-facing helpers built on top of them.

# ---- the tile's flat item list ------------------------------------------------------------------
#
# The segment table `seg` of an `EPBlock` has one column per band class; rows (1-based):
const SEG_ITEM, SEG_PAIR, SEG_NPAIRS, SEG_BKQ, SEG_BK, SEG_KQ_PT, SEG_KQ_SLOT, SEG_K_PT, SEG_K_SLOT, SEG_K_SHARED =
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10
const SEG_ROWS = 10

# The column of `seg` holding item (or pair) number `x` (1-based), by the offsets in row `row`: the last
# segment that starts before `x`. A scan for the few classes a tile has (at most `nw²`), a bisection
# above 16.
@inline function _segment_column(seg, nseg, row, x)
    if nseg <= 16
        c = 1
        while c < nseg && x > seg[row, c + 1]
            c += 1
        end
        return c
    end
    lo, hi = 1, nseg
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if x > seg[row, mid]
            lo = mid
        else
            hi = mid - 1
        end
    end
    lo
end

# `(m, n, ν, j)` of item `l` of a tile with `nm` modes and segment table `seg`.
@inline function _item_indices(seg, nseg, nm, l)
    c = _segment_column(seg, nseg, SEG_ITEM, l)
    bkq, bk = seg[SEG_BKQ, c], seg[SEG_BK, c]
    l0 = l - 1 - seg[SEG_ITEM, c]
    m = l0 % bkq + 1
    l1 = l0 ÷ bkq
    n = l1 % bk + 1
    l2 = l1 ÷ bk
    ν = l2 % nm + 1
    (m, n, ν, seg[SEG_PAIR, c] + l2 ÷ nm + 1)
end

# `(m, n, j, first_item, stride)` of entry `e` (a band pair of a pair, `n_items ÷ nm` of them): its mode `ν`
# is item `first_item + (ν - 1) * stride`.
@inline function _entry_indices(seg, nseg, nm, e)
    c = _segment_column(seg, nseg, SEG_ITEM, (e - 1) * nm + 1)
    bkq, bk = seg[SEG_BKQ, c], seg[SEG_BK, c]
    stride = bkq * bk
    e0 = e - 1 - seg[SEG_ITEM, c] ÷ nm
    pb = e0 % stride
    jj = e0 ÷ stride
    (pb % bkq + 1, pb ÷ bkq + 1, seg[SEG_PAIR, c] + jj + 1, seg[SEG_ITEM, c] + pb + stride * nm * jj + 1, stride)
end

"""
    item_indices(tile, l) -> (m, n, ν, j)

Band at k+q, band at k, phonon mode and pair of item `l` of the tile's flat list (`1 ≤ l ≤ length(tile.ep)`).
`@inline`, allocation-free, and usable inside a device kernel on `device_tile(tile)`.
"""
@inline item_indices(tile, l) = _item_indices(tile.seg, size(tile.seg, 2), tile.phs.nmodes, l)

"""
    entry_indices(tile, e) -> (m, n, j, first_item, stride)

The band pair `(m, n)` of pair `j` that is entry `e` of the tile (`length(tile.ep) ÷ nmodes` entries), for a
calculator that sums over the modes: mode `ν` of the entry is item `first_item + (ν - 1) * stride`.
"""
@inline entry_indices(tile, e) = _entry_indices(tile.seg, size(tile.seg, 2), tile.phs.nmodes, e)

"""
    device_tile(tile::EPBlock) -> NamedTuple

The arrays the accessors read (`seg`, the state maps, the frequencies `phs.e`, the energies and eigenvectors of the
gathered states, and `ik`, `ikq`), as a `NamedTuple` that a kernel or a device broadcast can capture (`Ref(device_tile(tile))`):
`item_indices`, `entry_indices`, `state_k`, `state_kq`, `e_k`, `e_kq`, `u_k` and `u_kq` take it in place of
the tile.
"""
function device_tile(tile::EPBlock)
    (; tile.seg, tile.imap_k, tile.imap_kq, phs = (; tile.phs.nmodes, tile.phs.e),
       els_k = (; tile.els_k.e, tile.els_k.u, tile.els_k.nw),
       els_kq = (; tile.els_kq.e, tile.els_kq.u, tile.els_kq.nw), tile.ik, tile.ikq)
end

# The shared outer point of a one-sided index: `ik::Int` under outer k, a per-pair vector under outer q.
@inline _pair_index(ik::Integer, j) = ik
@inline _pair_index(ik, j) = ik[j]

"""
    state_k(tile, n, j), state_kq(tile, m, j) -> Int

The index in the run's selection (`BandStates(els_k, sel_k)`, `BandStates(els_kq, sel_kq)`) of band `n` at k
or `m` at k+q of pair `j`: 0 if that band is not a state. Through `tile.imap_k` / `tile.imap_kq`.
"""
@inline state_k(tile, n, j) = tile.imap_k[n, _pair_index(tile.ik, j)]
@inline state_kq(tile, m, j) = tile.imap_kq[m, tile.ikq[j]]

# The segment column of pair `j`, and the 0-based position of `j` in it.
@inline function _pair_segment(tile, j)
    c = _segment_column(tile.seg, size(tile.seg, 2), SEG_PAIR, j)
    (c, j - 1 - tile.seg[SEG_PAIR, c])
end

"""
    e_k(tile, n, j), e_kq(tile, m, j), u_k(tile, a, n, j), u_kq(tile, a, m, j)

Energy of band `n` at k, `m` at k+q, and component `a` (Wannier function) of the eigenvector of that band,
for pair `j`, read from the tile's gathered states through the segment table.
"""
@inline function e_k(tile, n, j)
    c, jj = _pair_segment(tile, j)
    seg = tile.seg
    slot = seg[SEG_K_SLOT, c] + (seg[SEG_K_SHARED, c] == 1 ? 0 : jj * seg[SEG_BK, c]) + n
    tile.els_k.e[slot]
end
@inline function e_kq(tile, m, j)
    c, jj = _pair_segment(tile, j)
    tile.els_kq.e[tile.seg[SEG_KQ_SLOT, c] + jj * tile.seg[SEG_BKQ, c] + m]
end
@inline function u_k(tile, a, n, j)
    c, jj = _pair_segment(tile, j)
    seg = tile.seg
    slot = seg[SEG_K_SLOT, c] + (seg[SEG_K_SHARED, c] == 1 ? 0 : jj * seg[SEG_BK, c]) + n
    tile.els_k.u[a + tile.els_k.nw * (slot - 1)]
end
@inline function u_kq(tile, a, m, j)
    c, jj = _pair_segment(tile, j)
    slot = tile.seg[SEG_KQ_SLOT, c] + jj * tile.seg[SEG_BKQ, c] + m
    tile.els_kq.u[a + tile.els_kq.nw * (slot - 1)]
end

"""
    eachsegment(tile; ep = tile.ep, dg = tile.dg) -> iterator of (; c, items, pairs, b_kq, b_k, kq_pt, k_pt, k_shared, ep, dg)

Host iteration over the band classes of a tile (its segment table is on the host): the item range, the pair
range, the band counts, the point offsets of the class's pairs in `tile.els_kq` and `tile.els_k` (a shared
k point is `k_pt + 1` for every pair, else the pair `jj` of the class is `k_pt + jj`), and `ep` (and `dg`)
as `(b_kq, b_k, nmodes, n_pairs)` (`(…, 3)`) arrays of the class. Pass host copies as `ep` / `dg` for a tile
on a device.
"""
function eachsegment(tile::EPBlock; ep = tile.ep, dg = tile.dg)
    seg, nm = tile.seg_host, tile.phs.nmodes
    nseg = size(seg, 2)
    (begin
        l0, np, bkq, bk = seg[SEG_ITEM, c], seg[SEG_NPAIRS, c], seg[SEG_BKQ, c], seg[SEG_BK, c]
        items = l0 + 1:l0 + bkq * bk * nm * np
        (; c, items, pairs = seg[SEG_PAIR, c] + 1:seg[SEG_PAIR, c] + np, b_kq = bkq, b_k = bk,
           kq_pt = seg[SEG_KQ_PT, c], k_pt = seg[SEG_K_PT, c], k_shared = seg[SEG_K_SHARED, c] == 1,
           ep = reshape(view(ep, items), bkq, bk, nm, np),
           dg = dg === nothing ? nothing : reshape(view(dg, items, :), bkq, bk, nm, np, 3))
    end for c in 1:nseg)
end

"""
    eachitem(tile), eachentry(tile)

Host iteration, on a `CPUBackend`, over `(l, m, n, ν, j)` of every item in order, or over
`(e, m, n, j, first_item, stride)` of every entry. The loops run over the classes outside and the bands
inside, so nothing is decoded.
"""
function eachitem(tile::EPBlock)
    nm = tile.phs.nmodes
    Iterators.flatten(
        ((first(s.items) - 1 + m + s.b_kq * ((n - 1) + s.b_k * ((ν - 1) + nm * (jj - 1))), m, n, ν,
          first(s.pairs) + jj - 1)
         for jj in 1:length(s.pairs) for ν in 1:nm for n in 1:s.b_k for m in 1:s.b_kq)
        for s in eachsegment(tile))
end

function eachentry(tile::EPBlock)
    nm = tile.phs.nmodes
    Iterators.flatten(
        (let pb = (m - 1) + s.b_kq * (n - 1), stride = s.b_kq * s.b_k
             ((first(s.items) - 1) ÷ nm + pb + stride * (jj - 1) + 1, m, n, first(s.pairs) + jj - 1,
              first(s.items) + pb + stride * nm * (jj - 1), stride)
         end
         for jj in 1:length(s.pairs) for n in 1:s.b_k for m in 1:s.b_kq)
        for s in eachsegment(tile))
end

"""
    eph_window_scatter!(g2_out, tile::EPBlock, ni_stride, i0)

Device-resident scatter for a calculator that keeps `g2` on the device (no per-batch host streaming). For
every item `l = (m, n, ν, j)` of the tile, look up the state indices `i = state_k(tile, n, j)` and
`f = state_kq(tile, m, j)`; if both are states (`> 0`), write `g2 = |ep|²/(2ω)` (`ω = phs.e[ν, j]`, formed
here, as `epstate_set_g2!` does) into the mode-fastest linear slot `lin = ν + nm·(i-i0-1) + nm·ni_stride·(f-1)`
of the flat `g2_out`.

The output buffer indexes outer-k states along `i`, and there are two ways to size it:
- **Full buffer** — holds all `n_i` outer states at once: pass `ni_stride = n_i`, `i0 = 0`.
- **Per-batch buffer** — the outer-k e-ph loop walks the outer-k points in batches; a memory-bounded
  caller keeps only the CURRENT batch's outer states on the device (device use ∝ batch, not the
  whole grid) and flushes each batch to the host. Then the buffer's i-extent is the batch size, not
  `n_i`: pass `ni_stride =` that extent and `i0 =` the batch's global-i offset, so global state `i`
  writes to local row `i - i0`.

The target `lin` indices are unique across the run (distinct k → distinct i, distinct k+q → distinct f),
so the writes never collide (no atomics needed). Generic (CPU/fallback) method; the CUDA extension
provides a one-kernel `CuArray` method, one thread per item. Called from a device-resident calculator's
`run_calculator!(calc, ::EPBlock{OuterKLoop}, ctx)` (`G2Calculator`).
"""
function eph_window_scatter!(g2_out, tile::EPBlock, ni_stride::Int, i0::Int)
    (; phs) = tile
    nm = phs.nmodes
    for s in eachsegment(tile), j in s.pairs, ν in 1:nm, n in 1:s.b_k, m in 1:s.b_kq
        i = state_k(tile, n, j)
        f = state_kq(tile, m, j)
        if i > 0 && f > 0
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = abs2(s.ep[m, n, ν, j - first(s.pairs) + 1]) * inv(2 * phs.e[ν, j])
        end
    end
    nothing
end

"""
    eph_window_scatter_reim!(re_out, im_out, tile::EPBlock, ni_stride, i0)

Complex sibling of [`eph_window_scatter!`](@ref): same lookup and same linear slot, but it writes `real(ep)`
and `imag(ep)` of the raw matrix element into two real arrays instead of `|g|²/(2ω)` into one. A calculator
needs this when the *phase* of `g` enters, not only its magnitude -- a four-channel Nambu vertex, where the
`1/(2ω)` is applied downstream, once, as the channels are combined.

Generic (CPU/fallback) method; the CUDA extension provides a one-kernel `CuArray` method. See
[`eph_window_scatter!`](@ref) for `ni_stride`/`i0` and for why the target slots never collide.
"""
function eph_window_scatter_reim!(re_out, im_out, tile::EPBlock, ni_stride::Int, i0::Int)
    (; phs) = tile
    nm = phs.nmodes
    for s in eachsegment(tile), j in s.pairs, ν in 1:nm, n in 1:s.b_k, m in 1:s.b_kq
        i = state_k(tile, n, j)
        f = state_kq(tile, m, j)
        if i > 0 && f > 0
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            ep = s.ep[m, n, ν, j - first(s.pairs) + 1]
            re_out[lin] = real(ep)
            im_out[lin] = imag(ep)
        end
    end
    nothing
end

# `eph_window_scatter!` above is used by device-resident calculators that copy g2 (e.g.
# `G2Calculator`), not by the BTE calculator. The BTE analogue
# `bte_window_accumulate!` lives next to its sole caller `BoltzmannCalculator`
# (src/boltzmann/boltzmann_calculator.jl); its CUDA method is in the extension.

"""
    gather_pair_table!(out, table, iq_kk, iks_i, iks_f)

`out[ν, i, f] = table[ν, iq_kk[iks_i[i], iks_f[f]]]` for `out` `(nm, length(iks_i), length(iks_f))`:
a per-q table expanded onto state pairs. With a calculator's phonon table `ωph`, its q index
`iq_kk` and the k indices `el_i.iks` / `el_f.iks` of its states (`G2Calculator`,
`EPElementCalculator`), `out` is the phonon frequency of every state pair, bitwise the one inside
the run's `g2`; any other per-q table (e.g. integer frequency ids) gathers the same way. One
broadcast, so the same call runs on host arrays and, with every argument on the device, on the
device.
"""
function gather_pair_table!(out, table, iq_kk, iks_i, iks_f)
    nm = size(out, 1)
    size(out) == (nm, length(iks_i), length(iks_f)) || throw(DimensionMismatch(
        "out is $(size(out)), for $(length(iks_i)) outer and $(length(iks_f)) inner states"))
    size(table, 1) == nm || throw(DimensionMismatch(
        "the table has $(size(table, 1)) rows, out $nm"))
    # Nested dot calls fuse, so the pair's q index is never materialized.
    out .= getindex.(Ref(table), reshape(1:nm, :, 1, 1),
        getindex.(Ref(iq_kk), reshape(iks_i, 1, :, 1), reshape(iks_f, 1, 1, :)))
end

"""
    dedup_phonon_frequencies(ωq, omega_cutoff; bin = omega_bin) -> (idx, ωvals)

Group a phonon-frequency array of any shape (a per-pair `ωq[ν, i, f]`, or a phonon table
`ωph[ν, iq]` as [`phonon_index`](@ref) passes it) into its distinct values above
`omega_cutoff`. Returns `idx` (same shape as `ωq`, `Int32`) and `ωvals`, with the
convention that `ωvals[1] = 0` is a reserved sentinel and `ωvals[2:]` are the distinct
frequencies. Modes below the cutoff map to `idx = 1` (the sentinel), so a propagator
table's first row can be zeroed and a per-slice interaction build needs no cutoff branch.
Every other `idx` points at the matching `ωvals` entry (`≥ 2`).

Frequencies are grouped by the bin `floor(Int64, ω / bin)`, so symmetry-equivalent q points, whose
frequencies differ in the last bits, share an entry; `ωvals` holds the bin centre
`(k + 1/2) * bin`. The centre is a function of the bin alone, so it does not depend on the
order of the input or on which array is grouped, and the ids are in first-occurrence order.
`|ω - ωvals[idx]| <= bin/2 + eps(ω)`: this is a deliberate shift of the frequency, not
round-off, and `ωq` itself (and any `g2` built from it) is not touched. A symmetric cluster
straddling a bin edge stays two entries. `bin = 0` groups by exact equality and stores ω.
`maximum(abs, ωq) / bin` must stay below `2^52` so that the key and its bin centre are exact.

Threaded in three phases: (1) parallel chunk-local grouping (local ids written into `idx`, offset
by the sentinel), (2) serial in-order merge of the small chunk dictionaries into the global
`ωvals`, (3) parallel remap of `idx` to the global ids. Because the chunks are CONTIGUOUS index
ranges processed in order, the merged first-occurrence ordering, and therefore `idx` and
`ωvals`, is IDENTICAL to the serial version's.
"""
function dedup_phonon_frequencies(ωq::AbstractArray{FT}, omega_cutoff; bin = omega_bin) where {FT}
    bin >= 0 || throw(ArgumentError("bin must be non-negative, got $bin"))
    if iszero(bin)
        # Exact path: the key is ω itself and the stored value is ω.
        return _group_phonon_keys(ωq, ωq, omega_cutoff, bin)
    else
        # Binned path: the key is the integer bin index.
        maximum(abs, ωq; init = zero(FT)) / bin < 2.0^52 || throw(ArgumentError(
            "ω / bin reaches 2^52 (max ω $(maximum(abs, ωq)), bin $bin): the bin centre is not exact"))
        return _group_phonon_keys(ωq, floor.(Int64, Float64.(ωq) ./ bin), omega_cutoff, bin)
    end
end

# Three-phase grouping of `ωq` by `keys` (same shape); `bin = 0` stores the key, else its bin centre.
function _group_phonon_keys(ωq::AbstractArray{FT}, keys::AbstractArray{K}, omega_cutoff, bin) where {FT,K}
    idx = ones(Int32, size(ωq))      # default 1 => dropped (below cutoff)
    ωqv = vec(ωq)
    keysv = vec(keys)
    idxv = vec(idx)
    n = length(ωqv)
    nch = clamp(cld(n, 1_000_000), 1, Threads.nthreads())
    bounds = round.(Int, range(0, n; length = nch + 1))
    chunk_keys = Vector{Vector{K}}(undef, nch)
    # (1) chunk-local grouping: idx holds LOCAL first-occurrence ids, shifted by +1 so the
    # dropped-mode sentinel (idx = 1) stays unambiguous.
    Threads.@threads for c in 1:nch
        d = Dict{K,Int32}()
        ks = K[]
        @inbounds for I in (bounds[c]+1):bounds[c+1]
            ωqv[I] < omega_cutoff && continue  # leave idx[I] = 1 (sentinel)
            key = keysv[I]
            idxv[I] = get!(d, key) do
                push!(ks, key)
                Int32(length(ks) + 1)   # local id + sentinel shift, always >= 2
            end
        end
        chunk_keys[c] = ks
    end
    # (2) in-order merge -> global ωvals (first-occurrence order == the serial pass's).
    kindex = Dict{K,Int32}()
    ωvals = FT[zero(FT)]              # row 1: reserved zero sentinel (dropped modes)
    remaps = Vector{Vector{Int32}}(undef, nch)
    for c in 1:nch
        ks = chunk_keys[c]
        remap = Vector{Int32}(undef, length(ks))
        for (l, key) in enumerate(ks)
            remap[l] = get!(kindex, key) do
                push!(ωvals, iszero(bin) ? key : FT((key + 0.5) * bin))
                Int32(length(ωvals))  # position in ωvals, always >= 2
            end
        end
        remaps[c] = remap
    end
    # (3) local ids -> global ids.
    Threads.@threads for c in 1:nch
        remap = remaps[c]
        @inbounds for I in (bounds[c]+1):bounds[c+1]
            l = idxv[I]
            l > 1 && (idxv[I] = remap[l - 1])
        end
    end
    idx, ωvals
end

"""
    phonon_index(ωph, iq_kk, iks_i, iks_f, omega_cutoff; bin = omega_bin) -> (idx, ωvals)

The `(idx, ωvals)` of [`dedup_phonon_frequencies`](@ref) for every state pair `[ν, i, f]`, from a
calculator's phonon table `ωph[ν, iq]` and q index `iq_kk[ik, ikq]` (see `G2Calculator`), for outer
states at the k indices `iks_i` and inner states at `iks_f`. The table is deduplicated once and the
per-q ids are gathered onto the pairs, so `ωvals[idx[ν, i, f]]` is the pair's frequency (to within
`bin/2`), or the dropped-mode sentinel; the ids are numbered in table order.

The one guard on the table's shape is here: a q index past the table is an `ArgumentError`, and a
k index past `iq_kk` a `BoundsError` of the gather.
"""
function phonon_index(ωph::AbstractMatrix, iq_kk, iks_i, iks_f, omega_cutoff; bin = omega_bin)
    nm, nq = size(ωph)
    maximum(iq_kk; init = 0) <= nq || throw(ArgumentError(
        "the q index reaches q point $(maximum(iq_kk)), the phonon table holds $nq"))
    idq, ωvals = dedup_phonon_frequencies(ωph, omega_cutoff; bin)
    idx = Array{Int32,3}(undef, nm, length(iks_i), length(iks_f))
    gather_pair_table!(idx, idq, iq_kk, iks_i, iks_f)
    idx, ωvals
end
