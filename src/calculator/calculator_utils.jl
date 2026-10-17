# Utilities that calculators call from their `run_calculator!` methods, on any backend. Not used inside ElectronPhonon.jl
# itself and not exported; downstream calculators reach them as `ElectronPhonon.<name>`. The
# backend/device primitives (`to_device`, `batched_gemm!`, …) live in `common/gpu_utils.jl`; this
# file holds the higher-level, calculator-facing helpers built on top of them.

# The extents `(nbandkq, nbandk, nm, npairs)` of a scatter's block values `vals`, after checking
# that the other block arrays agree: `ikqs` of length `npairs` and index maps with at least the
# block's rows `nbandk` / `nbandkq` (a block may be narrower than the containers' box). The scatters
# loop over these extents unchecked (the device kernels with no bounds checks at all), so a
# mismatched array fails here instead.
function _scatter_extents(vals, ikqs, imap_i_col, imap_f)
    nbandkq, nbandk, nm, npairs = size(vals)
    length(ikqs) == npairs ||
        throw(DimensionMismatch("$(length(ikqs)) k+q indices for $npairs block points"))
    length(imap_i_col) >= nbandk ||
        throw(DimensionMismatch("the outer index map has $(length(imap_i_col)) rows, the block $nbandk"))
    size(imap_f, 1) >= nbandkq ||
        throw(DimensionMismatch("the inner index map has $(size(imap_f, 1)) rows, the block $nbandkq"))
    (nbandkq, nbandk, nm, npairs)
end

"""
    eph_window_scatter!(g2_out, g2vals, imap_i_col, imap_f, ikqs, ni_stride, i0)

Device-resident scatter for a calculator that keeps `g2` on the device (no per-batch host
streaming). For every `(m, n, ν, ipair)` entry of `g2vals` `(nbandkq, nbandk, nm, npairs)`, look
up the state indices `i = imap_i_col[n]` (in-window outer-k state) and `f = imap_f[m, ikqs[j]]`
(in-window k+q state); if both are in-window (`> 0`), write the value into the mode-fastest linear
slot `lin = ν + nm·(i-i0-1) + nm·ni_stride·(f-1)` of the flat `g2_out`. The extents are those of
`g2vals`; `ikqs` (length `npairs`) and the index maps (`nbandk` / `nbandkq` rows) must agree with
them, or a `DimensionMismatch` is thrown.

The output buffer indexes outer-k states along `i`, and there are two ways to size it:
- **Full buffer** — holds all `n_i` outer states at once: pass `ni_stride = n_i`, `i0 = 0`.
- **Per-batch buffer** — the outer-k e-ph loop walks the outer-k points in batches; a memory-bounded
  caller keeps only the CURRENT batch's outer states on the device (device use ∝ batch, not the
  whole grid) and flushes each batch to the host. Then the buffer's i-extent is the batch size, not
  `n_i`: pass `ni_stride =` that extent and `i0 =` the batch's global-i offset, so global state `i`
  writes to local row `i - i0`.

The target `lin` indices are unique across the run (distinct k → distinct i, distinct k+q →
distinct f), so the writes never collide (no atomics needed). Generic (CPU/fallback) method; the
CUDA extension provides a one-kernel `CuArray` method.

Called from a device-resident calculator's `run_calculator!(calc, ::EPBlock{OuterKLoop}, ctx)`
(`G2Calculator`) to scatter each block's `g2` into its window-mapped device output, with
`imap_i_col` / `imap_f` in the containers' box coordinates (see `_indmap_to_device`). The
"Scatter round-trip" testset of `test/test_gpu.jl` checks the non-collision invariant and the
CPU/CUDA agreement.
"""
function eph_window_scatter!(g2_out, g2vals, imap_i_col, imap_f, ikqs, ni_stride::Int, i0::Int)
    nbandkq, nbandk, nm, npairs = _scatter_extents(g2vals, ikqs, imap_i_col, imap_f)
    @inbounds for ipair in 1:npairs, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[ipair]]
        if i > 0 && f > 0
            # global outer state `i` lands at local row `i - i0` (see docstring for i0/ni_stride).
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, ipair]
        end
    end
    nothing
end

"""
    eph_window_scatter_reim!(re_out, im_out, epvals, imap_i_col, imap_f, ikqs, ni_stride, i0)

Complex sibling of [`eph_window_scatter!`](@ref): same window lookup and same linear slot, but it
writes `real(ep)` and `imag(ep)` of the raw matrix element `epvals` `(nbandkq, nbandk, nm,
npairs)` into two real arrays instead of `|g|²/(2ω)` into one. A calculator needs this when the
*phase* of `g` enters, not only its magnitude -- a four-channel Nambu vertex, where the `1/(2ω)` is
applied downstream, once, as the channels are combined.

Generic (CPU/fallback) method; the CUDA extension provides a one-kernel `CuArray` method. See
[`eph_window_scatter!`](@ref) for `ni_stride`/`i0` and for why the target slots never collide.
"""
function eph_window_scatter_reim!(re_out, im_out, epvals, imap_i_col, imap_f, ikqs,
                                  ni_stride::Int, i0::Int)
    nbandkq, nbandk, nm, npairs = _scatter_extents(epvals, ikqs, imap_i_col, imap_f)
    for ipair in 1:npairs, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[ipair]]
        if i > 0 && f > 0
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            ep = epvals[m, n, ν, ipair]
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
`maximum(abs, ωq) / bin` must stay below `2^53` so that the key is exact.

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
        maximum(abs, ωq; init = zero(FT)) / bin < 2.0^53 || throw(ArgumentError(
            "ω / bin reaches 2^53 (max ω $(maximum(abs, ωq)), bin $bin): the bin key is not exact"))
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
