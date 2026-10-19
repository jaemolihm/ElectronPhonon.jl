# Writing your own calculator

A **calculator** computes a physical property during one pass of an e-ph driver
(`run_eph_over_k_and_kq` and `run_eph_over_k_and_q`, outer loop over k, or `run_eph_over_q_and_k`,
outer loop over q). The driver builds the electron and phonon states and the e-ph matrix elements
and hands them to each calculator one **tile** at a time, an [`EPBlock`](@ref) of one outer point
with a tile of inner points (a flat list of the e-ph matrix elements, see below), together with a **context** of the loop order (`OuterKContext` or
`OuterQContext`). You subtype
`ElectronPhonon.AbstractCalculator` and implement a few methods; the same methods run on the CPU and
on a GPU when they are written with broadcasts and `alloc(backend, …)`.

The authoritative reference is the docstrings in `src/calculator/AbstractCalculator.jl`. This guide
is the tutorial; its examples are executed verbatim by `test/test_calculator_guide.jl`, so they
cannot rot.

## The surface

- `supports(calc, ::Type{OuterKLoop})` / `supports(calc, ::Type{OuterQLoop})` — the loop orders the
  calculator handles (default `false`; pass the type, not an instance). The driver refuses a
  calculator that does not support its order.
- `required_el_quantities(calc)`, `required_ph_quantities(calc)` — the state quantities it reads
  beyond the energies `e` and eigenvectors `u`, which the loop always provides on both electron sides
  and on the phonons, as field names of `BatchedElectronState` (`:vdiag`, `:v`, `:rbar`) and
  `BatchedPhononState` (`:vdiag`, …). Default: none.
- `setup_calculator!(calc, backend, els_k, els_kq, phs; order, sel_k, sel_kq, nchunks_threads,
  n_outer_batch, n_inner_tile, verbosity)` — once, before the loop (see below). `order` is
  `OuterKLoop()` or `OuterQLoop()`.
- `run_calculator!(calc, tile::EPBlock{OuterKLoop}, ctx)` (or `{OuterQLoop}`) — once per tile.
- `calculator_begin_batch!(calc, ctx)` / `calculator_end_batch!(calc, ctx)` — around every outer batch
  (`ctx.iks_batch` of an `OuterKContext` or `ctx.iqs_batch` of an `OuterQContext`, the outer indices
  of the batch). There is no default: define both, even as `= nothing`.
- `postprocess_calculator!(calc; kwargs...)` — once, after the loop.
- Optionally `calculator_bytes(calc, ::Type{<:EPBlock{Loop}}; nw, nmodes, nband_max_k, nband_max_kq,
  els_k, els_kq, phs, nchunks_threads) -> (; persistent, per_outer, per_pair)`, the device
  bytes the calculator allocates, so the loop sizes its tiles to the free memory. The loop counts
  `persistent` once, `per_outer` once per outer point of a batch and `per_pair` once per inner point
  of a tile and per thread chunk; a per-outer buffer the calculator holds per chunk multiplies by
  `nchunks_threads` itself. `els_k`, `els_kq`, `phs` are `nothing` in `estimate_device_memory`. Accept
  `kwargs...` for keywords added later.

## Minimal example: CPU, outer k, one thread

This calculator sums `wtq · |g|²/(2ω)` over the in-window bands, the modes and the q points, for each
k point (`g2_per_k`), and that sum weighted by `wtk` over the k points (`g2_avg`), which shows a
reduction over both momenta. It is as small as a calculator gets:

- it supports only the outer-k loop, so it runs with `run_eph_over_k_and_kq` and
  `run_eph_over_k_and_q`, but not with `run_eph_over_q_and_k`, which refuses it;
- it runs only on the CPU: its `run_calculator!` is a plain loop over the items of the tile;
- it refuses more than one thread chunk, so the tiles run one after another and it adds straight
  into its results. The run passes `nchunks_threads = 1`.

With `epw_folder` the folder of an EPW run:

<!-- doc-minimal:begin -->
```julia
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, EPBlock, omega_acoustic

# For each k: Σ over (q, m, n, ν) of wtq |ep[m, n, ν]|² / (2ω_ν(q)), with m and n the bands of
# k+q and k inside their windows. Modes with ω < omega_acoustic are skipped: at Γ the acoustic ω
# is ~0, where 1/(2ω) only amplifies roundoff.
mutable struct MinimalG2Calculator <: AbstractCalculator
    g2_per_k :: Vector{Float64}   # indexed by k point
    g2_avg   :: Float64           # Σ_k wtk g2_per_k[k]
    MinimalG2Calculator() = new(Float64[], 0.0)
end

# The outer-k drivers only: `run_eph_over_k_and_kq` and `run_eph_over_k_and_q`.
ElectronPhonon.supports(::MinimalG2Calculator, ::Type{OuterKLoop}) = true

# Once, before the loop. With more than one thread chunk, tiles of the same k would run at the same
# time and race on `g2_per_k[ik]` and `g2_avg`; the complete example below handles that.
function ElectronPhonon.setup_calculator!(c::MinimalG2Calculator, backend, els_k, els_kq, phs;
        nchunks_threads, kwargs...)
    nchunks_threads == 1 || throw(ArgumentError("MinimalG2Calculator needs nchunks_threads = 1"))
    c.g2_per_k = zeros(els_k.nk)
    c.g2_avg = 0.0
    c
end

# No per-batch work.
ElectronPhonon.calculator_begin_batch!(c::MinimalG2Calculator, ctx) = c
ElectronPhonon.calculator_end_batch!(c::MinimalG2Calculator, ctx) = c

# One tile: the outer k `tile.ik` with a tile of q points. An item `(l, m, n, ν, j)` is element `ep[l]`:
# band `m` at k+q and `n` at k, mode `ν`, of pair `j` = (k, q_j). Only the bands inside the windows are
# items. The sum over q is over the pairs of the tile and over the tiles of the same k; the sum over k
# is weighted by the outer k's weight `tile.wtk`.
function ElectronPhonon.run_calculator!(c::MinimalG2Calculator, tile::EPBlock{OuterKLoop}, ctx)
    (; ep, phs, wtq) = tile
    s = 0.0
    for (l, _, _, ν, j) in ElectronPhonon.eachitem(tile)
        ω = phs.e[ν, j]
        ω < omega_acoustic && continue
        s += wtq[j] * abs2(ep[l]) / (2ω)
    end
    c.g2_per_k[tile.ik] += s
    c.g2_avg += tile.wtk * s
    c
end

ElectronPhonon.postprocess_calculator!(c::MinimalG2Calculator; kwargs...) = c

# Run it: outer k on an 8³ grid (reduced by symmetry), inner k+q on the 8³ grid, one thread chunk.
model = load_model_from_epw_new(epw_folder, "temp", "pb"; epmat_outer_momentum = "el")
calc_minimal = MinimalG2Calculator()
out_minimal = ElectronPhonon.run_eph_over_k_and_kq(model, (8, 8, 8), (8, 8, 8);
    calculators = [calc_minimal], nchunks_threads = 1)
calc_minimal.g2_per_k    # one sum per k point of out_minimal.kpts
calc_minimal.g2_avg
```
<!-- doc-minimal:end -->

## Complete example: both loop orders and the GPU

The minimal example's `g2_per_k` and `g2_avg`, extended to

- any number of CPU thread chunks: every sum that tiles of different chunks add to goes to a
  per-chunk buffer (`g2_per_k_buffer`, `g2_avg_buffer`), reduced in `calculator_end_batch!` and
  `postprocess_calculator!`;
- both loop orders: `run_calculator!` and the batch brackets get an outer-q method;
- the GPU: the same methods, which read the tile one band class at a time with array broadcasts and
  reductions instead of item by item, and so run on either backend.

<!-- doc-example:begin -->
```julia
using ElectronPhonon
using LinearAlgebra: dot
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPBlock, OuterKContext,
    OuterQContext, alloc, omega_acoustic, eachsegment

# For each k: Σ over (q, m, n, ν) of wtq |ep[m, n, ν]|² / (2ω_ν(q)), with m and n the bands of
# k+q and k inside their windows. Modes with ω < omega_acoustic are skipped, as in the library's
# calculators: at Γ the acoustic ω is ~0, where 1/(2ω) only amplifies roundoff.
mutable struct EphG2SumCalculator <: AbstractCalculator
    g2_per_k        :: Vector{Float64}   # the result, indexed by k point
    g2_per_k_buffer :: Matrix{Float64}   # (chunk, outer k of the batch); outer-k runs only

    # Σ_k wtk g2_per_k[k], and its partial sum of each chunk.
    g2_avg          :: Float64
    g2_avg_buffer   :: Vector{Float64}

    # Scratch of each chunk, sized to one tile on the run's backend: the summand of the items, and the
    # per-pair sums of an outer-q tile on the backend and on the host.
    g2_scratch    :: Vector{Any}
    pair_sums_dev :: Vector{Any}
    pair_sums     :: Vector{Vector{Float64}}
    EphG2SumCalculator() = new(Float64[], zeros(0, 0), 0.0, Float64[], [], [], [])
end

ElectronPhonon.supports(::EphG2SumCalculator, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::EphG2SumCalculator, ::Type{OuterQLoop}) = true
# The loop always provides `e`, `u` and the e-ph matrix elements, which is all this calculator reads,
# so it defines no `required_el_quantities` / `required_ph_quantities`.

# Buffers are sized here, from the widths the loop chose; `run_calculator!` allocates none.
function ElectronPhonon.setup_calculator!(c::EphG2SumCalculator, backend, els_k, els_kq, phs;
        order, nchunks_threads, n_outer_batch, n_inner_tile, kwargs...)
    # The result: one sum per k point of the run.
    c.g2_per_k = zeros(els_k.nk)

    if order isa OuterKLoop
        # Outer k: the CPU thread chunks add to the sum of the same outer k concurrently, so each
        # chunk has its own row, one column per outer k of a batch. `calculator_end_batch!` sums
        # the rows. Outer q adds straight into `g2_per_k` and needs no buffer.
        c.g2_per_k_buffer = zeros(nchunks_threads, n_outer_batch)
    end

    # The reductions of `g2_avg`. The sum over q happens inside each tile, and across the tiles
    # of an outer k, giving one number per k. The sum over k is weighted by `wtk`: the outer k's
    # weight under OuterKLoop (`tile.wtk` a scalar), each pair's k weight under OuterQLoop
    # (`tile.wtk` a vector). Every tile of every chunk adds to the same scalar, so each chunk adds
    # to its own entry of `g2_avg_buffer`, summed in `postprocess_calculator!`.
    c.g2_avg = 0.0
    c.g2_avg_buffer = zeros(nchunks_threads)

    # Scratch per chunk, sized to the largest tile: at most `nband_max_kq * nband_max_k * nmodes` items
    # per pair, `n_inner_tile` pairs. The k+q box is the container's, or at most `nw` when k+q is solved
    # per tile. On a device `ctx.chunk` is always 1.
    nband_max_kq = els_kq === nothing ? els_k.nw : els_kq.nband_max
    n_items_max = nband_max_kq * els_k.nband_max * phs.nmodes * n_inner_tile
    c.g2_scratch = [alloc(backend, Float64, n_items_max) for _ in 1:nchunks_threads]
    c.pair_sums_dev = [alloc(backend, Float64, n_inner_tile) for _ in 1:nchunks_threads]
    c.pair_sums = [zeros(n_inner_tile) for _ in 1:nchunks_threads]
    c
end

# Before each outer-k batch: clear the outer-k buffer, which holds the sums of the current batch only.
function ElectronPhonon.calculator_begin_batch!(c::EphG2SumCalculator, ::OuterKContext)
    fill!(c.g2_per_k_buffer, 0)
    c
end

# Outer q has no per-batch buffer.
ElectronPhonon.calculator_begin_batch!(c::EphG2SumCalculator, ::OuterQContext) = c

# `run_calculator!` receives one tile: one outer point with a tile of inner points. Its e-ph matrix
# `tile.ep` is a flat list of items (band `m` at k+q, band `n` at k, mode `ν`, pair `j`), stored one
# band class at a time: the pairs whose k and k+q have the same band counts (windows) are one segment of
# the list, a `(b_kq, b_k, nmodes, n_pairs)` array. Only bands inside the windows are items, so there is
# no padding to mask. `eachsegment(tile)` iterates the segments on the host; each gives the `ep` of the
# class as such an array (a device array on a GPU) and its pairs `s.pairs`, which are the columns of
# the tile's per-pair arrays:
#
#               outer             pair j                             weight of pair j
#   OuterKLoop  k: tile.ik        k+q_j: tile.ikq[j], q_j: phs[:, j]   tile.wtq[j]
#   OuterQLoop  (none)            k_j: tile.ik[j], q_j: phs[:, j]      tile.wtq[j]
#
# A GPU cannot read a device array one element at a time (an error under `CUDA.allowscalar(false)`),
# so the methods below form the summand of a whole class with one broadcast and reduce it with `sum`;
# on the CPU the same code runs on host arrays.

# The summand of one band class: g2[m, n, ν, j] = w_j |ep[m, n, ν, j]|² / (2ω_ν), an array shaped like
# the class's `ep`, written into the scratch `g2_scratch`. The factors are reshaped so that their axes
# line up with ep[m, n, ν, j] (a reshape copies no data), and one fused broadcast forms it. Modes below
# omega_acoustic are set to 0 with `ifelse`, not by multiplying with a mask, which could give NaN.
@views function class_g2!(c::EphG2SumCalculator, tile::EPBlock, s, chunk)
    np = length(s.pairs)
    ω = reshape(tile.phs.e[:, s.pairs], 1, 1, tile.phs.nmodes, np)
    w = reshape(tile.wtq[s.pairs], 1, 1, 1, np)
    g2 = reshape(c.g2_scratch[chunk][1:length(s.ep)], size(s.ep))
    g2 .= ifelse.(ω .>= omega_acoustic, w .* abs2.(s.ep) ./ (2 .* ω), 0.0)
end

# Outer k. The whole tile adds to one number, the sum of the outer k. Tiles of other chunks add to the
# same k concurrently, so it goes to this chunk's row of `g2_per_k_buffer`, summed in
# `calculator_end_batch!`. The column is the position of the outer k in the batch: `tile.ik` is its index
# in the run, and the batch holds the outer k points `ctx.iks_batch`.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, tile::EPBlock{OuterKLoop},
                                        ctx::OuterKContext)
    s_tile = 0.0
    for s in eachsegment(tile)
        s_tile += sum(class_g2!(c, tile, s, ctx.chunk))
    end
    ik_batch = tile.ik - first(ctx.iks_batch) + 1   # position of the outer k in this batch
    c.g2_per_k_buffer[ctx.chunk, ik_batch] += s_tile
    c.g2_avg_buffer[ctx.chunk] += tile.wtk * s_tile
    c
end

# Outer q. Each pair adds to its own k point, `tile.ik[j]`. No other tile writes these k at the same time
# (the chunks split the k points, and one chunk runs its tiles one after another), so the sums go
# straight into the result. The same k appears in the pairs of different q. Per class:
#
# 1. `class_g2!` forms the summand of its items on the backend.
# 2. `sum!` reduces it over (m, n, ν) into one sum per pair, still on the backend. `sum!` reduces over
#    the axes where its output has extent 1.
# 3. The sums are copied to the host in one transfer and added to their k points, `tile.ik` copied to the
#    host. A k point can repeat in that list, so the sums are added one by one, not by one broadcast.
# 4. The k-weighted sum for `g2_avg` is a `dot` of the sums with the k weights `tile.wtk` on the
#    backend, one host number.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, tile::EPBlock{OuterQLoop},
                                        ctx::OuterQContext)
    for s in eachsegment(tile)
        g2 = class_g2!(c, tile, s, ctx.chunk)
        np = length(s.pairs)
        sums_dev = view(c.pair_sums_dev[ctx.chunk], 1:np)
        sum!(reshape(sums_dev, 1, 1, 1, np), g2)
        copyto!(c.pair_sums[ctx.chunk], 1, sums_dev, 1, np)
        for (jj, k) in enumerate(Array(tile.ik)[s.pairs])
            c.g2_per_k[k] += c.pair_sums[ctx.chunk][jj]
        end
        c.g2_avg_buffer[ctx.chunk] += dot(sums_dev, view(tile.wtk, s.pairs))
    end
    c
end

# After an outer-k batch: sum the chunks' rows into the k points of the batch.
@views function ElectronPhonon.calculator_end_batch!(c::EphG2SumCalculator, ctx::OuterKContext)
    c.g2_per_k[ctx.iks_batch] .= vec(sum(c.g2_per_k_buffer[:, 1:length(ctx.iks_batch)]; dims = 1))
    c
end

# Outer q: `run_calculator!` already added every pair to its k point; nothing to reduce.
ElectronPhonon.calculator_end_batch!(c::EphG2SumCalculator, ::OuterQContext) = c

# After the loop: sum the chunks' partial sums of `g2_avg`.
function ElectronPhonon.postprocess_calculator!(c::EphG2SumCalculator; kwargs...)
    c.g2_avg = sum(c.g2_avg_buffer)
    c
end
```
<!-- doc-example:end -->

## Run it with a driver

The drivers are public but not exported, so they are called as `ElectronPhonon.<name>`. With
`epw_folder` the folder of an EPW run:

<!-- doc-driver:begin -->
```julia
using ElectronPhonon.units: eV

# Pb from an EPW run, stored for the outer-k loops. Only the bands within 0.5 eV of the Fermi level
# are kept, on both electron sides.
model = load_model_from_epw_new(epw_folder, "temp", "pb"; epmat_outer_momentum = "el")
μ = 11.68eV
window = (μ - 0.5eV, μ + 0.5eV)

# Outer k on a k grid, inner k+q on a k+q grid. `symmetry = model.symmetry` (the default) reduces
# the outer k to the irreducible wedge: `out.kpts` holds the irreducible k points in the window, with
# their weights, and `calc.g2_per_k` one sum per point of `out.kpts`.
calc = EphG2SumCalculator()
out = ElectronPhonon.run_eph_over_k_and_kq(model, (8, 8, 8), (8, 8, 8);
    calculators = [calc], window_k = window, window_kq = window)

# Outer k on any list of k points, here a line from Γ to X, with inner q on a grid; k+q is solved
# per tile. A k point with no band in `window_k` is dropped, so `out_line.kpts` lists the k points
# that were run.
kpts_line = Kpoints([Vec3(x, 0.0, x) for x in range(0, 0.5, length = 11)])
calc_line = EphG2SumCalculator()
out_line = ElectronPhonon.run_eph_over_k_and_q(model, kpts_line, (8, 8, 8);
    calculators = [calc_line], window_k = window, window_kq = window)

# Outer q, from the model stored for the outer-q loop. Here `symmetry` would reduce the outer q
# points. That is exact only for a sum over k, and this calculator's result is per k, so the run
# keeps the full q grid with `symmetry = nothing`.
model_ph = load_model_from_epw_new(epw_folder, "temp", "pb"; epmat_outer_momentum = "ph")
calc_q = EphG2SumCalculator()
out_q = ElectronPhonon.run_eph_over_q_and_k(model_ph, (8, 8, 8), (8, 8, 8);
    calculators = [calc_q], symmetry = nothing, window_k = window, window_kq = window)

# The two grid runs agree on the k-weighted sum, up to the symmetry of the interpolated g2.
calc.g2_avg ≈ calc_q.g2_avg
```
<!-- doc-driver:end -->

The same runs on a GPU take `backend = ElectronPhonon.gpu_backend()` (with CUDA.jl loaded); the
calculator's methods then run on device arrays. For example, the outer-k grid run above:

```julia
using CUDA
calc_gpu = EphG2SumCalculator()
ElectronPhonon.run_eph_over_k_and_kq(model, (8, 8, 8), (8, 8, 8);
    calculators = [calc_gpu], window_k = window, window_kq = window,
    backend = ElectronPhonon.gpu_backend())
calc_gpu.g2_avg ≈ calc.g2_avg
```

## Run one k, q pair yourself

The engine stages used by the drivers are also public (unexported) API. An engine prepares the
states and reusable buffers; `stage2!` returns the complete tile (an `EPBlock`), including polar corrections.
There is no separate single-point calculation path or tile-getter function. With the calculator
defined above and a model loaded with `epmat_outer_momentum = "el"`:

<!-- doc-single-pair:begin -->
```julia
using ElectronPhonon: OuterKEngine, stage1!, stage2!, OuterKContext,
    setup_calculator!, calculator_begin_batch!, run_calculator!, calculator_end_batch!, postprocess_calculator!

calc = EphG2SumCalculator()
kpts = Kpoints(Vec3(0.25, 0.25, 0.25))   # one k point
qpts = Kpoints(Vec3(0.1, 0.1, 0.1))      # one q point

# The states at k, k + q and q, and the e-ph buffers. Construction runs no calculator hook.
eng = OuterKEngine(model, kpts, qpts; calculators = [calc], verbosity = 0)

# Size the calculator's buffers to the engine's states and widths, as a driver does.
setup_calculator!(calc, eng.backend, eng.els_k, eng.els_kq, eng.phs; order = OuterKLoop(),
    eng.sel_k, eng.sel_kq, nchunks_threads = length(eng.tiles),
    eng.n_outer_batch, eng.n_inner_tile, verbosity = 0)

# Stage 1 for the outer k points 1:1, and the calculator context of that batch.
stage1!(eng, 1:1)
ctx = OuterKContext(eng)

# Open the batch: the calculator clears its per-batch buffers.
calculator_begin_batch!(calc, ctx)

# Stage 2 for outer k 1 and inner q points 1:1: the tile, or `nothing` if the pair is skipped.
tile = stage2!(eng, 1, 1:1)
tile === nothing || run_calculator!(calc, tile, ctx)

# Close the batch: the calculator reduces its per-batch buffers into the result (here the
# per-chunk sums of the outer k into `g2_per_k`).
calculator_end_batch!(calc, ctx)

postprocess_calculator!(calc; eng.qpts, symmetry = nothing)
calc.g2_per_k
```
<!-- doc-single-pair:end -->

`OuterKEngine` defaults to inner q points, solving k+q within each tile. For a resident k+q grid,
pass `inner_loop_kq = true` and that grid as the third argument. A model loaded with
`epmat_outer_momentum = "ph"` instead uses `OuterQEngine(model, kpts, qpts; ...)`, with q as
the outer index and k as the inner range. The same stage calls work, with `OuterQContext(eng)`.

Pass your calculator to the engine constructor so its requested quantities and memory budget are
included; construction does **not** call setup or any lifecycle hook. You can also inspect matrix
elements without a calculator and request extra fields with `el_quantities` / `ph_quantities`.
Use `backend = ElectronPhonon.gpu_backend()` for device buffers and
`ElectronPhonon.synchronize(eng.backend)` before timing.

Point indices are into the engine's selected `eng.kpts`, `eng.kqpts` and `eng.qpts`, not necessarily
the original lists: energy windows and symmetry can change the selection. `stage1!` records the
current outer range; `stage2!(eng, ...)` requires its outer index to belong to it and returns the tile,
or `nothing` when every pair is skipped. The tile of a call borrows reusable storage: consume
it before the next stage call on that chunk, or copy the arrays to retain them. Different CPU chunks have
independent writable storage; use matching `chunk` in `stage2!` and `OuterKContext(eng; chunk)` /
`OuterQContext(eng; chunk)`. A context can also be built before the batch's stage 1, as the drivers
do, by passing the batch: `OuterKContext(eng; iks_batch = 1:1)`.

## The tile

An `EPBlock` is the e-ph matrix of one outer point with a tile of inner points ("pairs": `j` numbers
them), as a flat list of items `(m, n, ν, j)`: band `m` at k+q, band `n` at k, phonon mode `ν`, pair `j`.
The pairs are split by band class `(b_kq, b_k)`, the number of window bands at k+q and at k, and the
items of a class are stored back to back (`m` fastest, then `n`, `ν`, pair) in `tile.ep`, so the list
holds only real bands and no padding is ever an item. The classes differ in their strides, so decode an
item with `item_indices(tile, l)` (also inside a device kernel), or an entry (a band pair of a pair with
the modes summed inside) with `entry_indices(tile, e)`; on the host loop with `eachitem(tile)`,
`eachentry(tile)` or, one band class at a time, `eachsegment(tile)`, which gives each class's `ep` as a
`(b_kq, b_k, nmodes, n_pairs)` array. Read the states of an item with `state_k(tile, n, j)` and
`state_kq(tile, m, j)` (the state index in the run's selection, 0 if the band is not a state) and with
`e_k`, `e_kq`, `u_k`, `u_kq`; the raw `tile.els_k` / `tile.els_kq` hold class segments at different strides.

Under `OuterKLoop`: `ik::Int`, `wtk::Float64`, `ikq` the k+q indices of the pairs on the run's backend
(`nothing` under `run_eph_over_k_and_q`, where the k+q states are solved per tile), `iq` the q indices,
`wtq` the k+q weights, `phs.e[ν, j]` the frequencies of the pairs; `dg` is `(n_items, 3)` when the run
asks for the covariant derivative. Under `OuterQLoop` the tile holds the `(q, k)` pairs of one or several
q: `phs` (the frequencies, not the eigenvectors `u`: the e-ph matrix is already in the phonon basis, so read
the run's phonons at `iq` and do not list `:u` in `required_ph_quantities`, which an outer-q run refuses),
`iq`, `ik`, `wtq` and `xq` are per pair (vectors on the backend, `xq` on the host), so the same k can appear in several pairs, and
`ikq === nothing` when k+q is solved per tile. `ep` is the e-ph matrix
before the `1/(2ω)`; a calculator that needs `|g|²/(2ω)` forms it. The field list is in the `EPBlock`
docstring.

## The threading and batch contract

- **Any batch width.** The loop chooses `n_outer_batch` and `n_inner_tile` at runtime and passes
  them to `setup_calculator!`; size per-batch buffers to the first and per-tile scratch to the
  second. Every batch length ≥ 1 (`length(ctx.iks_batch)`, `length(ctx.iqs_batch)`) must work.
- **Writes.** Writes indexed by an inner-tile point are disjoint across the tiles that run at the same
  time (an outer-q tile can repeat a k point in pairs of different q; reduce those within the tile). Every other write
  (indexed by the outer point, or a reduction over the inner points) goes to a per-`ctx.chunk`
  partial, reduced in `calculator_end_batch!`, as in the example. Per-tile scratch is per chunk too: on
  the CPU the tiles of different chunks run concurrently. `ctx.chunk` is 1 on a device. Data of the
  tile's shared side (the outer k's energies under `OuterKLoop`, the q's frequencies under
  `OuterQLoop`) is not an inner-tile write either: concurrent tiles write the same slot, so record
  it once, in the brackets or at setup.
- **Brackets** run serially, once around every outer batch, on every backend.

## What `setup_calculator!` receives

`els_k`, `els_kq`, `phs` are the run's containers (`BatchedElectronState`, `BatchedPhononState`) on
the run's `backend`, holding the requested quantities; `els_kq` is `nothing` when k+q is solved per
tile.
`sel_k`, `sel_kq` are the `FilteredBandStates` selections they were built from: the selected states
with their weights (per state on a multigrid selection) and the below-window carrier count
`nstates_base`. `BandStates(els_k, sel_k)` gives the flattened per-state view (energies, velocities,
weights) that `BoltzmannCalculator`, `G2Calculator` and `EPElementCalculator` keep.

## Device buffers

Build whole-run buffers (energies, weights, outputs) in `setup_calculator!` with
`alloc(backend, …)` / `to_device(backend, …)`; on a `CPUBackend` they are host arrays and the same
code runs. The engine builds the state-index maps of the run's selections (0 for a band that is not a
state), so a kernel looks a state up from a tile's local band index with `state_k` / `state_kq` on
`device_tile(tile)`. Declare the bytes in `calculator_bytes`.

### Tiling a large outer-k output over the device: `TiledDeviceOutput`

A device-resident outer-k calculator whose output is indexed by the outer-k *state* (an
`(nmodes, n_i, n_f)` coupling, an `(n_i, n_f, nT)` scattering matrix) should not always hold the
whole thing on the device. `ElectronPhonon.TiledDeviceOutput` owns that bookkeeping:

- Construct it once in `setup_calculator!` from the full output shape, the axis tiled over outer-k
  states and the outer batch width: `TiledDeviceOutput{FT}((nmodes, n_i, n_f), 2, calc.el_i,
  n_outer_batch; narr = 2, force_stream_per_batch)`.
- It decides full-device-resident vs streaming one outer-k tile per batch from
  `free_bytes(ctx.backend)` (override with `force_stream_per_batch`), allocates lazily on the first
  batch, computes the outer-k tile ranges, zeros the active tile per batch, and does the contiguous
  device→host download.
- In `calculator_begin_batch!(calc, ctx)` call `tile_begin!(t, ctx)`; scatter into `device_array(t, k)`
  using `tile_offset(t)` / `tile_stride(t)` (`eph_window_scatter!`, `eph_window_scatter_reim!`, or
  your own); in `calculator_end_batch!(calc, ctx)` flush a streamed tile with `tile_download!(t)` and
  a small view-copy into your host output; in `postprocess_calculator!` copy a full-resident buffer
  back and `tile_free!(t)`.

`BoltzmannCalculator` (`src/boltzmann/boltzmann_calculator.jl`) and `G2Calculator` /
`EPElementCalculator` (`src/calculator/g2_calculator.jl`, `src/calculator/ep_element_calculator.jl`)
are worked references. See `README_GPU.md` for the device-loop details.
