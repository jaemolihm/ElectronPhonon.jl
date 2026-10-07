# Writing your own calculator

A **calculator** computes a physical property during one pass of an e-ph driver
(`run_eph_over_k_and_kq` and `run_eph_over_k_and_q`, outer loop over k, or `run_eph_over_q_and_k`,
outer loop over q). The driver builds the electron and phonon states and the e-ph matrix elements
and hands them to each calculator one **block** at a time, an [`EPBlock`](@ref) of one outer point
with a tile of inner points, together with a **context** of the loop order (`OuterKContext` or
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
- `run_calculator!(calc, block::EPBlock{OuterKLoop}, ctx)` (or `{OuterQLoop}`) — once per block.
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
- it runs only on the CPU: its `run_calculator!` is a plain loop over the block;
- it refuses more than one thread chunk, so the blocks run one after another and it adds straight
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

# Once, before the loop. With more than one thread chunk, blocks of the same k would run at the same
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

# One block: the outer k `block.ik` with a tile of q points. Pair `iq_tile` is (k, q_iq_tile):
# `ep[:, :, :, iq_tile]`, the k+q states `els_kq` and phonons `phs` at `iq_tile`, and the outer k
# `els_k` at 1 (the block's only k point). The sum over q is over the pairs of the block and over
# the blocks of the same k; the sum over k is weighted by the outer k's weight `block.wtk`.
function ElectronPhonon.run_calculator!(c::MinimalG2Calculator, block::EPBlock{OuterKLoop}, ctx)
    (; ep, els_k, els_kq, phs, wtq) = block
    s = 0.0
    for iq_tile in axes(ep, 4), ν in axes(ep, 3)
        ω = phs.e[ν, iq_tile]
        ω < omega_acoustic && continue
        for n in 1:els_k.nband[1], m in 1:els_kq.nband[iq_tile]
            s += wtq[iq_tile] * abs2(ep[m, n, ν, iq_tile]) / (2ω)
        end
    end
    c.g2_per_k[block.ik] += s
    c.g2_avg += block.wtk * s
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

- any number of CPU thread chunks: every sum that blocks of different chunks add to goes to a
  per-chunk buffer (`g2_per_k_buffer`, `g2_avg_buffer`), reduced in `calculator_end_batch!` and
  `postprocess_calculator!`;
- both loop orders: `run_calculator!` and the batch brackets get an outer-q method;
- the GPU: a second method per order, for a context on a `GPUBackend`, which forms the summand with
  one broadcast and reduces it on the device.

<!-- doc-example:begin -->
```julia
using ElectronPhonon
using LinearAlgebra: dot
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPBlock, OuterKContext,
    OuterQContext, CPUBackend, GPUBackend, alloc, omega_acoustic

# For each k: Σ over (q, m, n, ν) of wtq |ep[m, n, ν]|² / (2ω_ν(q)), with m and n the bands of
# k+q and k inside their windows. Modes with ω < omega_acoustic are skipped, as in the library's
# calculators: at Γ the acoustic ω is ~0, where 1/(2ω) only amplifies roundoff.
mutable struct EphG2SumCalculator <: AbstractCalculator
    g2_per_k        :: Vector{Float64}   # the result, indexed by k point
    g2_per_k_buffer :: Matrix{Float64}   # (chunk, outer k of the batch); outer-k runs only

    # Σ_k wtk g2_per_k[k], and its partial sum of each chunk.
    g2_avg          :: Float64
    g2_avg_buffer   :: Vector{Float64}

    # Scratch of the GPU methods, sized to one tile: the summand of a block, and the per-pair sums
    # of an outer-q block on the device and on the host. Empty on a CPU backend.
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

    # The reductions of `g2_avg`. The sum over q happens inside each block, and across the tiles
    # of an outer k, giving one number per k. The sum over k is weighted by `wtk`: the outer k's
    # weight under OuterKLoop (`block.wtk` a scalar), each pair's k weight under OuterQLoop
    # (`block.wtk` a vector). Every block of every chunk adds to the same scalar, so each chunk adds
    # to its own entry of `g2_avg_buffer`, summed in `postprocess_calculator!`.
    c.g2_avg = 0.0
    c.g2_avg_buffer = zeros(nchunks_threads)

    if backend isa GPUBackend
        # GPU scratch, sized to the largest block: `ep` is (nband_max_kq, nband_max_k, nmodes,
        # npairs) with npairs ≤ n_inner_tile. The k+q box is the container's, or at most `nw` when
        # k+q is solved per tile. On a device `ctx.chunk` is always 1.
        nband_max_kq = els_kq === nothing ? els_k.nw : els_kq.nband_max
        block_size = nband_max_kq * els_k.nband_max * phs.nmodes * n_inner_tile
        c.g2_scratch = [alloc(backend, Float64, block_size)]
        c.pair_sums_dev = [alloc(backend, Float64, n_inner_tile)]
        c.pair_sums = [zeros(n_inner_tile)]
    end
    c
end

# Before each outer-k batch: clear the outer-k buffer, which holds the sums of the current batch only.
function ElectronPhonon.calculator_begin_batch!(c::EphG2SumCalculator, ::OuterKContext)
    fill!(c.g2_per_k_buffer, 0)
    c
end

# Outer q has no per-batch buffer.
ElectronPhonon.calculator_begin_batch!(c::EphG2SumCalculator, ::OuterQContext) = c

# `run_calculator!` receives one block: one outer point with a tile of inner points. Pair `j` of the
# block is entry `[:, :, :, j]` of `ep`. Each array of the block has the pairs on its last axis,
# except the shared outer side, which has extent 1 there:
#
#               outer (extent 1)   inner, pair j                         weight of pair j
#   OuterKLoop  k: els_k[1]        k+q_j: els_kq[j], q_j: phs[j]         wtq[j]
#   OuterQLoop  q: phs[1]          k_j: els_k[j], k_j+q: els_kq[j]       wtq (the q weight)
#
# The shared side is still a one-point `BatchedElectronState` / `BatchedPhononState`, not a single
# state, so the block has one type for both orders and its extent 1 broadcasts over the pairs.
#
# There is one method per (loop order, backend). The CPU methods are plain loops. The GPU methods
# cannot read a device array one element at a time (an error under `CUDA.allowscalar(false)`), so
# they form the summand of the whole block with one broadcast (`block_g2!`) and reduce it on the
# device.

# The summand of a block on the GPU: g2[m, n, ν, j] = w_j |ep[m, n, ν, j]|² / (2ω_ν), an array shaped
# like `ep`, written into the scratch `g2_scratch`.
#
# Each factor is reshaped so that its axes line up with ep[m, n, ν, j], and one fused broadcast forms
# the whole block in a single kernel. The shared side has extent 1 on the pair axis (the outer k's
# `nband` under OuterKLoop, the outer q's `e` under OuterQLoop), so it broadcasts over the pairs, and
# one function serves both orders.
#
# The reshapes copy no data: a reshaped array shares its parent's memory (on the host it costs only
# a small array header, ~100 bytes), and a reshaped range stays lazy.
#
# Entries past a pair's band windows, and modes below omega_acoustic, are set to 0 with `ifelse`.
# The band padding is undefined, and multiplying it by a 0/1 mask could give NaN.
function block_g2!(c::EphG2SumCalculator, block, chunk)
    (; ep, els_k, els_kq, phs, wtq) = block
    nband_max_kq, nband_max_k, nmodes, npairs = size(ep)

    # Factors indexed as ep[m, n, ν, j].
    m = reshape(1:nband_max_kq, :, 1, 1, 1)                # band of k+q
    n = reshape(1:nband_max_k, 1, :, 1, 1)                 # band of k
    nband_kq = reshape(els_kq.nband, 1, 1, 1, :)           # window of k+q, per pair
    nband_k = reshape(els_k.nband, 1, 1, 1, :)             # window of k, per pair or shared
    ω = reshape(phs.e, 1, 1, nmodes, :)                    # ω_ν, per pair or shared
    w = wtq isa Number ? wtq : reshape(wtq, 1, 1, 1, :)    # weight of each pair

    # The summand, in the scratch viewed with the block's shape.
    g2 = reshape(view(c.g2_scratch[chunk], 1:length(ep)), size(ep))
    g2 .= ifelse.((m .<= nband_kq) .& (n .<= nband_k) .& (ω .>= omega_acoustic),
                  w .* abs2.(ep) ./ (2 .* ω), 0.0)
    g2
end

# Outer k, CPU. The whole block adds to one number, the sum of the outer k. Blocks of other chunks
# add to the same k concurrently, so it goes to this chunk's row of `g2_per_k_buffer`, summed in
# `calculator_end_batch!`. The column is the position of the outer k in the batch: `block.ik` is its index
# in the run, and the batch holds the outer k points `ctx.iks_batch`.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterKLoop},
                                        ctx::OuterKContext{CPUBackend})
    (; ep, els_k, els_kq, phs, wtq) = block
    nband_k = els_k.nband[1]              # the outer k, the block's only k point
    s = 0.0
    for iq_tile in axes(ep, 4)            # pair (k, q_iq_tile)
        for ν in axes(ep, 3)
            ω = phs.e[ν, iq_tile]
            ω < omega_acoustic && continue
            for n in 1:nband_k, m in 1:els_kq.nband[iq_tile]
                s += wtq[iq_tile] * abs2(ep[m, n, ν, iq_tile]) / (2ω)
            end
        end
    end
    ik_batch = block.ik - first(ctx.iks_batch) + 1   # position of the outer k in this batch
    c.g2_per_k_buffer[ctx.chunk, ik_batch] += s
    c.g2_avg_buffer[ctx.chunk] += block.wtk * s
    c
end

# Outer k, GPU: the same sum, as one device reduction of the block's summand.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterKLoop},
                                        ctx::OuterKContext{<:GPUBackend})
    g2 = block_g2!(c, block, ctx.chunk)
    s = sum(g2)
    ik_batch = block.ik - first(ctx.iks_batch) + 1   # position of the outer k in this batch
    c.g2_per_k_buffer[ctx.chunk, ik_batch] += s
    c.g2_avg_buffer[ctx.chunk] += block.wtk * s
    c
end

# Outer q, CPU. Each pair adds to its own k point, `block.ik[ik_tile]`. No other block writes these k at
# the same time (the chunks split the k points, and the q points of a batch run one after another),
# so the sums go straight into the result.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterQLoop},
                                        ctx::OuterQContext{CPUBackend})
    (; ep, els_k, els_kq, phs, wtq) = block
    for ik_tile in axes(ep, 4)            # pair (k_ik_tile, q)
        s = 0.0
        for ν in axes(ep, 3)
            ω = phs.e[ν, 1]               # the outer q, the block's only q point
            ω < omega_acoustic && continue
            for n in 1:els_k.nband[ik_tile], m in 1:els_kq.nband[ik_tile]
                s += wtq * abs2(ep[m, n, ν, ik_tile]) / (2ω)
            end
        end
        c.g2_per_k[block.ik[ik_tile]] += s
        c.g2_avg_buffer[ctx.chunk] += block.wtk[ik_tile] * s
    end
    c
end

# Outer q, GPU. The same per-pair sums as the CPU method, without reading the device arrays one
# element at a time:
#
# 1. `block_g2!` forms the summand g2[m, n, ν, ik_tile] of the whole block on the device.
# 2. `sum!` reduces it over (m, n, ν), into one sum per pair (per k point of the tile), still on the
#    device. `sum!` reduces over the axes where its output has extent 1.
# 3. The npairs sums are copied to the host in one transfer and added to their k points
#    `block.ik`, which is a host index list.
# 4. The k-weighted sum for `g2_avg` is a `dot` of the device sums with the device k weights
#    `block.wtk`. It runs on the device and returns one host number.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterQLoop},
                                        ctx::OuterQContext{<:GPUBackend})
    g2 = block_g2!(c, block, ctx.chunk)
    npairs = size(g2, 4)

    # Sum over (m, n, ν) of each pair, on the device.
    pair_sums_dev = view(c.pair_sums_dev[ctx.chunk], 1:npairs)
    sum!(reshape(pair_sums_dev, 1, 1, 1, npairs), g2)

    # Copy the sums to the host and add each to its k point.
    pair_sums = view(c.pair_sums[ctx.chunk], 1:npairs)
    copyto!(c.pair_sums[ctx.chunk], 1, pair_sums_dev, 1, npairs)
    view(c.g2_per_k, block.ik) .+= pair_sums

    # The k-weighted sum.
    c.g2_avg_buffer[ctx.chunk] += dot(pair_sums_dev, block.wtk)
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
calculator's GPU methods then run. For example, the outer-k grid run above:

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
states and reusable buffers; `stage2!` returns a complete `EPBlock`, including polar corrections.
There is no separate single-point calculation path or block-getter function. With the calculator
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

# Stage 2 for outer k 1 and inner q points 1:1: the block, or `nothing` if the pair is skipped.
block = stage2!(eng, 1, 1:1)
if block !== nothing
    run_calculator!(calc, block, ctx)
end

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
current outer range; `stage2!` requires its outer index to belong to it, and returns `nothing`
when every pair is skipped. Each returned block borrows reusable storage: consume it before
the next stage call on that chunk, or copy the arrays to retain them. Different CPU chunks have
independent writable storage; use matching `chunk` in `stage2!` and `OuterKContext(eng; chunk)` /
`OuterQContext(eng; chunk)`. A context can also be built before the batch's stage 1, as the drivers
do, by passing the batch: `OuterKContext(eng; iks_batch = 1:1)`.

## The block

Each array of an `EPBlock` has the pairs of the block on its last axis; the side shared by the whole
block has extent 1 there. Under `OuterKLoop`: `ep` is `(nband_max_kq, nband_max_k, nmodes, nq)`,
`els_k` the outer k (extent 1) at a box of its own band count (`nband_max_k = els_k.nband[1]`, so a
calculator's per-k box-shaped buffers are views at the block's extents), `els_kq` and `phs` the tile's k+q points and phonons, `ik::Int`,
`ikq` the k+q indices (a range, or a vector on the run's backend when the loop dropped pairs), `iq` the q indices,
`wtk::Float64`, `wtq` the k+q weights; under `run_eph_over_k_and_q` the inner points are the q
points, `ikq === nothing` and the k+q states are solved per tile. Under `OuterQLoop` the roles swap: `phs` has extent 1,
`iq::Int`, `ik` the k indices, and `ikq === nothing` when k+q is solved per tile; then
`nband_max_kq` is the block's largest k+q window, which differs between blocks. `ep` is the e-ph matrix before the `1/(2ω)`; a calculator that needs
`|g|²/(2ω)` forms it.

**Extents.** Every array of a block holds exactly the block's points, so take the extents from the
arrays themselves: a size mismatch between them is then detectable, and the scatter helpers check
the sizes of their inputs on entry (an `@inbounds` loop or a device kernel would not catch a read
past the block). The band axes are a box: local band `n` of point `j` is physical band
`iband_offset[j] + n` for `n ≤ nband[j]`, and everything past it (including box columns past band
`nw`) is undefined. Array bounds cannot catch a read there, so loop over `1:nband[j]`, look states
up through an index map that is 0 past it (`_indmap_to_device`), or select with `ifelse` on
`n ≤ nband[j]` as the example does; never multiply the padding by a 0/1 mask.

## The threading and batch contract

- **Any batch width.** The loop chooses `n_outer_batch` and `n_inner_tile` at runtime and passes
  them to `setup_calculator!`; size per-batch buffers to the first and per-tile scratch to the
  second. Every batch length ≥ 1 (`length(ctx.iks_batch)`, `length(ctx.iqs_batch)`) must work.
- **Writes.** Writes indexed by an inner-tile point are disjoint across blocks. Every other write
  (indexed by the outer point, or a reduction over the inner points) goes to a per-`ctx.chunk`
  partial, reduced in `calculator_end_batch!`, as in the example. Per-tile scratch is per chunk too: on
  the CPU the blocks of different chunks run concurrently. `ctx.chunk` is 1 on a device. Data of the
  block's shared side (the outer k's energies under `OuterKLoop`, the q's frequencies under
  `OuterQLoop`) is not an inner-tile write either: concurrent blocks write the same slot, so record
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

Build whole-run buffers (state-index maps, energies, weights) in `setup_calculator!` with
`alloc(backend, …)` / `to_device(backend, …)`; on a `CPUBackend` they are host arrays and the same
code runs. `_indmap_to_device(backend, states)` builds a state-index map in the box coordinates of
the containers built from that selection (0 for a band that is not a state and on the padding), so a
kernel looks a state up from a block's local band index and never reads the padding. Declare the bytes in
`calculator_bytes`.

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
