# Writing your own calculator

A **calculator** computes a physical property during one pass of an e-ph driver
(`run_eph_over_k_and_kq` and `run_eph_over_k_and_q`, outer loop over k, or `run_eph_over_q_and_k`,
outer loop over q). The driver builds the electron and phonon states and the e-ph matrix elements
and hands them to each calculator one **block** at a time, an [`EPBlock`](@ref) of one outer point
with a tile of inner points, together with a **`LoopContext`**. You subtype
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
- `setup_calculator!(calc, backend, els_k, els_kq, phs; sel_k, sel_kq, nchunks_threads,
  n_outer_batch, n_inner_tile, verbosity)` — once, before the loop (see below).
- `run_calculator!(calc, block::EPBlock{OuterKLoop}, ctx)` (or `{OuterQLoop}`) — once per block.
- `calculator_begin!(calc, ctx)` / `calculator_end!(calc, ctx)` — around every outer batch
  (`ctx.batch`, the outer indices of the batch). There is no default: define both, even as `= nothing`.
- `postprocess_calculator!(calc; kwargs...)` — once, after the loop.
- Optionally `calculator_bytes(calc, ::Type{<:EPBlock{O}}; nw, nmodes, nband_max_k, nband_max_kq,
  els_k, els_kq, phs, nchunks_threads) -> (; persistent, per_outer, per_pair)`, the device
  bytes the calculator allocates, so the loop sizes its tiles to the free memory. The loop counts
  `persistent` once, `per_outer` once per outer point of a batch and `per_pair` once per inner point
  of a tile and per thread chunk; a per-outer buffer the calculator holds per chunk multiplies by
  `nchunks_threads` itself. `els_k`, `els_kq`, `phs` are `nothing` in `estimate_device_memory`. Accept
  `kwargs...` for keywords added later.

## A complete minimal example

This calculator sums `wtq · |g|²/(2ω)` over the in-window bands, the modes and the q points, for each
k point. It supports both loop orders and runs on the CPU and on a GPU.

<!-- doc-example:begin -->
```julia
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPBlock, LoopContext, CPUBackend,
    alloc, omega_acoustic

# For each k: Σ over (q, m, n, ν) of wtq |ep[m, n, ν]|² / (2ω_ν(q)), with m and n the bands of
# k+q and k inside their windows. Modes with ω < omega_acoustic are skipped, as in the library's
# calculators: at Γ the acoustic ω is ~0, where 1/(2ω) only amplifies roundoff.
mutable struct EphG2SumCalculator <: AbstractCalculator
    g2_per_k     :: Vector{Float64}   # the result, indexed by k point
    partial_sums :: Matrix{Float64}   # (chunk, outer k of the batch), for the outer-k loop
    # Scratch of the GPU methods, one per thread chunk, sized to one tile: the summand of a block,
    # and the per-pair sums of an outer-q block on the backend and on the host.
    g2_scratch    :: Vector{Any}
    pair_sums_dev :: Vector{Any}
    pair_sums     :: Vector{Vector{Float64}}
    EphG2SumCalculator() = new(Float64[], zeros(0, 0), [], [], [])
end

ElectronPhonon.supports(::EphG2SumCalculator, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::EphG2SumCalculator, ::Type{OuterQLoop}) = true
# The loop always provides `e`, `u` and the e-ph matrix elements, which is all this calculator reads,
# so it defines no `required_el_quantities` / `required_ph_quantities`.

# Buffers are sized here, from the widths the loop chose; `run_calculator!` allocates none.
function ElectronPhonon.setup_calculator!(c::EphG2SumCalculator, backend, els_k, els_kq, phs;
        nchunks_threads, n_outer_batch, n_inner_tile, kwargs...)
    # The result: one sum per k point of the run.
    c.g2_per_k = zeros(els_k.nk)

    # Outer k: every chunk adds to the sum of the same outer k, so each chunk has its own partial,
    # one per outer k of a batch.
    c.partial_sums = zeros(nchunks_threads, n_outer_batch)

    # GPU scratch, sized to the largest block: `ep` is (nband_max_kq, nband_max_k, nmodes, npairs)
    # with npairs ≤ n_inner_tile. The k+q box is the container's, or at most `nw` when k+q is
    # solved per tile.
    nband_max_kq = els_kq === nothing ? els_k.nw : els_kq.nband_max
    block_size = nband_max_kq * els_k.nband_max * phs.nmodes * n_inner_tile
    c.g2_scratch = [alloc(backend, Float64, block_size) for _ in 1:nchunks_threads]
    c.pair_sums_dev = [alloc(backend, Float64, n_inner_tile) for _ in 1:nchunks_threads]
    c.pair_sums = [zeros(n_inner_tile) for _ in 1:nchunks_threads]
    c
end

ElectronPhonon.calculator_begin!(c::EphG2SumCalculator, ctx) = (fill!(c.partial_sums, 0.0); c)

# `run_calculator!` receives one block: one outer point with a tile of inner points. Pair `j` of the
# block is entry `[:, :, :, j]` of `ep`. Each array of the block has the pairs on its last axis,
# except the shared outer side, which has extent 1 there:
#
#               outer (extent 1)   inner, pair j                         weight of pair j
#   OuterKLoop  k: els_k[1]        k+q_j: els_kq[j], q_j: phs[j]         wtq[j]
#   OuterQLoop  q: phs[1]          k_j: els_k[j], k_j+q: els_kq[j]       wtq (the q weight)
#
# There is one method per (loop order, backend). The CPU methods are plain loops. The GPU methods
# cannot read a device array one element at a time (an error under `CUDA.allowscalar(false)`), so
# they form the summand of the whole block with one broadcast (`block_g2!`) and reduce it on the
# device. They are written for any backend, so they also run on a CPU backend.

# The summand of a block, w_j |ep[m, n, ν, j]|² / (2ω_ν), as an array shaped like `ep`. Every factor
# is reshaped to broadcast along ep[m, n, ν, j]; the shared side's extent 1 broadcasts over the
# pairs, so one function serves both orders. Entries past a pair's band windows and modes below
# omega_acoustic are 0, selected with `ifelse`: the band padding is undefined, and a 0/1 mask times
# an undefined entry could give NaN.
function block_g2!(c::EphG2SumCalculator, block, chunk)
    (; ep, els_k, els_kq, phs, wtq) = block
    nband_max_kq, nband_max_k, nmodes, npairs = size(ep)
    m = reshape(1:nband_max_kq, :, 1, 1, 1)         # band of k+q
    n = reshape(1:nband_max_k, 1, :, 1, 1)          # band of k
    nband_kq = reshape(els_kq.nband, 1, 1, 1, :)
    nband_k = reshape(els_k.nband, 1, 1, 1, :)
    ω = reshape(phs.e, 1, 1, nmodes, :)
    w = wtq isa Number ? wtq : reshape(wtq, 1, 1, 1, :)
    g2 = reshape(view(c.g2_scratch[chunk], 1:length(ep)), size(ep))
    g2 .= ifelse.((m .<= nband_kq) .& (n .<= nband_k) .& (ω .>= omega_acoustic),
                  w .* abs2.(ep) ./ (2 .* ω), 0.0)
    g2
end

# Outer k, CPU. The whole block adds to one number, the sum of the outer k. Blocks of other chunks
# add to the same k concurrently, so it goes to this chunk's partial, reduced in `calculator_end!`.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterKLoop},
                                        ctx::LoopContext{CPUBackend})
    (; ep, els_k, els_kq, phs, wtq) = block
    s = 0.0
    for j in axes(ep, 4)                  # pair j: (k, q_j)
        for ν in axes(ep, 3)
            ω = phs.e[ν, j]
            ω < omega_acoustic && continue
            for n in 1:els_k.nband[1], m in 1:els_kq.nband[j]
                s += wtq[j] * abs2(ep[m, n, ν, j]) / (2ω)
            end
        end
    end
    c.partial_sums[ctx.chunk, block.ik - first(ctx.batch) + 1] += s
    c
end

# Outer k, GPU: the same sum, as one device reduction of the block's summand.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterKLoop},
                                        ctx::LoopContext)
    g2 = block_g2!(c, block, ctx.chunk)
    c.partial_sums[ctx.chunk, block.ik - first(ctx.batch) + 1] += sum(g2)
    c
end

# Outer q, CPU. Each pair adds to its own k point, `block.ik[j]`. No other block writes these k at
# the same time (the chunks split the k points, and the q points of a batch run one after another),
# so the sums go straight into the result.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterQLoop},
                                        ctx::LoopContext{CPUBackend})
    (; ep, els_k, els_kq, phs, wtq) = block
    for j in axes(ep, 4)                  # pair j: (k_j, q)
        s = 0.0
        for ν in axes(ep, 3)
            ω = phs.e[ν, 1]
            ω < omega_acoustic && continue
            for n in 1:els_k.nband[j], m in 1:els_kq.nband[j]
                s += wtq * abs2(ep[m, n, ν, j]) / (2ω)
            end
        end
        c.g2_per_k[block.ik[j]] += s
    end
    c
end

# Outer q, GPU: sum the summand over (m, n, ν) of each pair on the device, copy the npairs sums to
# the host, and add them to their k points.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, block::EPBlock{OuterQLoop},
                                        ctx::LoopContext)
    g2 = block_g2!(c, block, ctx.chunk)
    npairs = size(g2, 4)
    pair_sums_dev = view(c.pair_sums_dev[ctx.chunk], 1:npairs)
    sum!(reshape(pair_sums_dev, 1, 1, 1, npairs), g2)
    copyto!(c.pair_sums[ctx.chunk], 1, pair_sums_dev, 1, npairs)
    view(c.g2_per_k, block.ik) .+= view(c.pair_sums[ctx.chunk], 1:npairs)
    c
end

function ElectronPhonon.calculator_end!(c::EphG2SumCalculator, ctx)
    if ctx.order isa OuterKLoop
        # Outer k: reduce the chunks' partials into the k points of the batch.
        c.g2_per_k[ctx.batch] .= vec(sum(view(c.partial_sums, :, 1:length(ctx.batch)); dims = 1))
    else
        # Outer q: `run_calculator!` already added every pair to its k point; nothing to reduce.
    end
    c
end

ElectronPhonon.postprocess_calculator!(c::EphG2SumCalculator; kwargs...) = c
```
<!-- doc-example:end -->

## Run it with a driver

The drivers are public but not exported, so they are called as `ElectronPhonon.<name>`. With
`epw_folder` the folder of an EPW run:

<!-- doc-driver:begin -->
```julia
# A model from an EPW run, stored for the outer-k loop.
model = load_model_from_epw_new(epw_folder, "temp", "pb"; epmat_outer_momentum = "el")

calc = EphG2SumCalculator()
out = ElectronPhonon.run_eph_over_k_and_kq(model, (8, 8, 8), (8, 8, 8);
    calculators = [calc], symmetry = nothing)
calc.g2_per_k    # one number per k point of out.kpts

# The same sums with q as the outer loop, from the model stored for the outer-q loop.
model_ph = load_model_from_epw_new(epw_folder, "temp", "pb"; epmat_outer_momentum = "ph")
calc_q = EphG2SumCalculator()
ElectronPhonon.run_eph_over_q_and_k(model_ph, (8, 8, 8), (8, 8, 8);
    calculators = [calc_q], symmetry = nothing)
```
<!-- doc-driver:end -->

Pass `backend = ElectronPhonon.gpu_backend()` to a driver to run on a GPU, and `window_k` /
`window_kq` to keep the bands near the Fermi level.

## Run one k, q pair yourself

The engine stages used by the drivers are also public (unexported) API. An engine prepares the
states and reusable buffers; `stage2!` returns a complete `EPBlock`, including polar corrections.
There is no separate single-point calculation path or block-getter function. With the calculator
defined above and a model loaded with `epmat_outer_momentum = "el"`:

<!-- doc-single-pair:begin -->
```julia
using ElectronPhonon: OuterKEngine, stage1!, stage2!, LoopContext,
    setup_calculator!, calculator_begin!, run_calculator!, calculator_end!, postprocess_calculator!

calc = EphG2SumCalculator()
kpts = Kpoints(Vec3(0.25, 0.25, 0.25))   # one k point
qpts = Kpoints(Vec3(0.1, 0.1, 0.1))      # one q point

# The states at k, k + q and q, and the e-ph buffers. Construction runs no calculator hook.
eng = OuterKEngine(model, kpts, qpts; calculators = [calc], verbosity = 0)

# Size the calculator's buffers to the engine's states and widths, as a driver does.
setup_calculator!(calc, eng.backend, eng.els_k, eng.els_kq, eng.phs;
    eng.sel_k, eng.sel_kq, nchunks_threads = length(eng.tiles),
    eng.n_outer_batch, eng.n_inner_tile, verbosity = 0)

# Stage 1 for the outer k points 1:1, and the calculator context of that batch.
stage1!(eng, 1:1)
ctx = LoopContext(eng)
calculator_begin!(calc, ctx)

# Stage 2 for outer k 1 and inner q points 1:1: the block, or `nothing` if the pair is skipped.
block = stage2!(eng, 1, 1:1)
if block !== nothing
    run_calculator!(calc, block, ctx)
end

calculator_end!(calc, ctx)
postprocess_calculator!(calc; qpts = eng.qpts, symmetry = nothing)
calc.g2_per_k
```
<!-- doc-single-pair:end -->

`OuterKEngine` defaults to inner q points, solving k+q within each tile. For a resident k+q grid,
pass `inner_loop_kq = true` and that grid as the third argument. A model loaded with
`epmat_outer_momentum = "ph"` instead uses `OuterQEngine(model, kpts, qpts; ...)`, with q as
the outer index and k as the inner range. The same stage calls and `LoopContext(eng)` work.

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
independent writable storage; use matching `chunk` in `stage2!` and `LoopContext(eng; chunk)`.

## The block

Each array of an `EPBlock` has the pairs of the block on its last axis; the side shared by the whole
block has extent 1 there. Under `OuterKLoop`: `ep` is `(nband_max_kq, nband_max_k, nmodes, nq)`,
`els_k` the outer k (extent 1), `els_kq` and `phs` the tile's k+q points and phonons, `ik::Int`,
`ikq` the k+q indices (a range, or a host vector when the loop dropped pairs), `iq` the q indices,
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
  second. Every `length(ctx.batch) ≥ 1` must work.
- **Writes.** Writes indexed by an inner-tile point are disjoint across blocks. Every other write
  (indexed by the outer point, or a reduction over the inner points) goes to a per-`ctx.chunk`
  partial, reduced in `calculator_end!`, as in the example. Per-tile scratch is per chunk too: on
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
weights) that `BoltzmannCalculator` and the MigdalEliashberg calculators keep.

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
  n_outer_batch; narr = 2, force_block)`.
- It decides full-device-resident vs per-tile block residency from `free_bytes(ctx.backend)`
  (override with `force_block`), allocates lazily on the first batch, computes the outer-k tile
  ranges, zeros the active tile per batch, and does the contiguous device→host download.
- In `calculator_begin!(calc, ctx)` call `tile_begin!(t, ctx)`; scatter into `device_array(t, k)`
  using `tile_offset(t)` / `tile_stride(t)` (`eph_window_scatter!`, `eph_window_scatter_reim!`, or
  your own); in `calculator_end!(calc, ctx)` flush a block tile with `tile_download!(t)` and a small
  view-copy into your host output; in `postprocess_calculator!` copy a full-resident buffer back and
  `tile_free!(t)`.

`BoltzmannCalculator` (`src/boltzmann/boltzmann_calculator.jl`) and `G2Calculator` /
`EPElementCalculator` (MigdalEliashberg.jl) are worked references. See `README_GPU.md` for the
device-loop details.
