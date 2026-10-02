# Writing your own calculator

A **calculator** computes a physical property during one pass of an e-ph driver
(`run_eph_over_k_and_kq`, outer loop over k, or `run_eph_over_q_and_k`, outer loop over q). The
driver builds the electron and phonon states and the e-ph matrix elements and hands them to each
calculator one **block** at a time, an [`EPBlock`](@ref) of one outer point with a tile of inner
points, together with a **`LoopContext`**. You subtype `ElectronPhonon.AbstractCalculator` and
implement a few methods; the same methods run on the CPU and on a GPU when they are written with
broadcasts and `alloc(backend, …)`.

The authoritative reference is the docstrings in `src/calculator/AbstractCalculator.jl`. This guide
is the tutorial; its example is executed verbatim by `test/test_calculator_guide.jl`, so it cannot
rot.

## The surface

- `supports(calc, ::Type{OuterKLoop})` / `supports(calc, ::Type{OuterQLoop})` — the loop orders the
  calculator handles (default `false`; pass the type, not an instance). The driver refuses a
  calculator that does not support its order.
- `required_el_quantities(calc)`, `required_ph_quantities(calc)` — the state quantities it reads
  beyond the energies `e` and eigenvectors `u`, which the loop always provides on both electron sides
  and on the phonons, as field names of `BatchedElectronState` (`:vdiag`, `:v`, `:rbar`) and
  `BatchedPhononState` (`:vdiag`, …). Default: none.
- `setup_calculator!(calc, backend, el_k, el_kq, ph; sel_k, sel_kq, nw, nmodes, nchunks_threads,
  n_outer_batch, n_inner_tile, verbosity)` — once, before the loop (see below).
- `run_calculator!(calc, block::EPBlock{OuterKLoop}, ctx)` (or `{OuterQLoop}`) — once per block.
- `calculator_begin!(calc, ctx)` / `calculator_end!(calc, ctx)` — around every outer batch
  (`ctx.batch`, the outer indices of the batch). There is no default: define both, even as `= nothing`.
- `postprocess_calculator!(calc; kwargs...)` — once, after the loop.
- Optionally `eph_batched_bytes_per_point(calc, ::Type{<:EPBlock{O}}; nw, nmodes, nband_max_k,
  nband_max_kq) -> (; persistent, per_outer, per_pair)`, the device bytes the calculator allocates,
  so the loop sizes its tiles to the free memory.

## A complete minimal example

This calculator sums `wtq · |g|²/(2ω)` over the in-window bands, the modes and the k+q points of
each outer k-point. It runs under `run_eph_over_k_and_kq`.

<!-- doc-example:begin -->
```julia
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, EPBlock, alloc

# One value per outer k: Σ over (k+q, m, n, ν) of wtq · |ep[m, n, ν]|² / (2ω[ν]).
mutable struct EphG2SumCalculator <: AbstractCalculator
    per_k :: Vector{Float64}   # result, indexed by outer-k index
    part  :: Matrix{Float64}   # (chunk, outer point of the batch) partial sums
    g2    :: Any               # per-tile scratch on the run's backend
    EphG2SumCalculator() = new(Float64[], zeros(0, 0), nothing)
end

ElectronPhonon.supports(::EphG2SumCalculator, ::Type{OuterKLoop}) = true
# The loop always provides `e`, `u` and the e-ph matrix elements, which is all this calculator reads,
# so it defines no `required_el_quantities` / `required_ph_quantities`.

# Buffers are sized here, from the widths the loop chose; `run_calculator!` allocates nothing.
function ElectronPhonon.setup_calculator!(c::EphG2SumCalculator, backend, el_k, el_kq, ph;
        nmodes, nchunks_threads, n_outer_batch, n_inner_tile, kwargs...)
    c.per_k = zeros(el_k.nk)
    c.part = zeros(nchunks_threads, n_outer_batch)
    c.g2 = alloc(backend, Float64, el_kq.nband_max, el_k.nband_max, nmodes, n_inner_tile)
    c
end

ElectronPhonon.calculator_begin!(c::EphG2SumCalculator, ctx) = (fill!(c.part, 0.0); c)

# One outer k (`p.ik`) with a tile of k+q points. Band entries past a point's window are undefined,
# so select the in-window ones with `ifelse` (never multiply by a 0/1 mask). The write is indexed by
# the outer point, not by an inner one, so it goes to this call's chunk slot.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, p::EPBlock{OuterKLoop}, ctx)
    (; ep, ph, el_k, el_kq, wtq) = p
    nbkq, nbk, nmodes, nq = size(ep)
    g2 = view(c.g2, :, :, :, 1:nq)
    g2 .= ifelse.((reshape(1:nbkq, nbkq, 1, 1, 1) .<= reshape(el_kq.nband, 1, 1, 1, nq)) .&
                  (reshape(1:nbk, 1, nbk, 1, 1) .<= reshape(el_k.nband, 1, 1, 1, 1)),
                  abs2.(ep) ./ (2 .* reshape(ph.e, 1, 1, nmodes, nq)), 0.0) .* reshape(wtq, 1, 1, 1, nq)
    c.part[ctx.chunk, p.ik - first(ctx.batch) + 1] += sum(g2)
    c
end

# After the batch: reduce the partials into the outer points of the batch.
function ElectronPhonon.calculator_end!(c::EphG2SumCalculator, ctx)
    c.per_k[ctx.batch] .= vec(sum(view(c.part, :, 1:length(ctx.batch)); dims = 1))
    c
end

ElectronPhonon.postprocess_calculator!(c::EphG2SumCalculator; kwargs...) = c
```
<!-- doc-example:end -->

Run it:

```julia
calc = EphG2SumCalculator()
run_eph_over_k_and_kq(model, (nk, nk, nk), (nk, nk, nk); calculators = [calc])
calc.per_k   # one number per outer k-point
```

## The block

Each array of an `EPBlock` has the pairs of the block on its last axis; the side shared by the whole
block has extent 1 there. Under `OuterKLoop`: `ep` is `(nband_max_kq, nband_max_k, nmodes, nq)`,
`el_k` the outer k (extent 1), `el_kq` and `ph` the tile's k+q points and phonons, `ik::Int`,
`ikq` the k+q indices (a range), `iq` the q indices, `wtk::Float64`, `wtq` the k+q weights. Under
`OuterQLoop` the roles swap: `ph` has extent 1, `iq::Int`, `ik` a range, and `ikq === nothing`
(k+q is solved per tile). `ep` is the e-ph matrix before the `1/(2ω)`; a calculator that needs
`|g|²/(2ω)` forms it.

**Extents.** Every array of a block holds exactly the block's points, so indexing past them is an
error rather than a read of another block's data. The band axes are a box: local band `n` of point
`j` is physical band `iband_offset[j] + n` for `n ≤ nband[j]`, and everything past it (including box
columns past band `nw`) is undefined. Array bounds cannot catch a read there, so loop over
`1:nband[j]`, look states up through an index map that is 0 past it (`_indmap_to_device`), or
select with `ifelse` on `n ≤ nband[j]` as the example does; never multiply the padding by a 0/1
mask.

## The threading and batch contract

- **Any batch width.** The loop chooses `n_outer_batch` and `n_inner_tile` at runtime and passes
  them to `setup_calculator!`; size per-batch buffers to the first and per-tile scratch to the
  second. Every `length(ctx.batch) ≥ 1` must work.
- **Writes.** Writes indexed by an inner-tile point are disjoint across blocks. Every other write
  (indexed by the outer point, or a reduction over the inner points) goes to a per-`ctx.chunk`
  partial, reduced in `calculator_end!`, as in the example. `ctx.chunk` is 1 on a device.
- **Brackets** run serially, once around every outer batch, on every backend.

## What `setup_calculator!` receives

`el_k`, `el_kq`, `ph` are the run's containers (`BatchedElectronState`, `BatchedPhononState`) on the
run's `backend`, holding the requested quantities; `el_kq` is `nothing` when k+q is solved per tile.
`sel_k`, `sel_kq` are the `FilteredBandStates` selections they were built from: the selected states
with their weights (per state on a multigrid selection) and the below-window carrier count
`nstates_base`. `BandStates(el_k, sel_k)` gives the flattened per-state view (energies, velocities,
weights) that `BoltzmannCalculator` and the MigdalEliashberg calculators keep.

## Device buffers

Build whole-run buffers (state-index maps, energies, weights) in `setup_calculator!` with
`alloc(backend, …)` / `to_device(backend, …)`; on a `CPUBackend` they are host arrays and the same
code runs. `_indmap_to_device(backend, states)` builds a state-index map in the box coordinates of
the containers built from that selection (0 for a band that is not a state and on the padding), so a
kernel looks a state up from a block's local band index and never reads the padding. Declare the bytes in
`eph_batched_bytes_per_point`.

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
