# Writing your own calculator

A **calculator** computes a physical property during one pass of an e-ph driver
(`run_eph_over_k_and_q`, `run_eph_over_q_and_k`, or `run_eph_over_k_and_kq`). The driver builds the
electron and phonon states and the e-ph matrix elements and hands them to each calculator as a
**payload** together with a **`LoopContext`**. You subtype `ElectronPhonon.AbstractCalculator` and
implement a few methods; nothing about the GPU is required for a CPU-only calculator.

The authoritative reference is the docstrings in `src/calculator/AbstractCalculator.jl`. This guide
is the tutorial; its example is executed verbatim by `test/test_calculator_guide.jl`, so it cannot
rot.

## The mandatory surface

A calculator must implement:

- `setup_calculator!(calc, kpts, qpts, el_states; kwargs...)` — run once, before the loop.
- `run_calculator!(calc, payload, ctx)` — one method per payload *type* the calculator consumes.
  The host per-(k,q) payload is `EPData`; the device-batched payloads (`EPDataQBatched`
  / `EPDataKBatched`) are only for the GPU path (see "Migrating to the GPU" below).
- `postprocess_calculator!(calc; kwargs...)` — run once, after the loop.
- `supports(calc, ::Type{T})` — declare the driver loop shapes (`OuterKLoop` / `OuterQLoop`) and
  the payload types the calculator handles. The default is `false`; the second argument must be a
  *type* (`supports(calc, OuterKLoop)`, not `supports(calc, OuterKLoop())` — passing an instance
  throws). The driver checks these up front and errors if it would hand the calculator a loop or
  payload it does not declare.

Optionally, `calculator_begin!(calc, scope, ctx)` / `calculator_end!(calc, scope, ctx)` bracket one
outer iteration (`OuterIteration()`) or one batch of outer iterations (`OuterIterationBatch()`);
their defaults are no-ops.

Which loop shape? `OuterKLoop` calculators run under `run_eph_over_k_and_q` /
`run_eph_over_k_and_kq` (outer loop over k); `OuterQLoop` calculators run under
`run_eph_over_q_and_k` (outer loop over q). A calculator declares whichever it is built for.

## A complete minimal CPU-only example

This calculator sums `|g|²/(2ω)` (weighted by the q-point weight) over all inner k+q points, bands,
and modes, giving one number per outer k-point. It runs under `run_eph_over_k_and_kq` (outer k).

<!-- doc-example:begin -->
```julia
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, EPData, OuterIteration

# One value per outer k: Σ over (k+q, m, n, ν) of wtq · g2[m, n, ν].
mutable struct EphG2SumCalculator <: AbstractCalculator
    per_k :: Vector{Float64}   # result, indexed by outer-k index
    chunk :: Vector{Float64}   # per-thread-chunk partial sum for the CURRENT outer k
    EphG2SumCalculator() = new(Float64[], Float64[])
end

# This calculator runs under the outer-k drivers and consumes the host per-(k,q) payload.
ElectronPhonon.supports(::EphG2SumCalculator, ::Type{OuterKLoop})    = true
ElectronPhonon.supports(::EphG2SumCalculator, ::Type{EPData}) = true

# Run once before the loop. `nchunks_threads` is the number of thread chunks the inner loop uses;
# allocate one partial-sum slot per chunk so concurrent calls never touch the same slot.
function ElectronPhonon.setup_calculator!(c::EphG2SumCalculator, kpts, qpts, el_states;
        nchunks_threads, kwargs...)
    c.per_k = zeros(kpts.n)
    c.chunk = zeros(nchunks_threads)
    c
end

# Serial, before the threaded inner loop for this outer k: zero the per-chunk partials.
function ElectronPhonon.calculator_begin!(c::EphG2SumCalculator, ::OuterIteration, ctx)
    fill!(c.chunk, 0.0)
    c
end

# Called CONCURRENTLY from @threads over inner-k+q chunks at fixed outer k. Accumulate into THIS
# chunk's slot (`p.id_chunk`) — never a shared slot — so there is no data race.
function ElectronPhonon.run_calculator!(c::EphG2SumCalculator, p::EPData, ctx)
    epdata = p.epdata
    s = 0.0
    for ν in 1:epdata.ph.nmodes, n in epdata.el_k.rng, m in epdata.el_kq.rng
        s += epdata.wtq * epdata.g2[m, n, ν]
    end
    c.chunk[p.id_chunk] += s
    c
end

# Serial, after the threaded region: reduce the per-chunk partials into this outer k's result.
# `ctx.outer_index` is the current outer-k index.
function ElectronPhonon.calculator_end!(c::EphG2SumCalculator, ::OuterIteration, ctx)
    c.per_k[ctx.outer_index] = sum(c.chunk)
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

## The threading contract (read this)

On all CPU paths `run_calculator!(calc, ::EPData, ctx)` is called **concurrently** from
`@threads` over inner (k+q or k) chunks at a fixed outer index. Mutating shared calculator state
from it races. Two safe patterns:

1. **Per-chunk buffers indexed by `payload.id_chunk`** (the example above): each concurrent call
   writes only its own chunk's slot. `id_chunk ∈ 1:nchunks_threads`, the count passed to
   `setup_calculator!`.
2. **Writes to disjoint slots**: if each call writes a distinct output location (e.g. a unique
   `(band, k+q)` entry), no per-chunk buffer is needed.

`calculator_begin!(calc, OuterIteration(), ctx)` and `calculator_end!(calc, OuterIteration(), ctx)`
run **serially**, outside the threaded region, once per outer iteration. Cross-chunk reductions
(summing the per-chunk partials, scattering into the final output for this outer index) belong
there — `ctx.outer_index` is the current outer index (`ik` for outer-k loops, `iq` for the outer-q
loop). `postprocess_calculator!` runs once after the whole loop.

`run_calculator!` is a dynamic dispatch per (k,q) per calculator (F11c): keep its body substantial
rather than trivial — if you need many tiny operations, batch them inside the calculator.

## What `setup_calculator!` receives

`setup_calculator!(calc, kpts, qpts, el_states; kwargs...)` is called with the outer k-points
`kpts`, the q-points `qpts` (may be `nothing` when phonons are computed on the fly), the electron
states at k `el_states`, and these keyword arguments (splat the rest with `kwargs...`):

- `nw`, `nmodes` — number of Wannier bands / phonon modes.
- `rng_band` — the in-window band range at k.
- `el_states_kq`, `kqpts` — electron states / k-points at k+q (may be `nothing`).
- `nelec_below_window_k`, `nelec_below_window_kq` — carrier counts below the window.
- `nchunks_threads` — number of inner-loop thread chunks (size per-chunk buffers to this).
- `verbosity`, `backend` — printing verbosity and the compute backend (`CPUBackend()` /
  `GPUBackend(proto)`). A CPU-only calculator ignores `backend`.

If your calculator only reads eigenvalues/eigenvectors of the outer-k states (not velocity or
position), override `required_el_k_quantities(calc) = ["eigenvalue", "eigenvector"]` so the outer-q
driver skips the velocity/position interpolation (the default is the full list).

## Migrating a calculator to the GPU

The GPU path is an explicit opt-in with no silent fallback: `run_eph_over_k_and_kq(...;
use_gpu=true)` (or `run_eph_over_q_and_k(...; use_gpu=true)`) requires **every** calculator to
support the corresponding device-batched payload, and a calculator that does not opt in errors
loudly. `use_gpu` is the only user-facing switch; internally the driver resolves a `backend` object
carried in `ctx.backend`.

To add a GPU path to an existing CPU calculator:

1. Declare the batched payload: `supports(calc, ::Type{EPDataQBatched}) = true` (outer-k) or
   `supports(calc, ::Type{EPDataKBatched}) = true` (outer-q).
2. Implement `run_calculator!(calc, p::EPDataQBatched, ctx)` (resp. `EPDataKBatched`).
   The payload carries the e-ph matrices for a whole batch **on the device** (`p.eps`, `p.g2s`,
   `p.ωqs`, batch indices …). Write it backend-generically — only `alloc(ctx.backend, T, dims...)`,
   `similar`, `copyto!`, broadcasting, `mul!`, and scatter-assignment — so the same method runs on
   CPU arrays and `CuArray`s and adds no CUDA dependency of its own.
3. Manage device buffers. `setup_calculator!` is passed the run's `backend`, so build whole-run
   device buffers (index maps, band energies, weights — anything intrinsic to the state sets) there
   with `alloc(backend, …)` / `to_device(backend, …)` (only when `backend isa GPUBackend`). Use the
   brackets only for per-iteration state: allocate/zero the per-iteration device accumulator in
   `calculator_begin!(calc, OuterIteration(), ctx)` (or `OuterIterationBatch()` for outer-k
   per-batch buffers) using `ctx.backend`, and copy the result device→host in the matching
   `calculator_end!`. Declare per-point device scratch via
   `eph_batched_bytes_per_point(calc, PayloadType; nw, nmodes)` so the loop's memory-adaptive batch
   sizing accounts for it.

   If a bracket is shared between the CPU and GPU loop shapes (e.g. the `OuterIteration` bracket,
   which the batched outer-k loop fires per k in addition to the CPU per-point loop), select the
   batched behavior from the loop **mode** carried in `ctx`, never the backend: either dispatch on
   `ctx::LoopContext{<:AbstractBackend, SingleMode}` vs `{<:AbstractBackend, BatchedMode}`, or branch
   on `ctx.mode isa ElectronPhonon.BatchedMode`. `ctx.backend` is only for allocation
   (`alloc(ctx.backend, …)`) / `free_bytes` / `synchronize`, not for telling the loop shapes apart.

### Tiling a large outer-k output over the device: `TiledDeviceOutput`

A device-resident outer-k calculator whose output is indexed by the outer-k *state* (so it grows
with the grid — e.g. an `(nmodes, n_i, n_f)` coupling, or an `(n_i, n_f, nT)` scattering matrix)
should not always hold the whole thing on the device. `ElectronPhonon.TiledDeviceOutput` is an
opt-in helper that owns exactly that bookkeeping, so `BoltzmannCalculator` and `EliashbergCalculator`
both use it instead of hand-rolling it:

- Construct it once (at `setup_calculator!`) from the full output shape and which axis is tiled over
  outer-k states — any dims, any tiled-axis position, and `narr` arrays of identical tiling per
  instance (`EliashbergCalculator` holds g2 and ωq in one instance):
  `TiledDeviceOutput{FT}((nmodes, n_i, n_f), 2, calc.el_i; narr = 2, force_block)`.
- It decides **full-device-resident vs per-tile block** residency from `free_bytes(ctx.backend)`
  (override with `force_block`), allocates lazily on the first batch, computes the outer-k tile
  ranges (with the contiguity guarantee), zeros the active tile per batch, and does the contiguous
  device→host download.
- In your `calculator_begin!(…, OuterIterationBatch(), ctx)` call `tile_begin!(t, ctx)`; scatter into
  `device_array(t, k)` using `tile_offset(t)` / `tile_stride(t)` (the scatter stays yours —
  `eph_window_scatter!` or your own); in `calculator_end!(…, OuterIterationBatch(), ctx)` flush a
  block tile with `tile_download!(t)` + a small view-copy into your host output; in
  `postprocess_calculator!` copy a full-resident buffer back and `tile_free!(t)`.

The lazily-allocated device buffers are held behind a `Union{Nothing, …}` / `Vector{Any}` handle;
type stability is preserved because the scatter kernel (a typed generic function) is the
function-boundary type barrier — everything the helper itself does runs once per batch (cold).

### Whole-run device buffers at setup

Band energies, integration weights, and state-index maps are intrinsic to the state sets, not to any
one loop iteration, so build them once in `setup_calculator!` from the `backend` it is handed (they
would otherwise pollute `LoopContext`, which describes only the current iteration). Upload arrays
with `to_device(backend, host_array)` and build the device state-index map with
`_indmap_to_device(backend, states, nw)`. Do this only for `backend isa GPUBackend` (the CPU path
uses the per-point `EPData` method and needs no device buffers).

`BoltzmannCalculator` (`src/boltzmann/boltzmann_calculator.jl`) and `EliashbergCalculator`
(MigdalEliashberg.jl) are worked references: each has both a CPU `EPData` method and a GPU
batched method sharing the same output arrays, and both use `TiledDeviceOutput`. See `README_GPU.md`
for the device-loop details.
