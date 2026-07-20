"""
    AbstractCalculator

A calculator computes properties of the system during a single pass of one of the e-ph drivers
(`run_eph_over_k_and_q`, `run_eph_over_q_and_k`, `run_eph_over_k_and_kq`). Once the electron and
phonon states and the e-ph matrix elements are formed, the driver hands them to each calculator as a
**payload** (a subtype of [`AbstractElPhPayload`]) together with a [`LoopContext`].

Users subtype `AbstractCalculator` and implement:
* `setup_calculator!(calc, kpts, qpts, el_states; kwargs...)` — run once, before the loop.
* `run_calculator!(calc, payload::AbstractElPhPayload, ctx::LoopContext)` — one method per payload
  type the calculator consumes (host per-(k,q), or one of the device-batched payloads).
* `postprocess_calculator!(calc; kwargs...)` — run once, after the loop.
* `supports(calc, ::Type{<:LoopTag})` / `supports(calc, ::Type{<:AbstractElPhPayload})` — declare
  which loop shapes and payloads the calculator handles (default `false`).

Optionally:
* `calculator_begin!(calc, scope, ctx)` / `calculator_end!(calc, scope, ctx)` — begin/end brackets
  around one outer iteration (`OuterIteration()`) or one batch of outer iterations
  (`OuterIterationBatch()`); default no-ops.
* `eph_batched_bytes_per_point(calc, PayloadType; nw, nmodes)` — per-point device scratch (bytes)
  the calculator holds, so the GPU loops' memory-adaptive batch sizing can account for it.
* `allowed_eph_phonon_basis(calc)` — phonon bases the calculator accepts.
* `required_el_k_quantities(calc)` — outer-k electron-state quantities the calculator needs (the
  outer-q driver computes the union; override to skip velocity/position).

See `docs/writing_a_calculator.md` for a worked example. The public (unexported) calculator API,
reachable as `ElectronPhonon.<name>`, is: `AbstractCalculator`, `supports`, `setup_calculator!`,
`run_calculator!`, `postprocess_calculator!`, `calculator_begin!`, `calculator_end!`, the loop tags
`OuterKLoop` / `OuterQLoop`, the scope tags `OuterIteration` / `OuterIterationBatch`, the payloads
`AbstractElPhPayload` / `EPData` / `EPDataQBatched` / `EPDataKBatched`,
`LoopContext` with the loop modes `SingleMode` / `BatchedMode` and the shared device stacks
`ElKDeviceStacks`, the backends `CPUBackend` / `GPUBackend` with `alloc` / `free_bytes` /
`synchronize`, `eph_window_scatter!`, `eph_batched_bytes_per_point`, `allowed_eph_phonon_basis`,
`required_el_k_quantities`, `required_el_k_device_stacks`, `_indmap_to_device`, the tiled
device-output helper `TiledDeviceOutput` (with `tile_begin!` / `tile_download!` / `tile_free!` /
`device_array` / `host_array` / `tile_offset` / `tile_length` / `tile_stride` / `is_block` /
`is_allocated`), the device-memory batch-sizing helpers `plan_batch` / `estimate_device_memory`, and
`to_device`.
"""
abstract type AbstractCalculator end


# =============================================================================
#  Payload family — generalizations of `EPState`
#
#  A payload carries all per-call data in typed fields (self-describing), and the loop-level state
#  lives in `LoopContext`. `run_calculator!` dispatches on the payload type, so the interface grows
#  by adding payload/scope *types*, never new hook *names*.

abstract type AbstractElPhPayload end

"""
    EPData{FT, DGT} <: AbstractElPhPayload

Host per-(k, q) point payload — a light immutable wrapper of the reused `EPState` buffer plus the
per-point indices. Immutable and small, so constructing one per (k, q) is free (stack-allocated).

Fields:
- `epdata`   :: `EPState{FT}` — the thread's reused e-ph data buffer (states + matrix elements).
- `ik`       :: outer k-point index.
- `iq`       :: q-point index, or `nothing` when phonons are not precomputed.
- `ikq`      :: k+q-point index, or `nothing` when k+q states are computed on the fly.
- `xk`, `xq` :: the k / q vectors.
- `id_chunk` :: CPU thread-chunk id (selects the calculator's per-thread buffer).
- `epdata_dg`:: covariant-derivative dg (`OffsetArray`), or `nothing`.
"""
struct EPData{FT, DGT} <: AbstractElPhPayload
    epdata    :: EPState{FT}
    ik        :: Int
    iq        :: Union{Int, Nothing}
    ikq       :: Union{Int, Nothing}
    xk        :: Vec3{FT}
    xq        :: Vec3{FT}
    id_chunk  :: Int
    epdata_dg :: DGT
end

"""
    EPDataQBatched{AT4C, AT4R, AT2, VI} <: AbstractElPhPayload

Device payload of the GPU outer-k loop (`run_eph_over_k_and_kq`, `use_gpu`): one outer k-point with a
batch of its k+q points. Runs on the backend of `ep_kq` (CPU or GPU); consumers stay backend-generic
(`similar`, `copyto!`, broadcasting, scatter) so no CUDA dependency leaks in.

Fields:
- `ep_kq`         :: `(nbandkq, nbandk, nmodes, nq)` complex — eigenbasis e-ph matrix (raw elements).
- `g2`            :: `(nbandkq, nbandk, nmodes, nq)` real — `|ep|²/(2ω)`, always formed by the loop's
  fused kernel (separate type param `AT4R` from `ep_kq`'s `AT4C`: complex vs real element type).
- `ωq`            :: `(nmodes, nq)` — phonon frequencies of this batch.
- `ik`            :: outer k index.
- `ikqs`          :: `(nq,)` k+q indices of this batch (device).
- `ibandk_offset` :: k-side window-projection band offset (0-based; 0 for full-band). `ep_kq`'s
  band-of-k axis `n` (1-based) is PHYSICAL band `ibandk_offset + n`.
"""
struct EPDataQBatched{AT4C, AT4R, AT2, VI} <: AbstractElPhPayload
    ep_kq         :: AT4C
    g2            :: AT4R
    ωq            :: AT2
    ik            :: Int
    ikqs          :: VI
    ibandk_offset :: Int
end

"""
    EPDataKBatched{AT4, AT3, AT2, AT1, VK} <: AbstractElPhPayload

Device payload of the GPU outer-q loop (`run_eph_over_q_and_k`, `use_gpu`): one phonon momentum q with
a batch of outer k-points. Runs on the backend of `ep_kq`; consumers stay backend-generic.

All fields are trimmed to the batch's actual width `nk` (the outer-k convention): the final partial
batch has `nk < n_batch_max`, and the loop hands width-`nk` views into its internally padded staging
buffers (a trailing-prefix device view, so it stays contiguous). A consumer therefore reads its own
size from any field (e.g. `size(ep_kq, 4)`) and never sees a padded tail. (The loop still zeros the
internal `wtk` padding as defense-in-depth, but that padding is not exposed here.)

Fields (`m` = k+q band, `n` = k band, `k` = batch column):
- `ep_kq` :: `(nw, nw, nmodes, nk)` — eigenbasis e-ph matrix. Out-of-window bands are already zeroed
  (the loop window-masks the eigenvector columns on both sides).
- `e_k`   :: `(nw, nk)` — k-side band energies (all `nw` bands).
- `e_kq`  :: `(nw, nk)` — k+q-side band energies.
- `uk`    :: `(nw, nw, nk)` — k-side eigenvectors, zero-padded outside the window.
- `ukq`   :: `(nw, nw, nk)` — k+q-side eigenvectors, zero-padded outside the window.
- `wtk`   :: `(nk,)` — k-point integration weights.
- `xks`   :: `(nk,)` — host `Vec3` list of the batch's k-vectors (for k-dependent phases).
- `iq`    :: q-point index.
"""
struct EPDataKBatched{AT4, AT3, AT2, AT1, VK} <: AbstractElPhPayload
    ep_kq :: AT4
    e_k   :: AT2
    e_kq  :: AT2
    uk    :: AT3
    ukq   :: AT3
    wtk   :: AT1
    xks   :: VK
    iq    :: Int
end


# =============================================================================
#  LoopContext — loop-level state, carried into every hook.

"""
    ElKDeviceStacks

Whole-run device stacks the outer-k GPU loop (`run_eph_over_k_and_kq`, `use_gpu`) builds once and
exposes to batched calculators via `LoopContext.el_k_stacks`, so a device-resident outer-k calculator
does not hand-upload its own copy. Built lazily and only for the quantities some calculator declares
through [`required_el_k_device_stacks`](@ref); an undeclared quantity's field is `nothing`. The
per-(band, k) energy grids are addressed by *physical* band `1:nw` (out-of-window entries are 0), so
a calculator derives its flattened per-state view by gathering through its own `(iband, ik)` index
map. Fields (any subset may be `nothing`):
- `e_k`  :: `(nw, nk)`   — outer-k band energies.
- `e_kq` :: `(nw, nkq)`  — k+q band energies.
- `wtk`  :: `(nk,)`      — outer-k integration weights.
- `wtq`  :: `(nkq,)`     — k+q integration weights.
"""
struct ElKDeviceStacks
    e_k  :: Any
    e_kq :: Any
    wtk  :: Any
    wtq  :: Any
end

# Loop-mode singletons name the SHAPE of the loop driving a hook, independent of the backend: a hook
# that must behave differently for the per-(k, q) host loop vs the device-batched loop dispatches on
# the mode, NOT the backend. The two are orthogonal — the batched loop still fires the per-iteration
# `OuterIteration` brackets, so `LoopContext{<:GPUBackend}` alone cannot tell a per-point hook from a
# per-batch one (it would run a CPU per-point reduction inside the batched loop).
abstract type LoopMode end
struct SingleMode <: LoopMode end     # per-(k, q) host inner loops (CPU paths)
struct BatchedMode <: LoopMode end   # device-batched loops (one outer index, a batch of inner ones)

"""
    LoopContext{BT <: AbstractBackend, MT <: LoopMode}

Loop-level state passed to every calculator hook. Replaces the ad-hoc per-hook kwargs (`ik`, `iq`,
`gpu_array`, `nk_batch_max`, `kstart`, `kend`) with one typed object.

`BT` is the backend type and `MT` the loop mode; the backend is first, so a partial annotation
`LoopContext{<:GPUBackend}` still names "any mode on a GPU backend". Backend-dependent hooks dispatch
on the *mode* (`LoopContext{<:AbstractBackend, SingleMode}` / `{<:AbstractBackend, BatchedMode}`), not
the backend, so the batched loop's per-iteration bracket does not collide with the per-point one.

Fields:
- `backend`     :: `CPUBackend()` or `GPUBackend(proto)` — allocation / free / synchronize routes.
- `mode`        :: `SingleMode()` (per-(k, q) host loop) or `BatchedMode()` (device-batched loop).
- `outer_index` :: current outer index (`ik` for outer-k loops, `iq` for the outer-q loop); `0` at
  batch scope.
- `batch`       :: outer-iteration range of the current batch (`1:0` on the CPU paths).
- `n_batch_max` :: loop batch cap, for device-buffer sizing.
- `el_k_stacks` :: the run's shared [`ElKDeviceStacks`](@ref), or `nothing` — populated only on the
  outer-k GPU loop when a calculator declares need (see [`required_el_k_device_stacks`](@ref)).
"""
struct LoopContext{BT <: AbstractBackend, MT <: LoopMode}
    backend     :: BT
    mode        :: MT
    outer_index :: Int
    batch       :: UnitRange{Int}
    n_batch_max :: Int
    el_k_stacks :: Union{Nothing, ElKDeviceStacks}
end

# Default the shared device stacks to `nothing`, so the CPU / outer-q / test construction sites that
# do not build them keep their 5-argument call unchanged (only the outer-k GPU loop passes the stacks).
LoopContext(backend::AbstractBackend, mode::LoopMode, outer_index::Integer, batch::UnitRange,
    n_batch_max::Integer) =
    LoopContext(backend, mode, outer_index, batch, n_batch_max, nothing)

# SingleMode context (CPU per-(k, q) host loops): there is no batch and no device stacks, so a
# SingleMode context cannot be handed a spurious batch — fill `batch = 1:0`, `n_batch_max = 0`,
# `el_k_stacks = nothing` automatically.
LoopContext(backend::AbstractBackend, ::SingleMode, outer_index::Integer) =
    LoopContext(backend, SingleMode(), outer_index, 1:0, 0, nothing)

# BatchedMode context at batch scope (device loops): there is no single outer index spanning the whole
# batch, so `outer_index = 0` is the "no single outer index — use `batch`" sentinel.
LoopContext(backend::AbstractBackend, ::BatchedMode, batch::UnitRange, n_batch_max::Integer,
    el_k_stacks::Union{Nothing, ElKDeviceStacks}) =
    LoopContext(backend, BatchedMode(), 0, batch, n_batch_max, el_k_stacks)


# =============================================================================
#  Capability trait — one extensible declaration replacing the `allow_*` quartet.

# Loop-shape tags: a calculator declares compatibility with a driver's loop shape.
abstract type LoopTag end
struct OuterKLoop <: LoopTag end    # run_eph_over_k_and_kq / run_eph_over_k_and_q (outer k)
struct OuterQLoop <: LoopTag end    # run_eph_over_q_and_k (outer q)

"""
    supports(calc, ::Type{T}) -> Bool

Declare that `calc` handles loop shape `T` (an `OuterKLoop` / `OuterQLoop` tag type) or payload `T`
(an `AbstractElPhPayload` type). Default `false` for both families. The drivers check this up front
and fail loudly if a calculator does not support the loop/payload they will hand it.

The second argument must be a *type* (e.g. `supports(calc, OuterKLoop)`), not an instance: a non-Type
argument throws, so a typo like `supports(calc, OuterKLoop())` fails loudly instead of silently
returning `false`.
"""
supports(::AbstractCalculator, ::Type{<:LoopTag}) = false
supports(::AbstractCalculator, ::Type{<:AbstractElPhPayload}) = false
supports(::AbstractCalculator, x) = error(
    "supports(calc, x) expects a loop-tag or payload TYPE (e.g. supports(calc, OuterKLoop) or " *
    "supports(calc, EPData)); got x::$(typeof(x)). Pass the Type, not an instance.")


# =============================================================================
#  Lifecycle: run-once initializers stay named; the per-iteration brackets unify.

"""
    allowed_eph_phonon_basis(calc::AbstractCalculator) -> Vector{Symbol}

Return the list of phonon bases the calculator supports for e-ph matrix elements.
- `:eigenmode`: e-ph coupling in phonon eigenmode basis (default)
- `:cartesian`: e-ph coupling in Cartesian displacement basis
"""
allowed_eph_phonon_basis(::AbstractCalculator) = [:eigenmode]

"""
    required_el_k_quantities(calc::AbstractCalculator) -> Vector{String}

Return the electron-state quantities at the outer k-points the calculator needs the driver to
compute. The outer-q driver (`run_eph_over_q_and_k`) computes the union over its calculators, so a
calculator that only reads eigenvalues/eigenvectors can override this to skip the
velocity/position interpolation (the dominant setup cost after the eigensolve). Default is the
conservative full list.
"""
required_el_k_quantities(::AbstractCalculator) = ["eigenvalue", "eigenvector", "velocity", "position"]

"""
    required_el_k_device_stacks(calc::AbstractCalculator) -> Vector{Symbol}

Whole-run device stacks the calculator's batched outer-k path needs the loop to build and expose via
`LoopContext.el_k_stacks` (see [`ElKDeviceStacks`](@ref)). The outer-k GPU loop
(`run_eph_over_k_and_kq`, `use_gpu`) builds the union over its calculators, so a calculator stops
hand-uploading data the loop already has. Valid symbols: `:e_k`, `:e_kq`, `:wtk`, `:wtq`. Default is
empty (no shared stacks needed).
"""
required_el_k_device_stacks(::AbstractCalculator) = Symbol[]

function setup_calculator!(::AbstractCalculator, kpts, qpts, el_states; kwargs...)
    error("setup_calculator! has to be implemented")
end

function postprocess_calculator!(::AbstractCalculator; kwargs...)
    error("postprocess_calculator! has to be implemented")
end

# Scope singletons name the iteration level being bracketed, so the call reads as a sentence:
# `calculator_begin!(calc, OuterIteration(), ctx)` = "at the beginning of one outer iteration".
struct OuterIteration end        # one iteration of the outer loop (one ik / one iq)
struct OuterIterationBatch end   # one batch of consecutive outer iterations (GPU loops)

# Begin/end brackets. Defaults are no-ops: a calculator that needs nothing per outer iteration /
# batch does not implement them (the mandatory surface is setup_calculator! / run_calculator! /
# postprocess_calculator!).
calculator_begin!(::AbstractCalculator, ::OuterIteration, ctx) = nothing
calculator_end!(::AbstractCalculator,   ::OuterIteration, ctx) = nothing
calculator_begin!(::AbstractCalculator, ::OuterIterationBatch, ctx) = nothing
calculator_end!(::AbstractCalculator,   ::OuterIterationBatch, ctx) = nothing


# =============================================================================
#  Execution hook — one function, dispatched on the payload type.
#
# `run_calculator!(calc, payload, ctx)`:
#   * `EPData`         — host per-(k, q) callback (CPU inner loops).
#   * `EPDataQBatched` — device, outer-k loop (one k, batch of k+q).
#   * `EPDataKBatched` — device, outer-q loop (one q, batch of k).
# There is deliberately no catch-all default: calling a payload a calculator does not implement is a
# `MethodError`, and the drivers reject unsupported calculators up front via `supports`. The empty
# generic-function declaration below creates the binding that calculators extend.
function run_calculator! end

"""
    eph_batched_bytes_per_point(calc, ::Type{<:AbstractElPhPayload}; nw, nmodes) -> Int

Device bytes of per-point scratch the calculator's batched `run_calculator!` path holds (its
workspace arrays sized `(…, batch)`, divided by the batch width). The GPU loops sum this over the
calculators and combine it with their own per-point staging cost to derive a memory-adaptive batch
width from `free_bytes`. Default `0` (no per-point device scratch).
"""
eph_batched_bytes_per_point(::AbstractCalculator, ::Type{<:AbstractElPhPayload}; nw, nmodes) = 0


# =============================================================================
#  Public (but unexported) calculator API. `public` (Julia ≥ 1.11) marks these names as the
#  supported interface without exporting them (users still reach them as `ElectronPhonon.<name>`).
#  Gated so the package still parses on older Julia (`Project.toml` compat is `julia = "1"`); the
#  `public` keyword lives inside a string so it is never parsed on a Julia that lacks it.
#  `eph_window_scatter!` (calculator_utils.jl) and the backend primitives (gpu_utils.jl) are marked
#  here too — `public`, like `export`, permits forward references to names defined later in the module.
if VERSION >= v"1.11.0-DEV.469"
    Core.eval(@__MODULE__, Meta.parse(
        "public AbstractCalculator, supports, setup_calculator!, run_calculator!, " *
        "postprocess_calculator!, calculator_begin!, calculator_end!, " *
        "OuterKLoop, OuterQLoop, OuterIteration, OuterIterationBatch, " *
        "AbstractElPhPayload, EPData, EPDataQBatched, EPDataKBatched, " *
        "LoopContext, SingleMode, BatchedMode, ElKDeviceStacks, CPUBackend, GPUBackend, alloc, free_bytes, synchronize, " *
        "eph_window_scatter!, eph_batched_bytes_per_point, allowed_eph_phonon_basis, " *
        "required_el_k_quantities, required_el_k_device_stacks, _indmap_to_device, " *
        "TiledDeviceOutput, tile_begin!, tile_download!, tile_free!, device_array, host_array, " *
        "tile_offset, tile_length, tile_stride, is_block, is_allocated, residency_use_block, " *
        "to_device, plan_batch, estimate_device_memory"))
end
