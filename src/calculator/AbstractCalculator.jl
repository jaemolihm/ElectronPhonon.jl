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
"""
abstract type AbstractCalculator end


# =============================================================================
#  Payload family — generalizations of `ElPhData`
#
#  A payload carries all per-call data in typed fields (self-describing), and the loop-level state
#  lives in `LoopContext`. `run_calculator!` dispatches on the payload type, so the interface grows
#  by adding payload/scope *types*, never new hook *names*.

abstract type AbstractElPhPayload end

"""
    ElPhDataPoint{FT, DGT} <: AbstractElPhPayload

Host per-(k, q) point payload — a light immutable wrapper of the reused `ElPhData` buffer plus the
per-point indices. Immutable and small, so constructing one per (k, q) is free (stack-allocated).

Fields:
- `epdata`   :: `ElPhData{FT}` — the thread's reused e-ph data buffer (states + matrix elements).
- `ik`       :: outer k-point index.
- `iq`       :: q-point index, or `nothing` when phonons are not precomputed.
- `ikq`      :: k+q-point index, or `nothing` when k+q states are computed on the fly.
- `xk`, `xq` :: the k / q vectors.
- `id_chunk` :: CPU thread-chunk id (selects the calculator's per-thread buffer).
- `epdata_dg`:: covariant-derivative dg (`OffsetArray`), or `nothing`.
"""
struct ElPhDataPoint{FT, DGT} <: AbstractElPhPayload
    epdata    :: ElPhData{FT}
    ik        :: Int
    iq        :: Union{Int, Nothing}
    ikq       :: Union{Int, Nothing}
    xk        :: Vec3{FT}
    xq        :: Vec3{FT}
    id_chunk  :: Int
    epdata_dg :: DGT
end

"""
    ElPhDataOuterKBatched{AT4C, AT4R, AT2, VI} <: AbstractElPhPayload

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
struct ElPhDataOuterKBatched{AT4C, AT4R, AT2, VI} <: AbstractElPhPayload
    ep_kq         :: AT4C
    g2            :: AT4R
    ωq            :: AT2
    ik            :: Int
    ikqs          :: VI
    ibandk_offset :: Int
end

"""
    ElPhDataOuterQBatched{AT4, AT3, AT2, AT1, VK} <: AbstractElPhPayload

Device payload of the GPU outer-q loop (`run_eph_over_q_and_k`, `use_gpu`): one phonon momentum q with
a batch of outer k-points. Runs on the backend of `ep_kq`; consumers stay backend-generic.

Fields (`m` = k+q band, `n` = k band, `k` = batch column):
- `ep_kq` :: `(nw, nw, nmodes, nk)` — eigenbasis e-ph matrix. Out-of-window bands are already zeroed
  (the loop window-masks the eigenvector columns on both sides).
- `e_k`   :: `(nw, nk)` — k-side band energies (all `nw` bands).
- `e_kq`  :: `(nw, nk)` — k+q-side band energies.
- `uk`    :: `(nw, nw, nk)` — k-side eigenvectors, zero-padded outside the window.
- `ukq`   :: `(nw, nw, nk)` — k+q-side eigenvectors, zero-padded outside the window.
- `wtk`   :: `(nk,)` — k-point integration weights; padded tail entries are 0, so a consumer may
  operate on the full (padded) batch without special-casing the final partial batch.
- `xks`   :: `(nk,)` — host `Vec3` list of the batch's k-vectors (for k-dependent phases).
- `iq`    :: q-point index.
"""
struct ElPhDataOuterQBatched{AT4, AT3, AT2, AT1, VK} <: AbstractElPhPayload
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
    LoopContext{BT <: AbstractBackend}

Loop-level state passed to every calculator hook. Replaces the ad-hoc per-hook kwargs (`ik`, `iq`,
`gpu_array`, `nk_batch_max`, `kstart`, `kend`) with one typed object.

Fields:
- `backend`     :: `CPUBackend()` or `GPUBackend(proto)` — allocation / free / synchronize routes.
- `outer_index` :: current outer index (`ik` for outer-k loops, `iq` for the outer-q loop); `0` at
  batch scope.
- `batch`       :: outer-iteration range of the current batch (`1:0` on the CPU paths).
- `n_batch_max` :: loop batch cap, for device-buffer sizing.
"""
struct LoopContext{BT <: AbstractBackend}
    backend     :: BT
    outer_index :: Int
    batch       :: UnitRange{Int}
    n_batch_max :: Int
end


# =============================================================================
#  Capability trait — one extensible declaration replacing the `allow_*` quartet.

# Loop-shape tags: a calculator declares compatibility with a driver's loop shape.
struct OuterKLoop end    # run_eph_over_k_and_kq / run_eph_over_k_and_q (outer k)
struct OuterQLoop end    # run_eph_over_q_and_k (outer q)

"""
    supports(calc, x) -> Bool

Declare that `calc` handles loop shape `x` (an `OuterKLoop` / `OuterQLoop` tag type) or payload `x`
(an `AbstractElPhPayload` type). Default `false`. The drivers check this up front and fail loudly if
a calculator does not support the loop/payload they will hand it.
"""
supports(::AbstractCalculator, ::Any) = false


# =============================================================================
#  Lifecycle: run-once initializers stay named; the per-iteration brackets unify.

"""
    allowed_eph_phonon_basis(calc::AbstractCalculator) -> Vector{Symbol}

Return the list of phonon bases the calculator supports for e-ph matrix elements.
- `:eigenmode`: e-ph coupling in phonon eigenmode basis (default)
- `:cartesian`: e-ph coupling in Cartesian displacement basis
"""
allowed_eph_phonon_basis(::AbstractCalculator) = [:eigenmode]

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
#   * `ElPhDataPoint`         — host per-(k, q) callback (CPU inner loops).
#   * `ElPhDataOuterKBatched` — device, outer-k loop (one k, batch of k+q).
#   * `ElPhDataOuterQBatched` — device, outer-q loop (one q, batch of k).
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
