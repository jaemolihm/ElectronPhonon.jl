"""
    AbstractCalculator

A calculator is a type for calculating properties of the system. In `run_eph_outer_k`,
once all the electron and phonon states and the e-ph matrix elements are calculated,
calculators are called and these information are passed as arguments.
Each calculator implement their own calculation of different properties of the system.

Users may subtype `AbstractCalculator` and define their own calculator.
Each Calculator should implement the following functions.
* `setup_calculator!(calc, kpts, qpts, el_states; kwargs...)`
* `run_calculator!(calc, epdata, iq; kwargs...)`
* `postprocess_calculator!(calc; kwargs...)`
* `allow_eph_outer_k(::AbstractCalculator)`
* `allow_eph_outer_q(::AbstractCalculator)`

Each function can take additional keyword arguments. The argument should specify the
used kwargs, and end with `kwargs...` to skip the unused ones.
"""
abstract type AbstractCalculator end

"""
Each calculator should allow one or both of these two options.
"""
allow_eph_outer_k(::AbstractCalculator) = false
allow_eph_outer_q(::AbstractCalculator) = false

"""
    allowed_eph_phonon_basis(calc::AbstractCalculator) -> Vector{Symbol}

Return the list of phonon bases the calculator supports for e-ph matrix elements.
- `:eigenmode`: e-ph coupling in phonon eigenmode basis (default)
- `:cartesian`: e-ph coupling in Cartesian displacement basis

Default implementation returns `[:eigenmode]` only.
"""
allowed_eph_phonon_basis(::AbstractCalculator) = [:eigenmode]


function setup_calculator!(::AbstractCalculator, kpts, qpts, el_states; kwargs...)
    error("setup_calculator! has to be implemented")
end

function setup_calculator_inner!(::AbstractCalculator; kwargs...)
    error("setup_calculator_inner! has to be implemented")
end

function run_calculator!(::AbstractCalculator, epdata, ik, iq, ikq; kwargs...)
    error("run_calculator! has to be implemented")
end

"""
    allow_eph_batched(calc::AbstractCalculator) -> Bool

Whether the calculator implements the batched device hook [`run_calculator_batched!`] used by
the GPU loop (`run_eph_over_k_and_kq` with `use_gpu = true`). When every calculator in a run
returns `true`, the GPU loop keeps the e-ph matrix for a whole `(k, {k+q})` batch on the
device and calls `run_calculator_batched!` once per batch — skipping the per-`(k,q)` host
`run_calculator!` callback and the device→host copy of the complex e-ph matrix. The default
opts out, so calculators keep using the host `run_calculator!` path.
"""
allow_eph_batched(::AbstractCalculator) = false

"""
    run_calculator_batched!(calc, ep_kq, ωq, ik, ikqs; kwargs...)

Batched hook invoked by `run_eph_over_k_and_kq` when `use_gpu = true`. That loop is an outer-k
loop with the inner k+q points batched, so this is called once per outer k-point `ik` with all
of that k's k+q points at once (the outer-q loop `run_eph_outer_q` has no batched hook). Consume
the e-ph matrix for the list of k+q points:
- `ep_kq` :: `(nbandkq, nbandk, nmodes, nq)` — eigenbasis e-ph matrix (the raw matrix elements).
- `ωq`    :: `(nmodes, nq)` — phonon frequencies for each q in the batch.
- `ik`    :: outer k-point index.
- `ikqs`  :: the `nq` k+q point indices of this batch (so the calculator can address its inner
  states).

Keyword argument:
- `g2` :: same shape as `ep_kq` — the loop passes `g2 = |ep|²/(2ω)` already formed (the GPU path
  folds it into the rotation kernel for free), so a calculator that needs `g2` should use this
  rather than recomputing it from `ep_kq`.

Runs on the backend of `ep_kq` (CPU or GPU); implementations should stay backend-generic
(`similar`, `copyto!`, broadcasting, scatter assignment) so no CUDA dependency leaks into the
calculator.
"""
function run_calculator_batched!(::AbstractCalculator, ep_kq, ωq, ik, ikqs; kwargs...)
    error("run_calculator_batched! has to be implemented (or set allow_eph_batched = false)")
end

function postprocess_calculator_inner!(::AbstractCalculator; kwargs...)
    error("postprocess_calculator_inner! has to be implemented")
end

function postprocess_calculator!(::AbstractCalculator; kwargs...)
    error("postprocess_calculator! has to be implemented")
end

# Optional per-outer-batch hooks for the batched GPU loop (`run_eph_over_k_and_kq`, use_gpu). That
# loop processes the outer-k points in consecutive BATCHES of `nk_batch_max` points (one batched
# interpolation per batch). These hooks let a calculator bound its device memory: rather than hold
# output for all outer-k states at once (device memory ∝ grid size), an *outer-batch-resident*
# calculator keeps only the CURRENT batch's output on the device and copies it to the host each batch.
# `setup_calculator_outer_batch!` (re)points/zeros the batch-sized device buffer at the start of a
# batch; `flush_calculator_outer_batch!` copies it to the host at the batch's end. Both are no-ops by
# default — a calculator that holds its whole output ignores them. `proto` is a device array (the e-ph
# matrix backend) for buffer allocation / free-memory queries; `kstart`/`kend` are the batch's outer-k range.
setup_calculator_outer_batch!(::AbstractCalculator; kwargs...) = nothing
flush_calculator_outer_batch!(::AbstractCalculator; kwargs...) = nothing
