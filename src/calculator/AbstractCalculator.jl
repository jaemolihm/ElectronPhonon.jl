"""
    AbstractCalculator

A calculator is a type for calculating properties of the system. In `run_eph_over_k_and_q`,
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

# Called once per outer-loop iteration, before the multithreaded / batched inner loop: `ik` on the
# outer-k loops, `iq` on the outer-q loop. On the GPU outer-q path it also brackets the per-q device
# accumulator — it receives `gpu_array` (a device array on the e-ph matrix backend) and `nk_batch_max`
# so a batched calculator can lazily allocate + zero its per-q device buffer here, with the matching
# device→host scatter in `postprocess_calculator_inner!` at the end of the q.
function setup_calculator_inner!(::AbstractCalculator; kwargs...)
    error("setup_calculator_inner! has to be implemented")
end

function run_calculator!(::AbstractCalculator, epdata, ik, iq, ikq; kwargs...)
    error("run_calculator! has to be implemented")
end

# The two GPU batched-calculator hooks are named for which momentum is the OUTER loop and which is
# batched on the INNER axis:
#   * `*_outer_k_batched*`  — the outer-k loop `run_eph_over_k_and_kq` (outer k, inner k+q batched)
#   * `*_outer_q_batched*`  — the outer-q loop `run_eph_over_q_and_k`   (outer q, inner k   batched)

"""
    allow_eph_outer_k_batched(calc::AbstractCalculator) -> Bool

Whether the calculator implements the batched device hook [`run_calculator_outer_k_batched!`] used
by the GPU outer-k loop (`run_eph_over_k_and_kq` with `use_gpu = true`) — outer k, inner k+q
batched. When every calculator in a run returns `true`, the GPU loop keeps the e-ph matrix for a
whole `(k, {k+q})` batch on the device and calls `run_calculator_outer_k_batched!` once per batch —
skipping the per-`(k,q)` host `run_calculator!` callback and the device→host copy of the complex
e-ph matrix. The default opts out, so calculators keep using the host `run_calculator!` path.
"""
allow_eph_outer_k_batched(::AbstractCalculator) = false

"""
    run_calculator_outer_k_batched!(calc, ep_kq, ωq, ik, ikqs; g2=nothing, ibandk_offset=0, kwargs...)

Batched hook invoked by `run_eph_over_k_and_kq` when `use_gpu = true` (outer k, inner k+q batched),
called once per outer k-point `ik` with all of that k's k+q points at once. Consume the e-ph matrix
for the list of k+q points:
- `ep_kq` :: `(nbandkq, nbandk, nmodes, nq)` — eigenbasis e-ph matrix (the raw matrix elements).
- `ωq`    :: `(nmodes, nq)` — phonon frequencies for each q in the batch.
- `ik`    :: outer k-point index.
- `ikqs`  :: the `nq` k+q point indices of this batch (so the calculator can address its inner
  states).

Keyword arguments:
- `g2`   :: same shape as `ep_kq` — the loop passes `g2 = |ep|²/(2ω)` already formed (the GPU path
  folds it into the rotation kernel for free), so a calculator that needs `g2` should use this
  rather than recomputing it from `ep_kq`.
- `ibandk_offset` :: k-side window-projection band offset, **0-based** — `ep_kq`'s band-of-k axis `n`
  (1-based, `1:nbandk`) corresponds to PHYSICAL band `ibandk_offset + n` (so `n = 1` is physical band
  `ibandk_offset + 1`). The loop rotates only an `nbandk`-wide contiguous eigenvector window around the
  in-window bands; `ibandk_offset = 0` for full-band runs (then physical band `= n`).

Runs on the backend of `ep_kq` (CPU or GPU); implementations should stay backend-generic
(`similar`, `copyto!`, broadcasting, scatter assignment) so no CUDA dependency leaks into the
calculator.
"""
function run_calculator_outer_k_batched!(::AbstractCalculator, ep_kq, ωq, ik, ikqs; kwargs...)
    error("run_calculator_outer_k_batched! has to be implemented (or set allow_eph_outer_k_batched = false)")
end

"""
    allow_eph_outer_q_batched(calc::AbstractCalculator) -> Bool

Whether the calculator implements the batched device hook [`run_calculator_outer_q_batched!`] used
by the GPU outer-q loop (`run_eph_over_q_and_k` with `use_gpu = true`) — outer q, inner k batched. When
every calculator in a run returns `true`, the GPU loop keeps the e-ph matrix for a whole
`(q, {k-batch})` on the device and calls `run_calculator_outer_q_batched!` once per k-batch —
skipping the per-`(k,q)` host `run_calculator!` callback and the device→host copy of the e-ph
matrix. The default opts out. This is the outer-q sibling of [`allow_eph_outer_k_batched`].
"""
allow_eph_outer_q_batched(::AbstractCalculator) = false

"""
    run_calculator_outer_q_batched!(calc, ep_kq, e_k, e_kq, uk, ukq, wtk, xks, iq; kwargs...)

Batched device hook for the GPU outer-q loop. For a fixed phonon momentum `q` (index `iq`),
consume the e-ph matrix and electron states for a batch of outer k-points at once:
- `ep_kq` :: `(nw, nw, nmodes, nkc)` — eigenbasis e-ph matrix on the device, index `[m, n, ν, k]`
  with `m` the k+q band, `n` the k band. Out-of-window bands are already zeroed (the loop
  window-masks the eigenvector columns on both sides), so out-of-window `(m,n)` contribute 0.
- `e_k`  :: `(nw, nkc)` — k-side band energies `e_k[n, k]` (device; all `nw` bands).
- `e_kq` :: `(nw, nkc)` — k+q-side band energies `e_kq[m, k]` (device).
- `uk`   :: `(nw, nw, nkc)` — k-side eigenvectors `uk[jw, n, k]`, zero-padded outside the window.
- `ukq`  :: `(nw, nw, nkc)` — k+q-side eigenvectors `ukq[iw, m, k]`, zero-padded outside the window.
- `wtk`  :: `(nkc,)` — k-point integration weights (device). Padded tail entries are 0, so a
  calculator may operate on the full (padded) batch without special-casing the final partial batch.
- `xks`  :: `(nkc,)` — the batch's k-vectors (host `Vec3` list), for any k-dependent phase factor.
- `iq`   :: q-point index.

The per-q lifecycle is bracketed by the generic [`setup_calculator_inner!`] (device buffer alloc /
zero at q start) and [`postprocess_calculator_inner!`] (device→host scatter at q end), the same
hooks the CPU loop uses per q. Runs on the backend of `ep_kq`; implementations should stay
backend-generic (`similar`, `copyto!`, broadcasting, `mul!`) so no CUDA dependency leaks into the
calculator.

Memory note: implementations typically preallocate device scratch sized `(…, k_batch)`, which also
scales with calculator-internal factors (number of frequencies, temperatures, stacked channels).
Declare that per-k footprint via [`eph_outer_q_batched_bytes_per_k`](@ref) so the loop's
memory-adaptive batch sizing accounts for it; otherwise a large `k_batch_size` can exhaust device
memory through the calculator's scratch alone.
"""
function run_calculator_outer_q_batched!(::AbstractCalculator, ep_kq, e_k, e_kq, uk, ukq, wtk, xks, iq; kwargs...)
    error("run_calculator_outer_q_batched! has to be implemented (or set allow_eph_outer_q_batched = false)")
end

"""
    eph_outer_q_batched_bytes_per_k(calc::AbstractCalculator; nw, nmodes) -> Int

Device bytes of per-k-point scratch that the calculator's [`run_calculator_outer_q_batched!`](@ref)
path holds for a k-batch (its workspace arrays sized `(…, k_batch)`, divided by the batch width).
The GPU outer-q loop sums this over the calculators and combines it with its own per-k staging cost
to derive a memory-adaptive batch width from `device_free_bytes`. Default `0`: the calculator
declares no per-k device scratch (the loop then budgets only its own buffers — override this if the
calculator allocates batch-sized device arrays, or a large batch can exhaust device memory).
"""
eph_outer_q_batched_bytes_per_k(::AbstractCalculator; nw, nmodes, kwargs...) = 0

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
# default — a calculator that holds its whole output ignores them. `gpu_array` is a device array (the e-ph
# matrix backend) for buffer allocation; `kstart`/`kend` are the batch's outer-k range.
setup_calculator_outer_batch!(::AbstractCalculator; kwargs...) = nothing
flush_calculator_outer_batch!(::AbstractCalculator; kwargs...) = nothing
