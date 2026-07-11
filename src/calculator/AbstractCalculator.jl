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
returns `true`, the GPU loop keeps the e-ph matrix for a whole `(k, {k+q})` chunk on the
device and calls `run_calculator_batched!` once per chunk — skipping the per-`(k,q)` host
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
- `ep_kq` :: `(nbandkq, nbandk, nmodes, nq)` — eigenbasis e-ph matrix (the raw matrix elements,
  *not* `g2`; the calculator forms `g2 = |ep|²/2ω` itself if it needs it).
- `ωq`    :: `(nmodes, nq)` — phonon frequencies for each q in the chunk.
- `ik`    :: outer k-point index.
- `ikqs`  :: the `nq` k+q point indices of this chunk (so the calculator can address its inner
  states).

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

"""
    eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                        nbandkq, nbandk, nm, nqc, n_i)

Device-resident scatter for a calculator that keeps `g2`/`ωq` on the device (no per-chunk
host streaming). For every `(m, n, ν, j)` entry of `g2vals` `(nbandkq, nbandk, nm, nqc)`, look
up the state indices `i = imap_i_col[n]` and `f = imap_f[m, ikqs[j]]`; if both are in-window
(`> 0`), write the value into the mode-fastest linear slot
`lin = ν + nm·(i−1) + nm·n_i·(f−1)` of the flat `g2_out` / `ωq_out`
(`ω = ωq[ν, j]`). The target `lin` indices are unique across the run (distinct k → distinct i,
distinct k+q → distinct f), so the writes never collide (no atomics needed). Generic
(CPU/fallback) method; the CUDA extension provides a one-kernel `CuArray` method.
"""
function eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, n_i::Int)
    @inbounds for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[j]]
        if i > 0 && f > 0
            lin = ν + nm * (i - 1) + nm * n_i * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, j]
            ωq_out[lin] = ωq[ν, j]
        end
    end
    nothing
end
