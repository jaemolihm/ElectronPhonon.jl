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


function setup_calculator!(::AbstractCalculator, kpts, qpts, el_states; kwargs...)
    error("setup_calculator! has to be implemented")
end

function setup_calculator_inner!(::AbstractCalculator; kwargs...)
    error("setup_calculator_inner! has to be implemented")
end

function run_calculator!(::AbstractCalculator, epdata, ik, iq, ikq; kwargs...)
    error("run_calculator! has to be implemented")
end

function postprocess_calculator_inner!(::AbstractCalculator; kwargs...)
    error("postprocess_calculator_inner! has to be implemented")
end

function postprocess_calculator!(::AbstractCalculator; kwargs...)
    error("postprocess_calculator! has to be implemented")
end
