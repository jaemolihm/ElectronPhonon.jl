
# Functions related to finding Fermi energy

using Roots

export find_fermi_energy

"""
    compute_ncarrier(μ, T, energy, weights)
Compute carrier density.
- μ: Chemical potential.
- T: temperature
- energy: band energy
- weights: k-point weights.
"""
function compute_ncarrier(μ, T, energy::AbstractArray{R, 2}, weights) where {R <: Real}
    nk = size(energy, 2)
    @assert length(weights) == nk
    ncarrier = zero(R)
    for ik in 1:nk
        for iband in 1:size(energy, 1)
            ncarrier += weights[ik] * occ_fermion(energy[iband, ik] - μ, T)
        end
    end
    ncarrier
end

function compute_ncarrier(μ, T, energy::AbstractArray{R, 1}, weights) where {R <: Real}
    n = length(energy)
    @assert length(weights) == n
    ncarrier = zero(R)
    for i in 1:n
        ncarrier += weights[i] * occ_fermion(energy[i] - μ, T)
    end
    ncarrier
end

"""
    find_chemical_potential(ncarrier, T, energy, weights)
Find chemical potential for target carrier density using bisection.
- `ncarrier`: target carrier density (electron or hole density)
- `T`: temperature
- `energy`: band energy
- `weights`: k-point weights.
"""
function find_chemical_potential(ncarrier, T, energy, weights)
    # FIXME: T=0 case
    # TODO: MPI

    # Get rough bounds for μ
    min_μ = minimum(energy) - 10
    max_μ = maximum(energy) + 10

    # Solve func(μ) = ncarrier(μ) - ncarrier = 0
    # I use let block to avoid type instability.
    # See https://github.com/JuliaLang/julia/issues/15276
    # and https://discourse.julialang.org/t/type-instability-of-nested-function/57007
    let ncarrier=ncarrier,
        func(μ) = compute_ncarrier(μ, T, energy, weights) - ncarrier
        Roots.find_zero(func, (min_μ, max_μ), Roots.Bisection(), atol=eps(ncarrier))
    end
end
