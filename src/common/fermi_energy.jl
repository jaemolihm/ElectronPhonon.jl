
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
function compute_ncarrier(μ, T, energy, weights)
    compute_ncarrier(μ, T, energy, weights, 1, size(energy, 1))
end

function compute_ncarrier(μ, T, energy, weights, iband_from, iband_to)
    nk = size(energy, 2)
    @assert length(weights) == nk
    ncarrier = eltype(energy)(0)
    for ik in 1:nk
        for iband in iband_from:iband_to
            ncarrier += weights[ik] * occ_fermion(energy[iband, ik] - μ, T)
        end
    end
    ncarrier
end

"""
    find_chemical_potential(ncarrier, T, energy, weights, nband_valence)
Find chemical potential for target carrier density using bisection.
- `ncarrier`: target carrier density (electron or hole density)
- `T`: temperature
- `energy`: band energy
- `weights`: k-point weights.
- `nband_valence`: Number of valence bands. It is assumed that `nband_valence` bands are
    occupied at every k points not included.
"""
function find_chemical_potential(ncarrier, T, energy, weights, nband_valence)
    # FIXME: T=0 case
    # TODO: MPI
    ncarrier += nband_valence * sum(weights)

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
