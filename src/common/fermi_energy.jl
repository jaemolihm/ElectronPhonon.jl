
# Functions related to finding Fermi energy

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
    find_chemical_potential(ncarrier_target, T, energy, weights)
Find chemical potential for target carrier density using bisection.
- ncarrier_target: target carrier density (electron or hole density)
- T: temperature
- energy: band energy
- weights: k-point weights.
- carrier_type: "e" or "h".
- nband_valence: Number of valence bands. Needed for hole carrier.
"""
function find_chemical_potential(ncarrier_target, T, energy, weights, carrier_type,
                                 nband_valence)
    if carrier_type == "e"
        ib_from = nband_valence + 1
        ib_to = size(energy, 1)
    elseif carrier_type == "h"
        ib_from = 1
        ib_to = nband_valence
        # We need the hole carrier density, not total electron density.
        # n(μ) = sum_k (params.nband_valence + n_carrier)
        # Hence, we add the contribution of the valence bands to ncarrier_target.
        ncarrier_target += nband_valence * sum(weights)
    else
        error("carrier_type must be e or h, not $carrier_type")
    end

    # Solve func(μ) = ncarrier(μ) - ncarrier_target = 0
    # I use let block to avoid type instability.
    # See https://github.com/JuliaLang/julia/issues/15276
    # and https://discourse.julialang.org/t/type-instability-of-nested-function/57007
    let ncarrier_target=ncarrier_target, ib_from=ib_from, ib_to=ib_to
        func(μ) = compute_ncarrier(μ, T, energy, weights, ib_from, ib_to) - ncarrier_target
        return bisect(func, -1.0e4, 1.0e4, tolf=ncarrier_target*1E-10)
    end
end
