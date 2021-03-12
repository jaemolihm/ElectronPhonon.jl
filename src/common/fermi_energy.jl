
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
    @assert length(weights) == size(energy, 2)
    nk = size(energy, 2)
    ncarrier = eltype(energy)(0)
    for ik in 1:nk
        for e in view(energy, :, ik)
            ncarrier += weights[ik] * occ_fermion(e - μ, T)
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
        energy_carrier = @view energy[nband_valence+1:end, :]
    elseif carrier_type == "h"
        energy_carrier = @view energy[1:nband_valence, :]
        # We need the hole carrier density, not total electron density.
        # n(μ) = sum_k (params.nband_valence + n_carrier)
        # Hence, we add the contribution of the valence bands to ncarrier_target.
        ncarrier_target += nband_valence * sum(weights)
    else
        error("carrier_type must be e or h, not $carrier_type")
    end

    # Solve func(μ) = ncarrier(μ) - ncarrier_target = 0
    func = μ -> compute_ncarrier(μ, T, energy_carrier, weights) - ncarrier_target
    bisect(func, -1.0e4, 1.0e4, tolf=ncarrier_target*1E-10)
end
