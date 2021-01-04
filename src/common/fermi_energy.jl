
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
            ncarrier += weights[ik] * occ_fermion((e - μ) / T)
        end
    end
    ncarrier
end

"""
    find_fermi_energy(ncarrier_target, T, energy, weights)
Find Fermi energy for target carrier density using bisection
- ncarrier_target: target carrier density
- T: temperature
- energy: band energy
- weights: k-point weights.
"""
function find_fermi_energy(ncarrier_target, T, energy, weights)
    # Solve func(mu) = ncarrier(mu) - ncarrier_target = 0
    func = μ -> compute_ncarrier(μ, T, energy, weights) - ncarrier_target
    bisect(func, -1.0e4, 1.0e4, tolf=ncarrier_target*1E-10)
end
