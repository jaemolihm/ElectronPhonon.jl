
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
function compute_ncarrier(μ, T, energy::AbstractMatrix, weights)
    nk = size(energy, 2)
    @assert length(weights) == nk
    ncarrier = zero(eltype(energy))
    for ik in 1:nk
        for iband in 1:size(energy, 1)
            ncarrier += weights[ik] * occ_fermion(energy[iband, ik] - μ, T)
        end
    end
    ncarrier
end

function compute_ncarrier(μ, T, energy::AbstractVector, weights)
    sum(@. weights * occ_fermion(energy - μ, T))
end

"""
    compute_ncarrier_hole(μ, T, energy::AbstractVector, weights)
Compute hole density.
"""
function compute_ncarrier_hole(μ, T, energy::AbstractVector, weights)
    sum(@. weights * (1 - occ_fermion(energy - μ, T)))
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

    # Solve func(μ) = ncarrier(μ) - ncarrier = 0
    func(μ) = compute_ncarrier(μ, T, energy, weights) - ncarrier
    Roots.bisection(func, -Inf, Inf)
end

"""
    find_chemical_potential_semiconductor(ncarrier, T, energy_e, energy_h, weights_e, weights_h)
Find chemical potential for target carrier density using bisection. Minimize floating point
error by computing doped carrier density, not the total carrier density.
"""
function find_chemical_potential_semiconductor(ncarrier, T, energy_e, energy_h, weights_e, weights_h)
    # FIXME: T=0 case
    # TODO: MPI

    # Solve func(μ) = ncarrier_electron(μ) - ncarrier_hole(μ) - ncarrier = 0
    func(μ) = (  compute_ncarrier(μ, T, energy_e, weights_e)
               - compute_ncarrier_hole(μ, T, energy_h, weights_h) - ncarrier)
    Roots.bisection(func, -Inf, Inf)
end
