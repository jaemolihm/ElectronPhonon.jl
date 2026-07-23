
using Base.Threads

export PhononSelfEnergyParams
export PhononSelfEnergy
export compute_phonon_selfen!

Base.@kwdef struct PhononSelfEnergyParams{T <: Real}
    Tlist::Vector{T} # Temperature
    μ::T # Fermi level
    smearing::T # Smearing parameter for delta function
    spin_degeneracy::T # degeneracy of electron bands (e.g. spin degeneracy)
end

# Data and buffers for phonon self-energy
Base.@kwdef struct PhononSelfEnergy{T <: Real}
    imsigma::Array{T, 3}
end

function PhononSelfEnergy{FT}(nmodes::Int, nq::Int, ntemperatures::Int) where FT
    PhononSelfEnergy{FT}(
        imsigma = zeros(FT, nmodes, nq, ntemperatures),
    )
end

"""
# Compute phonon self-energy for given k and q point data in epstate
# TODO: Real part
"""
@timing "selfen_ph" function compute_phonon_selfen!(phself, epstate,
        params::PhononSelfEnergyParams, iq)
    el_k_occ = epstate.el_k.occupation
    el_kq_occ = epstate.el_kq.occupation

    μ = params.μ
    inv_smear = 1 / params.smearing

    for (iT, T) in enumerate(params.Tlist)
        set_occupation!(epstate.el_k, μ, T)
        set_occupation!(epstate.el_kq, μ, T)

        # Calculate imaginary part of phonon self-energy
        for imode in 1:epstate.nmodes
            omega = epstate.ph.e[imode]
            if omega < omega_acoustic
                continue
            end

            @inbounds for ib in epstate.el_k.rng, jb in epstate.el_kq.rng
                delta_e = epstate.el_kq.e[jb] - epstate.el_k.e[ib] - omega
                delta = gaussian(delta_e * inv_smear) * inv_smear
                phself.imsigma[imode, iq, iT] += (epstate.g2[jb, ib, imode]
                    * epstate.wtk * π * (el_k_occ[ib] - el_kq_occ[jb]) * delta)
            end
        end # mode
    end # temperature
end
