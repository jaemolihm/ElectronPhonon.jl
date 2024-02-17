
using Base.Threads
using OffsetArrays

# TODO: Make the buffers module const ?

export ElectronSelfEnergyParams
export ElectronSelfEnergy
export compute_electron_selfen!

Base.@kwdef struct ElectronSelfEnergyParams{T <: Real}
    Tlist::Vector{T} # Temperature
    μ::T # Fermi level
    smearing::T # Smearing parameter for delta function
end

# Data and buffers for self-energy of electron
Base.@kwdef struct ElectronSelfEnergy{T}
    imsigma::T
end

function ElectronSelfEnergy{FT}(rng_band, nk::Int, ntemps::Int) where FT
    # TODO: Remove nmodes
    nband = length(rng_band)
    ElectronSelfEnergy(
        imsigma = OffsetArray(zeros(FT, nband, nk, ntemps), rng_band, :, :)
    )
end

"""
# Compute electron self-energy for given k and q point data in epdata
# TODO: Real part
"""
@timing "selfen_el" function compute_electron_selfen!(elself, epdata,
        params::ElectronSelfEnergyParams, ik)
    el_kq_occ = epdata.el_kq.occupation

    μ = params.μ
    inv_smear = 1 / params.smearing

    for (iT, T) in enumerate(params.Tlist)

        set_occupation!(epdata.el_kq, μ, T)

        # Calculate imaginary part of electron self-energy
        for imode in 1:epdata.nmodes
            omega = epdata.ph.e[imode]
            if (omega < omega_acoustic)
                continue
            end
            occ_ph = occ_boson(epdata.ph.e[imode], T)

            @inbounds for ib in epdata.el_k.rng, jb in epdata.el_kq.rng
                delta_e1 = epdata.el_k.e[ib] - (epdata.el_kq.e[jb] - omega)
                delta_e2 = epdata.el_k.e[ib] - (epdata.el_kq.e[jb] + omega)
                delta1 = gaussian(delta_e1 * inv_smear) * inv_smear
                delta2 = gaussian(delta_e2 * inv_smear) * inv_smear
                fcoeff1 = occ_ph + el_kq_occ[jb]
                fcoeff2 = occ_ph + 1.0 - el_kq_occ[jb]
                elself.imsigma[ib, ik, iT] += (epdata.g2[jb, ib, imode] * epdata.wtq
                    * π * (fcoeff1 * delta1 + fcoeff2 * delta2)
                )
            end
        end
    end
end
