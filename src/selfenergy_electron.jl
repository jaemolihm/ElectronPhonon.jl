
using Base.Threads

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
Base.@kwdef struct ElectronSelfEnergy{T <: Real}
    imsigma::Array{T, 3}

    # buffers
    focc_kq::Vector{Vector{T}}
end

function ElectronSelfEnergy(T, nband::Int, nmodes::Int, nk::Int, ntemperatures::Int)
    data = ElectronSelfEnergy{T}(
        focc_kq=[Vector{T}(undef, nband) for _ in 1:nthreads()],
        imsigma=zeros(T, nband, nk, ntemperatures),
    )
    data
end

"""
# Compute electron self-energy for given k and q point data in epdata
# TODO: Real part
"""
@timing "selfen_el" function compute_electron_selfen!(elself, epdata,
        params::ElectronSelfEnergyParams, ik)
    ph_occ = epdata.ph.occupation
    focc_kq = elself.focc_kq[Threads.threadid()]

    μ = params.μ
    inv_smear = 1 / params.smearing

    for (iT, T) in enumerate(params.Tlist)

        set_occupation!(epdata.ph, T)

        for ib in epdata.el_kq.rng
            focc_kq[ib] = occ_fermion((epdata.el_kq.e[ib] - μ) / T)
        end

        # Calculate imaginary part of electron self-energy
        for imode in 1:epdata.nmodes
            omega = epdata.ph.e[imode]
            if (omega < EPW.omega_acoustic)
                continue
            end

            @inbounds for ib in epdata.el_k.rng, jb in epdata.el_kq.rng
                delta_e1 = epdata.el_k.e[ib] - (epdata.el_kq.e[jb] - omega)
                delta_e2 = epdata.el_k.e[ib] - (epdata.el_kq.e[jb] + omega)
                delta1 = gaussian(delta_e1 * inv_smear) * inv_smear
                delta2 = gaussian(delta_e2 * inv_smear) * inv_smear
                fcoeff1 = ph_occ[imode] + focc_kq[jb]
                fcoeff2 = ph_occ[imode] + 1.0 - focc_kq[jb]
                elself.imsigma[ib, ik, iT] += (epdata.g2[jb, ib, imode] * epdata.wtq
                    * π * (fcoeff1 * delta1 + fcoeff2 * delta2))
            end
        end
    end
end
