
import Base.@kwdef
using Base.Threads

export ElectronSelfEnergy
export compute_electron_selfen!

# Data and buffers for self-energy of electron
@kwdef struct ElectronSelfEnergy{T <: Real}
    imsigma::Array{T, 2}

    # buffers
    nocc_q::Vector{Vector{T}}
    focc_kq::Vector{Vector{T}}
end

function ElectronSelfEnergy(T, nband::Int, nmodes::Int, nk::Int)
    ElectronSelfEnergy{T}(
        imsigma=zeros(T, nband, nk),
        nocc_q=[zeros(T, nmodes) for i=1:nthreads()],
        focc_kq=[zeros(T, nband) for i=1:nthreads()],
    )
end

"""
# Compute electron self-energy for given k and q point data in epdata
# TODO: Real part
"""
@timing "selfen_el" function compute_electron_selfen!(elself, epdata, ik; efermi, temperature, smear)
    inv_smear = 1 / smear

    nocc_q = elself.nocc_q[threadid()]
    focc_kq = elself.focc_kq[threadid()]

    nocc_q .= occ_boson.(epdata.omega ./ temperature)
    @views focc_kq[epdata.rngkq] .= occ_fermion.((epdata.ekq[epdata.rngkq] .- efermi) ./ temperature)

    # Calculate imaginary part of electron self-energy
    for imode in 1:epdata.nmodes
        omega = epdata.omega[imode]
        if (omega < omega_acoustic)
            continue
        end

        @inbounds for ib in epdata.rngk, jb in epdata.rngkq
            delta_e1 = epdata.ek[ib] - (epdata.ekq[jb] - omega)
            delta_e2 = epdata.ek[ib] - (epdata.ekq[jb] + omega)
            delta1 = gaussian(delta_e1 * inv_smear) * inv_smear
            delta2 = gaussian(delta_e2 * inv_smear) * inv_smear
            fcoeff1 = nocc_q[imode] + focc_kq[jb]
            fcoeff2 = nocc_q[imode] + 1.0 - focc_kq[jb]
            elself.imsigma[ib, ik] += (epdata.g2[jb, ib, imode] * epdata.wtq
                * π * (fcoeff1 * delta1 + fcoeff2 * delta2))
        end
    end
end
