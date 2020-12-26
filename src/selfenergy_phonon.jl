
import Base.@kwdef
using Base.Threads

export PhononSelfEnergy
export compute_phonon_selfen!

# Data and buffers for self-energy of electron
@kwdef struct PhononSelfEnergy{T <: Real}
    imsigma::Array{T, 2}

    # buffers
    nocc_q::Vector{Vector{T}}
    focc_k::Vector{Vector{T}}
    focc_kq::Vector{Vector{T}}
end

function PhononSelfEnergy(T, nw::Int, nmodes::Int, nq::Int)
    PhononSelfEnergy{T}(
        imsigma=zeros(T, nmodes, nq),
        nocc_q=[zeros(T, nmodes) for i=1:nthreads()],
        focc_k=[zeros(T, nw) for i=1:nthreads()],
        focc_kq=[zeros(T, nw) for i=1:nthreads()],
    )
end

"""
# Compute phonon self-energy for given k and q point data in epdata
# TODO: Real part
"""
function compute_phonon_selfen!(phself, epdata, iq; efermi, temperature, smear)
    inv_smear = 1 / smear

    nocc_q = phself.nocc_q[threadid()]
    focc_k = phself.focc_k[threadid()]
    focc_kq = phself.focc_kq[threadid()]

    nocc_q .= occ_boson.(epdata.omega ./ temperature)
    focc_k .= occ_fermion.((epdata.ek_full .- efermi) ./ temperature)
    focc_kq .= occ_fermion.((epdata.ekq_full .- efermi) ./ temperature)

    # Calculate imaginary part of phonon self-energy
    for imode in 1:epdata.nmodes
        omega = epdata.omega[imode]
        if (omega < omega_acoustic)
            continue
        end

        @inbounds for ib in 1:epdata.nw, jb in 1:epdata.nw
            delta_e = epdata.ekq_full[jb] - epdata.ek_full[ib] - omega
            delta = gaussian(delta_e * inv_smear) * inv_smear
            phself.imsigma[imode, iq] += (epdata.g2[jb, ib, imode] * epdata.wtk
                * π * (focc_k[ib] - focc_kq[jb]) * delta)
        end
    end
end
