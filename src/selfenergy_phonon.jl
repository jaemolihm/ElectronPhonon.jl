
using Base.Threads

export PhononSelfEnergyParams
export PhononSelfEnergy
export compute_phonon_selfen!

Base.@kwdef struct PhononSelfEnergyParams{T <: Real}
    Tlist::Vector{T} # Temperature
    μ::T # Fermi level
    smearing::T # Smearing parameter for delta function
    spin_degeneracy::T # Spin degeneracy
end

# Data and buffers for self-energy of electron
Base.@kwdef struct PhononSelfEnergy{T <: Real}
    imsigma::Array{T, 3}

    # buffers
    nocc_q::Vector{Vector{T}}
    focc_k::Vector{Vector{T}}
    focc_kq::Vector{Vector{T}}
end

function PhononSelfEnergy(T, nband::Int, nmodes::Int, nq::Int, ntemperatures::Int)
    PhononSelfEnergy{T}(
        nocc_q=[Vector{T}(undef, nmodes) for i=1:Threads.nthreads()],
        focc_k=[Vector{T}(undef, nband) for i=1:Threads.nthreads()],
        focc_kq=[Vector{T}(undef, nband) for i=1:Threads.nthreads()],
        imsigma=zeros(T, nmodes, nq, ntemperatures),
    )
end

"""
# Compute phonon self-energy for given k and q point data in epdata
# TODO: Real part
"""
@timing "selfen_ph" function compute_phonon_selfen!(phself, epdata,
        params::PhononSelfEnergyParams, iq)
    nocc_q = phself.nocc_q[Threads.threadid()]
    focc_k = phself.focc_k[Threads.threadid()]
    focc_kq = phself.focc_kq[Threads.threadid()]

    μ = params.μ
    inv_smear = 1 / params.smearing

    for (iT, T) in enumerate(params.Tlist)
        nocc_q .= occ_boson.(epdata.omega ./ T)
        for ib in epdata.rngk
            focc_k[ib] = occ_fermion((epdata.ek[ib] - μ) / T)
        end
        for ib in epdata.rngkq
            focc_kq[ib] = occ_fermion((epdata.ekq[ib] - μ) / T)
        end

        # Calculate imaginary part of phonon self-energy
        for imode in 1:epdata.nmodes
            omega = epdata.omega[imode]
            if (omega < omega_acoustic)
                continue
            end

            @inbounds for ib in epdata.rngk, jb in epdata.rngkq
                delta_e = epdata.ekq[jb] - epdata.ek[ib] - omega
                delta = gaussian(delta_e * inv_smear) * inv_smear
                phself.imsigma[imode, iq, iT] += (epdata.g2[jb, ib, imode]
                    * epdata.wtk * π * (focc_k[ib] - focc_kq[jb]) * delta)
            end
        end # mode
    end # temperature
end
