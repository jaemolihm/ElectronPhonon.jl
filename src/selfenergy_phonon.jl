
using Base.Threads

export PhononSelfEnergyParams
export PhononSelfEnergy
export compute_phonon_selfen!

Base.@kwdef struct PhononSelfEnergyParams{T <: Real}
    Tlist::Vector{T} # Temperature
    μ::T # Fermi level
    smearing::T # Smearing parameter for delta function
    # TODO: Should spin_degeneracy be Int or Float?
    spin_degeneracy::T # Spin degeneracy
end

# Data and buffers for self-energy of electron
Base.@kwdef struct PhononSelfEnergy{T <: Real}
    imsigma::Array{T, 3}

    # buffers
    focc_k::Vector{Vector{T}}
    focc_kq::Vector{Vector{T}}
end

function PhononSelfEnergy(T, nband::Int, nmodes::Int, nq::Int, ntemperatures::Int)
    PhononSelfEnergy{T}(
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
    focc_k = phself.focc_k[Threads.threadid()]
    focc_kq = phself.focc_kq[Threads.threadid()]

    μ = params.μ
    inv_smear = 1 / params.smearing

    for (iT, T) in enumerate(params.Tlist)
        for ib in epdata.el_k.rng
            focc_k[ib] = occ_fermion((epdata.el_k.e[ib] - μ) / T)
        end
        for ib in epdata.el_kq.rng
            focc_kq[ib] = occ_fermion((epdata.el_kq.e[ib] - μ) / T)
        end

        # Calculate imaginary part of phonon self-energy
        for imode in 1:epdata.nmodes
            omega = epdata.ph.e[imode]
            if (omega < omega_acoustic)
                continue
            end

            @inbounds for ib in epdata.el_k.rng, jb in epdata.el_kq.rng
                delta_e = epdata.el_kq.e[jb] - epdata.el_k.e[ib] - omega
                delta = gaussian(delta_e * inv_smear) * inv_smear
                phself.imsigma[imode, iq, iT] += (epdata.g2[jb, ib, imode]
                    * epdata.wtk * π * (focc_k[ib] - focc_kq[jb]) * delta)
            end
        end # mode
    end # temperature
end
