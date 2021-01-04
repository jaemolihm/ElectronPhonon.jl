
# Functions related to calculation of electron conductivity
using Parameters
using Printf

export TransportParams
export TransportSERTA
export transport_set_μ!
export compute_lifetime_serta!

@with_kw struct TransportParams{T <: Real}
    Tlist::Vector{T} # Temperature
    n::T # Carrier density
    carrier_type::String # e (electron) or h (hole)
    nband_valence::Int # Number of valence bands (used only for semiconductors)
    smearing::T # Smearing parameter for delta function
    spin_degeneracy::Int # Spin degeneracy
    μlist::Vector{T} = zeros(T, length(Tlist)) # Chemical poetntial
end

# Data and buffers for SERTA (self-energy relaxation-time approximation) conductivity
Base.@kwdef struct TransportSERTA{T <: Real}
    inv_τ::Array{T, 3}

    # thread-safe buffers
    nocc_q::Vector{Vector{T}}
    focc_kq::Vector{Vector{T}}
end

function TransportSERTA(T, nband::Int, nmodes::Int, nk::Int, ntemperatures::Int)
    TransportSERTA{T}(
        inv_τ=zeros(T, nband, nk, ntemperatures),
        nocc_q=[zeros(T, nmodes) for i=1:nthreads()],
        focc_kq=[zeros(T, nband) for i=1:nthreads()],
    )
end

function transport_set_μ!(parameters::TransportParams, energy, weights, volume)
    if parameters.carrier_type == "e"
        @views e_carrier = energy[parameters.nband_valence+1:end, :]
    elseif parameters.carrier_type == "h"
        @views e_carrier = energy[1:parameters.nband_valence, :]
    else
        error("carrier_type must be e or h, not $carrier_type")
    end

    @info @sprintf "n = %.1e cm^-3" parameters.n / (volume/unit_to_aru(:cm)^3)

    for (iT, T) in enumerate(parameters.Tlist)
        ncarrier_target = parameters.n / parameters.spin_degeneracy
        μ = find_fermi_energy(ncarrier_target, T, e_carrier, weights)
        parameters.μlist[iT] = μ
        @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
    end
    nothing
end


"""
    compute_lifetime_serta!(transdata::TransportSERTA, epdata, params::TransportParams, ik)
Compute electron inverse lifetime for given k and q point data in epdata
"""
function compute_lifetime_serta!(transdata::TransportSERTA, epdata, params::TransportParams, ik)
    inv_smear = 1 / params.smearing

    nocc_q = transdata.nocc_q[threadid()]
    focc_kq = transdata.focc_kq[threadid()]

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        nocc_q .= occ_boson.(epdata.omega ./ T)
        for ib in epdata.rngkq
            focc_kq[ib] = occ_fermion((epdata.ekq[ib] - μ) / T)
        end

        # Calculate inverse electron lifetime
        for imode in 1:epdata.nmodes
            omega = epdata.omega[imode]
            if (omega < omega_acoustic)
                continue
            end

            @inbounds for ib in epdata.rngk, jb in epdata.rngkq
                # 1: phonon absorption. e_k + phonon -> e_kq
                # 2: phonon emission.   e_k -> e_kq + phonon
                delta_e1 = epdata.ek[ib] - (epdata.ekq[jb] - omega)
                delta_e2 = epdata.ek[ib] - (epdata.ekq[jb] + omega)
                delta1 = gaussian(delta_e1 * inv_smear) * inv_smear
                delta2 = gaussian(delta_e2 * inv_smear) * inv_smear
                fcoeff1 = nocc_q[imode] + focc_kq[jb]
                fcoeff2 = nocc_q[imode] + 1.0 - focc_kq[jb]

                transdata.inv_τ[ib, ik, iT] += (2π * epdata.wtq
                    * epdata.g2[jb, ib, imode]
                    * (fcoeff1 * delta1 + fcoeff2 * delta2))
            end
        end # modes
    end # temperatures
end

"""
    compute_lifetime_serta!(transdata::TransportSERTA, epdata, params::TransportParams, ik)
Compute electron inverse lifetime for given k and q point data in epdata
"""
function compute_mobility_serta!(inv_τ, energy, vel_diag, weights,
        params::TransportParams, window=(-Inf, Inf))

    nband, nk = size(inv_τ)
    @assert size(energy) == (nband, nk)
    @assert size(vel_diag) == (3, nband, nk)

    σlist = zeros(eltype(inv_τ), 3, 3, length(params.Tlist))

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        for ik in 1:nk
            for iband in 1:nband
                enk = energy[iband, ik]
                @views vnk = vel_diag[:, iband, ik]

                # Skip if enk is outside the window
                if enk < window[1] || enk > window[2]
                    continue
                end

                focc = occ_fermion((enk - μ) / T)
                dfocc = occ_fermion_derivative((enk - μ) / T) / T
                τ = 1 / inv_τ[iband, ik, iT]

                for j=1:3, i=1:3
                    σlist[i, j, iT] += weights[ik] * dfocc * τ * vnk[i] * vnk[j]
                end
            end # ib
        end # ik
    end # temperatures
    σlist *= params.spin_degeneracy
    σlist
end

function transport_print_mobility(σlist, transport_params, volume)
    carrier_density_SI = transport_params.n / volume * unit_to_aru(:cm)^3
    charge_density_SI = carrier_density_SI * units.e_SI

    σ_Si = σlist .* (units.e_SI^2 / volume * unit_to_aru(:ħ) * unit_to_aru(:cm))

    println("======= Electron mobility =======")
    println("Carrier density (cm^-3) =  $carrier_density_SI")
    for iT in 1:length(transport_params.Tlist)
        println("T (K) = $(transport_params.Tlist[iT] / unit_to_aru(:K))")
        println("mobility (cm^2/Vs) = ")
        for i in 1:3
            @printf "%10.3f %10.3f %10.3f\n" (σ_Si[:, i, iT] ./ charge_density_SI)...
        end
        println()
    end
end
