
# Functions related to calculation of electron conductivity
using Base.Threads
using Parameters
using Printf

export TransportParams

@with_kw struct TransportParams{T <: Real}
    Tlist::Vector{T} # Temperature
    n::T # Carrier density
    carrier_type::String # e (electron) or h (hole)
    nband_valence::Int # Number of valence bands (used only for semiconductors)
    smearing::T # Smearing parameter for delta function
    spin_degeneracy::Int # Spin degeneracy
    μlist::Vector{T} = zeros(T, length(Tlist)) # Output. Chemical poetntial
end

# Data and buffers for SERTA (self-energy relaxation-time approximation) conductivity
struct TransportSERTA{T <: Real}
    inv_τ::Array{T, 3}

    # thread-safe buffers
    nocc_q::Vector{Vector{T}}
    focc_kq::Vector{Vector{T}}
end

function TransportSERTA(T, nband::Int, nmodes::Int, nk::Int, ntemperatures::Int)
    data = TransportSERTA{T}(
        zeros(T, nband, nk, ntemperatures),
        [zeros(T, nmodes)],
        [zeros(T, nband)],
    )
    Threads.resize_nthreads!(data.nocc_q)
    Threads.resize_nthreads!(data.focc_kq)
    data
end

# TODO: Add test for electron and hole case
function transport_set_μ!(params::TransportParams, energy, weights, volume)
    ncarrier_target = params.n / params.spin_degeneracy

    mpi_isroot() && @info @sprintf "n = %.1e cm^-3" params.n / (volume/unit_to_aru(:cm)^3)

    for (iT, T) in enumerate(params.Tlist)
        μ = find_chemical_potential(ncarrier_target, T, energy, weights,
            params.carrier_type, params.nband_valence)
        params.μlist[iT] = μ
        mpi_isroot() && @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
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
    compute_mobility_serta!(params::TransportParams, inv_τ, energy, vel_diag,
        weights, window=(-Inf, Inf))
Compute electron inverse lifetime for given k and q point data in epdata
"""
function compute_mobility_serta!(params::TransportParams, inv_τ, el_states, iband_offset,
    weights, window=(-Inf, Inf))

    nband, nk = size(inv_τ)
    @assert length(el_states) == nk

    σlist = zeros(eltype(inv_τ), 3, 3, length(params.Tlist))

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        for ik in 1:nk
            el = el_states[ik]
            for iband in 1:el.nband
                # iband = 1 is the first band inside the window for the state el.
                # This is el.rng[1]-th band for the whole nw bands.
                # inv_τ is defined for states inside the fsthick window, which ignores
                # first iband_offset bands.
                # So, iband = 1 corresponds to iband_epdata as follows.
                iband_epdata = iband + el.rng[1] - iband_offset - 1

                enk = el.e[iband]
                @views vnk = el.vdiag[:, iband]

                # Skip if enk is outside the window
                if enk < window[1] || enk > window[2]
                    continue
                end

                focc = occ_fermion((enk - μ) / T)
                dfocc = occ_fermion_derivative((enk - μ) / T) / T
                τ = 1 / inv_τ[iband_epdata, ik, iT]

                for j=1:3, i=1:3
                    σlist[i, j, iT] += weights[ik] * dfocc * τ * vnk[i] * vnk[j]
                end
            end # ib
        end # ik
    end # temperatures
    σlist .*= params.spin_degeneracy
    return σlist
end

function transport_print_mobility(σlist, transport_params, volume)
    carrier_density_SI = transport_params.n / volume * unit_to_aru(:cm)^3
    charge_density_SI = carrier_density_SI * units.e_SI

    σ_Si = σlist .* (units.e_SI^2 / volume * unit_to_aru(:ħ) * unit_to_aru(:cm))

    println("======= Electron mobility =======")
    println("Carrier density (cm^-3) =  $carrier_density_SI")
    for iT in 1:length(transport_params.Tlist)
        println("T (K)  = $(transport_params.Tlist[iT] / unit_to_aru(:K))")
        @printf "μ (eV) = %.4f\n" transport_params.μlist[iT] / unit_to_aru(:eV)
        println("mobility (cm^2/Vs) = ")
        for i in 1:3
            @printf "%10.3f %10.3f %10.3f\n" (σ_Si[:, i, iT] ./ charge_density_SI)...
        end
        println()
    end
end
