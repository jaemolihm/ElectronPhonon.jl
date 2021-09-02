
# Functions related to calculation of electron conductivity
using Base.Threads
using Printf

export ElectronTransportParams
export compute_conductivity_serta!
export transport_print_mobility

# TODO: Allow multiple carrier density

"""
    ElectronTransportParams{T <: Real}
Parameters for electron transport calculation. Arguments:
* `Tlist::Vector{T}`: list of temperatures
* `n::T`: Number of carriers per unit cell, relative to the reference configuration where `nband_valence` bands are filled.
* `nband_valence::Int`: Number of valence bands (used only for semiconductors)
* `volume::T`: Volume of the unit cell
* `smearing::Tuple{Symbol, T}`: (:Mode, smearing). Smearing parameter for delta function. Mode can be Gaussian, Lorentzian, Tetrahedron, and GaussianTetrahedron.
* `spin_degeneracy::Int`: Spin degeneracy.
* `μlist::Vector{T}`: Chemical potential.
"""
Base.@kwdef struct ElectronTransportParams{T <: Real}
    Tlist::Vector{T}
    n::T
    nband_valence::Int = 0
    volume::T
    smearing::Tuple{Symbol, T}
    spin_degeneracy::Int
    μlist::Vector{T} = fill(T(NaN), length(Tlist))
end

# Data and buffers for SERTA (self-energy relaxation-time approximation) conductivity
Base.@kwdef struct TransportSERTA{T <: Real}
    inv_τ::Array{T, 3}
end

function TransportSERTA(T, nband::Int, nmodes::Int, nk::Int, ntemperatures::Int)
    data = TransportSERTA{T}(
        inv_τ=zeros(T, nband, nk, ntemperatures),
    )
    data
end

# TODO: Add test for electron and hole case
function transport_set_μ!(params, energy, weights)
    ncarrier_target = params.n / params.spin_degeneracy

    mpi_isroot() && @info @sprintf "n = %.1e cm^-3" params.n / (params.volume/unit_to_aru(:cm)^3)

    for (iT, T) in enumerate(params.Tlist)
        μ = find_chemical_potential(ncarrier_target, T, energy, weights, params.nband_valence)
        params.μlist[iT] = μ
        mpi_isroot() && @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
    end
    nothing
end

"""
    compute_lifetime_serta!(transdata::TransportSERTA, epdata, params::ElectronTransportParams, ik)
Compute electron inverse lifetime for given k and q point data in epdata
"""
@timing "compute_τ_serta" function compute_lifetime_serta!(transdata::TransportSERTA,
        epdata, params::ElectronTransportParams, ik)
    if params.smearing[1] !== :Gaussian
        error("$(params.smearing[1]) not implemented. Only Gaussian smearing is implemented.")
    end
    inv_smear = 1 / params.smearing[2]

    ph_occ = epdata.ph.occupation
    el_kq_occ = epdata.el_kq.occupation

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        set_occupation!(epdata.ph, T)
        set_occupation!(epdata.el_kq, μ, T)

        # Calculate inverse electron lifetime
        for imode in 1:epdata.nmodes
            omega = epdata.ph.e[imode]
            if omega < omega_acoustic
                continue
            end

            @inbounds for ib in epdata.el_k.rng, jb in epdata.el_kq.rng
                # 1: phonon absorption. e_k + phonon -> e_kq
                # 2: phonon emission.   e_k -> e_kq + phonon
                delta_e1 = epdata.el_k.e[ib] - (epdata.el_kq.e[jb] - omega)
                delta_e2 = epdata.el_k.e[ib] - (epdata.el_kq.e[jb] + omega)
                delta1 = gaussian(delta_e1 * inv_smear) * inv_smear
                delta2 = gaussian(delta_e2 * inv_smear) * inv_smear
                fcoeff1 = ph_occ[imode] + el_kq_occ[jb]
                fcoeff2 = ph_occ[imode] + 1.0 - el_kq_occ[jb]

                transdata.inv_τ[ib, ik, iT] += (2π * epdata.wtq
                    * epdata.g2[jb, ib, imode]
                    * (fcoeff1 * delta1 + fcoeff2 * delta2))
            end
        end # modes
    end # temperatures
end

"""
    compute_conductivity_serta!(params::ElectronTransportParams, inv_τ, energy, vel_diag,
        weights, window=(-Inf, Inf))
Compute electron inverse lifetime for given k and q point data in epdata
"""
function compute_conductivity_serta!(params::ElectronTransportParams{R}, inv_τ,
        el_states::Vector{ElectronState{R}}, weights, window=(-Inf, Inf)) where {R <: Real}

    nband, nk = size(inv_τ)
    @assert length(el_states) == nk

    σlist = zeros(eltype(inv_τ), 3, 3, length(params.Tlist))

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        for ik in 1:nk
            el = el_states[ik]

            for iband in el.rng
                enk = el.e[iband]
                vnk = el.vdiag[iband]

                # Skip if enk is outside the window
                if enk < window[1] || enk > window[2]
                    @assert false, "this should not happen. enk must be inside the window"
                    continue
                end

                dfocc = -occ_fermion_derivative(enk - μ, T)
                τ = 1 / inv_τ[iband, ik, iT]

                for j=1:3, i=1:3
                    σlist[i, j, iT] += weights[ik] * dfocc * τ * vnk[i] * vnk[j]
                end
            end # iband
        end # ik
    end # temperatures
    σlist .*= params.spin_degeneracy / params.volume
    return σlist
end

"""
    transport_print_mobility(σlist, params::ElectronTransportParams; do_print=true)
Utility to calculate and print mobility in SI units.
"""
function transport_print_mobility(σlist, params::ElectronTransportParams; do_print=true)
    carrier_density_SI = params.n / params.volume * unit_to_aru(:cm)^3
    charge_density_SI = carrier_density_SI * units.e_SI

    σ_SI = σlist .* (units.e_SI^2 * unit_to_aru(:ħ) * unit_to_aru(:cm))
    mobility_SI = σ_SI ./ abs(charge_density_SI)

    if do_print
        println("======= Electron mobility =======")
        println("Carrier density (cm^-3) =  $carrier_density_SI")
        for iT in 1:length(params.Tlist)
            println("T (K)  = $(params.Tlist[iT] / unit_to_aru(:K))")
            @printf "μ (eV) = %.4f\n" params.μlist[iT] / unit_to_aru(:eV)
            println("mobility (cm^2/Vs) = ")
            for i in 1:3
                @printf "%10.3f %10.3f %10.3f\n" mobility_SI[:, i, iT]...
            end
            println()
        end
    end
    mobility_SI
end
