
# Functions related to calculation of electron conductivity
using Base.Threads
using Printf

export ElectronTransportParams
export compute_conductivity_serta!
export transport_print_mobility

"""
    ElectronTransportParams{T <: Real}
Parameters for electron transport calculation. Arguments:
* `Tlist::Vector{T}`: list of temperatures
* `nlist::Vector{T}`: Number of electrons per unit cell, relative to the reference configuration where
    `spin_degeneracy * nband_valence` bands are filled. Includes the spin degeneracy.
* `nband_valence::Int`: Number of valence bands, excluding spin degeneracy (used only for semiconductors)
* `volume::T`: Volume of the unit cell
* `smearing::Tuple{Symbol, T}`: `(:Mode, smearing)`. Smearing parameter for delta function.
    Mode can be Gaussian, Lorentzian, Tetrahedron, and GaussianTetrahedron.
* `spin_degeneracy::Int`: Spin degeneracy.
* `μlist::Vector{T}`: Chemical potential.
* `type::Symbol`: Type of the carrier. `:Metal` or `:Semiconductor`. Defaults to `:Metal` if
    abs(n) >= 1 and to `:Semiconductor` otherwise.

**NOTE**: If `type == :Semiconductor`, it is assumed that the maximum of the `nband_valence`-th band
is below the minimum of the `nband_valence+1`-th band for the calculation of the chemical potential.
If this condition does not hold, manually set `type = :Metal`.
"""
Base.@kwdef struct ElectronTransportParams{T <: Real}
    Tlist::Vector{T}
    nlist::Vector{T}
    nband_valence::Int = 0
    volume::T
    smearing::Tuple{Symbol, T}
    spin_degeneracy::Int
    μlist::Vector{T} = fill(convert(eltype(Tlist), NaN), length(Tlist))
    type::Symbol = maximum(abs.(nlist)) >= 1 ? :Metal : :Semiconductor
    # FIXME: Check length of Tlist and nlist is equal
    # TODO: Simple constructor with single T or n
end

# Data and buffers for SERTA (self-energy relaxation-time approximation) conductivity
Base.@kwdef struct TransportSERTA{T}
    inv_τ::T
end

function TransportSERTA{FT}(rng_band, nk::Int, ntemps::Int) where FT
    nband = length(rng_band)
    TransportSERTA(
        inv_τ = OffsetArray(zeros(FT, nband, nk, ntemps), rng_band, :, :)
    )
end

# TODO: Add unit test for electron and hole case
"""
    transport_set_μ!(params, energy, weights, nelec_below_window=0; do_print=true)
- `nelec_below_window`: Number of electron per unit cell from states below the window. Spin
    degeneracy factor not multiplied.
"""
function transport_set_μ!(params, energy, weights, nelec_below_window=0; do_print=true)
    for i in axes(params.Tlist, 1)
        n = params.nlist[i]
        T = params.Tlist[i]
        # Since n is the difference of number of electrons per cell from nband_valence,
        # nband_valence should be added for the real target ncarrier.
        # Also, nelec_below_window is the contribution to the ncarrier from occupied states
        # outside the window (i.e. not included in `energy`). So it is subtracted.
        ncarrier_target = n / params.spin_degeneracy + params.nband_valence - nelec_below_window

        μ = find_chemical_potential(ncarrier_target, T, energy, weights)
        params.μlist[i] = μ
        if do_print && mpi_isroot()
            @info @sprintf "n = %.1e cm^-3" n / (params.volume/unit_to_aru(:cm)^3)
            @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
        end
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

                transdata.inv_τ[ib + epdata.el_k.nband_ignore, ik, iT] += (
                    2π * epdata.wtq * epdata.g2[jb, ib, imode]
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

    σ = zeros(eltype(inv_τ), 3, 3, length(params.Tlist))

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        for ik in 1:nk
            el = el_states[ik]

            for iband in el.rng
                iband_full = iband + el.nband_ignore
                enk = el.e[iband]
                vnk = el.vdiag[iband]

                # Skip if enk is outside the window
                if enk < window[1] || enk > window[2]
                    @assert false, "this should not happen. enk must be inside the window"
                    continue
                end

                dfocc = -occ_fermion_derivative(enk - μ, T)
                τ = 1 / inv_τ[iband_full, ik, iT]

                for j=1:3, i=1:3
                    σ[i, j, iT] += weights[ik] * dfocc * τ * vnk[i] * vnk[j]
                end
            end # iband
        end # ik
    end # temperatures
    σ .*= params.spin_degeneracy / params.volume
    return σ
end

"""
    transport_print_mobility(σ, params::ElectronTransportParams; do_print=true)
Utility to calculate and print mobility in SI units.
TODO: Allow printing if σ is not 3*3.
FIXME: Is abs(charge_density_SI) correct?
"""
function transport_print_mobility(σ, params::ElectronTransportParams; do_print=true)
    carrier_density_SI = params.nlist ./ params.volume .* unit_to_aru(:cm)^3
    charge_density_SI = carrier_density_SI .* units.e_SI

    σ_SI = σ .* (units.e_SI^2 * unit_to_aru(:ħ) * unit_to_aru(:cm))
    mobility_SI = σ_SI ./ reshape(abs.(charge_density_SI), 1, 1, :)

    if do_print
        if params.type === :Semiconductor
            println("======= Electrical mobility =======")
            for i in axes(params.Tlist, 1)
                println("Carrier density (cm⁻³) =  $(carrier_density_SI[i])")
                println("T (K)  = $(params.Tlist[i] / unit_to_aru(:K))")
                @printf "μ (eV) = %.4f\n" params.μlist[i] / unit_to_aru(:eV)
                println("mobility (cm²/Vs) = ")
                for a in 1:3
                    @printf "%10.3f %10.3f %10.3f\n" mobility_SI[:, a, i]...
                end
                println()
            end
        elseif params.type === :Metal
            println("======= Electrical conductivity =======")
            for i in axes(params.Tlist, 1)
                println("Carrier density (cm⁻³) =  $(carrier_density_SI[i])")
                println("T (K)  = $(params.Tlist[i] / unit_to_aru(:K))")
                @printf "μ (eV) = %.4f\n" params.μlist[i] / unit_to_aru(:eV)
                println("conductivity (1/(Ω*cm)) = ")
                for a in 1:3
                    @printf "%12.3f %12.3f %12.3f\n" σ_SI[:, a, i]...
                end
                println()
            end
        else
            error("params.type = $(params.type) not identified")
        end
    end
    (;σ_SI, mobility_SI)
end
