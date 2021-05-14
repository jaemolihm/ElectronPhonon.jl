"""
Electron conductivity in the self-energy relaxation time (SERTA) approximation
"""

using Parameters: @with_kw

export ElectronTransportParams
export bte_compute_μ!
export compute_lifetime_serta!

# TODO: Merge with transport_electron.jl (or deprecate the latter)

"""
    ElectronTransportParams{T <: Real}
Parameters for electron transport calculation. Arguments:
* `Tlist::Vector{T}`: list of temperatures
* `n::T`: Carrier density
* `nband_valence::Int`: Number of valence bands (used only for semiconductors)
* `smearing::Tuple{Symbol, T}`: (:Mode, smearing). Smearing parameter for delta function. Mode can be Gaussian, Lorentzian, and Tetrahedron.
* `spin_degeneracy::Int`: Spin degeneracy.
* `μlist::Vector{T}`: Chemical potential. Defaults to 0.
"""
@with_kw struct ElectronTransportParams{T <: Real}
    Tlist::Vector{T}
    n::T
    nband_valence::Int
    smearing::Tuple{Symbol, T}
    spin_degeneracy::Int
    μlist::Vector{T} = zeros(T, length(Tlist))
end

function bte_compute_μ!(params, el::BTStates{R}, volume) where {R <: Real}
    ncarrier_target = params.n / params.spin_degeneracy

    mpi_isroot() && @info @sprintf "n = %.1e cm^-3" params.n / (volume/unit_to_aru(:cm)^3)

    for (iT, T) in enumerate(params.Tlist)
        μ = find_chemical_potential(ncarrier_target, T, el.e, el.k_weight, 0)
        params.μlist[iT] = μ
        mpi_isroot() && @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
    end
    nothing
end

function compute_lifetime_serta!(inv_τ, btmodel, params, recip_lattice, ngrid)
    # TODO: Clean input params recip_lattice and ngrid
    # TODO: add smearing_mode field in params. (:Gaussian, :Lorentzian, :Tetrahedron)
    R = eltype(inv_τ)
    η = params.smearing[2]
    inv_η = 1 / η
    el_i = btmodel.el_i
    el_f = btmodel.el_f
    ph = btmodel.ph
    scat = btmodel.scattering

    for iscat in 1:scat.n
        ind_el_i = scat.ind_el_i[iscat]
        ind_el_f = scat.ind_el_f[iscat]
        ind_ph = scat.ind_ph[iscat]
        sign_ph = scat.sign_ph[iscat]
        g2 = scat.mel[iscat]

        ω_ph = ph.e[ind_ph]
        if ω_ph < omega_acoustic
            continue
        end
        e_i = el_i.e[ind_el_i]
        e_f = el_f.e[ind_el_f]

        # sign_ph = +1: phonon emission.   e_k -> e_kq + phonon
        # sign_ph = -1: phonon absorption. e_k + phonon -> e_kq
        delta_e = e_i - e_f - sign_ph * ω_ph

        # For electron final state occupation, use e_k - sign_ph * ω_ph instead of e_kq,
        # using energy conservation. The former is better because the phonon velocity is
        # much smaller than the electron velocity, so that it changes less w.r.t q.
        e_f_occupation = e_i - sign_ph * ω_ph
        # e_f_occupation = e_f

        if params.smearing[1] == :Gaussian
            delta = gaussian(delta_e * inv_η) * inv_η
        elseif params.smearing[1] == :Lorentzian
            delta = η / (delta_e^2 + η^2) / π
        elseif params.smearing[1] == :Tetrahedron
            v_cart = - el_f.vdiag[ind_el_f] - sign_ph * ph.vdiag[ind_ph]
            v_delta_e = recip_lattice' * v_cart
            delta = delta_parallelepiped(zero(R), delta_e, v_delta_e, 1 ./ ngrid)
        end

        coeff1 = 2π * el_f.k_weight[ind_el_f] * g2 * delta

        for iT in 1:length(params.Tlist)
            T = params.Tlist[iT]
            μ = params.μlist[iT]
            n_ph = occ_boson(ω_ph, T)
            f_kq = occ_fermion(e_f_occupation - μ, T)

            fcoeff = sign_ph == 1 ? n_ph + 1 - f_kq : n_ph + f_kq

            inv_τ[ind_el_i, iT] += coeff1 * fcoeff
        end
    end
    inv_τ
end


function compute_mobility_serta!(params, inv_τ, el::BTStates{R}, ngrid, recip_lattice) where {R}
    @assert el.n == size(inv_τ, 1)

    σlist = zeros(eltype(inv_τ), 3, 3, length(params.Tlist))

    emax = maximum(el.e)
    emin = minimum(el.e)

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        # Maximum occupation for given energies
        if emax < μ
            dfocc_max = -EPW.occ_fermion_derivative(emax - μ, T)
        elseif emin > μ
            dfocc_max = -EPW.occ_fermion_derivative(emin - μ, T)
        else
            # energy range include enk - μ = 0.
            dfocc_max = -EPW.occ_fermion_derivative(zero(T), T)
        end
        # @info dfocc_max

        cnt = 0
        for i = 1:el.n
            enk = el.e[i]
            vnk = el.vdiag[i]

            dfocc = -occ_fermion_derivative(enk - μ, T)
            # if dfocc > dfocc_max / 1E4
            #     # Near the Fermi level. Use subsampling.
            #     dfocc = -occ_fermion_derivative_smear(enk, vnk, μ, T, ngrid, recip_lattice, (8, 8, 8))
            #     cnt += 1
            # end
            τ = 1 / inv_τ[i, iT]

            for b=1:3, a=1:3
                σlist[a, b, iT] += el.k_weight[i] * dfocc * τ * vnk[a] * vnk[b]
            end
        end # i
        @info "cnt = $cnt"
    end # temperatures
    σlist .*= params.spin_degeneracy
    σlist
end
