"""
Electron conductivity in the self-energy relaxation time (SERTA) approximation
"""

using Printf
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

function bte_compute_μ!(params, el::BTStates{R}, volume; do_print=true) where {R <: Real}
    ncarrier_target = params.n / params.spin_degeneracy

    do_print && mpi_isroot() && @info @sprintf "n = %.1e cm^-3" params.n / (volume/unit_to_aru(:cm)^3)

    for (iT, T) in enumerate(params.Tlist)
        μ = find_chemical_potential(ncarrier_target, T, el.e, el.k_weight, 0)
        params.μlist[iT] = μ
        do_print && mpi_isroot() && @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
    end
    nothing
end


function compute_lifetime_serta!(inv_τ, btmodel, params, recip_lattice)
    el_i = btmodel.el_i
    el_f = btmodel.el_f
    ph = btmodel.ph
    scat = btmodel.scattering
    compute_lifetime_serta!(inv_τ, el_i, el_f, ph, scat, params, recip_lattice)
end

function compute_lifetime_serta!(inv_τ, el_i, el_f, ph, scat, params, recip_lattice)
    # TODO: Clean input param recip_lattice
    FT = eltype(inv_τ)
    η = params.smearing[2]
    inv_η = 1 / η
    ngrid = el_f.ngrid

    # iscat_print_step = max(1, round(Int, scat.n / 10))
    for iscat in 1:scat.n
        # if mod(iscat, iscat_print_step) == 0
        #     @printf("%.1f %% done\n", iscat / scat.n * 100)
        # end
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
        # e_f_occupation = e_i - sign_ph * ω_ph

        # FIXME: The above is not done because it changed mobility a lot (525 to 12000) for cubicBN test (test/boltzmann/test_mobility.jl)
        e_f_occupation = e_f

        if params.smearing[1] == :Gaussian
            delta = gaussian(delta_e * inv_η) * inv_η
        elseif params.smearing[1] == :Lorentzian
            delta = η / (delta_e^2 + η^2) / π
        elseif params.smearing[1] == :Tetrahedron
            v_cart = - el_f.vdiag[ind_el_f] - sign_ph * ph.vdiag[ind_ph]
            v_delta_e = recip_lattice' * v_cart
            delta = delta_parallelepiped(zero(FT), delta_e, v_delta_e, 1 ./ ngrid)
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
        # @info "cnt = $cnt"
    end # temperatures
    σlist .*= params.spin_degeneracy
    σlist
end

"""
    occ_fermion_derivative_smear(e, vcart, μ, T, ngrid, recip_lattice, npoints=(4, 4, 4))
Use the band velocity to linearly interpolate energy and compute the occupation factor
by sampling points inside the box of size `1/ngrid` using a regular grid of size `npoints`.
"""
function occ_fermion_derivative_smear(e, vcart, μ, T, ngrid, recip_lattice, npoints=(4, 4, 4))
    dkmin = (-0.5, -0.5, -0.5) ./ ngrid
    dkmax = (0.5, 0.5, 0.5) ./ ngrid
    ddk = (dkmax .- dkmin) ./ npoints

    v = recip_lattice' * vcart

    # Simple numerical integration of f(k)
    dfocc_smear = 0.0
    @inbounds for i1 in 1:npoints[1], i2 in 1:npoints[2], i3 in 1:npoints[3]
        k = dkmin .+ ddk .* (i1-0.5, i2-0.5, i3-0.5)
        dfocc_smear += EPW.occ_fermion_derivative(e + dot(v, k) - μ, T)
    end
    dfocc_smear /= prod(npoints)
    dfocc_smear
end





function debug_compute_lifetime_serta!(inv_τ, btmodel, params, recip_lattice, ngrid, mode)
    # TODO: Clean input params recip_lattice and ngrid
    R = eltype(inv_τ)
    η = params.smearing[2]
    inv_η = 1 / η

    el_i = btmodel.el_i
    el_f = btmodel.el_f
    ph = btmodel.ph
    scat = btmodel.scattering

    nsample_1d = 5
    nsamples = 0
    ksamples = zeros(R, 3, 2*(nsample_1d+1)^2)
    ω_ph_sampling = zeros(R, 2*(nsample_1d+1)^2)
    cnt = 0

    for iscat in 1:scat.n
        ind_el_i = scat.ind_el_i[iscat]
        # # DEBUG
        # if ind_el_i != 1
        #     continue
        # end
        ind_el_f = scat.ind_el_f[iscat]
        ind_ph = scat.ind_ph[iscat]
        sign_ph = scat.sign_ph[iscat]
        g2 = scat.mel[iscat]
        if mode == "acoustic" && ph.iband[ind_ph] > 3
            continue
        end
        if mode == "optical" && ph.iband[ind_ph] <= 3
            continue
        end

        ω_ph = ph.e[ind_ph]
        if ω_ph < omega_acoustic
            continue
        end
        e_i = el_i.e[ind_el_i]
        e_f = el_f.e[ind_el_f]

        # sign_ph = +1: phonon emission.   e_k -> e_kq + phonon
        # sign_ph = -1: phonon absorption. e_k + phonon -> e_kq
        delta_e = e_i - e_f - sign_ph * ω_ph
        # delta_e = e_i - e_f - sign_ph * 0.004 # DEBUG
        # delta_e = e_i - e_f # DEBUG

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

        # If phonon frequency is much smaller than the temperature, use tetrahedron sampling
        occ_use_sampling = η < 0 && delta > 1E-8 && ω_ph < maximum(params.Tlist)
        occ_use_sampling = false
        if occ_use_sampling
            cnt += 1
            _, nsamples = delta_parallelepiped_sampling!(zero(R), delta_e, v_delta_e,
                1 ./ ngrid, nsample_1d, ksamples)
            v_ph = recip_lattice' * ph.vdiag[ind_ph]
            @views @inbounds for i in 1:nsamples
                ω_ph_sampling[i] = ω_ph + dot(v_ph, ksamples[:, i])
            end
        end

        coeff1 = 2π * el_f.k_weight[ind_el_f] * g2 * delta

        for iT in 1:length(params.Tlist)
            T = params.Tlist[iT]
            μ = params.μlist[iT]
            if occ_use_sampling
                @views n_ph = sum(occ_boson.(ω_ph_sampling[1:nsamples], T)) / nsamples
                # @info occ_boson(ω_ph, T), n_ph, nsamples
            else
                n_ph = occ_boson(ω_ph, T)
            end
            f_kq = occ_fermion(e_f_occupation - μ, T)

            fcoeff = sign_ph == 1 ? n_ph + 1 - f_kq : n_ph + f_kq

            inv_τ[ind_el_i, iT] += coeff1 * fcoeff
        end
    end
    if η < 0
        @info "Sampling: $cnt times / Total: $(scat.n)"
    end
    inv_τ
end