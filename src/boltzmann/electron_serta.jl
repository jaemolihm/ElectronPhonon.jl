"""
Electron conductivity in the self-energy relaxation time (SERTA) approximation
"""

export bte_compute_μ!
export compute_lifetime_serta!

# TODO: Merge with transport_electron.jl (or deprecate the latter)


function bte_compute_μ!(params::TransportParams{R}, el::BTStates{R}, volume) where {R <: Real}
    ncarrier_target = params.n / params.spin_degeneracy

    mpi_isroot() && @info @sprintf "n = %.1e cm^-3" params.n / (volume/unit_to_aru(:cm)^3)

    for (iT, T) in enumerate(params.Tlist)
        μ = find_chemical_potential(ncarrier_target, T, el.e, el.k_weight, 0)
        params.μlist[iT] = μ
        mpi_isroot() && @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
    end
    nothing
end

function compute_lifetime_serta!(inv_τ, el_i, el_f, ph, scat, params::TransportParams{R}, recip_lattice, ngrid) where {R}
    # TODO: Clean input params recip_lattice and ngrid
    inv_η = 1 / params.smearing

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

        if params.smearing > 0
            # Gaussian smearing
            delta = gaussian(delta_e * inv_η) * inv_η
        else
            # tetrahedron
            v_cart = - el_f.vdiag[ind_el_f] - sign_ph * ph.vdiag[ind_ph]
            v_delta_e = v_cart' * recip_lattice
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


function compute_mobility_serta!(params::TransportParams{R}, inv_τ, el::BTStates{R}) where {R}
    @assert el.n == size(inv_τ, 1)

    σlist = zeros(eltype(inv_τ), 3, 3, length(params.Tlist))

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]

        for i = 1:el.n
            enk = el.e[i]
            vnk = el.vdiag[i]

            dfocc = -EPW.occ_fermion_derivative(enk - μ, T)
            τ = 1 / inv_τ[i, iT]

            for j=1:3, i=1:3
                σlist[i, j, iT] += el.k_weight[i] * dfocc * τ * vnk[i] * vnk[j]
            end
        end # i
    end # temperatures
    σlist .*= params.spin_degeneracy
    σlist
end
