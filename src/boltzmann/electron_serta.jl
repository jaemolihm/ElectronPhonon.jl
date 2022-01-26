"""
Electron conductivity in the self-energy relaxation time (SERTA) approximation
FIXME: Add test for compute_lifetime_serta_mode!
"""

using Printf
using TetrahedronIntegration

export bte_compute_μ!
export compute_lifetime_serta!
export compute_conductivity_serta!
export compute_transport_distribution_function

# TODO: Merge with transport_electron.jl (or deprecate the latter)

"""
    bte_compute_μ!(params, el::BTStates{R}; do_print=true) where {R <: Real}
"""
function bte_compute_μ!(params, el::BTStates{R}; do_print=true) where {R <: Real}
    # Since params.n is the difference of number of electrons per cell from nband_valence,
    # nband_valence should be added for the real target ncarrier.
    # Also, el.nstates_base is the contribution to the ncarrier from occupied states
    # outside the window (i.e. not included in `energy`). So it is subtracted.
    ncarrier_target = params.n / params.spin_degeneracy + params.nband_valence - el.nstates_base

    do_print && mpi_isroot() && @info @sprintf "n = %.1e cm^-3" params.n / (params.volume/unit_to_aru(:cm)^3)

    for (iT, T) in enumerate(params.Tlist)
        μ = find_chemical_potential(ncarrier_target, T, el.e, el.k_weight)
        params.μlist[iT] = μ
        do_print && mpi_isroot() && @info @sprintf "T = %.1f K , μ = %.4f eV" T/unit_to_aru(:K) μ/unit_to_aru(:eV)
    end
    params.μlist
end


function compute_lifetime_serta!(inv_τ, btmodel, params, recip_lattice; mode_resolved=false)
    el_i = btmodel.el_i
    el_f = btmodel.el_f
    ph = btmodel.ph
    scat = btmodel.scattering
    compute_lifetime_serta!(inv_τ, el_i, el_f, ph, scat, params, recip_lattice; mode_resolved)
end

function compute_lifetime_serta!(inv_τ, el_i, el_f, ph, scat, params, recip_lattice; mode_resolved=false)
    # TODO: Clean input param recip_lattice
    inv_τ_iscat = zeros(eltype(inv_τ), length(params.Tlist))

    # iscat_print_step = max(1, round(Int, scat.n / 10))
    for (iscat, s) in enumerate(scat)
        # if mod(iscat, iscat_print_step) == 0
        #     @printf("%.1f %% done\n", iscat / scat.n * 100)
        # end
        _compute_lifetime_serta_single_scattering!(inv_τ_iscat, el_i, el_f, ph, params, s, recip_lattice)

        if mode_resolved
            @views inv_τ[s.ind_el_i, s.ind_ph, :] .+= inv_τ_iscat
        else
            @views inv_τ[s.ind_el_i, :] .+= inv_τ_iscat
        end
    end
    inv_τ
end

function _compute_lifetime_serta_single_scattering!(inv_τ::AbstractArray{FT}, el_i, el_f, ph, params, s, recip_lattice) where {FT}
    inv_τ .= 0

    η = params.smearing[2]
    ngrid = el_f.ngrid

    ind_el_i = s.ind_el_i
    ind_el_f = s.ind_el_f
    ind_ph = s.ind_ph
    sign_ph = s.sign_ph
    g2 = s.mel

    ω_ph = ph.e[ind_ph]
    # Skip if phonon frequency is too close to 0 (acoustic phonon at q=0)
    ω_ph < omega_acoustic && return

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
        inv_η = 1 / η
        delta = gaussian(delta_e * inv_η) * inv_η
    elseif params.smearing[1] == :Lorentzian
        delta = η / (delta_e^2 + η^2) / π
    elseif params.smearing[1] == :Tetrahedron
        v_cart = - el_f.vdiag[ind_el_f] - sign_ph * ph.vdiag[ind_ph]
        v_delta_e = recip_lattice' * v_cart
        delta = delta_parallelepiped(zero(FT), delta_e, v_delta_e, 1 ./ ngrid)
    elseif params.smearing[1] == :GaussianTetrahedron
        v_cart = - el_f.vdiag[ind_el_f] - sign_ph * ph.vdiag[ind_ph]
        v_delta_e = recip_lattice' * v_cart
        delta = gaussian_parallelepiped(η, delta_e, v_delta_e, 1 ./ ngrid) / η
    end
    delta < eps(FT) && return

    coeff1 = 2FT(π) * el_f.k_weight[ind_el_f] * g2 * delta

    for iT in 1:length(params.Tlist)
        T = params.Tlist[iT]
        μ = params.μlist[iT]
        n_ph = occ_boson(ω_ph, T)
        f_kq = occ_fermion(e_f_occupation - μ, T)

        fcoeff = sign_ph == 1 ? n_ph + 1 - f_kq : n_ph + f_kq

        inv_τ[iT] = coeff1 * fcoeff
    end
end


# Conductivity

function compute_conductivity_serta!(params, inv_τ, el::BTStates{R}, ngrid, recip_lattice) where {R}
    @assert el.n == size(inv_τ, 1)

    σ = zeros(eltype(inv_τ), 3, 3, length(params.Tlist))

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
                σ[a, b, iT] += el.k_weight[i] * dfocc * τ * vnk[a] * vnk[b]
            end
        end # i
        # @info "cnt = $cnt"
    end # temperatures
    σ .*= params.spin_degeneracy / params.volume
    σ
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


"""
    compute_transport_distribution_function(elist::AbstractVector{R}, smearing, el, inv_τ, params) where {R}
Compute the transport distribution function ``Σ^{a,b}(elist)``, where
``Σ^{a,b}(e) = 1/volume * ∑_{n,k} v^a_{nk} * v^b_{nk} * df_{nk} / τ_{nk} * δ(e - e_{nk})``.
``Σ^{a,b}(e)`` satisfies ``σ^{a,b} = ∫de (-df(e)/de) Σ^{a,b}(e)``.
We use Gaussian smearing for the delta function using `smearing`.
"""
function compute_transport_distribution_function(elist, smearing, el, inv_τ, params, symmetry=nothing) where {R}
    Tlist = params.Tlist
    μlist = params.μlist
    e = el.e
    w = el.k_weight
    vv_nosym = collect(reshape(reinterpret(Float64, [v * v' for v in el.vdiag]), 3, 3, :))
    vv = symmetrize_array(vv_nosym, symmetry, order=2)
    Σ_tdf = zeros(eltype(elist), length(elist), 3, 3, length(Tlist))
    σ_list = zeros(eltype(elist), el.n)
    for iT in 1:length(Tlist)
        T = Tlist[iT]
        μ = μlist[iT]
        dfocc = -EPW.occ_fermion_derivative.(e .- μ, T)
        @views for a in 1:3, b in 1:3
            @. σ_list = w * dfocc * vv[a, b, :] / inv_τ[:, iT]
            for i in 1:el.n
                @. Σ_tdf[:, a, b, iT] += σ_list[i] * gaussian((elist - e[i]) / smearing) / smearing
            end
        end
    end
    Σ_tdf .*= params.spin_degeneracy / params.volume
    Σ_tdf
end

# TODO: Add utility plot_transport_distribution_function or so (or add automatic elist and smearing)


# Running SERTA mobility calculation
# FIXME: remove recip_lattice argument (add it to transport_params?)
function run_serta(filename, transport_params, symmetry, recip_lattice, ::Type{FT}=Float64; do_print=false) where FT
    # Read btedata
    fid = h5open(filename, "r")
    el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.BTStates{FT})
    el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.BTStates{FT})
    ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{FT})

    # Compute chemical potential
    bte_compute_μ!(transport_params, el_i; do_print)

    inv_τ = zeros(FT, el_i.n, length(transport_params.Tlist))

    # Compute lifetime
    group_scattering = open_group(fid, "scattering")
    mpi_isroot() && println("Original grid: Total $(length(group_scattering)) groups of scattering")
    @time for (ig, key) in enumerate(keys(group_scattering))
        mpi_isroot() && mod(ig, 100) == 0 && println("Calculating scattering for group $ig")
        g = open_group(group_scattering, key)
        scat = load_BTData(g, EPW.ElPhScatteringData{FT})
        compute_lifetime_serta!(inv_τ, el_i, el_f, ph, scat, transport_params, recip_lattice)
    end

    close(fid)

    σ = compute_conductivity_serta!(transport_params, inv_τ, el_i, el_i.ngrid, recip_lattice)
    σ = symmetrize_array(σ, symmetry, order=2)
    σ_SI, mobility_SI = transport_print_mobility(σ, transport_params; do_print)

    (; inv_τ, σ, σ_SI, mobility_SI, el_i)
end


# FIXME: qpts should be read from file
# FIXME: remove recip_lattice argument (add it to transport_params?)
function run_serta_subgrid(filename_original, filename_subgrid, transport_params, symmetry, qpts, recip_lattice, ::Type{FT}=Float64; do_print=false) where FT
    # Read original grid btedata
    fid = h5open(filename_original, "r")
    el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.BTStates{FT})
    el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.BTStates{FT})
    ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{FT})

    # Read subgrid btedata
    fid_sub = h5open(filename_subgrid, "r")
    el_i_sub = load_BTData(fid_sub["initialstate_electron"], EPW.BTStates{FT})
    el_f_sub = load_BTData(fid_sub["finalstate_electron"], EPW.BTStates{FT})
    ph_sub = load_BTData(fid_sub["phonon"], EPW.BTStates{FT})
    iq_subgrid_to_grid = read(fid_sub, "iq_subgrid_to_grid")

    # Compute chemical potential
    bte_compute_μ!(transport_params, el_i; do_print)

    # Check the initial states are identical for grid and subgrid models
    @assert all(isapprox.(el_i.e, el_i_sub.e))

    # For the original grid, filter only states not subdivided
    ind_ph_map = states_index_map(ph)
    ind_ph_filter = fill(true, ph.n)
    for iq in unique(iq_subgrid_to_grid)
        xq = qpts.vectors[iq]
        xq_int = mod.(round.(Int, xq .* ph.ngrid), ph.ngrid)
        ind_ph_list = ind_ph_map[CI(xq_int...)]
        for imode in 1:ph.nband
            if ind_ph_list[imode] > 0
                ind_ph_filter[ind_ph_list[imode]] = false
            end
        end
    end

    inv_τ = zeros(FT, el_i.n, length(transport_params.Tlist))

    # Contribution from the original grid. Reading scattering one ik at a time
    group_scattering = open_group(fid, "scattering")
    mpi_isroot() && println("Original grid: Total $(length(group_scattering)) groups of scattering")
    @time for (ig, key) in enumerate(keys(group_scattering))
        mpi_isroot() && mod(ig, 100) == 0 && println("Calculating scattering for group $ig")
        scat = load_BTData(open_group(group_scattering, key), EPW.ElPhScatteringData{FT})
        scat_exclude_subgrid = filter(s -> ind_ph_filter[s.ind_ph], scat)
        compute_lifetime_serta!(inv_τ, el_i, el_f, ph, scat_exclude_subgrid, transport_params, recip_lattice)
    end
    inv_τ_only_original = copy(inv_τ)

    # Contribution from the q subgrid. Reading scattering one ik at a time
    group_scattering_sub = open_group(fid_sub, "scattering")
    mpi_isroot() && println("Subgrid: Total $(length(group_scattering_sub)) groups of scattering")
    @time for (ig, key) in enumerate(keys(group_scattering_sub))
        mpi_isroot() && mod(ig, 100) == 0 && println("Calculating scattering for group $ig")
        scat = load_BTData(open_group(group_scattering_sub, key), EPW.ElPhScatteringData{FT})
        compute_lifetime_serta!(inv_τ, el_i_sub, el_f_sub, ph_sub, scat, transport_params, recip_lattice)
    end

    close(fid)
    close(fid_sub)

    σ = compute_conductivity_serta!(transport_params, inv_τ, el_i, el_i.ngrid, recip_lattice)
    σ = symmetrize_array(σ, symmetry, order=2)
    σ_SI, mobility_SI = transport_print_mobility(σ, transport_params; do_print)

    (; inv_τ, σ, σ_SI, mobility_SI, el_i, inv_τ_only_original)
end
