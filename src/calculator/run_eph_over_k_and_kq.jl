function run_eph_over_k_and_kq(
        model       :: Model{FT},
        kpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        kqpts_input :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        ;
        calculators = [],
        mpi_comm_k = nothing,
        mpi_comm_q = nothing,
        fourier_mode = "gridopt",
        window_k  = (-Inf, Inf),
        window_kq = (-Inf, Inf),
        el_kq_from_unfolding = false,
        skip_eph = false,
        symmetry = model.symmetry,
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        nchunks_threads = nthreads(),  # Number of chunks for multithreading
        covariant_derivative_of_g = false,  # Compute cov. derivative of g
    ) where {FT}

    if model.epmat_outer_momentum != "el"
        throw(ArgumentError("model.epmat_outer_momentum must be el to use run_eph_outer_k"))
    end
    for calc in calculators
        if !allow_eph_outer_k(calc)
            throw(ArgumentError("$calc does not allow eph_outer_k. Use run_eph_outer_q instead."))
        end
    end

    mpi_comm_k === nothing || error("mpi_comm_k not implemented")
    mpi_comm_q === nothing || error("mpi_comm_q not implemented")

    setup = _setup_eph_over_k_and_kq(model, kpts_input, kqpts_input;
        mpi_comm_k, mpi_comm_q, fourier_mode, window_k, window_kq,
        el_kq_from_unfolding, symmetry, calculators, nchunks_threads,
        covariant_derivative_of_g,
    )

    _loop_eph_over_k_and_kq(model,
        setup.kpts, setup.qpts, setup.kqpts,
        setup.el_k_save, setup.el_kq_save,
        setup.ph_save, setup.precompute_ph,
        setup.epdatas, setup.ep_ekpRs, setup.epmat, setup.ep_ekpR_obj,
        setup.dyn_threads,
        setup.epmat_R, setup.epobj_ekpR_R, setup.ep_ekpR_Rs;
        calculators, skip_eph,
        energy_conservation, screening_params,
        progress_print_step, nchunks_threads,
        covariant_derivative_of_g, symmetry,
    )

    (; setup.kpts, setup.qpts, setup.el_k_save, setup.el_kq_save, setup.ph_save)
end


# _setup_eph_over_k_and_kq and _loop_eph_over_k_and_kq are split from
# run_eph_over_k_and_kq so that all variables captured by the @threads closure in
# _loop_eph_over_k_and_kq are typed function arguments, avoiding Core.Box wrapping.
function _setup_eph_over_k_and_kq(
        model       :: Model{FT},
        kpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        kqpts_input :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        ;
        mpi_comm_k = nothing,
        mpi_comm_q = nothing,
        fourier_mode = "gridopt",
        window_k  = (-Inf, Inf),
        window_kq = (-Inf, Inf),
        el_kq_from_unfolding = false,
        symmetry = nothing,
        calculators = [],
        nchunks_threads = nthreads(),
        covariant_derivative_of_g = false,
    ) where {FT}

    (; nw, nmodes) = model

    # Generate k points
    @time kpts, iband_min, iband_max, nelec_below_window_k = filter_kpoints(
        kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode)
    @time el_k_save  = compute_electron_states(model, kpts,  ["eigenvalue", "eigenvector", "velocity", "position"], window_k;  fourier_mode)
    nk = kpts.n

    if el_kq_from_unfolding
        # To ensure gauge consistency between symmetry-equivalent k points, we explicitly compute
        # electron states only for k+q in the irreducible BZ and unfold them to the full BZ.
        @time kqpts_irr, iband_kq_min, iband_kq_max, nelec_below_window_kq = filter_kpoints(
            kqpts_input, nw, model.el_ham, window_kq, mpi_comm_q; symmetry, fourier_mode)

        kqpts, ik_to_ikirr_isym_kq = unfold_kpoints(kqpts_irr, symmetry)

        el_kq_save_irr = compute_electron_states(model, kqpts_irr, ["eigenvalue", "eigenvector", "velocity", "position"], window_kq; fourier_mode)

        el_kq_save = unfold_ElectronStates(model, el_kq_save_irr, kqpts_irr, kqpts, ik_to_ikirr_isym_kq, symmetry; fourier_mode)

        # el_kq_save_irr is not used anymore.
        el_kq_save_irr !== el_kq_save && empty!(el_kq_save_irr)

    else
        @time kqpts, iband_kq_min, iband_kq_max, nelec_below_window_kq = filter_kpoints(
            kqpts_input, nw, model.el_ham, window_kq, mpi_comm_q; fourier_mode)

        @time el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity", "position"], window_kq; fourier_mode)
    end


    # Precompute qpts and phonon states if k and k+q meshes are commensurate
    if all(kpts.ngrid .> 0) && all(mod.(kqpts.ngrid, kpts.ngrid) .== 0)
        # kqpts is denser than kpts
        precompute_ph = true
        qpts = combine_kpoint_grids(kqpts, kpts, -, kqpts.ngrid)

    elseif all(kpts.ngrid .> 0) && all(mod.(kpts.ngrid, kqpts.ngrid) .== 0)
        # kpts is denser than kqpts
        precompute_ph = true
        qpts = combine_kpoint_grids(kqpts, kpts, -, kpts.ngrid)

    else
        precompute_ph = false
    end


    # Maximum number of electron bands to decide the size of e-ph matrix buffer.
    nband_max = max(maximum(el.nband for el in el_k_save),
                    maximum(el.nband for el in el_kq_save))


    epdatas = Channel{ElPhData{FT}}(nthreads())
    foreach(1:nthreads()) do _
        put!(epdatas, ElPhData{FT}(nw, nmodes, nband_max))
    end

    # E-ph matrix in electron Bloch, phonon Wannier representation
    ep_ekpR_obj = get_next_wannier_object(model.epmat)
    epmat = get_interpolator(model.epmat; fourier_mode, threads = true)
    ep_ekpRs = get_interpolator_channel(ep_ekpR_obj; fourier_mode)

    # Setup WannierObject and interpolator for im * Rₑ * g(Rₑ, Rₚ)
    if covariant_derivative_of_g
        epmat_R_obj = ElectronPhonon.wannier_object_multiply_R(model.epmat, model.lattice);
        epmat_R = get_interpolator(epmat_R_obj; fourier_mode, threads = true);

        epobj_ekpR_R = get_next_wannier_object(epmat_R_obj);
        ep_ekpR_Rs = get_interpolator_channel(epobj_ekpR_R; fourier_mode);

        # Tight-binding approximation: dgᵃ_{ijν}(Rₑ, Rₚ) += im * (rᵃ_j - rᵃ_i) g_{ijν}(Rₑ, Rₚ)
        # epmat        : (i, j, nmodes, Rₚ, Rₑ)
        # epobj_ekpR_R : (i, j, nmodes, Rₚ, 3, Rₑ)
        @views for ire in axes(epmat_R_obj.op_r, 2)
            nrp = length(epmat_R_obj.irvec_next)
            tmp_g  = Base.ReshapedArray(model.epmat.op_r[:, ire], (nw, nw, nmodes, nrp), ())
            tmp_gR = Base.ReshapedArray(epmat_R_obj.op_r[:, ire], (nw, nw, nmodes, nrp, 3), ())
            for idir in 1:3, iw in 1:nw
                ri = model.wann_centers[iw][idir]
                tmp_gR[iw, :, :, :, idir] .-= im .* ri .* tmp_g[iw, :, :, :]
                tmp_gR[:, iw, :, :, idir] .+= im .* ri .* tmp_g[:, iw, :, :]
            end
        end
    else
        epmat_R = nothing
        epobj_ekpR_R = nothing
        ep_ekpR_Rs = nothing
    end


    # Precompute phonon states if precompute_ph == true
    if precompute_ph
        @time ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode)
        dyn_threads = nothing
    else
        qpts = nothing
        ph_save = nothing
        dyn_threads = get_interpolator_channel(model.ph_dyn; fourier_mode)
    end


    # Initialize calculators
    for calc in calculators
        setup_calculator!(
            calc, kpts, qpts, el_k_save
            ;
            rng_band = iband_min:iband_max,
            el_states_kq = el_kq_save,
            model.nw,
            model.nmodes,
            kqpts,
            nelec_below_window_k,
            nelec_below_window_kq,
            nchunks_threads,
        )
    end

    if mpi_isroot()
        @info "Number of k points = $(kpts.n)"
        @info "Number of k+q points = $(kqpts.n)"
        precompute_ph && @info "Number of q points = $(qpts.n)"
    end

    return (;
        kpts, qpts, kqpts,
        el_k_save, el_kq_save,
        ph_save, precompute_ph,
        nband_max,
        epdatas, ep_ekpRs, epmat, ep_ekpR_obj,
        dyn_threads,
        epmat_R, epobj_ekpR_R, ep_ekpR_Rs,
        iband_min, iband_max,
        nelec_below_window_k, nelec_below_window_kq,
    )
end


function _loop_eph_over_k_and_kq(
        model       :: Model{FT},
        kpts, qpts, kqpts,
        el_k_save, el_kq_save,
        ph_save, precompute_ph,
        epdatas, ep_ekpRs, epmat, ep_ekpR_obj,
        dyn_threads,
        epmat_R, epobj_ekpR_R, ep_ekpR_Rs;
        calculators = [],
        skip_eph = false,
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        nchunks_threads = nthreads(),
        covariant_derivative_of_g = false,
        symmetry = nothing,
    ) where {FT}

    (; nw, nmodes) = model
    nk = kpts.n

    for ik in 1:nk
        if mod(ik, progress_print_step) == 0 && mpi_isroot()
            mpi_isroot() && @info "$(now()) ik = $ik / $nk"
            flush(stdout)
            flush(stderr)
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        for epdata in epdatas.data
            epdata.el_k = el_k
        end

        if !skip_eph
            get_eph_RR_to_kR!(ep_ekpR_obj, epmat, xk, no_offset_view(el_k.u))
        end

        if covariant_derivative_of_g
            get_fourier!(epmat_R.out, epmat_R, xk);
            # (iw_jw_imode, Rₚ, idir) -> (iw_jw_imode_idir, Rₚ)
            tmp = Base.ReshapedArray(epmat_R.out, (nw*nw*nmodes, length(epobj_ekpR_R.irvec), 3), ())
            epobj_ekpR_R.op_r .= reshape(permutedims(tmp, (1, 3, 2)), (nw*nw*nmodes*3, length(epobj_ekpR_R.irvec)))
        end

        # Multithreading setup
        for calc in calculators
            setup_calculator_inner!(calc; ik)
        end

        @threads for (id_chunk, ikqs) in enumerate(chunks(1:kqpts.n; n=nchunks_threads))
        # @time for (id_chunk, ikqs) in enumerate(collect(chunks(1:kqpts.n; n=nchunks_threads))[1:1])
            epdata = take!(epdatas)
            ep_ekpR = take!(ep_ekpRs)

            if covariant_derivative_of_g
                ep_ekpR_R = take!(ep_ekpR_Rs)
            else
                ep_ekpR_R = nothing
            end

            if ! precompute_ph
                dyn = take!(dyn_threads)
            else
                dyn = nothing
            end

            _run_eph_over_k_and_kq_inner(model, epdata, ik, ep_ekpR, el_kq_save,
                xk, ph_save, dyn, kpts, qpts, kqpts, ikqs, precompute_ph, id_chunk,
                energy_conservation, screening_params, skip_eph;
                ep_ekpR_R, calculators,
            )

            put!(ep_ekpRs, ep_ekpR)
            put!(epdatas, epdata)
            if ! precompute_ph
                put!(dyn_threads, dyn)
            end
            if covariant_derivative_of_g
                put!(ep_ekpR_Rs, ep_ekpR_R)
            end
        end # ikq chunk

        # Multithreading collect
        for calc in calculators
            postprocess_calculator_inner!(calc; ik)
        end

    end # ik

    for calc in calculators
        postprocess_calculator!(calc; qpts, symmetry)
    end
end


function _run_eph_over_k_and_kq_inner(model, epdata, ik, ep_ekpR, el_kq_save,
        xk, ph_save, dyn, kpts, qpts, kqpts, ikqs, precompute_ph, id_chunk,
        energy_conservation, screening_params, skip_eph;
        ep_ekpR_R, calculators,
    )

    (; nw, nmodes) = model

    ϵs = zeros(ComplexF64, model.nmodes)

    for ikq in ikqs
        xkq = kqpts.vectors[ikq]
        xq = xkq - xk
        xq = normalize_kpoint_coordinate(xq .+ 1/2) .- 1/2

        epdata.el_kq = el_kq_save[ikq]
        epdata.wtk = kpts.weights[ik]
        epdata.wtq = kqpts.weights[ikq]

        # Use precomputed data for the phonon state at q

        if precompute_ph
            # Use precomputed data for the phonon state at q
            iq = xk_to_ik(xq, qpts)
            if iq === nothing
                throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
            end
            epdata.ph = ph_save[iq]
        else
            # Compute phonon state at q.
            iq = nothing
            set_eigen!(epdata.ph, xq, dyn, model.mass, model.polar_phonon)
            if ! skip_eph
                set_eph_dipole_coeff!(epdata.ph, xq, model.polar_eph)
            end
        end

        # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
        check_energy_conservation_all(epdata, kqpts.ngrid, model.recip_lattice, energy_conservation...) || continue

        epdata_set_mmat!(epdata)

        # Compute electron-phonon coupling
        if !skip_eph
            get_eph_kR_to_kq!(epdata, ep_ekpR, xq)

            if ep_ekpR_R !== nothing
                # This must be done before the long-range calculation

                get_fourier!(ep_ekpR_R.out, ep_ekpR_R, xq)
                dg_wan = Base.ReshapedArray(ep_ekpR_R.out, (nw, nw, nmodes, 3), ())
                dg = zeros(ComplexF64, (epdata.el_kq.nband, epdata.el_k.nband, nmodes, 3))

                # Apply electron gauge matrices (Wannier to eigenstate)
                tmp1 = zeros(ComplexF64, nw, nw)
                @views for idir in 1:3, imode in 1:nmodes
                    tmp1 .= dg_wan[:, :, imode, idir]
                    dg[:, :, imode, idir] .= no_offset_view(epdata.el_kq.u)' * tmp1 * no_offset_view(epdata.el_k.u)
                end

                # Apply phonon gauge matrix (Wannier to eigenstate)
                tmp2 = zeros(ComplexF64, size(dg, 1), size(dg, 3))
                @views for idir in 1:3
                    for iw in axes(dg, 2)
                        tmp2 .= dg[:, iw, :, idir]
                        dg[:, iw, :, idir] .= tmp2 * epdata.ph.u
                    end
                end

                # One could compute the Berry connection term as below. However, there are two issues.
                # 1. One needs to sum all bands (or WFs) to compute matrix multiplication g * rbar.
                #    But this is not currently possible as we truncate g by the window already
                #    at the level of g(k, Rₚ).
                # 2. Calculating covatiant derivative of g using WFs are not exact in any case
                #    because one in principle needs terms like <u_k+q+b|dV|u_k> in plane wave.
                #    (or compute [r, dV] directly in plane wave)
                # Therefore, we just stick to the simple diagonal tight-binding approximation,
                # which is implemented by adding im * (rj - ri) * g_{ij} to dg
                # (i.e. derivative in tight-binding gauge, where phase factor is e^{i*k*(R + rj - ri)}).
                # # Add Berry connection term : im * (g * rbar_k - rbar_kq * g)
                # @views for idir in 1:3
                #     ξk  = no_offset_view(getindex.(epdata.el_k.rbar,  idir))
                #     ξkq = no_offset_view(getindex.(epdata.el_kq.rbar, idir))
                #     for imode in 1:nmodes
                #         g = no_offset_view(epdata.ep[:, :, imode])
                #         dg[:, :, imode, idir] .+= im .* (g * ξk .- ξkq * g)
                #     end
                # end

                # For debugging
                # if ik == ikq
                #     dk_dir = model.recip_lattice * Vec3(0, 0, 1)
                #     print("$(abs(dk_dir' * dg[1, 1, 6, :])), ")
                # end

                epdata_dg = OffsetArray(dg, epdata.el_kq.rng, epdata.el_k.rng, :, :)

            else
                epdata_dg = nothing

            end

            if screening_params !== nothing
                # FIXME: screening should go into calculator
                (; T, μ) = calculators[1].occ[1]
                xq_ = ElectronPhonon.normalize_kpoint_coordinate(xq .+ 0.5) .- 0.5
                ϵs .= epsilon_lindhard.(Ref(model.recip_lattice * xq_), epdata.ph.e, T, μ, Ref(screening_params))
                ϵs .= real.(ϵs)
            else
                ϵs .= 1
            end
            epdata_compute_eph_dipole!(epdata, ϵs; model)
            epdata_set_g2!(epdata)
        end

        # TODO: Screening

        # Now, we are done with matrix elements. All data saved in epdata.

        for calc in calculators
            # FIXME: Find out better way to pass epdata_dg
            run_calculator!(calc, epdata, ik, iq, ikq; xq, xk, id_chunk, epdata_dg)
        end

    end # ikq
end
