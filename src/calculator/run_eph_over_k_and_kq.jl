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
        use_gpu = false,   # Run the e-ph Wannier->Bloch interpolation on the GPU
        nq_batch_max = nothing,  # GPU: k+q points per batched kR->kq kernel (nothing = all k+q in one batch)
        nk_batch_max = 256,   # GPU: number of outer k points processed per batched RR->kR kernel
    ) where {FT}

    if model.epmat_outer_momentum != "el"
        throw(ArgumentError("model.epmat_outer_momentum must be el to use run_eph_outer_k"))
    end
    for calc in calculators
        if !allow_eph_outer_k(calc)
            throw(ArgumentError("$calc does not allow eph_outer_k. Use run_eph_outer_q instead."))
        end
    end

    # Outer-k MPI decomposition (multi-CPU/GPU): `mpi_comm_k` splits the OUTER k-points across ranks
    # — each rank computes the e-ph coupling only for its k-slice, yielding a rank-local
    # `calc.g2[:, i_local, :]` / `el_i`. k+q, the q-grid, and the phonon states stay FULL on every
    # rank (so any k can scatter to any k+q). `filter_kpoints` does the split + load-balance in the
    # shared `_setup`; the EliashbergCalculator is already rank-local; and the CPU and GPU inner
    # loops are UNCHANGED (they iterate whatever `kpts` they are handed). The solver then decomposes
    # over the matching outer states i (see MigdalEliashberg). q-decomposition (mpi_comm_q) is a
    # separate, not-yet-implemented scheme. See GPU_PROGRESS.md "Multi-GPU / MPI".
    mpi_comm_q === nothing || error("mpi_comm_q not implemented (use mpi_comm_k for outer-k decomposition)")
    (mpi_comm_k === nothing || !el_kq_from_unfolding) || throw(ArgumentError(
        "mpi_comm_k requires el_kq_from_unfolding = false (k+q stays full per rank; the unfolding " *
        "path is not split-aware)."))

    # Symmetry (IBZ outer k) is supported on the GPU path, but only with directly-computed k+q
    # (el_kq_from_unfolding = false): the IBZ reduction happens in the shared setup and the GPU loop
    # is symmetry-agnostic (validated: filter/states/scatter match CPU). GPU unfolding of the k+q
    # electron states is not implemented. Checked before setup so the unfolding branch is not entered.
    (!use_gpu || symmetry === nothing || !el_kq_from_unfolding) || throw(ArgumentError(
        "use_gpu supports symmetry only with el_kq_from_unfolding = false (GPU k+q unfolding not implemented)."))

    setup = _setup_eph_over_k_and_kq(model, kpts_input, kqpts_input;
        mpi_comm_k, mpi_comm_q, fourier_mode, window_k, window_kq,
        el_kq_from_unfolding, symmetry, calculators, nchunks_threads,
        covariant_derivative_of_g, use_gpu,
    )

    if use_gpu
        # GPU path: minimal scope. Extra flags must be off; the loop asserts the
        # rest (no polar, full bands, commensurate grids, no symmetry/screening/MPI).
        covariant_derivative_of_g && throw(ArgumentError("use_gpu does not support covariant_derivative_of_g"))
        skip_eph && throw(ArgumentError("use_gpu requires skip_eph = false"))
        _loop_eph_over_k_and_kq_gpu(model,
            setup.kpts, setup.qpts, setup.kqpts,
            setup.el_k_save, setup.el_kq_save,
            setup.ph_save, setup.precompute_ph,
            setup.epdatas;
            calculators,
            energy_conservation, screening_params,
            progress_print_step, nq_batch_max, nk_batch_max, symmetry,
        )
    else
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
    end

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
        use_gpu = false,
    ) where {FT}

    (; nw, nmodes) = model

    # Generate k points
    @time kpts, iband_min, iband_max, nelec_below_window_k = filter_kpoints(
        kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode, use_gpu)
    @time el_k_save  = compute_electron_states(model, kpts,  ["eigenvalue", "eigenvector", "velocity", "position"], window_k;  fourier_mode, use_gpu)
    nk = kpts.n

    if el_kq_from_unfolding
        # To ensure gauge consistency between symmetry-equivalent k points, we explicitly compute
        # electron states only for k+q in the irreducible BZ and unfold them to the full BZ.
        @time kqpts_irr, iband_kq_min, iband_kq_max, nelec_below_window_kq = filter_kpoints(
            kqpts_input, nw, model.el_ham, window_kq, mpi_comm_q; symmetry, fourier_mode, use_gpu)

        kqpts, ik_to_ikirr_isym_kq = unfold_kpoints(kqpts_irr, symmetry)

        el_kq_save_irr = compute_electron_states(model, kqpts_irr, ["eigenvalue", "eigenvector", "velocity", "position"], window_kq; fourier_mode, use_gpu)

        el_kq_save = unfold_ElectronStates(model, el_kq_save_irr, kqpts_irr, kqpts, ik_to_ikirr_isym_kq, symmetry; fourier_mode)

        # el_kq_save_irr is not used anymore.
        el_kq_save_irr !== el_kq_save && empty!(el_kq_save_irr)

    else
        @time kqpts, iband_kq_min, iband_kq_max, nelec_below_window_kq = filter_kpoints(
            kqpts_input, nw, model.el_ham, window_kq, mpi_comm_q; fourier_mode, use_gpu)

        @time el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity", "position"], window_kq; fourier_mode, use_gpu)
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
        @time ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode, use_gpu)
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


# =============================================================================
#  GPU calculator loop (see README_GPU.md).
#
#  This mirrors `_loop_eph_over_k_and_kq` but moves the e-ph Wannier->Bloch interpolation
#  onto the device using the batched drivers from `wannier_to_bloch_batched.jl`. The code here is
#  backend-generic: it only calls `to_device` and the generic batched drivers, so no CUDA
#  code lives in the base package (the device methods are provided by the CUDA extension). It
#  is a separate function from the CPU `_loop_eph_over_k_and_kq` because its control flow —
#  batched over k-batches and q-batches with device staging — differs from the per-(k,q) CPU loop,
#  not because it holds any device-specific code.
#
#  Minimal first step — the rest is asserted off (not implemented on the GPU path):
#    * no polar / long-range terms (so ϵs = 1, no dipole correction)
#    * full bands, no energy window  ->  nband == nw at every k / k+q (uniform batch shapes)
#    * commensurate k / k+q grids (precompute_ph) so phonon states are precomputed
#    * no covariant derivative of g, no screening, no symmetry/unfolding, single node (no MPI),
#      and energy_conservation = (:None, 0.0)
#
#  Buffer reuse: device buffers are allocated once before the k loop and reused for every
#  (k, q), so the loop itself allocates almost nothing.
function _loop_eph_over_k_and_kq_gpu(
        model       :: Model{FT},
        kpts, qpts, kqpts,
        el_k_save, el_kq_save,
        ph_save, precompute_ph,
        epdatas;
        calculators = [],
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        nq_batch_max::Union{Int, Nothing} = nothing,
        nk_batch_max::Int = 256,
        symmetry = nothing,
    ) where {FT}

    (; nw, nmodes) = model
    nk = kpts.n
    nkq = kqpts.n

    # ----- scope asserts (minimal GPU step) -----
    precompute_ph || throw(ArgumentError(
        "use_gpu requires commensurate k / k+q grids so phonon states are precomputed (precompute_ph)."))
    (!model.polar_phonon.use && !model.polar_eph.use) || throw(ArgumentError(
        "use_gpu does not support polar / long-range terms. Use the CPU path."))
    energy_conservation === (:None, 0.0) || throw(ArgumentError(
        "use_gpu supports only energy_conservation = (:None, 0.0)."))
    screening_params === nothing || throw(ArgumentError("use_gpu does not support screening_params."))
    # symmetry (IBZ outer k) is allowed: the reduction is done in the shared setup and this loop is
    # symmetry-agnostic — `symmetry` is only passed through to `postprocess_calculator!`. The
    # dispatcher gates the unsupported el_kq_from_unfolding = true case.

    # Default (nq_batch_max === nothing): process all k+q in a single batch per k. Fewer, larger
    # kR->kq / calculator kernels — the GPU e-ph path is launch-bound for small nw/nmodes, so one
    # big batch is ~1.8× faster than the old 1024 default at 16³, and the per-batch device buffers
    # stay modest even at 50³ (≲ a few hundred MB). Pass an Int to cap the batch if memory-limited.
    nq_batch_max = nq_batch_max === nothing ? nkq : min(nq_batch_max, nkq)
    nk_batch_max = min(nk_batch_max, nk)

    # The GPU path is always device-native/batched: the e-ph matrix for a whole (k, {k+q}) batch
    # stays on the device and each calculator does its reduction/scatter there (no per-(k,q) host
    # callback, no host g2, no complex-ep D2H). Every calculator must therefore implement the
    # batched device hook; a non-batched calculator fails loudly rather than silently falling back
    # to a slow host path — a hard-to-spot performance cliff (see README_GPU.md "Decisions").
    isempty(calculators) && throw(ArgumentError("use_gpu requires at least one calculator."))
    all(allow_eph_batched, calculators) || throw(ArgumentError(
        "use_gpu requires every calculator to implement the batched device hook " *
        "(allow_eph_batched = true); the GPU path does not fall back to the host run_calculator! loop."))

    # The k+q side of the interpolation runs full-band (uniform nw×nw, using the full eigenvectors
    # `el.u_full`); the energy window is applied in the calculator scatter, where out-of-window
    # states have imap == 0 and are skipped, so a windowed run needs no special handling there.
    fullband = all(el.nband == nw for el in el_k_save) && all(el.nband == nw for el in el_kq_save)

    # ----- k-side window projection -----
    # The OUTER-k side does NOT need all nw bands: the calculator scatter keeps only in-window
    # (band, k) pairs, so rotate the k side by only an `nbandk_max`-wide CONTIGUOUS window of
    # eigenvector columns per k (`u_full[:, nb0+1 : nb0+nbandk_max]`, positioned to contain that k's
    # in-window range `rng`). Every downstream per-(k,q) object — the kR→kq GEMM, both gauge
    # rotations, g2, and the scatter — shrinks by nw/nbandk_max, the dominant ∝ nk·nq cost at narrow
    # windows (e.g. TaAs ±0.1 eV: nbandk_max ≈ 4 of nw = 32). Out-of-window bands inside the projected
    # window scatter to imap == 0 and are skipped exactly as before; the calculators only need the
    # per-k physical-band offset `ibandk_offset` (0-based: ep_kq band n ↔ physical band
    # ibandk_offset + n) to address their imaps. Full-band runs have
    # nbandk_max = nw, ibandk_offset = 0 — the path (shapes, GEMMs, results) is unchanged.
    # ibandk_offsets[ik]: 0-based window start = (first in-window band − 1), clamped so the
    # nbandk_max-wide window [offset+1, offset+nbandk_max] stays within [1, nw]. The clamp only
    # bites when a k's bands sit near the top edge: e.g. nw=4, nbandk_max=3, rng=3:4 → first−1=2
    # exceeds nw−nbandk_max=1, so offset=1 (window 2:4 ⊇ 3:4); offset=2 would give 3:5, off the top.
    nbandk_max = fullband ? nw : max(maximum(el -> el.nband, el_k_save; init = 1), 1)
    ibandk_offsets = fullband ? zeros(Int, nk) :
        [clamp(first(el.rng) - 1, 0, nw - nbandk_max) for el in el_k_save]

    # ----- device interpolators (allocated once) -----
    epmat_dev = to_device(model.epmat)
    epmat_itp = BatchedWannierInterpolator(epmat_dev; batch_size = nk_batch_max)
    ep_ekpR_dev = to_device(get_next_wannier_object(model.epmat))
    ndata_ekpR = nw * nbandk_max * nmodes
    # Set ndata BEFORE building the interpolator (its Fourier cache is sized to parent.ndata):
    # under the k-side projection only the first nw·nbandk_max·nmodes rows of op_r are filled/read.
    ep_ekpR_dev.ndata = ndata_ekpR
    ep_ekpR_itp = BatchedWannierInterpolator(ep_ekpR_dev; batch_size = nq_batch_max)
    nr_ep = ep_ekpR_dev.nr

    # ----- persistent workspace (allocated once, reused across all (k, q)) -----
    # All device staging is sized to the full batch and used as plain CuArrays (not
    # batch-sliced views), so the batched drivers' reshape/cuBLAS calls stay on dense arrays.

    # RR->kR over a batch of `nk_batch_max` outer-k at once: one batched kernel per batch instead of one
    # launch-bound single-k call per k. `ep_ekpR_all` holds g(k, R_ep) for the whole batch; each
    # k's slice is then copied into `ep_ekpR_dev` (device→device) for the inner kR->kq driver.
    uks_dev     = similar(epmat_dev.op_r, Complex{FT}, nw, nbandk_max, nk_batch_max)
    uks_host    = Array{Complex{FT}}(undef, nw, nbandk_max, nk_batch_max)
    ep_ekpR_all = similar(epmat_dev.op_r, Complex{FT}, ndata_ekpR, nr_ep, nk_batch_max)
    ks_batch     = Vector{Vec3{FT}}(undef, nk_batch_max)

    uphs_dev = similar(epmat_dev.op_r, Complex{FT}, nmodes, nmodes, nq_batch_max)
    epkq_dev = similar(epmat_dev.op_r, Complex{FT}, nw, nbandk_max, nmodes, nq_batch_max)

    # In-place scratch for the per-k kR->kq driver (g / tmp), reused across all (k, q) so the
    # driver allocates nothing per call. Sized for the max batch width `nq_batch_max`; the driver
    # uses the first `nq_batch` columns for a partial final batch.
    kRkq_ws = KRtoKQWorkspace(ep_ekpR_dev.op_r, ndata_ekpR, nw, nbandk_max, nmodes, nq_batch_max)

    # Collect the k+q electron eigenvectors on the host and copy to the device once (they do not
    # depend on the outer k), reused across all k. Each q-batch reads a contiguous slice directly.
    ukqs_all_dev = similar(epmat_dev.op_r, Complex{FT}, nw, nw, nkq)
    let ukqs_all_host = Array{Complex{FT}}(undef, nw, nw, nkq)
        for ikq in 1:nkq
            @views ukqs_all_host[:, :, ikq] .= el_kq_save[ikq].u_full
        end
        copyto!(ukqs_all_dev, ukqs_all_host)
    end
    # Phonon eigenvectors `u` and frequencies `e` depend only on iq, so (like ukqs_all_dev above)
    # collect the full q-grid stacks on the host and copy to the device once, then gather per
    # batch on the device by index (below).
    uph_all_dev = similar(epmat_dev.op_r, Complex{FT}, nmodes, nmodes, qpts.n)
    ωq_all_dev  = similar(epmat_dev.op_r, FT, nmodes, qpts.n)
    let uph_all_host = Array{Complex{FT}}(undef, nmodes, nmodes, qpts.n),
        ωq_all_host  = Array{FT}(undef, nmodes, qpts.n)
        for iq in 1:qpts.n
            @views uph_all_host[:, :, iq] .= ph_save[iq].u
            @views ωq_all_host[:, iq]     .= ph_save[iq].e
        end
        copyto!(uph_all_dev, uph_all_host)
        copyto!(ωq_all_dev, ωq_all_host)
    end

    qs_batch  = Vector{Vec3{FT}}(undef, nq_batch_max)
    iqs_batch  = Vector{Int}(undef, nq_batch_max)
    iqs_batch_dev    = similar(epmat_dev.op_r, Int, nq_batch_max)

    # q-index lookup without an O(prod(ngrid)) dense table (ngrid can be ~1e6/dim with few points).
    # `combine_kpoint_grids` sorts `qpts` by hash, so a FULL grid has iq == hash+1 (pure arithmetic);
    # otherwise fall back to the O(n) hash Dict `GridKpoints` carries. Decide once:
    qpts_is_full = qpts.n == prod(qpts.ngrid) &&
        all(_hash_xk(qpts.vectors[iq], qpts) == iq - 1 for iq in 1:qpts.n)

    # Integer grid-coord hash for iq, replacing the per-(k,q) float normalize + `_hash_xk`:
    #   hc_i = mod(xkqs_int[i,ikq] - xks_int[i,ik], ng_i),  hash = (hc1*ng2 + hc2)*ng3 + hc3,
    # reproducing `_hash_xk` bit-identically with no Float64 in the hot loop. Requires every k and
    # k+q to lie exactly on the q-grid (ngrid a multiple of both meshes) — guaranteed by precompute_ph,
    # asserted above. `qs_batch` reads qpts.vectors[iq] ≡ (xkq-xk) mod G; the Fourier phase is periodic
    # in q→q+G, so the interpolated e-ph matrix is unchanged.
    ng1, ng2, ng3 = qpts.ngrid
    shq = qpts.shift
    xkqs_int = Matrix{Int}(undef, 3, nkq)
    xks_int  = Matrix{Int}(undef, 3, nk)
    for ikq in 1:nkq, d in 1:3
        xkqs_int[d, ikq] = round(Int, (kqpts.vectors[ikq][d] - shq[d]) * qpts.ngrid[d])
    end
    for ik in 1:nk, d in 1:3
        xks_int[d, ik] = round(Int, kpts.vectors[ik][d] * qpts.ngrid[d])
    end

    ikqs_host = Vector{Int}(undef, nq_batch_max)
    ωq_dev    = similar(epmat_dev.op_r, FT, nmodes, nq_batch_max)
    ikqs_dev  = similar(epmat_dev.op_r, Int, nq_batch_max)
    g2_dev    = similar(epmat_dev.op_r, FT, nw, nbandk_max, nmodes, nq_batch_max)

    epdata = take!(epdatas)

    for kstart in 1:nk_batch_max:nk
        kend = min(kstart + nk_batch_max - 1, nk)
        iks_batch = kstart:kend
        nk_batch = length(iks_batch)

        # Stack U(k) and the k list for this outer-k batch (pad the partial tail with valid
        # duplicated data so the batched RR->kR runs on dense `nk_batch_max`-sized arrays).
        for (ik_ind, ik) in enumerate(iks_batch)
            # k-side window projection: the `nbandk_max` contiguous eigenvector columns around this
            # k's in-window range (all nw columns when full-band). The window selection itself
            # still happens in the calculator scatter (imap == 0 outside), offset by ibandk_offsets[ik].
            nb0 = ibandk_offsets[ik]
            @views uks_host[:, :, ik_ind] .= el_k_save[ik].u_full[:, nb0+1:nb0+nbandk_max]
            ks_batch[ik_ind] = kpts.vectors[ik]
        end
        for ik_ind in (nk_batch+1):nk_batch_max
            @views uks_host[:, :, ik_ind] .= uks_host[:, :, nk_batch]
            ks_batch[ik_ind] = ks_batch[nk_batch]
        end
        copyto!(uks_dev, uks_host)

        # One batched RR->kR over the whole batch: g(k, R_ep) for all k in the batch.
        get_eph_RR_to_kR_batched!(ep_ekpR_all, epmat_itp, ks_batch, uks_dev)

        # Outer-batch-resident calculators (re)point/zero their per-batch device buffer here, before
        # this batch's scatters; no-op (default hooks) for calculators that hold their whole output.
        for calc in calculators
            setup_calculator_outer_batch!(calc; kstart, kend, proto = epmat_dev.op_r)
        end

    for (ik_ind, ik) in enumerate(iks_batch)
        if mod(ik, progress_print_step) == 0 && mpi_isroot()
            @info "$(now()) ik = $ik / $nk"
            flush(stdout); flush(stderr)
        end
        el_k = el_k_save[ik]
        epdata.el_k = el_k

        # Load this k's g(k, R_ep) into the interpolator's parent (cheap device→device copy);
        # the inner kR->kq driver reads `ep_ekpR_dev.op_r` fresh. Under the k-side projection
        # only the first ndata_ekpR rows are used (op_r itself is full-band-sized).
        @views ep_ekpR_dev.op_r[1:ndata_ekpR, :] .= ep_ekpR_all[:, :, ik_ind]
        ep_ekpR_dev._id += 1

        for calc in calculators
            setup_calculator_inner!(calc; ik)
        end

        qstart = 1
        while qstart <= nkq
            qend = min(qstart + nq_batch_max - 1, nkq)
            nq_batch = qend - qstart + 1

            # Build the per-q index list for this batch (host-side integers only); the phonon
            # eigenvectors `u` and frequencies `e` are gathered on the device by iq below.
            for j in 1:nq_batch
                ikq = qstart + j - 1
                # Integer grid-coord hash for iq (no Float64 normalize/_hash_xk per pair).
                h1 = mod(xkqs_int[1, ikq] - xks_int[1, ik], ng1)
                h2 = mod(xkqs_int[2, ikq] - xks_int[2, ik], ng2)
                h3 = mod(xkqs_int[3, ikq] - xks_int[3, ik], ng3)
                hash = (h1 * ng2 + h2) * ng3 + h3
                # Full sorted grid → iq = hash+1 (no table); else O(n) Dict (no ngrid³ array).
                iq = qpts_is_full ? hash + 1 : get(qpts._xk_hash_to_ik, hash, 0)
                # Guard both paths: Dict miss → iq==0; full-grid fast path → iq>qpts.n if off-grid.
                (iq < 1 || iq > qpts.n) && throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
                # q-vector from the O(n) qpts.vectors (a cached gather — cheaper than recomputing it
                # with 3 float divisions; qpts.vectors is O(n), never ngrid³). ≡ (xkq-xk) mod G.
                qs_batch[j] = qpts.vectors[iq]
                iqs_batch[j] = iq
                ikqs_host[j] = ikq
            end
            rng_q = 1:nq_batch   # this batch's columns within the nq_batch_max-sized device buffers

            # Gather this batch's phonon eigenvectors/frequencies on the device by iq
            # (uphs_dev[:,:,j] = uph_all_dev[:,:,iq[j]]); ωq gathered here too so the fused kernel
            # can fold g2 = |ep|²/(2ω) in the same pass. Everything below runs at width nq_batch
            # via views into the nq_batch_max-sized buffers, so there is no padded tail.
            # H2D copy of just the first nq_batch indices (5-arg contiguous copy — copying a host↔
            # device SubArray view instead would fall back to scalar indexing); the device gather
            # then reads only `view(iqs_batch_dev, rng_q)`, so the untouched tail is never used.
            # `@inbounds`: bounds-checking a device INDEX ARRAY costs a Bool map+reduce kernel and
            # a D2H round trip of the result per (k, batch); `iq` is already validated on the host
            # (the guard above), so the device-side re-check is pure overhead.
            copyto!(iqs_batch_dev, 1, iqs_batch, 1, nq_batch)
            @inbounds @views uphs_dev[:, :, rng_q] .= uph_all_dev[:, :, view(iqs_batch_dev, rng_q)]
            @inbounds @views ωq_dev[:, rng_q]      .= ωq_all_dev[:, view(iqs_batch_dev, rng_q)]
            # k+q rotations: a contiguous slice of the prebuilt device stack (no copy).
            ukqs_used = view(ukqs_all_dev, :, :, qstart:qend)

            # One batched Wannier->Bloch over this batch's q: ep_kq(q) (nw, nw, nmodes), folding
            # g2 = |ep|²/(2ω) into the same fused kernel pass.
            get_eph_kR_to_kq_batched!(view(epkq_dev, :, :, :, rng_q), ep_ekpR_itp, view(qs_batch, rng_q),
                view(uphs_dev, :, :, rng_q), ukqs_used; ws=kRkq_ws,
                g2_out = view(g2_dev, :, :, :, rng_q), ωq = view(ωq_dev, :, rng_q))

            # Hand the batch's e-ph matrix (still on the device) to each calculator, which forms
            # g2 / scatters it on the device; no D2H of the e-ph matrix here. 5-arg contiguous H2D
            # copy of the first nq_batch k+q indices (a SubArray-view copy would go scalar).
            copyto!(ikqs_dev, 1, ikqs_host, 1, nq_batch)
            for calc in calculators
                run_calculator_batched!(calc,
                    view(epkq_dev, :, :, :, rng_q), view(ωq_dev, :, rng_q),
                    ik, view(ikqs_dev, rng_q); g2 = view(g2_dev, :, :, :, rng_q),
                    ibandk_offset = ibandk_offsets[ik])
            end

            qstart = qend + 1
        end # q batch

        for calc in calculators
            postprocess_calculator_inner!(calc; ik)
        end
    end # ik within batch

    # Outer-batch-resident calculators D2H this batch's device buffer into their host output here
    # (one contiguous copy per batch, not per k); no-op (default hooks) for full-resident calculators.
    for calc in calculators
        flush_calculator_outer_batch!(calc; kstart, kend)
    end

    # Bound the host look-ahead to one k-batch: a device-resident calculator never D2H-syncs per k,
    # so without this the host can race across all batches, keeping every batch's RR->kR scratch +
    # per-k transients live in the memory pool at once. Draining at each batch boundary caps the
    # transient working set with negligible utilization cost. No-op on the CPU backend.
    device_synchronize(epmat_dev.op_r)
    end # k batch

    put!(epdatas, epdata)

    for calc in calculators
        postprocess_calculator!(calc; qpts, symmetry)
    end
end
