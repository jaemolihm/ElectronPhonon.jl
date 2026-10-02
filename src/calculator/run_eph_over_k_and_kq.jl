"""
    run_eph_over_k_and_kq(model, kpts, kqpts; calculators, backend, batched, kwargs...)

Sweep the outer k points and, for each, the inner k+q points, handing each calculator the e-ph
coupling as an [`EPBlock`](@ref)`{OuterKLoop}`: one outer k with a tile of k+q points.

* `backend :: AbstractBackend = CPUBackend()` — array placement. Pass
  `backend = ElectronPhonon.gpu_backend()` for a GPU run (requires a loaded GPU extension, e.g. CUDA).
* `batched :: Union{Nothing, Bool} = nothing` — the batched loop (`nothing` or `true`) or the
  per-(k,q) host [`EPData`](@ref) loop (`false`, CPU only, not working until the loops are merged:
  ElectronPhonon.jl issue #72).
* `fourier_mode = "gridopt"` — `"gridopt"` or `"normal"` on a `CPUBackend` (a batched mode is an
  `ArgumentError`).
* `fill_padding_nan = false` — fill the state containers' box entries outside each window with NaN,
  so that a calculator that reads them fails (a test switch).

The batched path has a narrower scope than the per-point one (no polar/long-range, no screening,
`energy_conservation = (:None, 0.0)`, commensurate grids, no `covariant_derivative_of_g`, no
`skip_eph`, no `el_kq_from_unfolding` under symmetry); it asserts each of these.

Two further keywords let a run take its electron eigenpairs from a shared cache instead of
diagonalizing H(k) itself:

* `el_k_eigenpairs`, `el_kq_eigenpairs :: Union{Nothing, Eigenpairs}` — a cache from
  [`electron_eigenpairs`](@ref) for the outer k and the inner k+q side. They give runs over
  *overlapping* k-point sets a shared eigenvector gauge, which a band-resolved `g2 = |g|²` needs
  inside a degenerate multiplet. Each cache must cover every k-point *this rank* visits on its side
  (a missing point is an error, not a silent recompute) and be resident on the run's `backend`; both
  are read-only, so one cache serves runs with different windows and quantity lists.

Runs sharing a k+q cache must make the same `el_kq_from_unfolding` choice: unfolding carries the
cached *irreducible* eigenvector rotated by the symmetry operation, a direct run the cached
eigenvector at the point itself, and the two are a gauge apart. Nothing checks this.

* `ph_eigenpairs :: Union{Nothing, Eigenpairs}` — the same for the phonons: a cache with
  `nbasis = nmodes` over the run's own q points, replacing the dynamical-matrix diagonalization in
  [`compute_phonon_states`](@ref). Build it with [`phonon_eigenpairs`](@ref), or assemble it from
  an earlier run's returned `(qpts, ph_save)`; either pins the phonon eigenmode basis of two runs
  to each other inside a degenerate multiplet. The q points a run visits are
  `combine_kpoint_grids(kpts, kqpts)`, not an argument, so a cache must cover that set.

Returns `(; kpts, qpts, el_k, el_kq, ph)`, the run's state containers (`BatchedElectronState`,
`BatchedPhononState`) holding the quantities the loop and the calculators requested.
"""
function run_eph_over_k_and_kq(
        model       :: Model{FT},
        kpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints, FilteredBandStates},
        kqpts_input :: Union{NTuple{3,Int}, Kpoints, GridKpoints, FilteredBandStates},
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
        backend :: AbstractBackend = CPUBackend(),   # Where arrays live (gpu_backend() for a GPU run)
        batched :: Union{Nothing, Bool} = nothing,   # Loop shape (nothing = batched)
        nq_batch_max = nothing,  # Batched: k+q points per batched kR->kq kernel (nothing = all k+q in one batch)
        # Batched: number of outer k points per batched RR->kR kernel and per calculator bracket.
        # Calculators still see one block per outer k, but a k's q-tiles are not contiguous (the
        # q-tile loop is outside the k loop); this also sets how many outer k reuse one kR->kq phase
        # tile.
        nk_outer_batch_max = 256,
        el_k_eigenpairs  :: Union{Nothing, Eigenpairs} = nothing,
        el_kq_eigenpairs :: Union{Nothing, Eigenpairs} = nothing,
        ph_eigenpairs    :: Union{Nothing, Eigenpairs} = nothing,
        fill_padding_nan :: Bool = false,
        verbosity::Int = 1,
    ) where {FT}

    if model.epmat_outer_momentum != "el"
        throw(ArgumentError("model.epmat_outer_momentum must be el to use run_eph_over_k_and_kq"))
    end
    # The per-point loop queries interpolators one k at a time, which the batched Fourier modes
    # (a k-list registered in advance) do not serve. A non-CPU backend ignores `fourier_mode` in the loop.
    (backend isa CPUBackend && fourier_mode ∉ ("normal", "gridopt")) && throw(ArgumentError(
        "fourier_mode = \"$fourier_mode\" is not supported by run_eph_over_k_and_kq on a CPU backend: " *
        "its loop queries interpolators one k at a time. Use \"gridopt\" (the default) or \"normal\"."))
    screening_params === nothing || error(
        "screening_params is not supported: dielectric screening is currently disabled (ϵ ≡ 1). " *
        "Pass screening_params = nothing.")

    # Resolve the loop shape once, here, so nothing below this entry sees anything but a `Bool`. An
    # EXPLICIT `batched = false` on a GPU backend is an error rather than silently overridden.
    batched_resolved = batched === nothing ? true : batched
    (backend isa CPUBackend || batched_resolved) || throw(ArgumentError(
        "batched = false is not supported on a GPU backend: the per-(k,q) host `EPData` payload " *
        "cannot be built from device arrays. Pass backend = CPUBackend() for the per-point path."))

    for calc in calculators
        if !supports(calc, OuterKLoop)
            throw(ArgumentError("$calc does not support the outer-k loop. Use run_eph_over_q_and_k instead."))
        end
        # The per-point path hands each calculator the per-(k,q) host `EPData`.
        if !batched_resolved && !supports(calc, EPData)
            throw(ArgumentError("$calc does not declare support for the per-(k,q) host payload; " *
                "define supports(::$(typeof(calc)), ::Type{EPData}) = true."))
        end
    end

    # Outer-k MPI decomposition (multi-CPU/GPU): `mpi_comm_k` splits the OUTER k-points across ranks
    # — each rank computes the e-ph coupling for its k-slice only (rank-local `calc.g2` / `el_i`),
    # while k+q, the q-grid, and the phonon states stay FULL per rank (so any k can scatter to any
    # k+q). `filter_kpoints` does the split + load-balance in `_setup`; the CPU and GPU inner loops
    # are unchanged (they iterate whatever `kpts` they receive). `mpi_comm_q` is a separate scheme,
    # not yet implemented.
    mpi_comm_q === nothing || error("mpi_comm_q not implemented (use mpi_comm_k for outer-k decomposition)")
    (mpi_comm_k === nothing || !el_kq_from_unfolding) || throw(ArgumentError(
        "mpi_comm_k requires el_kq_from_unfolding = false (k+q stays full per rank; the unfolding " *
        "path is not split-aware)."))

    # Symmetry (IBZ outer k) is supported on the batched path, but only with directly-computed k+q
    # (el_kq_from_unfolding = false): the IBZ reduction happens in the shared setup and the batched
    # loop is symmetry-agnostic (validated: filter/states/scatter match the per-point path). Unfolding
    # of the k+q electron states is not implemented there. Checked before setup so the unfolding
    # branch is not entered.
    (!batched_resolved || symmetry === nothing || !el_kq_from_unfolding) || throw(ArgumentError(
        "the batched path supports symmetry only with el_kq_from_unfolding = false " *
        "(batched k+q unfolding not implemented)."))

    # A prebuilt k+q FilteredBandStates is consumed as-is (the caller already built the full-BZ selection,
    # e.g. via unfold_band_states), so the internal IBZ+unfold path does not run — el_kq_from_unfolding
    # is meaningless there.
    (!(kqpts_input isa FilteredBandStates) || !el_kq_from_unfolding) || throw(ArgumentError(
        "el_kq_from_unfolding = true is not supported when the k+q argument is a prebuilt FilteredBandStates; " *
        "build the full-BZ k+q selection explicitly (e.g. unfold_band_states) and pass el_kq_from_unfolding = false."))

    setup = _setup_eph_over_k_and_kq(model, kpts_input, kqpts_input;
        mpi_comm_k, mpi_comm_q, fourier_mode, window_k, window_kq,
        el_kq_from_unfolding, symmetry, calculators, nchunks_threads,
        covariant_derivative_of_g, backend, batched = batched_resolved, verbosity,
        el_k_eigenpairs, el_kq_eigenpairs, ph_eigenpairs, fill_padding_nan,
    )

    if batched_resolved
        # Batched path: minimal scope. Extra flags must be off; the loop asserts the
        # rest (no polar, full bands, commensurate grids, no screening).
        covariant_derivative_of_g && throw(ArgumentError(
            "the batched path does not support covariant_derivative_of_g"))
        skip_eph && throw(ArgumentError("the batched path requires skip_eph = false"))
        _loop_eph_over_k_and_kq_batched(model,
            setup.kpts, setup.qpts, setup.kqpts,
            setup.el_k, setup.el_kq, setup.sel_k, setup.sel_kq,
            setup.ph, setup.precompute_ph,
            setup.epmat_dev, setup.backend;
            setup.el_quantities, setup.ph_quantities, calculators,
            energy_conservation, screening_params, nchunks_threads,
            progress_print_step, nq_batch_max, nk_outer_batch_max, symmetry, verbosity,
        )
        return (; setup.kpts, setup.qpts, setup.el_k, setup.el_kq, setup.ph)
    else
        _loop_eph_over_k_and_kq(model,
            setup.kpts, setup.qpts, setup.kqpts,
            setup.el_k_save, setup.el_kq_save,
            setup.ph_save, setup.precompute_ph,
            setup.epstates, setup.ep_ekpRs, setup.epmat, setup.ep_ekpR_obj,
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
        kpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints, FilteredBandStates},
        kqpts_input :: Union{NTuple{3,Int}, Kpoints, GridKpoints, FilteredBandStates},
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
        backend :: AbstractBackend = CPUBackend(),
        batched :: Bool = false,
        el_k_eigenpairs  :: Union{Nothing, Eigenpairs} = nothing,
        el_kq_eigenpairs :: Union{Nothing, Eigenpairs} = nothing,
        ph_eigenpairs    :: Union{Nothing, Eigenpairs} = nothing,
        fill_padding_nan :: Bool = false,
        verbosity::Int = 1,
    ) where {FT}

    (; nw, nmodes) = model

    # The quantities the batched loop builds the containers with: its own (the electron and phonon
    # eigenvectors) and the calculators'.
    el_quantities = batched ? union([:u], required_el_quantities.(calculators)...) : nothing
    ph_quantities = union([:u], required_ph_quantities.(calculators)...)

    # Outer k and k+q setup via the shared role helpers. Each yields a `FilteredBandStates` selection
    # (a prebuilt one passed through verbatim, or filtered from a grid — the k+q grid path also
    # IBZ-reduces + unfolds under symmetry) plus its `kpts` and computed electron states: per-point
    # `ElectronState`s, or on the batched path a container built from the selection.
    (; kpts, iband_min, iband_max, el_k_save, el_k, sel_k) = _setup_electron_k(model, kpts_input;
        window_k, mpi_comm_k, symmetry, fourier_mode, backend, verbosity, el_k_eigenpairs,
        el_quantities, fill_padding_nan)
    nk = kpts.n

    el_kq_quantities = ["eigenvalue", "eigenvector", "velocity", "position"]
    (; kqpts, el_kq_save, el_kq, sel_kq) = _setup_electron_kq(model, kqpts_input;
        window_kq, mpi_comm_q, symmetry, el_kq_from_unfolding, el_kq_quantities,
        fourier_mode, backend, verbosity, el_kq_eigenpairs, el_quantities, fill_padding_nan)


    # Precompute qpts and phonon states if k and k+q meshes are commensurate
    if all(kpts.ngrid .> 0) && all(mod.(kqpts.ngrid, kpts.ngrid) .== 0)
        # kqpts is denser than kpts
        precompute_ph = true
        qpts = maybe_time(verbosity) do
            combine_kpoint_grids(kqpts, kpts, -, kqpts.ngrid)
        end

    elseif all(kpts.ngrid .> 0) && all(mod.(kpts.ngrid, kqpts.ngrid) .== 0)
        # kpts is denser than kqpts
        precompute_ph = true
        qpts = maybe_time(verbosity) do
            combine_kpoint_grids(kqpts, kpts, -, kpts.ngrid)
        end

    else
        precompute_ph = false
        ph_eigenpairs === nothing || throw(ArgumentError(
            "ph_eigenpairs requires commensurate k / k+q grids: on incommensurate grids the " *
            "phonon states are solved per (k, q) inside the loop, so there is no q-point set for " *
            "a cache to cover."))
    end


    # The per-point loop takes/puts one EPState buffer per thread, sized to its largest window.
    epstates = batched ? nothing : get_epstates_channel(FT, nw, nmodes,
        max(maximum(el.nband for el in el_k_save), maximum(el.nband for el in el_kq_save)))

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


    # Precompute phonon states if precompute_ph == true: a container on the batched path.
    if precompute_ph
        ph_save, ph = maybe_time(verbosity) do
            if batched
                nothing, compute_phonon_states_batched(model, qpts, ph_quantities; fourier_mode,
                                                       backend, eigenpairs = ph_eigenpairs)
            else
                # FIXME: Compute velocity_diagonal only if needed by calculator.
                compute_phonon_states(model, qpts,
                    ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"];
                    fourier_mode, eigenpairs = ph_eigenpairs), nothing
            end
        end
        dyn_threads = nothing
    else
        qpts = nothing
        ph_save = nothing
        ph = nothing
        dyn_threads = get_interpolator_channel(model.ph_dyn; fourier_mode)
    end


    # `model.epmat` is uploaded to the backend ONCE here — a separate device object that
    # `_loop_eph_over_k_and_kq_batched` reuses rather than re-uploading. On a `CPUBackend`
    # `to_device` is the identity (`common/gpu_utils.jl`), so `epmat_dev === model.epmat`.
    # `backend` is carried in `LoopContext` and passed to `setup_calculator!`.
    epmat_dev = batched ? to_device(backend, model.epmat) : nothing

    # The per-point loop initializes the calculators here; the batched loop does, once it has
    # chosen its batch widths.
    batched || _setup_calculators!(calculators, backend, nothing, kpts, qpts, el_k_save;
        nw, nmodes, rng_band = iband_min:iband_max, el_states_kq = el_kq_save, kqpts,
        sel_k, sel_kq, nchunks_threads, verbosity,
    )

    if verbosity > 0 && mpi_isroot()
        @info "Number of k points = $(kpts.n)"
        @info "Number of k+q points = $(kqpts.n)"
        precompute_ph && @info "Number of q points = $(qpts.n)"
    end

    return (;
        kpts, qpts, kqpts,
        el_k_save, el_kq_save,
        ph_save, precompute_ph,
        el_k, el_kq, ph, el_quantities, ph_quantities,
        epstates, ep_ekpRs, epmat, ep_ekpR_obj,
        dyn_threads,
        epmat_R, epobj_ekpR_R, ep_ekpR_Rs,
        epmat_dev, backend,
        iband_min, iband_max,
        sel_k, sel_kq,
    )
end


function _loop_eph_over_k_and_kq(
        model       :: Model{FT},
        kpts, qpts, kqpts,
        el_k_save, el_kq_save,
        ph_save, precompute_ph,
        epstates, ep_ekpRs, epmat, ep_ekpR_obj,
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
    backend = CPUBackend()

    for ik in 1:nk
        if mod(ik, progress_print_step) == 0 && mpi_isroot()
            mpi_isroot() && @info "$(now()) ik = $ik / $nk"
            flush(stdout)
            flush(stderr)
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        for epstate in epstates.data
            epstate.el_k = el_k
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
        ctx = LoopContext(backend, SingleMode(), ik)
        foreach(c -> calculator_begin!(c, OuterIteration(), ctx), calculators)

        @threads for (id_chunk, ikqs) in enumerate(chunks(1:kqpts.n; n=nchunks_threads))
        # @time for (id_chunk, ikqs) in enumerate(collect(chunks(1:kqpts.n; n=nchunks_threads))[1:1])
            epstate = take!(epstates)
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

            _run_eph_over_k_and_kq_inner(model, epstate, ik, ep_ekpR, el_kq_save,
                xk, ph_save, dyn, kpts, qpts, kqpts, ikqs, precompute_ph, id_chunk,
                energy_conservation, screening_params, skip_eph, ctx;
                ep_ekpR_R, calculators,
            )

            put!(ep_ekpRs, ep_ekpR)
            put!(epstates, epstate)
            if ! precompute_ph
                put!(dyn_threads, dyn)
            end
            if covariant_derivative_of_g
                put!(ep_ekpR_Rs, ep_ekpR_R)
            end
        end # ikq chunk

        # Multithreading collect
        foreach(c -> calculator_end!(c, OuterIteration(), ctx), calculators)

    end # ik

    foreach(c -> postprocess_calculator!(c; qpts, symmetry), calculators)
end


function _run_eph_over_k_and_kq_inner(model :: Model{FT}, epstate, ik, ep_ekpR, el_kq_save,
        xk, ph_save, dyn, kpts, qpts, kqpts, ikqs, precompute_ph, id_chunk,
        energy_conservation, screening_params, skip_eph, ctx;
        ep_ekpR_R, calculators,
    ) where {FT}

    (; nw, nmodes) = model

    ϵs = zeros(Complex{FT}, model.nmodes)

    for ikq in ikqs
        xkq = kqpts.vectors[ikq]
        xq = xkq - xk
        xq = normalize_kpoint_coordinate(xq .+ 1/2) .- 1/2

        epstate.el_kq = el_kq_save[ikq]
        epstate.wtk = kpts.weights[ik]
        epstate.wtq = kqpts.weights[ikq]

        # Use precomputed data for the phonon state at q

        if precompute_ph
            # Use precomputed data for the phonon state at q
            iq = xk_to_ik_unsafe(xq, qpts)
            if iq === nothing
                throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
            end
            epstate.ph = ph_save[iq]
        else
            # Compute phonon state at q.
            iq = nothing
            set_eigen!(epstate.ph, dyn, model.mass, model.polar_phonon, xq)
            if ! skip_eph
                set_eph_dipole_coeff!(epstate.ph, model.polar_eph, xq)
            end
        end

        # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
        check_energy_conservation_all(epstate, kqpts.ngrid, model.recip_lattice, energy_conservation...) || continue

        epstate_set_mmat!(epstate)

        # Compute electron-phonon coupling
        if !skip_eph
            get_eph_kR_to_kq!(epstate, ep_ekpR, xq)

            if ep_ekpR_R !== nothing
                # This must be done before the long-range calculation

                get_fourier!(ep_ekpR_R.out, ep_ekpR_R, xq)
                dg_wan = Base.ReshapedArray(ep_ekpR_R.out, (nw, nw, nmodes, 3), ())
                dg = zeros(ComplexF64, (epstate.el_kq.nband, epstate.el_k.nband, nmodes, 3))

                # Apply electron gauge matrices (Wannier to eigenstate)
                tmp1 = zeros(ComplexF64, nw, nw)
                @views for idir in 1:3, imode in 1:nmodes
                    tmp1 .= dg_wan[:, :, imode, idir]
                    dg[:, :, imode, idir] .= no_offset_view(epstate.el_kq.u)' * tmp1 * no_offset_view(epstate.el_k.u)
                end

                # Apply phonon gauge matrix (Wannier to eigenstate)
                tmp2 = zeros(ComplexF64, size(dg, 1), size(dg, 3))
                @views for idir in 1:3
                    for iw in axes(dg, 2)
                        tmp2 .= dg[:, iw, :, idir]
                        dg[:, iw, :, idir] .= tmp2 * epstate.ph.u
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
                #     ξk  = no_offset_view(getindex.(epstate.el_k.rbar,  idir))
                #     ξkq = no_offset_view(getindex.(epstate.el_kq.rbar, idir))
                #     for imode in 1:nmodes
                #         g = no_offset_view(epstate.ep[:, :, imode])
                #         dg[:, :, imode, idir] .+= im .* (g * ξk .- ξkq * g)
                #     end
                # end

                # For debugging
                # if ik == ikq
                #     dk_dir = model.recip_lattice * Vec3(0, 0, 1)
                #     print("$(abs(dk_dir' * dg[1, 1, 6, :])), ")
                # end

                epstate_dg = OffsetArray(dg, epstate.el_kq.rng, epstate.el_k.rng, :, :)

            else
                epstate_dg = nothing

            end

            _apply_screening!(ϵs, calculators, model, xq, epstate, screening_params)
            epstate_compute_eph_dipole!(epstate, ϵs; model)
            epstate_set_g2!(epstate)
        end

        # TODO: Screening

        # Now, we are done with matrix elements. All data saved in epstate.

        # FIXME: Find out better way to pass epstate_dg (now a typed payload field)
        payload = EPData(epstate, ik, iq, ikq, xk, xq, id_chunk, epstate_dg)
        foreach(c -> run_calculator!(c, payload, ctx), calculators)

    end # ikq
end


# =============================================================================
#  Batched calculator loop (see README_GPU.md).
#
#  This mirrors `_loop_eph_over_k_and_kq` but moves the e-ph Wannier->Bloch interpolation
#  onto the backend using the batched drivers from `wannier_to_bloch_batched.jl`. The code here is
#  backend-generic: it only calls `to_device` and the generic batched drivers, so no CUDA
#  code lives in the base package (the device methods are provided by the CUDA extension). It
#  is a separate function from `_loop_eph_over_k_and_kq` because its control flow —
#  batched over k-batches and q-batches with device staging — differs from the per-(k,q) loop,
#  not because it holds any device-specific code.
#
#  Backend: nothing here is device-specific, so the loop runs on `GPUBackend` and `CPUBackend`
#  alike. On the CPU the k-batch loop is serial, `batched_gemm!` degrades to a `mul!` loop, and
#  `plan_batch` returns the requested cap verbatim (`free_bytes(::CPUBackend) == typemax(Int)`).
#  Naming: the `_dev` suffix on the locals below means "on `backend`", which is the host under
#  `CPUBackend()`.
#
#  Supported (all handled upstream in the shared `_setup`, so this loop itself is agnostic to them):
#    * energy windows (window_k / window_kq): both sides carry only their in-window bands (box
#      storage, see `BatchedElectronState`)
#    * IBZ outer-k symmetry
#    * outer-k MPI decomposition (mpi_comm_k)
#  Not supported — asserted off on the batched path (see the scope-assert block below):
#    * polar / long-range terms
#    * incommensurate k / k+q grids: requires precompute_ph (phonon states precomputed)
#    * covariant derivative of g
#    * screening
#    * el_kq_from_unfolding (directly-computed k+q only)
#    * energy_conservation other than (:None, 0.0)
#
#  Calculator brackets: one `calculator_begin!/end!(calc, ctx)` pair around every outer-k batch
#  (`ctx.batch`); a k's work is spread over the q-tiles, so it does not finish at a single point.
#
#  Buffer reuse: device buffers are allocated once before the k loop and reused for every
#  (k, q), so the loop itself allocates almost nothing.
#
#  Loop shape: `k-batch -> q-tile -> k -> q(device)`. The q-tile loop sits OUTSIDE the per-k loop
#  because the kR->kq Fourier phase is built from the fixed k+q list (the k+q convention, see
#  `get_eph_RR_to_kR_batched!`) and is therefore the same for every k of the batch: one built tile
#  is reused `nk_batch` times.

# ---- per-(k, q-tile) `iq` index build ------------------------------------------------------------
#
# Fill `iqs[1:nq]` with the index into `qpts` of `x_{k+q} - x_k` for outer k `ik` and every k+q of
# the tile `qstart .+ (0:nq-1)`; the caller uploads it for the device phonon copy. A standalone
# function so the `ik`/`qstart`/`nq` it works on are typed arguments rather than `Core.Box`-wrapped
# captures of the three enclosing loop bodies.
function _fill_iqs!(iqs, qpts, xkqs_int, xks_int, ik, qstart, nq)
    ng1, ng2, ng3 = qpts.ngrid
    k1, k2, k3 = xks_int[1, ik], xks_int[2, ik], xks_int[3, ik]
    for j in 1:nq
        ikq = qstart + j - 1
        # Integer grid-coord hash for iq (no Float64 normalize/_hash_xk per pair). Both operands
        # are pre-reduced into 0:ng-1 at setup, so the fold is a compare-and-add.
        h1 = _wrap_reduced(xkqs_int[1, ikq] - k1, ng1)
        h2 = _wrap_reduced(xkqs_int[2, ikq] - k2, ng2)
        h3 = _wrap_reduced(xkqs_int[3, ikq] - k3, ng3)
        hash = (h1 * ng2 + h2) * ng3 + h3
        iq = _ik_from_hash(qpts, hash)
        # 0 = miss on either index.
        (iq < 1 || iq > qpts.n) && throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
        iqs[j] = iq
    end
    iqs
end

function _loop_eph_over_k_and_kq_batched(
        model       :: Model{FT},
        kpts, qpts, kqpts,
        el_k, el_kq, sel_k, sel_kq,
        ph, precompute_ph,
        epmat_dev, backend;
        el_quantities, ph_quantities,
        calculators = [],
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        nchunks_threads = nthreads(),
        progress_print_step = 20,
        nq_batch_max::Union{Int, Nothing} = nothing,
        nk_outer_batch_max::Int = 256,
        symmetry = nothing,
        verbosity::Int = 1,
    ) where {FT}

    (; nw, nmodes) = model
    nk = kpts.n
    nkq = kqpts.n

    # ----- scope asserts (minimal batched step) -----
    precompute_ph || throw(ArgumentError(
        "the batched path requires commensurate k / k+q grids so phonon states are precomputed (precompute_ph)."))
    (!model.polar_phonon.use && !model.polar_eph.use) || throw(ArgumentError(
        "the batched path does not support polar / long-range terms. Use the per-point path " *
        "(backend = CPUBackend(), batched = false)."))
    energy_conservation === (:None, 0.0) || throw(ArgumentError(
        "the batched path supports only energy_conservation = (:None, 0.0)."))
    screening_params === nothing || throw(ArgumentError(
        "the batched path does not support screening_params."))
    # symmetry (IBZ outer k) is allowed: the reduction is done in the shared setup and this loop is
    # symmetry-agnostic — `symmetry` is only passed through to `postprocess_calculator!`. The
    # dispatcher gates the unsupported el_kq_from_unfolding = true case.

    # Default (nq_batch_max === nothing): size the q-tile to the free device memory (below), which
    # for small nw/nmodes lands on all k+q in a single tile. Fewer, larger kR->kq / calculator
    # kernels — the GPU e-ph path is launch-bound for small nw/nmodes, so one big tile is ~1.8×
    # faster than the old 1024 default at 16³, and one tile also means one kR->kq phase build for
    # the whole outer-k batch. Passing an Int caps the tile harder; the memory-adaptive cap (§7)
    # then takes the smaller of the two so a large-nw run cannot OOM.
    nq_batch_user = nq_batch_max   # nothing = size to memory (capped at nkq); Int = hard upper cap
    nk_batch_max = min(nk_outer_batch_max, nk)

    # Every calculator computes on the device-resident blocks; there is no per-point fallback.
    isempty(calculators) && throw(ArgumentError("the batched path requires at least one calculator."))

    # ----- window projection -----
    # Both electron sides are the containers' boxes: `nbandk_max` / `nbandkq_max` eigenvector
    # columns per point, starting at the point's first in-window band. Every per-(k, q) object (the
    # kR→kq GEMM, both gauge rotations, the calculators' work) shrinks by nw / nband_max, the dominant
    # ∝ nk·nq cost at narrow windows (e.g. TaAs ±0.1 eV: nbandk_max ≈ 4 of nw = 32). Box columns past
    # a point's window are undefined, and the calculators do not read them.
    nbandk_max = el_k.nband_max
    nbandkq_max = el_kq.nband_max

    # ----- device interpolators (allocated once) -----
    # `epmat_dev` (device e-ph object) was uploaded ONCE in the shared setup and threaded here through
    # `backend`; the loop reuses it rather than re-uploading. `backend` is carried in LoopContext below.
    itp_epmat = BatchedWannierInterpolator(epmat_dev; backend, batch_size = nk_batch_max)
    # g(k, R_ep) is born partial-width: under the k-side eigenvector-window projection only the
    # first nw·nbandk_max·nmodes rows carry data. It is consumed directly out of `ep_ekpR_all`
    # (below), so there is no child WannierObject and no second interpolator here.
    ndata_ekpR = nw * nbandk_max * nmodes
    nr_ep = length(model.epmat.irvec_next)

    # ----- memory-adaptive q-batch size (§7) -----
    # Every per-q staging buffer scales with the q-batch width, so cap it at what free device memory
    # allows (30% headroom for the batched drivers' recycled temporaries). The whole-run + per-k-batch
    # commitments allocated after this point are subtracted first; `epmat_dev` / `itp_epmat` are
    # already live, so `free_bytes` reflects them. All buffer byte accounting lives in
    # `_outer_k_staging_bytes` (shared with `estimate_device_memory`); `nq_batch_user`
    # (Int, or nkq when nothing) stays a hard cap.
    # The k, k+q and phonon stacks were built in the setup, so `free_bytes` already accounts for
    # them.
    per_point, committed = _outer_k_staging_bytes(; nw, nbandk_max, nbandkq_max, nmodes, nr_ep, nk,
        nkq, nk_stack = 0, nkq_stack = 0, nq_grid = 0, nk_batch_max, calculators,
        nr_epmat = epmat_dev.nr, FT)
    nq_batch_cap = nq_batch_user === nothing ? nkq : min(nq_batch_user, nkq)
    nq_batch_max = plan_batch(backend, per_point, committed, nq_batch_cap; what = "outer-k")
    if verbosity > 0 && mpi_isroot()
        @info "batched outer-k staging: committed = $(round(committed / 1e9, digits = 2)) GB, " *
              "$(round(per_point / 1e3, digits = 1)) kB/q; q-batch size = $nq_batch_max"
    end

    foreach(c -> setup_calculator!(c, backend, el_k, el_kq, ph; sel_k, sel_kq, nw, nmodes,
        nchunks_threads, n_outer_batch = nk_batch_max, n_inner_tile = nq_batch_max, verbosity),
        calculators)

    # ----- persistent workspace (allocated once, reused across all (k, q)) -----
    # All device staging is sized to the full batch and used as plain CuArrays (not
    # batch-sliced views), so the batched drivers' reshape/cuBLAS calls stay on dense arrays.
    # Every buffer below comes from `alloc(backend, ...)`, so the backend is the single authority on
    # where "device" is.

    # RR->kR over a batch of `nk_batch_max` outer-k at once: one batched kernel per batch instead of one
    # launch-bound single-k call per k. `ep_ekpR_all` holds g(k, R_ep) for the whole batch; the inner
    # kR->kq driver reads each k's slice `ep_ekpR_all[:, :, ik_ind]` directly.
    # The outer k-batch's states, copied from `el_k` per batch, with every quantity it holds.
    el_k_tile   = BatchedElectronState(backend, nw, nbandk_max, nk_batch_max, el_quantities; FT)
    ep_ekpR_all = alloc(backend, Complex{FT}, ndata_ekpR, nr_ep, nk_batch_max)
    ks_batch     = Vector{Vec3{FT}}(undef, nk_batch_max)

    # The q-tile's phonons, copied from `ph` by `iq` per (k, q-tile), with every quantity it holds.
    ph_tile  = BatchedPhononState(backend, nmodes, nq_batch_max, ph_quantities; FT)
    epkq_dev = alloc(backend, Complex{FT}, nbandkq_max, nbandk_max, nmodes, nq_batch_max)

    # In-place scratch for the per-k kR->kq driver (g / tmp), reused across all (k, q) so the
    # driver allocates nothing per call. Sized for the max batch width `nq_batch_max`; the driver
    # uses the first `nq_batch` columns for a partial final batch.
    kRkq_ws = KRtoKQWorkspace(epmat_dev.op_r, ndata_ekpR, nbandkq_max, nbandk_max, nmodes, nq_batch_max)

    # The k+q states (independent of the outer k) and the phonons are resident on the backend,
    # built in the setup; each q-tile reads a contiguous slice of the former and copies the latter
    # by `iq`. The k+q weights are uploaded once.
    wtkq_dev = to_device_copy(backend, collect(FT, kqpts.weights))

    # Grid coordinates as (3 × n) real device matrices, uploaded once. Both phase builds below read
    # them directly, so nothing on the phase path is staged on the host or copied H2D inside the loop.
    # Negated once here rather than conjugating the phase tile every batch: the convention needs
    # conj(exp(2πi R_p·x_k)) = exp(2πi R_p·(−x_k)), and the two are bitwise identical (FP negation is
    # exact, and `cispi` is exactly symmetric). This is the only consumer of the k coordinates.
    # Out-of-place on purpose: on `CPUBackend` `_kpoints_to_device_matrix` returns a view onto
    # `kpts.vectors`, so negating in place would corrupt the k-points.
    mxk_dev = _kpoints_to_device_matrix(backend, kpts) .* -1
    xkq_dev = _kpoints_to_device_matrix(backend, kqpts)

    # The two Fourier phase matrices of the k+q convention (see `get_eph_RR_to_kR_batched!`):
    #   P_mk[ip, k] = exp(2πi R_p · (−x_k))    — folded into g(k, R_ep) once per outer-k batch
    #   P_kq[ip, j] = exp(2πi R_p · x_{k+q_j}) — the kR->kq phase, INDEPENDENT of the outer k
    # so one built P_kq tile serves every k of the batch. That reuse factor is `nk_batch`, i.e.
    # `nk_outer_batch_max`: lowering that cap shrinks this saving proportionally.
    # Not a `TiledDeviceOutput`: that tiles the OUTPUT side over the outer-state axis and owns a
    # host mirror plus a per-batch D2H, whereas P_kq is an input-side, q-indexed,
    # write-once-read-many device buffer with no host side and no D2H at all.
    # FIXME: this driver still hand-assembles the Fourier layer's internals -- the R-vector
    # matrix here, the P_mk/P_kq buffers below, and the bare `build_fourier_phase!` calls.
    # Accepted deliberately ([D5] in plans/wannier_interp_reorg.md: the shortest form was a
    # renamed function, not a phase object that would own these), so read that item before
    # reopening it. The hoist itself is correct and must stay.
    irvecp_mat = _irvec_to_device_matrix(backend, model.epmat.irvec_next, FT)
    P_mk  = alloc(backend, Complex{FT}, nr_ep, nk_batch_max)
    P_kq = alloc(backend, Complex{FT}, nr_ep, nq_batch_max)
    # Defensive: only columns 1:nk_batch are rewritten per batch, so a partial final batch leaves the
    # tail columns holding whatever the previous batch wrote. Nothing reads them — the k loop runs
    # `1:nk_batch` — and 1 is the identity of the convention multiply, so the padded (never-read)
    # slice of ep_ekpR_all stays meaningful whether or not it has been written yet.
    fill!(P_mk, 1)

    # `iq` index staging for one (k, q-tile), on the host and on the backend.
    iqs_batch     = Vector{Int}(undef, nq_batch_max)
    iqs_batch_dev = alloc(backend, Int, nq_batch_max)

    # Integer grid-coord hash for iq, replacing the per-(k,q) float normalize + `_hash_xk`:
    #   hc_i = fold(xkqs_int[i,ikq] - xks_int[i,ik], ng_i),  hash = (hc1*ng2 + hc2)*ng3 + hc3,
    # reproducing `_hash_xk` bit-identically with no Float64 in the hot loop. Requires every k and
    # k+q to lie exactly on the q-grid (ngrid a multiple of both meshes) — guaranteed by precompute_ph,
    # asserted above. Only `iq` is needed: the q-VECTOR no longer enters the interpolation (the
    # kR->kq phase is built from x_{k+q}), so this loop copies phonon data by index only.
    # Both coordinate lists are reduced into `0:ng-1` here, once, so the per-pair fold is
    # `_wrap_reduced` (a compare-and-add) instead of `mod` (a runtime integer division). The q-grid
    # shift is subtracted on the k+q side only: q = x_{k+q} - x_k - shift, and folding it into one
    # operand keeps it out of the pair loop.
    shq = qpts.shift
    xkqs_int = Matrix{Int}(undef, 3, nkq)
    xks_int  = Matrix{Int}(undef, 3, nk)
    for ikq in 1:nkq
        xkqs_int[:, ikq] .= _grid_coords_reduced(kqpts.vectors[ikq], qpts.ngrid, shq)
    end
    for ik in 1:nk
        xks_int[:, ik] .= _grid_coords_reduced(kpts.vectors[ik], qpts.ngrid, zero(Vec3{FT}))
    end

    for kstart in 1:nk_batch_max:nk
        kend = min(kstart + nk_batch_max - 1, nk)
        iks_batch = kstart:kend
        nk_batch = length(iks_batch)

        # Gather U(k) (the k-side box) and the k list for this outer-k batch, padding the partial
        # tail with the last k so the batched RR->kR runs on dense `nk_batch_max`-sized arrays.
        # A full batch is a range, copied with no index upload.
        iks_padded = nk_batch == nk_batch_max ? iks_batch :
            [iks_batch; fill(kend, nk_batch_max - nk_batch)]
        copy_batched_electron_states!(el_k_tile, el_k, iks_padded)
        uks_dev = el_k_tile.u
        for (ik_ind, ik) in enumerate(iks_padded)
            ks_batch[ik_ind] = kpts.vectors[ik]
        end

        if mpi_isroot() && div(kend, progress_print_step) > div(kstart - 1, progress_print_step)
            @info "$(now()) ik = $kstart:$kend / $nk"
            flush(stdout); flush(stderr)
        end

        # One batched RR->kR over the whole batch: g(k, R_ep) for all k in the batch, stored in the
        # k+q convention (multiplied by P_mk, the phase at −x_k) so the kR->kq phase below is
        # k-independent.
        @views build_fourier_phase!(P_mk[:, 1:nk_batch], irvecp_mat, mxk_dev[:, iks_batch])
        get_eph_RR_to_kR_batched!(ep_ekpR_all, itp_epmat, ks_batch, uks_dev;
            additional_phase = P_mk)

        ctx = LoopContext(backend, OuterKLoop(), iks_batch, 1)
        foreach(c -> calculator_begin!(c, ctx), calculators)

        qstart = 1
        while qstart <= nkq
            qend = min(qstart + nq_batch_max - 1, nkq)
            nq_batch = qend - qstart + 1
            rng_q = 1:nq_batch   # this tile's columns within the nq_batch_max-sized device buffers

            # kR->kq phase for this q-tile, built ONCE and reused by every k of the outer-k batch —
            # the reason the q-tile loop sits outside the k loop. In the k+q convention it reads
            # x_{k+q} directly, so it is a contiguous slice of the fixed k+q list.
            @views build_fourier_phase!(P_kq[:, rng_q], irvecp_mat, xkq_dev[:, qstart:qend])
            # The k+q side of the tile: a contiguous slice of the resident container (no copy). Its
            # index list is the plain range of the tile, `isbits`, so it rides in the kernel launch
            # parameters instead of costing a device buffer and a global load per thread.
            el_kq_block = view_batched_electron_states(el_kq, qstart:qend)
            wtkq_block = view(wtkq_dev, qstart:qend)
            ep = view(epkq_dev, :, :, :, rng_q)

            for (ik_ind, ik) in enumerate(iks_batch)
                # This (k, tile)'s q indices, checked on the host by `_fill_iqs!` and copied once into
                # the persistent device buffer (5-arg contiguous copy), which `copy_batched_phonon_states!` takes
                # as it is. Everything below runs at width nq_batch via views into the
                # nq_batch_max-sized buffers, so there is no padded tail.
                _fill_iqs!(iqs_batch, qpts, xkqs_int, xks_int, ik, qstart, nq_batch)
                copyto!(iqs_batch_dev, 1, iqs_batch, 1, nq_batch)
                iqs = view(iqs_batch_dev, rng_q)
                copy_batched_phonon_states!(ph_tile, ph, iqs)
                ph_block = view_batched_phonon_states(ph_tile, rng_q)

                # One batched Wannier->Bloch over this tile's q: ep (nbandkq_max, nbandk_max, nmodes, q).
                get_eph_kR_to_kq_batched!(ep, view(ep_ekpR_all, :, :, ik_ind), view(P_kq, :, rng_q),
                    ph_block.u, el_kq_block.u; ws = kRkq_ws)

                block = EPBlock{OuterKLoop}(ep, nothing,
                    view_batched_electron_states(el_k_tile, ik_ind:ik_ind), el_kq_block, ph_block,
                    kpts.weights[ik], wtkq_block, kpts.vectors[ik],
                    view(qpts.vectors, view(iqs_batch, rng_q)), ik, qstart:qend, iqs)
                foreach(c -> run_calculator!(c, block, ctx), calculators)
            end # ik

            qstart = qend + 1
        end # q tile

        foreach(c -> calculator_end!(c, ctx), calculators)

        # Bound the host look-ahead to one k-batch: a device-resident calculator never D2H-syncs per k,
        # so without this the host can race across all batches, keeping every batch's RR->kR scratch +
        # per-k transients live in the memory pool at once. Draining at each batch boundary caps the
        # transient working set with negligible utilization cost. No-op on the CPU backend.
        synchronize(backend)
    end # k batch

    foreach(c -> postprocess_calculator!(c; qpts, symmetry), calculators)
end
