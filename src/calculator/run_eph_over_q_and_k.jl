# TODO: Control dipole and quadrupole terms
# TODO: Implement MPI
# TODO: nband_max ?

using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads
using OffsetArrays: no_offset_view


"""
    run_eph_over_q_and_k(model, kpts, qpts; calculators, kwargs...)

* `el_kq_from_unfolding`: If true, compute the electron states at k+q by computing the
    states at k+q in the irreducible BZ and unfolding them to the full BZ. This is useful to
    ensure gauge consistency between symmetry-equivalent k points.
    To enable this option, `kpts` and `qpts` must have same grid size.
"""
function run_eph_over_q_and_k(
        model       :: Model{FT},
        kpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        qpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        ;
        calculators = [],
        mpi_comm_k = nothing,
        mpi_comm_q = nothing,
        fourier_mode = "gridopt",
        window_k  = (-Inf, Inf),
        window_kq = (-Inf, Inf),
        skip_eph = false,
        el_kq_from_unfolding = false,
        precompute_el_kq = el_kq_from_unfolding,
        use_symmetry = true,
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        keep_all_qpts = false,
        nchunks_threads = nthreads(),  # Number of chunks for multithreading
        eph_phonon_basis::Symbol = :eigenmode,  # :eigenmode or :cartesian
        verbosity::Int = 1,
        eph_buffers::Union{Nothing, EphOuterQLoopBuffers} = nothing,
        use_gpu = false,          # Run the inner-k e-ph interpolation + calculator on the GPU
        nk_batch_max = 2^15,      # GPU: max number of outer k points processed per batch
    ) where {FT}

    if model.epmat_outer_momentum != "ph"
        throw(ArgumentError("model.epmat_outer_momentum must be ph to use run_eph_over_q_and_k"))
    end
    for calc in calculators
        if !supports(calc, OuterQLoop)
            throw(ArgumentError("$calc does not support the outer-q loop. Use run_eph_over_k_and_q instead."))
        end
    end

    # Validate eph_phonon_basis compatibility with calculators
    for calc in calculators
        if eph_phonon_basis ∉ allowed_eph_phonon_basis(calc)
            throw(ArgumentError("Calculator $calc does not support eph_phonon_basis = :$eph_phonon_basis. " *
                                "Allowed: $(allowed_eph_phonon_basis(calc))"))
        end
    end

    # Validate eph_phonon_basis compatibility with dipole screening
    if eph_phonon_basis == :cartesian && screening_params !== nothing
        throw(ArgumentError("eph_phonon_basis = :cartesian is incompatible with dipole screening (screening_params). " *
                            "Dipole screening is mode-dependent and requires eigenmode basis."))
    end

    mpi_comm_k === nothing || error("mpi_comm_k not implemented")
    mpi_comm_q === nothing || error("mpi_comm_q not implemented")

    # Handle el_kq_from_unfolding override before any use of precompute_el_kq
    if el_kq_from_unfolding && !precompute_el_kq
        println("el_kq_from_unfolding requires precompute_el_kq = true. Overwrite precompute_el_kq.")
        precompute_el_kq = true
    end

    setup = _setup_eph_over_q_and_k(model, kpts_input, qpts_input;
        mpi_comm_k, mpi_comm_q, fourier_mode, window_k, window_kq,
        el_kq_from_unfolding, precompute_el_kq, use_symmetry,
        keep_all_qpts, eph_phonon_basis, calculators, nchunks_threads,
        verbosity, eph_buffers, use_gpu,
    )

    if use_gpu
        # GPU path: setup stays on the host (filter/states/setup_calculator!); the inner-k e-ph
        # interpolation, k+q eigensolve, and calculator reduction run on the device. Extra flags
        # must be off; `_loop_eph_over_q_and_k_gpu` asserts the rest (no polar/screening, full-band via
        # window masking, directly-computed k+q, energy_conservation = (:None, 0.0)).
        precompute_el_kq && throw(ArgumentError(
            "use_gpu does not support precompute_el_kq (k+q states are eigensolved on the device)."))
        _loop_eph_over_q_and_k_gpu(model,
            setup.kpts, setup.qpts,
            setup.el_k_save, setup.ph_save, setup.eph_buffers,
            setup.el_ham_dev, setup.backend;
            calculators, skip_eph, window_kq,
            energy_conservation, screening_params,
            progress_print_step, eph_phonon_basis, verbosity, nk_batch_max,
        )
    else
        _loop_eph_over_q_and_k(model,
            setup.kpts, setup.qpts, setup.kqpts,
            setup.el_k_save, setup.el_kq_save, setup.ph_save,
            setup.precompute_el_kq, setup.eph_buffers;
            calculators, skip_eph, window_kq,
            energy_conservation, screening_params,
            progress_print_step, nchunks_threads,
            eph_phonon_basis, verbosity,
        )
    end

    (; setup.kpts, setup.qpts, setup.el_k_save, setup.el_kq_save, setup.ph_save)
end


# _setup_eph_over_q_and_k and _loop_eph_over_q_and_k are split from run_eph_over_q_and_k so that all
# variables captured by the @threads closure in _loop_eph_over_q_and_k are typed function
# arguments, avoiding Core.Box wrapping.
function _setup_eph_over_q_and_k(
        model       :: Model{FT},
        kpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        qpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        ;
        mpi_comm_k = nothing,
        mpi_comm_q = nothing,
        fourier_mode = "gridopt",
        window_k  = (-Inf, Inf),
        window_kq = (-Inf, Inf),
        el_kq_from_unfolding = false,
        precompute_el_kq = el_kq_from_unfolding,
        use_symmetry = true,
        keep_all_qpts = false,
        eph_phonon_basis::Symbol = :eigenmode,
        calculators = [],
        nchunks_threads = nthreads(),
        verbosity::Int = 1,
        eph_buffers::Union{Nothing, EphOuterQLoopBuffers} = nothing,
        use_gpu = false,
    ) where {FT}

    (; nw, nmodes) = model

    symmetry = use_symmetry ? model.symmetry : nothing

    # Generate k points and electron states at k (shared setup core; use_gpu: batched device
    # eigensolve for the window test). The outer-q-batched GPU path consumes only e_full / u_full /
    # rng from el_k_save, so it skips the velocity/position interpolation+rotation (the dominant
    # setup cost after the eigensolve); all other paths keep the full quantity list.
    # TODO: calculators should declare whether they need velocity and/or position, instead of the
    #       driver hard-coding this for the outer-q-batched path.
    el_k_quantities = (use_gpu && !isempty(calculators) && all(c -> supports(c, ElPhDataOuterQBatched), calculators)) ?
        ["eigenvalue", "eigenvector"] : ["eigenvalue", "eigenvector", "velocity", "position"]
    (; kpts, iband_min, iband_max, nelec_below_window_k, el_k_save) = _setup_eph_common(
        model, kpts_input; window_k, mpi_comm_k, symmetry, fourier_mode, use_gpu, verbosity, el_k_quantities)
    nk = kpts.n

    # Generate q points
    if qpts_input isa ElectronPhonon.AbstractKpoints
        mpi_comm_q === nothing || error("qpts_input as Kpoints with mpi_comm_q not implemented")
        qpts = qpts_input
    elseif qpts_input isa NTuple{3,Int}
        qpts = kpoints_grid(qpts_input, mpi_comm_q)
    else
        error("type of qpts_input is wrong")
    end

    if !keep_all_qpts
        qpts = maybe_time(verbosity) do
            filter_qpoints(qpts, kpts, nw, model.el_ham, window_kq; fourier_mode)
        end
    end
    nq = qpts.n


    if el_kq_from_unfolding && (kpts.ngrid !== qpts.ngrid)
        throw(ArgumentError("To use el_kq_from_unfolding, kpts and qpts must have same grid size"))
    end


    ph_save = maybe_time(verbosity) do
        compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode, eph_phonon_basis)
    end


    # If precompute_el_kq, generate a Kpoint for k+q and compute electron states therein.
    # Otherwise, it is computed on the fly for each k and each q.
    if precompute_el_kq
        shift_kq = kpts.shift + qpts.shift
        kqpts, iband_min_kq, iband_max_kq, nelec_below_window_kq = maybe_time(verbosity) do
            filter_kpoints(qpts.ngrid, nw, model.el_ham, window_kq; shift=shift_kq, fourier_mode)
        end
        kqpts = GridKpoints(kqpts)

        if el_kq_from_unfolding
            kqpts_irr, ik_to_ikirr_isym_kq = fold_kpoints(kqpts, symmetry)
        else
            kqpts_irr, ik_to_ikirr_isym_kq = nothing, nothing
        end
        el_kq_save = _compute_el_kq_states(model, kqpts, kqpts_irr, ik_to_ikirr_isym_kq,
            symmetry, el_kq_from_unfolding, window_kq; fourier_mode)
    else
        kqpts = nothing
        nelec_below_window_kq = nothing
        el_kq_save = nothing
    end


    # Maximum number of electron bands to decide the size of e-ph matrix buffer.
    # If k+q are not precomputed, we don't know the maximum number of bands, so use nw.
    nband_max = precompute_el_kq ? max(maximum(el.nband for el in el_k_save),
                                       maximum(el.nband for el in el_kq_save)) : nw


    if eph_buffers === nothing
        eph_buffers = EphOuterQLoopBuffers(model;
            nchunks_threads, precompute_el_kq, fourier_mode, nband_max)
    end

    if verbosity > 0 && mpi_isroot()
        @info "Number of k points = $nk"
        precompute_el_kq && @info "Number of k+q points = $(kqpts.n)"
        @info "Number of q points = $nq"
    end


    # Backend: one resolution point. On the GPU path upload `model.el_ham` ONCE here (for the k+q
    # eigensolve in the loop) and wrap it as the backend prototype; the loop reuses this device object
    # rather than re-uploading. `backend` is carried in `LoopContext` and passed to `setup_calculator!`.
    el_ham_dev = use_gpu ? to_device(model.el_ham) : nothing
    backend = use_gpu ? GPUBackend(el_ham_dev.op_r) : CPUBackend()

    # Chemical potential is solved inside each calculator's `setup_calculator!` (via
    # `set_chemical_potential!`), not here. On the GPU path a calculator can run its ncarrier sums on
    # the device via `backend.proto` (the bisection sweeps all in-window states many times and
    # dominates the setup at dense grids); the generic `compute_ncarrier` broadcast+sum works for
    # every occ_type on the device, so no occ_type is special-cased.
    _setup_calculators!(calculators, kpts, qpts, el_k_save;
        nw, nmodes, rng_band = iband_min:iband_max, el_states_kq = el_kq_save, kqpts,
        nelec_below_window_k, nelec_below_window_kq, nchunks_threads, verbosity, backend,
    )

    return (;
        kpts, qpts, kqpts,
        el_k_save, el_kq_save, ph_save,
        precompute_el_kq, nband_max,
        eph_buffers, el_ham_dev, backend,
        iband_min, iband_max,
        nelec_below_window_k, nelec_below_window_kq,
    )
end


function _loop_eph_over_q_and_k(
        model       :: Model{FT},
        kpts, qpts, kqpts,
        el_k_save, el_kq_save, ph_save,
        precompute_el_kq,
        eph_buffers :: EphOuterQLoopBuffers{FT};
        calculators = [],
        skip_eph = false,
        window_kq = (-Inf, Inf),
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        nchunks_threads = nthreads(),
        eph_phonon_basis::Symbol = :eigenmode,
        verbosity::Int = 1,
    ) where {FT}

    (; epdatas, ep_eRpq_obj, ep_eRpqs, epmat, ham_threads, vel_threads) = eph_buffers
    nk = kpts.n
    nq = qpts.n
    backend = CPUBackend()

    for iq in 1:nq
        if verbosity > 0 && mod(iq, progress_print_step) == 0 && mpi_isroot()
            @info "iq = $iq"
            flush(stdout)
            flush(stderr)
        end
        xq = qpts.vectors[iq]
        ph = ph_save[iq]

        # Use precomputed data for the phonon state at q
        for epdata in epdatas.data
            epdata.ph = ph
        end

        if !skip_eph
            get_eph_RR_to_Rq!(ep_eRpq_obj, epmat, xq, ph, eph_phonon_basis)
        end

        # Multithreading setup
        ctx_q = LoopContext(backend, iq, 1:0, 0)
        foreach(c -> calculator_begin!(c, OuterIteration(), ctx_q), calculators)

        @threads for (id_chunk, iks) in enumerate(chunks(1:nk; n=nchunks_threads))
            epdata = take!(epdatas)
            ep_eRpq = take!(ep_eRpqs)

            if !precompute_el_kq
                ham = take!(ham_threads)
                vel = take!(vel_threads)
            end

            ϵs = zeros(Complex{FT}, model.nmodes)

            for ik in iks
                xk = kpts.vectors[ik]
                xkq = xk + xq

                epdata.wtk = kpts.weights[ik]
                epdata.wtq = qpts.weights[iq]

                # Use precomputed data for the electron state at k
                epdata.el_k = el_k_save[ik]

                if precompute_el_kq
                    # Use precomputed data for the electron state at k+q
                    ikq = xk_to_ik(xkq, kqpts)
                    ikq === nothing && continue
                    epdata.el_kq = el_kq_save[ikq]
                else
                    # Compute electron state at k+q.
                    ikq = nothing
                    set_eigen!(epdata.el_kq, ham, xkq)

                    # Set energy window, skip if no state is inside the window
                    set_window!(epdata.el_kq, window_kq)
                    length(epdata.el_kq.rng) == 0 && continue

                    set_velocity_diag!(epdata.el_kq, vel, xkq, model.el_velocity_mode)
                    # TODO: full velocity
                end

                # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
                check_energy_conservation_all(epdata, qpts.ngrid, model.recip_lattice, energy_conservation...) || continue

                epdata_set_mmat!(epdata)

                # Compute electron-phonon coupling
                if !skip_eph
                    get_eph_Rq_to_kq!(epdata, ep_eRpq, xk)
                    _apply_screening!(ϵs, calculators, model, xq, epdata, screening_params)
                    epdata_compute_eph_dipole!(epdata, ϵs; model)
                    epdata_set_g2!(epdata)
                end

                # TODO: Screening

                # Now, we are done with matrix elements. All data saved in epdata.

                payload = ElPhDataPoint(epdata, ik, iq, ikq, xk, xq, id_chunk, nothing)
                foreach(c -> run_calculator!(c, payload, ctx_q), calculators)

            end # ik

            put!(ep_eRpqs, ep_eRpq)
            put!(epdatas, epdata)
            if !precompute_el_kq
                put!(ham_threads, ham)
                put!(vel_threads, vel)
            end
        end # ik chunk

        # Multithreading collect
        foreach(c -> calculator_end!(c, OuterIteration(), ctx_q), calculators)

    end # iq

    postprocess_calculator!.(calculators; qpts, model.symmetry)
end


# =============================================================================
#  GPU calculator loop for the outer-q e-ph sweep.
#
#  Mirrors `_loop_eph_over_k_and_kq_gpu` (run_eph_over_k_and_kq.jl): the shared `_setup_eph_over_q_and_k`
#  runs on the host, and this loop moves the inner-k work — the e-ph Rq→kq interpolation, the k+q
#  eigensolve, and the calculator reduction — onto the device via the generic batched drivers
#  (`get_eph_Rq_to_kq_batched!`, `eigen_batched`, `get_fourier_batched!`) and the outer-q batched
#  calculator hook. The code is backend-generic: it calls only `to_device` and the batched drivers,
#  so no CUDA code lives here (the device methods live in the CUDA extension).
#
#  Per q: interpolate g(R_el, q) on the HOST (`get_eph_RR_to_Rq!`, nq is small), upload it,
#  then loop over the outer k points in batches. Each batch: batched k+q eigensolve (window-masked
#  by zeroing out-of-window eigenvector columns), batched Rq→kq e-ph interpolation, and one call to
#  `run_calculator!(calc, ::ElPhDataOuterQBatched, ctx)`. The per-q device accumulator is bracketed
#  by the `calculator_begin!/end!(…, OuterIteration(), ctx)` brackets, the same ones the CPU loop uses.
#
#  Scope (asserted below; the CPU path handles the rest): no screening,
#  energy_conservation = (:None, 0.0), skip_eph = false, and every calculator supports the
#  `ElPhDataOuterQBatched` payload. Windows are supported via eigenvector-column masking (out-of-window
#  states contribute exactly 0), so the batched full-band nw×nw shapes stay uniform. Polar models are
#  supported: the `polar_eph` dipole term is added on the device per batch (`add_eph_dipole_batched!`).
function _loop_eph_over_q_and_k_gpu(
        model       :: Model{FT},
        kpts, qpts,
        el_k_save, ph_save,
        eph_buffers :: EphOuterQLoopBuffers{FT},
        el_ham_dev, backend;
        calculators = [],
        skip_eph = false,
        window_kq = (-Inf, Inf),
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        eph_phonon_basis::Symbol = :eigenmode,
        verbosity::Int = 1,
        nk_batch_max = 2^15,
    ) where {FT}

    (; nw, nmodes) = model
    nk = kpts.n
    nq = qpts.n

    # ----- scope asserts -----
    skip_eph && throw(ArgumentError("use_gpu requires skip_eph = false."))
    energy_conservation === (:None, 0.0) || throw(ArgumentError(
        "use_gpu supports only energy_conservation = (:None, 0.0)."))
    # screening_params === nothing also guarantees ϵ ≡ 1 in the polar dipole term below.
    screening_params === nothing || throw(ArgumentError("use_gpu does not support screening_params."))
    (!isempty(calculators) && all(c -> supports(c, ElPhDataOuterQBatched), calculators)) || throw(ArgumentError(
        "use_gpu (outer-q) requires every calculator to support the ElPhDataOuterQBatched payload. " *
        "Use the CPU path otherwise."))

    # `el_ham_dev` (device Hamiltonian for the k+q eigensolve) was uploaded ONCE in the shared setup
    # and threaded here through `backend`; the loop reuses it rather than re-uploading.

    # ----- device clone of the electron-Wannier / phonon-Bloch e-ph object -----
    # `get_eph_RR_to_Rq!` runs on the HOST per q into `ep_eRpq_obj`; its op_r is then uploaded
    # into this device clone. Reusing the clone across q is correct because `get_fourier_batched!`
    # recomputes its phases and reads `parent.op_r` fresh on every call (it never consults `_id`).
    ep_eRpq_obj = eph_buffers.ep_eRpq_obj
    epmat = eph_buffers.epmat
    ep_eRpq_dev = to_device(ep_eRpq_obj)
    ndata_eRpq = nw^2 * nmodes
    @assert ep_eRpq_dev.ndata == ndata_eRpq

    # ----- k-side eigenvectors / energies / weights uploaded ONCE -----
    # `Uk` is zero-padded outside each k's in-window range `rng`, so the full nw×nw Rq→kq rotation
    # reproduces the CPU's windowed rotation with zeros outside.
    # TODO: this whole-grid stack (16·nw²·nk bytes) stays device-resident for the entire run and is
    #       NOT covered by the memory-adaptive batch sizing below — for large nw on a dense grid it,
    #       not the per-batch scratch, is the memory limit. Stream the k side per batch (upload only
    #       the batch's `Uk`, like the outer-k loop's per-batch `uks_host`) if it does not fit.
    Uk_all_dev  = alloc(backend, Complex{FT}, nw, nw, nk)
    ek_all_dev  = alloc(backend, FT, nw, nk)
    wtk_all_dev = alloc(backend, FT, nk)
    let Uk_host = zeros(Complex{FT}, nw, nw, nk), ek_host = zeros(FT, nw, nk), wtk_host = zeros(FT, nk)
        for ik in 1:nk
            el = el_k_save[ik]
            @views Uk_host[:, el.rng, ik] .= el.u_full[:, el.rng]
            ek_host[:, ik] .= el.e_full
            wtk_host[ik] = kpts.weights[ik]
        end
        copyto!(Uk_all_dev, Uk_host)
        copyto!(ek_all_dev, ek_host)
        copyto!(wtk_all_dev, wtk_host)
    end

    # ----- memory-adaptive k-batch size -----
    # Every per-batch device buffer scales with the batch size: the loop's own staging plus each
    # calculator's per-k device scratch (`eph_batched_bytes_per_point`). Cap the batch at what
    # the free device memory allows (30% headroom for the batched drivers' recycled temporaries);
    # `nk_batch_max` stays a hard cap (the only control on the CPU backend, where `free_bytes`
    # is unbounded).
    # TODO: make memory management more structured — the per-k byte estimate is hand-counted and
    #       omits the whole-grid k-side stack; a single helper that owns the device buffers and their
    #       size accounting (and the k-side streaming fallback) would be more robust.
    use_polar_eph = model.polar_eph.use

    bytes_per_k = 16 * nw^2 * (5 * nmodes + 8 + (use_polar_eph ? 1 : 0)) +
        24 * (length(model.el_ham.irvec) + length(ep_eRpq_obj.irvec)) +
        sum(Int[eph_batched_bytes_per_point(calc, ElPhDataOuterQBatched; nw, nmodes) for calc in calculators])
    free = free_bytes(backend)
    nk_batch_mem = free == typemax(Int) ? nk : max(1, (free ÷ 10 * 7) ÷ bytes_per_k)
    nk_batch_cap = min(Int(nk_batch_max), nk)
    nk_batch_max = min(nk_batch_cap, nk_batch_mem)
    if verbosity > 0 && mpi_isroot() && nk_batch_max < nk_batch_cap
        @info "GPU outer-q: memory-adaptive k-batch size = $nk_batch_max " *
              "($(round(bytes_per_k / 1e3, digits = 1)) kB/k, $(round(free / 1e9, digits = 1)) GB free)"
    end

    itp_el_ham = BatchedWannierInterpolator(el_ham_dev; batch_size = nk_batch_max)
    itp_ep_eRpq = BatchedWannierInterpolator(ep_eRpq_dev; batch_size = nk_batch_max)

    # ----- persistent per-batch device workspace (sized to nk_batch_max) -----
    ep_ws     = RqToKQWorkspace(ep_eRpq_dev.op_r, ndata_eRpq, nw, nw, nmodes, nk_batch_max)
    ep_batch  = alloc(backend, Complex{FT}, nw, nw, nmodes, nk_batch_max)
    Hkq_flat  = alloc(backend, Complex{FT}, nw * nw, nk_batch_max)
    Uk_batch  = alloc(backend, Complex{FT}, nw, nw, nk_batch_max)
    Ukq_batch = alloc(backend, Complex{FT}, nw, nw, nk_batch_max)
    ek_batch  = alloc(backend, FT, nw, nk_batch_max)
    wtk_batch = alloc(backend, FT, nk_batch_max)
    ks_batch  = Vector{Vec3{FT}}(undef, nk_batch_max)
    kqs_batch = Vector{Vec3{FT}}(undef, nk_batch_max)

    # ----- polar e-ph dipole (long-range) scratch -----
    # `ph.eph_dipole_coeff` is host-precomputed per q by the shared setup; `add_eph_dipole_batched!`
    # applies the term on the device.
    mmats_batch = use_polar_eph ? alloc(backend, Complex{FT}, nw, nw, nk_batch_max) : nothing
    coeffs_dev  = use_polar_eph ? alloc(backend, Complex{FT}, nmodes) : nothing

    wmin, wmax = window_kq

    for iq in 1:nq
        if verbosity > 0 && mod(iq, progress_print_step) == 0 && mpi_isroot()
            @info "iq = $iq"; flush(stdout); flush(stderr)
        end
        xq = qpts.vectors[iq]
        ph = ph_save[iq]

        # HOST: interpolate g(R_el, q) in the requested phonon basis, then upload to the device clone.
        get_eph_RR_to_Rq!(ep_eRpq_obj, epmat, xq, ph, eph_phonon_basis)
        copyto!(ep_eRpq_dev.op_r, ep_eRpq_obj.op_r)
        use_polar_eph && copyto!(coeffs_dev, ph.eph_dipole_coeff)

        # Per-q calculator begin: allocate (first q) + zero the device accumulator (OuterIteration
        # bracket, same as the CPU loop; ctx carries backend + n_batch_max for the device buffer).
        ctx_q = LoopContext(backend, iq, 1:0, nk_batch_max)
        foreach(c -> calculator_begin!(c, OuterIteration(), ctx_q), calculators)

        for kstart in 1:nk_batch_max:nk
            kend = min(kstart + nk_batch_max - 1, nk)
            iks_batch = kstart:kend
            nk_batch = length(iks_batch)

            # k / k+q lists (host), tail padded with the last valid vector so the batched Fourier /
            # eigensolve see finite data (their results in the tail are discarded via wtk = 0).
            for (ik_ind, ik) in enumerate(iks_batch)
                ks_batch[ik_ind]  = kpts.vectors[ik]
                kqs_batch[ik_ind] = ks_batch[ik_ind] + xq
            end
            for ik_ind in (nk_batch + 1):nk_batch_max
                ks_batch[ik_ind]  = ks_batch[nk_batch]
                kqs_batch[ik_ind] = kqs_batch[nk_batch]
            end

            # k-side data: device→device slice, tail padded. Unlike the outer-k loop (whose padded
            # duplicates are harmless because each state scatters to a unique in-window index), here
            # a padded k column WOULD double-count into the q-summed χ, so its integration weight is
            # zeroed — the calculator multiplies by `wtk`, making the padded columns contribute 0.
            @views copyto!(Uk_batch[:, :, 1:nk_batch], Uk_all_dev[:, :, iks_batch])
            @views copyto!(ek_batch[:, 1:nk_batch],    ek_all_dev[:, iks_batch])
            @views copyto!(wtk_batch[1:nk_batch],      wtk_all_dev[iks_batch])
            if nk_batch < nk_batch_max
                @views Uk_batch[:, :, (nk_batch + 1):nk_batch_max] .= Uk_batch[:, :, nk_batch:nk_batch]
                @views ek_batch[:, (nk_batch + 1):nk_batch_max]    .= ek_batch[:, nk_batch:nk_batch]
                @views wtk_batch[(nk_batch + 1):nk_batch_max]      .= 0
            end

            # k+q eigensolve on the device (batched). No gauge fixing needed: χ is gauge-invariant.
            # TODO: `eigen_batched` allocates (E, U) each batch; an in-place variant into ek/Ukq
            #       scratch would remove the per-batch allocation.
            get_fourier_batched!(Hkq_flat, itp_el_ham, kqs_batch)
            Ekq, Ukq = eigen_batched(reshape(Hkq_flat, nw, nw, nk_batch_max))   # (nw,·), (nw,nw,·)

            # k+q window mask: zero eigenvector COLUMNS m outside [wmin, wmax] (Ekq[m,k]). This
            # zeroes ep_kq[m,·] and every k+q-side matrix element for out-of-window m, so those
            # (m,n) pairs contribute exactly 0 — reproducing the CPU's `for m in el_kq.rng` loop.
            mask_kq = (Ekq .>= wmin) .& (Ekq .<= wmax)                    # (nw, ·) Bool
            Ukq_batch .= Ukq .* reshape(mask_kq, 1, nw, nk_batch_max)

            # Batched Rq→kq e-ph interpolation: ep_batch[m,n,ν,k] = Ukq(k)' * g(k) * Uk(k).
            get_eph_Rq_to_kq_batched!(ep_batch, itp_ep_eRpq, ks_batch, Uk_batch, Ukq_batch; ws = ep_ws)

            use_polar_eph && add_eph_dipole_batched!(ep_batch, coeffs_dev, Ukq_batch, Uk_batch, mmats_batch)

            payload = ElPhDataOuterQBatched(ep_batch, ek_batch, Ekq,
                Uk_batch, Ukq_batch, wtk_batch, ks_batch, iq)
            foreach(c -> run_calculator!(c, payload, ctx_q), calculators)
        end # k batch

        # Per-q calculator end: device→host scatter into the per-q output arrays (OuterIteration bracket).
        foreach(c -> calculator_end!(c, OuterIteration(), ctx_q), calculators)

        # Bound the host look-ahead to one q so per-q device scratch does not pile up in the pool.
        synchronize(backend)
    end # iq

    postprocess_calculator!.(calculators; qpts, model.symmetry)
end


# =============================================================================
# Deprecated driver names — forwarders, removed after one release. Explicit @warn (maxlog=1) because
# Base.@deprecate depwarns are invisible in ordinary script runs (Julia ≥ 1.5). Both drivers renamed
# to the run_eph_over_<outer>_and_<inner> scheme. Delete this whole block when the old names go.
function run_eph_outer_k(args...; kwargs...)
    @warn "run_eph_outer_k is deprecated; use run_eph_over_k_and_q (identical arguments)." maxlog=1
    run_eph_over_k_and_q(args...; kwargs...)
end
function run_eph_outer_q(args...; kwargs...)
    @warn "run_eph_outer_q is deprecated; use run_eph_over_q_and_k (identical arguments)." maxlog=1
    run_eph_over_q_and_k(args...; kwargs...)
end
