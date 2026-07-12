# TODO: Control dipole and quadrupole terms
# TODO: Implement MPI
# TODO: nband_max ?

using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads
using OffsetArrays: no_offset_view


"""
    run_eph_outer_q(model, kpts, qpts; calculators, kwargs...)

* `el_kq_from_unfolding`: If true, compute the electron states at k+q by computing the
    states at k+q in the irreducible BZ and unfolding them to the full BZ. This is useful to
    ensure gauge consistency between symmetry-equivalent k points.
    To enable this option, `kpts` and `qpts` must have same grid size.
"""
function run_eph_outer_q(
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
        k_chunk_size = 1 << 15,   # GPU: number of outer k points processed per batched chunk
    ) where {FT}

    if model.epmat_outer_momentum != "ph"
        throw(ArgumentError("model.epmat_outer_momentum must be ph to use run_eph_outer_q"))
    end
    for calc in calculators
        if !allow_eph_outer_q(calc)
            throw(ArgumentError("$calc does not allow run_eph_outer_q. Use run_eph_outer_k instead."))
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

    setup = _setup_eph_outer_q(model, kpts_input, qpts_input;
        mpi_comm_k, mpi_comm_q, fourier_mode, window_k, window_kq,
        el_kq_from_unfolding, precompute_el_kq, use_symmetry,
        keep_all_qpts, eph_phonon_basis, calculators, nchunks_threads,
        verbosity, eph_buffers, use_gpu,
    )

    if use_gpu
        # GPU path: setup stays on the host (filter/states/setup_calculator!); the inner-k e-ph
        # interpolation, k+q eigensolve, and calculator reduction run on the device. Extra flags
        # must be off; `_loop_eph_outer_q_gpu` asserts the rest (no polar/screening, full-band via
        # window masking, directly-computed k+q, energy_conservation = (:None, 0.0)).
        precompute_el_kq && throw(ArgumentError(
            "use_gpu does not support precompute_el_kq (k+q states are eigensolved on the device)."))
        _loop_eph_outer_q_gpu(model,
            setup.kpts, setup.qpts,
            setup.el_k_save, setup.ph_save, setup.eph_buffers;
            calculators, skip_eph, window_kq,
            energy_conservation, screening_params,
            progress_print_step, eph_phonon_basis, verbosity, k_chunk_size,
        )
    else
        _loop_eph_outer_q(model,
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


# _setup_eph_outer_q and _loop_eph_outer_q are split from run_eph_outer_q so that all
# variables captured by the @threads closure in _loop_eph_outer_q are typed function
# arguments, avoiding Core.Box wrapping.
function _setup_eph_outer_q(
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

    # Generate k points (use_gpu: batched device eigensolve for the window test)
    if verbosity > 0
        @time kpts, iband_min, iband_max, nelec_below_window_k = filter_kpoints(
            kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode, use_gpu)
    else
        kpts, iband_min, iband_max, nelec_below_window_k = filter_kpoints(
            kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode, use_gpu)
    end
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
        if verbosity > 0
            @time qpts = filter_qpoints(qpts, kpts, nw, model.el_ham, window_kq; fourier_mode)
        else
            qpts = filter_qpoints(qpts, kpts, nw, model.el_ham, window_kq; fourier_mode)
        end
    end
    nq = qpts.n


    if el_kq_from_unfolding && (kpts.ngrid !== qpts.ngrid)
        throw(ArgumentError("To use el_kq_from_unfolding, kpts and qpts must have same grid size"))
    end


    # Compute and save electron state at k.
    # GPU batched-q path: the loop consumes only e_full / u_full / rng from el_k_save, and
    # batched-q calculators are (per the run_calculator_batched_q! contract) given k-side states
    # with eigenvalue/eigenvector only — so skip the velocity/position interpolation+rotation, the
    # dominant setup cost after the eigensolve. All other paths keep the full quantity list.
    el_k_quantities = (use_gpu && !isempty(calculators) && all(allow_eph_batched_q, calculators)) ?
        ["eigenvalue", "eigenvector"] : ["eigenvalue", "eigenvector", "velocity", "position"]
    if verbosity > 0
        @time el_k_save = compute_electron_states(model, kpts, el_k_quantities, window_k; fourier_mode, use_gpu)
        @time ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode, eph_phonon_basis)
    else
        el_k_save = compute_electron_states(model, kpts, el_k_quantities, window_k; fourier_mode, use_gpu)
        ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode, eph_phonon_basis)
    end


    # If precompute_el_kq, generate a Kpoint for k+q and compute electron states therein.
    # Otherwise, it is computed on the fly for each k and each q.
    if precompute_el_kq
        shift_kq = kpts.shift + qpts.shift
        if verbosity > 0
            @time kqpts, iband_min_kq, iband_max_kq, nelec_below_window_kq = filter_kpoints(
                qpts.ngrid, nw, model.el_ham, window_kq; shift=shift_kq, fourier_mode)
        else
            kqpts, iband_min_kq, iband_max_kq, nelec_below_window_kq = filter_kpoints(
                qpts.ngrid, nw, model.el_ham, window_kq; shift=shift_kq, fourier_mode)
        end
        kqpts = GridKpoints(kqpts)

        if el_kq_from_unfolding
            # To ensure gauge consistency between symmetry-equivalent k points, we explicitly compute
            # electron states only for k+q in the irreducible BZ and unfold them to the full BZ.
            kqpts_irr, ik_to_ikirr_isym_kq = fold_kpoints(kqpts, symmetry)
            el_kq_save_irr = compute_electron_states(model, kqpts_irr, ["eigenvalue", "eigenvector", "velocity", "position"], window_kq; fourier_mode)

            el_kq_save = unfold_ElectronStates(model, el_kq_save_irr, kqpts_irr, kqpts, ik_to_ikirr_isym_kq, symmetry; fourier_mode)

            # el_kq_save_irr is not used anymore.
            el_kq_save_irr !== el_kq_save && empty!(el_kq_save_irr)
        else
            el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity", "position"], window_kq; fourier_mode)
        end
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


    # GPU: pre-solve the chemical potential on the device for calculators whose occ still needs
    # it. The μ bisection sweeps all in-window states ~60 times; at dense grids it dominates the
    # setup even with the threaded host reduction (6.5 s of the 7.6 s setup at nk=80 full window).
    # Upload the flat state energies/weights once and let the existing find_chemical_potential
    # run its ncarrier sums as device broadcast+sum (the generic compute_ncarrier method; no CUDA
    # code here). occ.μlist caches the result, so the set_chemical_potential! inside each
    # calculator's setup_calculator! below then no-ops — calculators are unchanged. Device-safe
    # subset only: :FermiDirac (exp-based kernel) metals; anything else (e.g. :MV needs
    # SpecialFunctions.erf, not kernel-safe; the :Semiconductor solve needs host band masks)
    # silently keeps the existing host path. The device sum order shifts μ at the ~1e-15 level.
    if use_gpu
        e_dev = w_dev = el_flat = nothing
        for calc in calculators
            hasproperty(calc, :occ) || continue
            occ = calc.occ
            occ isa ElectronOccupationParams || continue
            chemical_potential_is_computed(occ) && continue
            (occ.occ_type === :FermiDirac && occ.type === :Metal) || continue
            if el_flat === nothing
                el_flat, _ = electron_states_to_BTStates(el_k_save, kpts, nelec_below_window_k)
                proto = to_device(model.el_ham).op_r
                e_dev = similar(proto, FT, length(el_flat.e));        copyto!(e_dev, el_flat.e)
                w_dev = similar(proto, FT, length(el_flat.k_weight)); copyto!(w_dev, el_flat.k_weight)
            end
            ncarrier_target = @. (occ.nlist + occ.nelec) / occ.spin_degeneracy - el_flat.nstates_base
            for i in eachindex(occ.Tlist)
                occ.μlist[i] = find_chemical_potential(ncarrier_target[i], occ.Tlist[i], e_dev, w_dev;
                                                       occ.occ_type)
                mpi_isroot() && @info "n = $(occ.nlist[i]) , T = $(round(occ.Tlist[i]/unit_to_aru(:K), digits=1)) K " *
                    "($(occ.occ_type)) , μ = $(round(occ.μlist[i]/unit_to_aru(:eV), digits=4)) eV (device solve)"
            end
        end
    end

    setup_calculator!.(calculators, Ref(kpts), Ref(qpts), Ref(el_k_save);
        nw, nmodes, rng_band = iband_min:iband_max,
        el_states_kq = el_kq_save, kqpts, nelec_below_window_k, nelec_below_window_kq,
        nchunks_threads, verbosity,
    )

    return (;
        kpts, qpts, kqpts,
        el_k_save, el_kq_save, ph_save,
        precompute_el_kq, nband_max,
        eph_buffers,
        iband_min, iband_max,
        nelec_below_window_k, nelec_below_window_kq,
    )
end


function _loop_eph_outer_q(
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
    (; nmodes) = model
    nk = kpts.n
    nq = qpts.n

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
            u_ph_for_eph = (eph_phonon_basis == :eigenmode) ? ph.u : I(nmodes)
            get_eph_RR_to_Rq!(ep_eRpq_obj, epmat, xq, u_ph_for_eph)
        end

        # Multithreading setup
        setup_calculator_inner!.(calculators; iq)

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

                run_calculator!.(calculators, Ref(epdata), Ref(ik), Ref(iq), Ref(ikq); xq, xk, id_chunk)

            end # ik

            put!(ep_eRpqs, ep_eRpq)
            put!(epdatas, epdata)
            if !precompute_el_kq
                put!(ham_threads, ham)
                put!(vel_threads, vel)
            end
        end # ik chunk

        # Multithreading collect
        postprocess_calculator_inner!.(calculators; iq)

    end # iq

    postprocess_calculator!.(calculators; qpts, model.symmetry)
end


# =============================================================================
#  GPU calculator loop for the outer-q e-ph sweep.
#
#  Mirrors `_loop_eph_over_k_and_kq_gpu` (run_eph_over_k_and_kq.jl): the shared `_setup_eph_outer_q`
#  runs on the host, and this loop moves the inner-k work — the e-ph Rq→kq interpolation, the k+q
#  eigensolve, and the calculator reduction — onto the device via the generic batched drivers
#  (`get_eph_Rq_to_kq_batched!`, `eigen_batched`, `get_fourier_batched!`) and the batched-q
#  calculator hook. The code is backend-generic: it calls only `to_device` and the batched drivers,
#  so no CUDA code lives here (the device methods live in the CUDA extension).
#
#  Per q: interpolate g(R_el, q) on the HOST (`get_eph_RR_to_Rq!`, nq is small), upload it, then loop
#  over outer k in chunks. Each chunk: batched k+q eigensolve (window-masked by zeroing out-of-window
#  eigenvector columns), batched Rq→kq e-ph interpolation, and one call to `run_calculator_batched_q!`.
#
#  Scope (asserted below; the CPU path handles the rest): no screening,
#  energy_conservation = (:None, 0.0), skip_eph = false, and every calculator implements
#  `allow_eph_batched_q`. Windows are supported via eigenvector-column masking (out-of-window states
#  contribute exactly 0), so the batched full-band nw×nw shapes stay uniform. Polar models are
#  supported: `polar_phonon` only affects the HOST phonon setup (`ph_save` from the shared
#  `_setup_eph_outer_q`, consumed here exactly as by the CPU loop), and the `polar_eph` dipole term
#  is added on the device per chunk (see the `use_polar_eph` block below).
#
#  Window masking: the k side uses the host `el_k_save[ik].rng` (its eigenvectors are uploaded once,
#  zero-padded outside `rng`); the k+q side is masked here from the device eigenvalues `Wkq`. The
#  final partial k-chunk is padded to `kb` with duplicated (finite) data and its integration weight
#  set to 0, so the padded columns contribute 0 without any partial-chunk special-casing downstream.
function _loop_eph_outer_q_gpu(
        model       :: Model{FT},
        kpts, qpts,
        el_k_save, ph_save,
        eph_buffers :: EphOuterQLoopBuffers{FT};
        calculators = [],
        skip_eph = false,
        window_kq = (-Inf, Inf),
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        eph_phonon_basis::Symbol = :eigenmode,
        verbosity::Int = 1,
        k_chunk_size = 1 << 15,
    ) where {FT}

    (; nw, nmodes) = model
    nk = kpts.n
    nq = qpts.n

    # ----- scope asserts -----
    skip_eph && throw(ArgumentError("use_gpu requires skip_eph = false."))
    energy_conservation === (:None, 0.0) || throw(ArgumentError(
        "use_gpu supports only energy_conservation = (:None, 0.0)."))
    # screening_params === nothing also guarantees ϵs ≡ 1 in the polar dipole term below (the CPU
    # loop divides the dipole coefficient by ϵs; with no screening that division is a no-op).
    screening_params === nothing || throw(ArgumentError("use_gpu does not support screening_params."))
    (!isempty(calculators) && all(allow_eph_batched_q, calculators)) || throw(ArgumentError(
        "use_gpu (outer-q) requires every calculator to implement allow_eph_batched_q " *
        "(run_calculator_batched_q!). Use the CPU path otherwise."))

    # ----- device Hamiltonian for the k+q eigensolve (allocated once) -----
    el_ham_dev = to_device(model.el_ham)
    proto = el_ham_dev.op_r

    # ----- device clone of the electron-Wannier / phonon-Bloch e-ph object -----
    # `get_eph_RR_to_Rq!` runs on the HOST per q into `ep_eRpq_obj`; its op_r is then uploaded into
    # this device clone. The per-q `copyto!` into `op_r` is what makes reusing the clone (and its
    # interpolator) across q correct: `get_fourier_batched!` recomputes its phases and reads
    # `parent.op_r` fresh on every call (it never consults `_id` — only the non-batched gridopt
    # interpolators cache against it).
    ep_eRpq_obj = eph_buffers.ep_eRpq_obj
    epmat = eph_buffers.epmat
    ep_eRpq_dev = to_device(ep_eRpq_obj)
    ndata_eRpq = nw^2 * nmodes
    @assert ep_eRpq_dev.ndata == ndata_eRpq

    # ----- k-side eigenvectors / energies / weights uploaded ONCE -----
    # `uk` is zero-padded outside each k's in-window range `rng` (columns outside `rng` are 0), so
    # the full nw×nw Rq→kq rotation reproduces the CPU's windowed rotation with zeros outside.
    # Memory ceiling: this whole-grid stack stays device-resident for the entire run —
    # 16·nw²·nk bytes (nw=3, nk≈1.2M → ~0.2 GB; nw=12 → ~2.8 GB; nw=40 → ~30 GB). For large nw on
    # dense grids this upload, not the chunk scratch, is the limit; the fallback is to stream the
    # k side per chunk (upload only the chunk's `uk`, like the outer-k loop's per-tile `uks_host`)
    # — not implemented here.
    Uk_all_dev  = similar(proto, Complex{FT}, nw, nw, nk)
    ek_all_dev  = similar(proto, FT, nw, nk)
    wtk_all_dev = similar(proto, FT, nk)
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

    # ----- memory-adaptive k-chunk width -----
    # Every per-chunk device buffer scales linearly with the chunk width: the loop's own staging
    # (Rq→kq workspace, ep_chunk, H(k+q) + eigensolve, U/e/wt slices, the two interpolator caches
    # + phase buffers) plus each calculator's batched-q scratch, declared via
    # `eph_batched_q_bytes_per_k` (dominant for χ-like calculators: it scales with nω·nocc etc.).
    # Derive the width from the free device memory AFTER the fixed whole-run uploads above, keeping
    # 30% headroom for the batched drivers' internal temporaries (eigensolve copy, broadcast
    # temporaries) recycled by the device memory pool. `k_chunk_size` remains a hard cap (and the
    # only control on the CPU backend, where `device_free_bytes` is unbounded).
    use_polar_eph = model.polar_eph.use

    bytes_per_k_loop = 16 * nw^2 * (5 * nmodes + 8 + (use_polar_eph ? 1 : 0)) +
        24 * (length(model.el_ham.irvec) + length(ep_eRpq_obj.irvec))
    bytes_per_k = bytes_per_k_loop +
        sum(Int[eph_batched_q_bytes_per_k(calc; nw, nmodes) for calc in calculators])
    free = device_free_bytes(proto)
    kb_mem = free == typemax(Int) ? nk : max(1, (free ÷ 10 * 7) ÷ bytes_per_k)
    kb = min(Int(k_chunk_size), nk, kb_mem)
    if verbosity > 0 && mpi_isroot() && kb < min(Int(k_chunk_size), nk)
        @info "GPU outer-q: memory-adaptive k-chunk width kb = $kb " *
              "($(round(bytes_per_k / 1e3, digits = 1)) kB/k, $(round(free / 1e9, digits = 1)) GB free)"
    end

    el_ham_itp = BatchedWannierInterpolator(el_ham_dev; batch_size = kb)
    ep_eRpq_itp = BatchedWannierInterpolator(ep_eRpq_dev; batch_size = kb)

    # ----- persistent per-chunk device workspace (sized to kb) -----
    ep_ws     = RqToKQWorkspace(ep_eRpq_dev.op_r, ndata_eRpq, nw, nw, nmodes, kb)
    ep_chunk  = similar(proto, Complex{FT}, nw, nw, nmodes, kb)
    Hkq_flat  = similar(proto, Complex{FT}, nw * nw, kb)
    Uk_chunk  = similar(proto, Complex{FT}, nw, nw, kb)
    Ukq_chunk = similar(proto, Complex{FT}, nw, nw, kb)
    ek_chunk  = similar(proto, FT, nw, kb)
    wtk_chunk = similar(proto, FT, kb)
    xk_chunk  = Vector{Vec3{FT}}(undef, kb)
    xkq_chunk = Vector{Vec3{FT}}(undef, kb)

    # ----- polar e-ph dipole (long-range) buffers -----
    # CPU counterpart: `epdata_compute_eph_dipole!(epdata, ϵs; model)` with ϵs ≡ 1 (guaranteed by
    # the screening_params === nothing assert above), whose active branch adds
    # `(coeff[ν] / ϵs[ν]) · Σ_iw conj(ukq[iw,m]) uk[iw,n]` to ep[m,n,ν] for in-window (m,n), with
    # `coeff = ph.eph_dipole_coeff` host-precomputed per q by the shared setup in the requested
    # eph_phonon_basis. The quadrupole (`coeff_r`) block is commented out on the CPU — a no-op —
    # and is ported as a no-op here.
    if use_polar_eph
        mmat_chunk = similar(proto, Complex{FT}, nw, nw, kb)
        coeff_dev  = similar(proto, Complex{FT}, nmodes)
    end

    wmin, wmax = window_kq

    for iq in 1:nq
        if verbosity > 0 && mod(iq, progress_print_step) == 0 && mpi_isroot()
            @info "iq = $iq"; flush(stdout); flush(stderr)
        end
        xq = qpts.vectors[iq]
        ph = ph_save[iq]

        # HOST: interpolate g(R_el, q) in the requested phonon basis, then upload to the device clone.
        u_ph_for_eph = (eph_phonon_basis == :eigenmode) ? ph.u : Matrix{Complex{FT}}(I, nmodes, nmodes)
        get_eph_RR_to_Rq!(ep_eRpq_obj, epmat, xq, u_ph_for_eph)
        copyto!(ep_eRpq_dev.op_r, ep_eRpq_obj.op_r)
        use_polar_eph && copyto!(coeff_dev, ph.eph_dipole_coeff)

        # Per-q calculator begin: allocate (first q) + zero the device accumulator.
        for calc in calculators
            setup_calculator_batched_q!(calc; iq, proto, k_chunk_size = kb)
        end

        for kstart in 1:kb:nk
            kend = min(kstart + kb - 1, nk)
            nkc = kend - kstart + 1

            # k / k+q lists (host), tail padded with the last valid vector so the batched Fourier /
            # eigensolve see finite data (their results in the tail are discarded via wtk = 0).
            for (a, ik) in enumerate(kstart:kend)
                xk_chunk[a]  = kpts.vectors[ik]
                xkq_chunk[a] = xk_chunk[a] + xq
            end
            for a in (nkc + 1):kb
                xk_chunk[a]  = xk_chunk[nkc]
                xkq_chunk[a] = xkq_chunk[nkc]
            end

            # k-side data: device→device slice, tail padded (weights → 0 so padded columns vanish).
            @views copyto!(Uk_chunk[:, :, 1:nkc], Uk_all_dev[:, :, kstart:kend])
            @views copyto!(ek_chunk[:, 1:nkc],    ek_all_dev[:, kstart:kend])
            @views copyto!(wtk_chunk[1:nkc],      wtk_all_dev[kstart:kend])
            if nkc < kb
                @views Uk_chunk[:, :, (nkc + 1):kb] .= Uk_chunk[:, :, nkc:nkc]
                @views ek_chunk[:, (nkc + 1):kb]    .= ek_chunk[:, nkc:nkc]
                @views wtk_chunk[(nkc + 1):kb]      .= 0
            end

            # k+q eigensolve on the device (batched). No gauge fixing needed: χ is gauge-invariant.
            get_fourier_batched!(Hkq_flat, el_ham_itp, xkq_chunk)
            Wkq, Ukq_raw = eigen_batched(reshape(Hkq_flat, nw, nw, kb))   # (nw,kb), (nw,nw,kb)

            # k+q window mask: zero eigenvector COLUMNS m outside [wmin, wmax] (Wkq[m,k]). This
            # zeroes ep_kq[m,·] and every k+q-side matrix element for out-of-window m, so those
            # (m,n) pairs contribute exactly 0 — reproducing the CPU's `for m in el_kq.rng` loop.
            maskkq = (Wkq .>= wmin) .& (Wkq .<= wmax)                      # (nw, kb) Bool
            Ukq_chunk .= Ukq_raw .* reshape(maskkq, 1, nw, kb)

            # Batched Rq→kq e-ph interpolation: ep_chunk[m,n,ν,k] = ukq(k)' * g(k) * uk(k).
            get_eph_Rq_to_kq_batched!(ep_chunk, ep_eRpq_itp, xk_chunk, Uk_chunk, Ukq_chunk; ws = ep_ws)

            if use_polar_eph
                # Polar dipole term: ep[m,n,ν,k] += coeff[ν] · mmat[m,n,k] with the overlap
                # mmat[m,n,k] = Σ_iw conj(Ukq[iw,m,k]) Uk[iw,n,k]. The zero-padded eigenvector
                # columns make mmat vanish for out-of-window m or n, matching the CPU rng loops
                # (ϵs ≡ 1, so no division; see the buffer comment above).
                batched_gemm!('C', 'N', Ukq_chunk, Uk_chunk, mmat_chunk)
                ep_chunk .+= reshape(coeff_dev, 1, 1, nmodes) .* reshape(mmat_chunk, nw, nw, 1, kb)
            end

            for calc in calculators
                run_calculator_batched_q!(calc, ep_chunk, ek_chunk, Wkq,
                    Uk_chunk, Ukq_chunk, wtk_chunk, xk_chunk, iq)
            end
        end # k chunk

        # Per-q calculator end: device→host scatter into the per-q output arrays.
        for calc in calculators
            flush_calculator_batched_q!(calc; iq)
        end

        # Bound the host look-ahead to one q so per-q device scratch does not pile up in the pool.
        device_synchronize(proto)
    end # iq

    postprocess_calculator!.(calculators; qpts, model.symmetry)
end
