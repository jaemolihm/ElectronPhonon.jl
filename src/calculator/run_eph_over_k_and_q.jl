# TODO: Control dipole and quadrupole terms
# TODO: Implement MPI
# TODO: nband_max ?

using Dates: now
using Base.Threads: nthreads, threadid, @threads
using ChunkSplitters
using OffsetArrays: no_offset_view


"""
* `el_kq_from_unfolding`: If true, compute the electron states at k+q by computing the
    states at k+q in the irreducible BZ and unfolding them to the full BZ. This is useful to
    ensure gauge consistency between symmetry-equivalent k points.
    To enable this option, `kpts` and `qpts` must have same grid size.
"""
function run_eph_over_k_and_q(
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
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        symmetry = model.symmetry,
        nchunks_threads = nthreads(),  # Number of chunks for multithreading
        verbosity::Int = 1,
    ) where {FT}

    if model.epmat_outer_momentum != "el"
        throw(ArgumentError("model.epmat_outer_momentum must be el to use run_eph_over_k_and_q"))
    end
    screening_params === nothing || error(
        "screening_params is not supported: dielectric screening is currently disabled (ϵ ≡ 1). " *
        "Pass screening_params = nothing.")
    for calc in calculators
        if !supports(calc, OuterKLoop)
            throw(ArgumentError("$calc does not support the outer-k loop. Use run_eph_over_q_and_k instead."))
        end
        if !supports(calc, EPData)
            throw(ArgumentError("$calc does not declare support for the per-(k,q) host payload; " *
                "define supports(::$(typeof(calc)), ::Type{EPData}) = true."))
        end
    end

    mpi_comm_k === nothing || error("mpi_comm_k not implemented")
    mpi_comm_q === nothing || error("mpi_comm_q not implemented")

    # Handle el_kq_from_unfolding override before any use of precompute_el_kq
    if el_kq_from_unfolding && !precompute_el_kq
        println("el_kq_from_unfolding requires precompute_el_kq = true. Overwrite precompute_el_kq.")
        precompute_el_kq = true
    end

    setup = _setup_eph_over_k_and_q(model, kpts_input, qpts_input;
        mpi_comm_k, mpi_comm_q, fourier_mode, window_k, window_kq,
        el_kq_from_unfolding, precompute_el_kq, symmetry,
        calculators, nchunks_threads, verbosity,
    )

    _loop_eph_over_k_and_q(model,
        setup.kpts, setup.qpts, setup.kqpts,
        setup.el_k_save, setup.el_kq_save, setup.ph_save,
        setup.precompute_el_kq,
        setup.epdatas, setup.ep_ekpRs, setup.epmat, setup.ep_ekpR_obj,
        setup.ham_threads, setup.vel_threads, setup.pos_threads;
        calculators, skip_eph, window_kq,
        energy_conservation, screening_params,
        progress_print_step, nchunks_threads, symmetry,
    )

    (; setup.kpts, setup.qpts, setup.el_k_save, setup.ph_save)
end


# _setup_eph_over_k_and_q and _loop_eph_over_k_and_q are split from run_eph_over_k_and_q so that all
# variables captured by the @threads closure in _loop_eph_over_k_and_q are typed function
# arguments, avoiding Core.Box wrapping.
function _setup_eph_over_k_and_q(
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
        symmetry = nothing,
        calculators = [],
        nchunks_threads = nthreads(),
        verbosity::Int = 1,
    ) where {FT}

    (; nw, nmodes) = model

    # Generate k points and electron states at k (shared setup core)
    (; kpts, iband_min, iband_max, el_k_save, sel_k) = _setup_electron_k(
        model, kpts_input; window_k, mpi_comm_k, symmetry, fourier_mode, verbosity)
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
    qpts = maybe_time(verbosity) do
        filter_qpoints(qpts, kpts, nw, model.el_ham, window_kq; fourier_mode)
    end
    nq = qpts.n


    if el_kq_from_unfolding && (kpts.ngrid !== qpts.ngrid)
        throw(ArgumentError("To use el_kq_from_unfolding, kpts and qpts must have same grid size"))
    end


    ph_save = maybe_time(verbosity) do
        compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode)
    end


    # If precompute_el_kq, generate a Kpoint for k+q and compute electron states therein.
    # Otherwise, it is computed on the fly for each k and each q.
    if precompute_el_kq
        shift_kq = kpts.shift + qpts.shift
        sel_kq = maybe_time(verbosity) do
            filter_electron_states(qpts.ngrid, nw, model.el_ham, window_kq; shift=shift_kq, fourier_mode)
        end
        kqpts = GridKpoints(sel_kq.kpts)

        if el_kq_from_unfolding
            kqpts_irr, ik_to_ikirr_isym_kq = fold_kpoints(kqpts, symmetry)
        else
            kqpts_irr, ik_to_ikirr_isym_kq = nothing, nothing
        end
        el_kq_save = _compute_electron_states_kq(model, kqpts, kqpts_irr, ik_to_ikirr_isym_kq,
            symmetry, el_kq_from_unfolding, window_kq;
            quantities=["eigenvalue", "eigenvector", "velocity", "position"], fourier_mode)
    else
        kqpts = nothing
        el_kq_save = nothing
        sel_kq = nothing
    end


    # Maximum number of electron bands to decide the size of e-ph matrix buffer.
    # If k+q are not precomputed, we don't know the maximum number of bands, so use nw.
    nband_max = precompute_el_kq ? max(maximum(el.nband for el in el_k_save),
                                       maximum(el.nband for el in el_kq_save)) : nw


    epdatas = _make_epdatas_channel(FT, nw, nmodes, nband_max)

    # E-ph matrix in electron Bloch, phonon Wannier representation
    ep_ekpR_obj = get_next_wannier_object(model.epmat)
    epmat = get_interpolator(model.epmat; fourier_mode, threads = true)
    ep_ekpRs = get_interpolator_channel(ep_ekpR_obj; fourier_mode)


    if !precompute_el_kq
        ham_threads = get_interpolator_channel(model.el_ham; fourier_mode)
        vel_threads = if model.el_velocity_mode === :Direct
            get_interpolator_channel(model.el_vel; fourier_mode)
        else
            get_interpolator_channel(model.el_ham_R; fourier_mode)
        end
        pos_threads = get_interpolator_channel(model.el_pos; fourier_mode)
    else
        ham_threads = nothing
        vel_threads = nothing
        pos_threads = nothing
    end

    if verbosity > 0 && mpi_isroot()
        @info "Number of k points = $nk"
        precompute_el_kq && @info "Number of k+q points = $(kqpts.n)"
        @info "Number of q points = $nq"
    end

    # Backend: one resolution point. This driver has no GPU loop, so it is always the CPU backend;
    # it is carried in `LoopContext` and passed to `setup_calculator!` for interface uniformity.
    backend = CPUBackend()

    _setup_calculators!(calculators, kpts, qpts, el_k_save;
        nw, nmodes, rng_band = iband_min:iband_max, el_states_kq = el_kq_save, kqpts,
        sel_k, sel_kq, nchunks_threads, verbosity, backend,
    )

    return (;
        kpts, qpts, kqpts,
        el_k_save, el_kq_save, ph_save,
        precompute_el_kq, nband_max,
        epdatas, ep_ekpRs, epmat, ep_ekpR_obj,
        ham_threads, vel_threads, pos_threads,
        iband_min, iband_max,
    )
end


function _loop_eph_over_k_and_q(
        model       :: Model{FT},
        kpts, qpts, kqpts,
        el_k_save, el_kq_save, ph_save,
        precompute_el_kq,
        epdatas, ep_ekpRs, epmat, ep_ekpR_obj,
        ham_threads, vel_threads, pos_threads;
        calculators = [],
        skip_eph = false,
        window_kq = (-Inf, Inf),
        energy_conservation = (:None, 0.0),
        screening_params = nothing,
        progress_print_step = 20,
        nchunks_threads = nthreads(),
        symmetry = nothing,
    ) where {FT}

    nk = kpts.n
    nq = qpts.n
    backend = CPUBackend()

    for ik in 1:nk
        if mod(ik, progress_print_step) == 0 && mpi_isroot()
            mpi_isroot() && @info "$(now()) ik = $ik / $nk"
            flush(stdout)
            flush(stderr)
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        # Use precomputed data for the electron state at k
        for epdata in epdatas.data
            epdata.el_k = el_k
        end

        if !skip_eph
            get_eph_RR_to_kR!(ep_ekpR_obj, epmat, xk, no_offset_view(el_k.u))
        end

        # Multithreading setup
        ctx_k = LoopContext(backend, SingleMode(), ik)
        foreach(c -> calculator_begin!(c, OuterIteration(), ctx_k), calculators)

        @threads for (id_chunk, iqs) in enumerate(chunks(1:nq; n=nchunks_threads))
            epdata = take!(epdatas)
            ep_ekpR = take!(ep_ekpRs)

            if !precompute_el_kq
                ham = take!(ham_threads)
                vel = take!(vel_threads)
                pos = take!(pos_threads)
            end

            ϵs = zeros(Complex{FT}, model.nmodes)

            for iq in iqs
                xq = qpts.vectors[iq]
                xkq = xk + xq

                epdata.wtk = kpts.weights[ik]
                epdata.wtq = qpts.weights[iq]

                # Use precomputed data for the phonon state at q
                epdata.ph = ph_save[iq]

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
                    set_position!(epdata.el_kq, pos, xkq)
                    # TODO: full velocity
                end

                # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
                check_energy_conservation_all(epdata, qpts.ngrid, model.recip_lattice, energy_conservation...) || continue

                epdata_set_mmat!(epdata)

                # Compute electron-phonon coupling
                if !skip_eph
                    get_eph_kR_to_kq!(epdata, ep_ekpR, xq)
                    _apply_screening!(ϵs, calculators, model, xq, epdata, screening_params)
                    epdata_compute_eph_dipole!(epdata, ϵs; model)
                    epdata_set_g2!(epdata)
                end

                # TODO: Screening

                # Now, we are done with matrix elements. All data saved in epdata.

                payload = EPData(epdata, ik, iq, ikq, xk, xq, id_chunk, nothing)
                foreach(c -> run_calculator!(c, payload, ctx_k), calculators)

            end # iq

            put!(ep_ekpRs, ep_ekpR)
            put!(epdatas, epdata)
            if !precompute_el_kq
                put!(ham_threads, ham)
                put!(vel_threads, vel)
                put!(pos_threads, pos)
            end
        end # iq chunk

        # Multithreading collect
        foreach(c -> calculator_end!(c, OuterIteration(), ctx_k), calculators)

    end # ik

    postprocess_calculator!.(calculators; qpts, symmetry)
end
