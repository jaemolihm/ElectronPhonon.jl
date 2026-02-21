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
        verbosity,
    )

    _loop_eph_outer_q(model,
        setup.kpts, setup.qpts, setup.kqpts,
        setup.el_k_save, setup.el_kq_save, setup.ph_save,
        setup.precompute_el_kq,
        setup.epdatas, setup.ep_eRpqs, setup.epmat, setup.ep_eRpq_obj,
        setup.ham_threads, setup.vel_threads;
        calculators, skip_eph, window_kq,
        energy_conservation, screening_params,
        progress_print_step, nchunks_threads,
        eph_phonon_basis, verbosity,
    )

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
    ) where {FT}

    (; nw, nmodes) = model

    symmetry = use_symmetry ? model.symmetry : nothing

    # Generate k points
    if verbosity > 0
        @time kpts, iband_min, iband_max, nelec_below_window_k = filter_kpoints(
            kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode)
    else
        kpts, iband_min, iband_max, nelec_below_window_k = filter_kpoints(
            kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode)
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


    # Compute and save electron state at k
    if verbosity > 0
        @time el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity", "position"], window_k; fourier_mode)
        @time ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode, eph_phonon_basis)
    else
        el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity", "position"], window_k; fourier_mode)
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


    epdatas = Channel{ElPhData{FT}}(nthreads())
    foreach(1:nthreads()) do _
        put!(epdatas, ElPhData{FT}(nw, nmodes, nband_max))
    end

    # E-ph matrix in electron Bloch, phonon Wannier representation
    ep_eRpq_obj = get_next_wannier_object(model.epmat)
    ep_eRpqs = get_interpolator_channel(ep_eRpq_obj; fourier_mode)

    epmat = get_interpolator(model.epmat; fourier_mode, threads = true)


    if !precompute_el_kq
        ham_threads = get_interpolator_channel(model.el_ham; fourier_mode)
        vel_threads = if model.el_velocity_mode === :Direct
            get_interpolator_channel(model.el_vel; fourier_mode)
        else
            get_interpolator_channel(model.el_ham_R; fourier_mode)
        end
    else
        ham_threads = nothing
        vel_threads = nothing
    end

    if verbosity > 0 && mpi_isroot()
        @info "Number of k points = $nk"
        precompute_el_kq && @info "Number of k+q points = $(kqpts.n)"
        @info "Number of q points = $nq"
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
        epdatas, ep_eRpqs, epmat, ep_eRpq_obj,
        ham_threads, vel_threads,
        iband_min, iband_max,
        nelec_below_window_k, nelec_below_window_kq,
    )
end


function _loop_eph_outer_q(
        model       :: Model{FT},
        kpts, qpts, kqpts,
        el_k_save, el_kq_save, ph_save,
        precompute_el_kq,
        epdatas, ep_eRpqs, epmat, ep_eRpq_obj,
        ham_threads, vel_threads;
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
