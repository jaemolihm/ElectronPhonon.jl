# TODO: Control dipole and quadrupole terms
# TODO: Implement MPI
# TODO: nband_max ?

using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads
using ElectronPhonon: WannierObject, fold_kpoints, unfold_ElectronStates, check_energy_conservation_all, epsilon_lindhard
using OffsetArrays: no_offset_view


"""
    run_eph_outer_q(model, kpts, qpts, calculators; kwargs...)

* `el_kq_from_unfolding`: If true, compute the electron states at k+q by computing the
    states at k+q in the irreducible BZ and unfolding them to the full BZ. This is useful to
    ensure gauge consistency between symmetry-equivalent k points.
    To enable this option, `kpts` and `qpts` must have same grid size.
"""
function run_eph_outer_q(
        model       :: Model{FT},
        kpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        qpts_input  :: Union{NTuple{3,Int}, Kpoints, GridKpoints},
        calculators :: AbstractVector,
        ;
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
    ) where {FT}

    if model.epmat_outer_momentum != "ph"
        throw(ArgumentError("model.epmat_outer_momentum must be ph to use run_eph_outer_q"))
    end
    for calc in calculators
        if !allow_eph_outer_q(calc)
            throw(ArgumentError("$calc does not allow run_eph_outer_q. Use run_eph_outer_k instead."))
        end
    end

    mpi_comm_k === nothing || error("mpi_comm_k not implemented")
    mpi_comm_q === nothing || error("mpi_comm_q not implemented")

    (; nw, nmodes) = model

    nchunks_threads = 2 * nthreads()  # Number of chunks for multithreading

    symmetry = use_symmetry ? model.symmetry : nothing

    # Generate k points
    @time kpts, iband_min, iband_max, nelec_below_window_k = filter_kpoints(
        kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode)
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
    @time qpts = filter_qpoints(qpts, kpts, nw, model.el_ham, window_kq; fourier_mode)
    nq = qpts.n


    if el_kq_from_unfolding && (kpts.ngrid !== qpts.ngrid)
        throw(ArgumentError("To use el_kq_from_unfolding, kpts and qpts must have same grid size"))
    end
    if el_kq_from_unfolding && !precompute_el_kq
        println("el_kq_from_unfolding requires precompute_el_kq = true. Overwrite precompute_el_kq.")
        precompute_el_kq = true
    end


    # Compute and save electron state at k
    @time el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"], window_k; fourier_mode)
    @time ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode)


    # If precompute_el_kq, generate a Kpoint for k+q and compute electron states therein.
    # Otherwise, it is computed on the fly for each k and each q.
    if precompute_el_kq
        shift_kq = kpts.shift + qpts.shift
        @time kqpts, iband_min_kq, iband_max_kq, nelec_below_window_kq = filter_kpoints(
            qpts.ngrid, nw, model.el_ham, window_kq; shift=shift_kq, fourier_mode)
        kqpts = GridKpoints(kqpts)

        if el_kq_from_unfolding
            # To ensure gauge consistency between symmetry-equivalent k points, we explicitly compute
            # electron states only for k+q in the irreducible BZ and unfold them to the full BZ.
            kqpts_irr, ik_to_ikirr_isym_kq = fold_kpoints(kqpts, symmetry)
            el_kq_save_irr = compute_electron_states(model, kqpts_irr, ["eigenvalue", "eigenvector", "velocity"], window_kq; fourier_mode)

            el_kq_save = unfold_ElectronStates(model, el_kq_save_irr, kqpts_irr, kqpts, ik_to_ikirr_isym_kq, symmetry; fourier_mode)

            # el_kq_save_irr is not used anymore.
            el_kq_save_irr !== el_kq_save && empty!(el_kq_save_irr)
        else
            el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity"], window_kq; fourier_mode)
        end
    else
        kqpts = nothing
        nelec_below_window_kq = nothing
        el_kq_save = nothing
    end


    # Maximum number of electron bands to decide the size of e-ph matrix buffer.
    if precompute_el_kq
        nband_max = max(maximum(el.nband for el in el_k_save),
                        maximum(el.nband for el in el_kq_save))
    else
        # Since k+q are not computed, we don't know the maximum number of bands.
        # Hence, set it to the largest possible value, nw.
        nband_max = nw
    end


    epdatas = Channel{ElPhData{FT}}(nthreads())
    foreach(1:nthreads()) do _
        put!(epdatas, ElPhData{FT}(nw, nmodes, nband_max))
    end

    # E-ph matrix in electron Bloch, phonon Wannier representation
    ep_eRpq_obj = WannierObject(model.epmat.irvec_next,
                                zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))
    ep_eRpqs = get_interpolator_channel(ep_eRpq_obj; fourier_mode)

    epmat = get_interpolator(model.epmat; fourier_mode, threads = true)


    if !precompute_el_kq
        ham_threads = get_interpolator_channel(model.el_ham; fourier_mode)
        vel_threads = if model.el_velocity_mode === :Direct
            get_interpolator_channel(model.el_vel; fourier_mode)
        else
            get_interpolator_channel(model.el_ham_R; fourier_mode)
        end
    end

    if mpi_isroot()
        @info "Number of k points = $nk"
        precompute_el_kq && @info "Number of k+q points = $(kqpts.n)"
        @info "Number of q points = $nq"
    end


    setup_calculator!.(calculators, Ref(kpts), Ref(qpts), Ref(el_k_save);
        rng_band = iband_min:iband_max,
        el_states_kq = el_kq_save, kqpts, nelec_below_window_k, nelec_below_window_kq,
        nchunks_threads
    )


    for iq in 1:nq
        if mod(iq, progress_print_step) == 0 && mpi_isroot()
            mpi_isroot() && @info "iq = $iq"
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
            get_eph_RR_to_Rq!(ep_eRpq_obj, epmat, xq, ph.u)
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
                    epdata_compute_eph_dipole!(epdata, ϵs)
                    epdata_set_g2!(epdata)
                end

                # TODO: Screening

                # Now, we are done with matrix elements. All data saved in epdata.

                run_calculator!.(calculators, Ref(epdata), Ref(ik), Ref(iq), Ref(ikq); xq, xk, id_chunk)

            end # iq

            put!(ep_eRpqs, ep_eRpq)
            put!(epdatas, epdata)
            if !precompute_el_kq
                put!(ham_threads, ham)
                put!(vel_threads, vel)
            end
        end # iq chunk

        # Multithreading collect
        postprocess_calculator_inner!.(calculators; iq)

    end # ik

    postprocess_calculator!.(calculators; qpts, model.symmetry)

    (; kpts, qpts, el_k_save, ph_save)
end
