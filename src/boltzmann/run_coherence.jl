using JLD2

export run_transport_coherence

# TODO: Merge with run_transport

function run_transport_coherence(
        model::ModelEPW,
        kgrid::NTuple{3,Int},
        qgrid::NTuple{3,Int};
        mpi_comm_k=nothing,
        mpi_comm_q=nothing,
        fourier_mode="gridopt",
        window_k=(-Inf,Inf),
        window_kq=(-Inf,Inf),
        folder,
        energy_conservation=(:None, 0.0),
        use_irr_k=false,
        shift_q=(0, 0, 0),
        average_degeneracy=false,
    )
    FT = Float64
    mpi_comm_k !== nothing && error("mpi_comm_k not implemented")
    mpi_comm_q !== nothing && error("mpi_comm_q not implemented")
    kgrid != qgrid && error("kgrid and qgrid must be the same (otherwise not implemented")
    model.epmat_outer_momentum != "el" && error("model.epmat_outer_momentum must be el")
    use_irr_k && error("use_irr_k not implemented")
    shift_q != (0, 0, 0) && error("shift_q not implemented")
    all(window_k .≈ window_kq) || error("window_k and window_kq must be the same (otherwise not implemented")

    # mod.(qgrid, kgrid) == (0, 0, 0) || error("q grid must be an integer multiple of k grid.")

    nw = model.nw

    @timing "setup kgrid" begin
        # Generate k points
        mpi_isroot() && println("Setting k-point grid")
        symmetry = use_irr_k ? model.symmetry : nothing
        kpts, iband_min_k, iband_max_k, nstates_base_k = filter_kpoints(kgrid, nw,
            model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode)

        # # Generate k+q points
        # mpi_isroot() && println("Setting k+q-point grid")
        # shift_kq = shift_q ./ qgrid
        # kqpts, iband_min_kq, iband_max_kq, nstates_base_kq = filter_kpoints(qgrid, nw,
        #     model.el_ham, window_kq, mpi_comm_k, shift=shift_kq; fourier_mode)
        # if mpi_comm_k !== nothing
        #     # k+q points are not distributed over mpi_comm_k in the remaining part.
        #     kqpts = mpi_allgather(kqpts, mpi_comm_k)
        # end

        # TODO: Currently, for coherence transport, kpts and kqpts should be identical.
        # The reason is that the off-diagonal matrix elements are hard to unfold or interpolate.
        # In the future, it may be implemented.
        kqpts = kpts
        iband_min_kq = iband_min_k
        iband_max_kq = iband_max_k
        nstates_base_kq = nstates_base_k
    end

    iband_min = min(iband_min_k, iband_min_kq)
    iband_max = max(iband_max_k, iband_max_kq)

    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1

    qpts = EPW.add_two_kpoint_grids(kqpts, kpts, -, kqpts.ngrid)

    # Move xq inside [-0.5, 0.5]^3. This doesn't change the Fourier transform but
    # makes the long-range part more robust.
    sort!(shift_center!(qpts, (0, 0, 0)))

    btedata_prefix = joinpath(folder, "btedata_coherence")
    compute_electron_phonon_bte_data_coherence(model, btedata_prefix, window_k, window_kq,
        kpts, kqpts, qpts, nband, nband_ignore, nstates_base_k, nstates_base_kq, energy_conservation,
        average_degeneracy, mpi_comm_k, mpi_comm_q, fourier_mode)

    (;nband, nband_ignore, kpts, qpts, kqpts)
end

function compute_electron_phonon_bte_data_coherence(model, btedata_prefix, window_k, window_kq, kpts,
        kqpts, qpts, nband, nband_ignore, nstates_base_k, nstates_base_kq, energy_conservation,
        average_degeneracy, mpi_comm_k, mpi_comm_q, fourier_mode)
    FT = Float64

    nw = model.nw
    nmodes = model.nmodes
    nk = kpts.n
    nq = qpts.n
    nkq = kqpts.n

    mpi_isroot() && println("Calculating electron and phonon states")
    g = nothing
    # TODO: parallelize this part
    @timing "hdf init" begin
        # Open HDF5 file for writing BTEdata
        fid_btedata = jldopen("$btedata_prefix.rank$(mpi_myrank(mpi_comm_k)).jld2", "w")

        # Calculate initial (k) and final (k+q) electron states, write to HDF5 file
        mpi_isroot() && println("Calculating electron states at k")
        el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"], window_k, nband, nband_ignore, "gridopt")
        # for (el, xk) in zip(el_k_save, kpts.vectors)
        #     set_gauge_to_diagonalize_velocity_matrix!(el, xk, 1, model)
        # end
        el_k_boltzmann, _ = electron_states_to_QMEStates(el_k_save, kpts, nstates_base_k)
        fid_btedata["initialstate_electron"] = el_k_boltzmann

        # NOTE: Currently, for coherence transport, the set of electron states at k and k+q
        # are chosen to be identical to ensure consistent gauge
        # mpi_isroot() && println("Calculating electron states at k+q")
        el_kq_save = el_k_save
        el_kq_boltzmann = el_k_boltzmann
        # el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_kq, nband, nband_ignore, "gridopt")
        # el_kq_boltzmann, imap_el_kq = electron_states_to_BTStates(el_kq_save, kqpts, nstates_base_kq)
        fid_btedata["finalstate_electron"] = el_kq_boltzmann

        # Write phonon states to HDF5 file
        mpi_isroot() && println("Calculating phonon states")
        ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"], "gridopt")
        ph_boltzmann, _ = phonon_states_to_BTStates(ph_save, qpts)
        fid_btedata["phonon"] = ph_boltzmann
    end

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epdatas = [ElPhData{Float64}(nw, nmodes, nband, nband_ignore)]
    Threads.resize_nthreads!(epdatas)
    epobj_ekpR = WannierObject(model.epmat.irvec_next, zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    # Setup for collecting scattering processes
    max_nscat = nkq * nmodes * nband^2
    bt_mel = zeros(Complex{FT}, max_nscat)
    bt_econv_p = falses(max_nscat)
    bt_econv_m = falses(max_nscat)
    bt_ib = zeros(Int16, max_nscat) # For band indices, use Int16 assuming they do not exceed 32767
    bt_jb = zeros(Int16, max_nscat)
    bt_imode = zeros(Int16, max_nscat)
    bt_ik = zeros(Int, max_nscat)
    bt_ikq = zeros(Int, max_nscat)

    # Setup for collecting scattering processes
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k   points = $nk")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k+q points = $nkq")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of q   points = $nq")
    flush(stdout)
    flush(stderr)

    nscat_tot = 0
    for ik in 1:nk
        if mod(ik, 100) == 0
            println("ik = $ik")
            # mpi_isroot() && println("ik = $ik")
            flush(stdout)
            flush(stderr)
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        for epdata in epdatas
            copyto!(epdata.el_k, el_k)
        end

        get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, get_u(el_k), fourier_mode)

        bt_nscat = 0

        # Threads.@threads :static for ikq in 1:nkq
        for ikq in 1:nkq
            tid = Threads.threadid()
            epdata = epdatas[tid]

            epdata.wtk = kpts.weights[ik]
            epdata.wtq = kqpts.weights[ikq]

            xkq = kqpts.vectors[ikq]

            # Find xq in qpts. Since xq can be shifted by a lattice vector, take xq from qpts.vectors
            iq = xk_to_ik(xkq - xk, qpts)
            xq = qpts.vectors[iq]

            # Copy saved electron and phonon states to epdata
            copyto!(epdata.ph, ph_save[iq])
            copyto!(epdata.el_kq, el_kq_save[ikq])

            el_kq = epdata.el_kq
            ph = epdata.ph

            # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
            check_energy_conservation_all(epdata, kqpts.ngrid, model.recip_lattice, energy_conservation...) || continue

            # Compute electron-phonon coupling
            get_eph_kR_to_kq!(epdata, epobj_ekpR, xq, fourier_mode)
            if any(abs.(xq) .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                model.polar_eph.use && epdata_compute_eph_dipole!(epdata)
            end
            epdata_set_g2!(epdata)

            # Average g2 over degenerate electron bands
            if average_degeneracy
                epdata_g2_degenerate_average!(epdata)
            end

            @timing "bt_push" @inbounds for imode in 1:nmodes, jb in el_kq.rng, ib in el_k.rng
                # Save only if the scattering satisfies energy conservation
                econv_p, econv_m = (check_energy_conservation(el_k, el_kq, ph, ib, jb, imode, sign_ph,
                kqpts.ngrid, model.recip_lattice, energy_conservation...) for sign_ph in (1, -1))
                if econv_p || econv_m
                    bt_nscat += 1
                    ω = epdata.ph.e[imode]
                    bt_mel[bt_nscat] = epdata.ep[jb, ib, imode] / sqrt(2ω)
                    bt_econv_p[bt_nscat] = econv_p
                    bt_econv_m[bt_nscat] = econv_m
                    bt_ib[bt_nscat] = ib
                    bt_jb[bt_nscat] = jb
                    bt_imode[bt_nscat] = imode
                    bt_ik[bt_nscat] = ik
                    bt_ikq[bt_nscat] = ikq
                end
                # TODO: Save bt_econv_p and bt_econv_m also in BTScattering
            end
        end # ikq

        @timing "bt_dump" begin
            fid_btedata["scattering/ik$ik/mel"] = bt_mel[1:bt_nscat]
            fid_btedata["scattering/ik$ik/econv_p"] = bt_econv_p[1:bt_nscat]
            fid_btedata["scattering/ik$ik/econv_m"] = bt_econv_m[1:bt_nscat]
            fid_btedata["scattering/ik$ik/ib"] = bt_ib[1:bt_nscat]
            fid_btedata["scattering/ik$ik/jb"] = bt_jb[1:bt_nscat]
            fid_btedata["scattering/ik$ik/imode"] = bt_imode[1:bt_nscat]
            fid_btedata["scattering/ik$ik/ik"] = bt_ik[1:bt_nscat]
            fid_btedata["scattering/ik$ik/ikq"] = bt_ikq[1:bt_nscat]
        end

        nscat_tot += bt_nscat
    end # ik

    close(fid_btedata)
    @info "nscat_tot = $nscat_tot"
    nothing
end

