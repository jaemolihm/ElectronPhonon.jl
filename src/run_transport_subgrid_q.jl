
using Base.Threads
using HDF5

export run_transport_subgrid_q

# FIXME: Merge various compute_electron_phonon_bte_* routines.
# TODO: Save qpts to file

"""
For given `kpts` and `qpts`, subdivide q points with ``|q| < subgrid_q_max`` by `subgrid_scale`
and calculate e-ph scattering data.
The q point grid must be a multiple of the k point grid.
- `subgrid_q_max`: maximum |q_cart| for subdividing. In (2π / model.alat) units.
FIXME: |q_cart| < subgrid_q_max or all(abs.(q) < subgrid_q_max)? Crystal vs cartesian?
TODO:
- Cleanup input parameters nband
- Implement mpi_comm_k
- Implement mpi_comm_q
"""
function run_transport_subgrid_q(
        model::ModelEPW{FT},
        kpts::AbstractKpoints,
        qpts_original::AbstractKpoints,
        nband,
        subgrid_q_max,
        subgrid_scale;
        mpi_comm_k=nothing,
        mpi_comm_q=nothing,
        fourier_mode="gridopt",
        window_k=(-Inf,Inf),
        window_kq=(-Inf,Inf),
        folder,
        energy_conservation=(:None, 0.0),
    ) where FT
    if mpi_comm_k !== nothing
        error("mpi_comm_q not implemented")
    end
    if mpi_comm_q !== nothing
        error("mpi_comm_q not implemented")
    end
    if mod.(qpts_original.ngrid, kpts.ngrid) != (0, 0, 0)
        throw(ArgumentError("Input q grid must be an integer multiple of k grid."))
    end

    # Find list of q points to make subgrid
    do_subgrid = [norm(model.recip_lattice * xq) / (2π / model.alat) < subgrid_q_max for xq in qpts_original.vectors]
    iqs_to_subgrid = (1:qpts_original.n)[do_subgrid]
    qpts_to_subgrid = EPW.get_filtered_kpoints(qpts_original, do_subgrid)

    # Compute q point object for the subgrid
    qpts = EPW.kpoints_create_subgrid(qpts_to_subgrid, subgrid_scale)
    indmap = sortperm(qpts)
    sort!(qpts)
    iq_subgrid_to_grid = repeat(iqs_to_subgrid, inner=prod(subgrid_scale))[indmap]

    # Map k and q points to k+q points
    mpi_isroot() && println("Finding the list of k+q points")
    kqpts = add_two_kpoint_grids(kpts, qpts, +, qpts.ngrid)
    sort!(kqpts)

    btedata_prefix = joinpath(folder, "btedata_subgrid")

    # Since the number of subgrid q points are typically smaller than the number of k points,
    # it is better to use outer_q than to use outer_k.
    # To do so, one needs to use model with epmat_outer_momentum == "ph".
    compute_func = if model.epmat_outer_momentum == "ph"
        compute_electron_phonon_bte_data_outer_q
    else
        compute_electron_phonon_bte_data_outer_k
    end
    compute_func(model, btedata_prefix, window_k, window_kq, kpts, kqpts, qpts, nband, energy_conservation, mpi_comm_k, mpi_comm_q, fourier_mode)

    fid_btedata = h5open("$btedata_prefix.rank$(mpi_myrank(mpi_comm_k)).h5", "cw")
    fid_btedata["iq_subgrid_to_grid"] = iq_subgrid_to_grid
    close(fid_btedata)

    (kpts=kpts, qpts=qpts, kqpts=kqpts, nband=nband)
end


function compute_electron_phonon_bte_data_outer_q(model::ModelEPW{FT}, btedata_prefix, window_k, window_kq,
    kpts, kqpts, qpts, nband, energy_conservation,
    mpi_comm_k, mpi_comm_q, fourier_mode) where FT

    if model.epmat_outer_momentum != "ph"
        throw(ArgumentError("model.epmat_outer_momentum must be ph"))
    end

    nw = model.nw
    nmodes = model.nmodes
    nk = kpts.n
    nq = qpts.n
    nkq = kqpts.n

    mpi_isroot() && println("Calculating electron and phonon states")
    # TODO: parallelize this part
    @timing "hdf init" begin
        # Open HDF5 file for writing BTEdata
        fid_btedata = h5open("$btedata_prefix.rank$(mpi_myrank(mpi_comm_k)).h5", "w")
        #     # Write some attributes to file
        #     g = create_group(fid_btedata, "electron")
        #     write_attribute(fid_btedata["electron"], "nk", nk)
        #     write_attribute(fid_btedata["electron"], "nbandk_max", nband)
        #     fid_btedata["electron/weights"] = kpts.weights
        #     write_attribute(fid_btedata["electron"], "nelec_below_window", nelec_below_window)

        # Calculate initial (k) and final (k+q) electron states, write to HDF5 file
        mpi_isroot() && println("Calculating electron states at k")
        el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_k; fourier_mode)
        el_k_boltzmann, imap_el_k = electron_states_to_BTStates(el_k_save, kpts)
        g = create_group(fid_btedata, "initialstate_electron")
        dump_BTData(g, el_k_boltzmann)

        mpi_isroot() && println("Calculating electron states at k+q")
        el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_kq; fourier_mode)
        el_kq_boltzmann, imap_el_kq = electron_states_to_BTStates(el_kq_save, kqpts)
        g = create_group(fid_btedata, "finalstate_electron")
        dump_BTData(g, el_kq_boltzmann)

        # Write phonon states to HDF5 file
        mpi_isroot() && println("Calculating phonon states")
        ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode)
        ph_boltzmann, imap_ph = phonon_states_to_BTStates(ph_save, qpts)
        g = create_group(fid_btedata, "phonon")
        dump_BTData(g, ph_boltzmann)
    end

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epdatas = [ElPhData{FT}(nw, nmodes, nband)]
    Threads.resize_nthreads!(epdatas)
    epobj_eRpq = WannierObject(model.epmat.irvec_next, zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    # Setup for collecting scattering processes
    @timing "bt init" begin
        max_nscat = nk * nmodes * nband^2 * 2
        bt_scat = ElPhScatteringData{FT}(max_nscat)
    end

    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k   points = $nk")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k+q points = $nkq")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of q   points = $nq")
    flush(stdout)
    flush(stderr)

    nscat_tot = 0
    for iq in 1:nq
        if mod(iq, 100) == 0
            println("iq = $iq")
            # mpi_isroot() && println("iq = $iq")
            flush(stdout)
            flush(stderr)
        end
        xq = qpts.vectors[iq]
        ph = ph_save[iq]

        for epdata in epdatas
            epdata.ph = ph
        end

        get_eph_RR_to_Rq!(epobj_eRpq, model.epmat, xq, ph.u, fourier_mode)

        bt_nscat = 0

        # Threads.@threads :static for ikq in 1:nkq
        for ik in 1:nk
            tid = Threads.threadid()
            epdata = epdatas[tid]

            epdata.wtk = kpts.weights[ik]
            epdata.wtq = qpts.weights[iq]

            xk = kpts.vectors[ik]

            # Reusing k+q states: map xkq to ikq, the index of k+q point in the global list
            ikq = xk_to_ik(xk + xq, kqpts)

            # Copy saved electron and phonon states to epdata
            epdata.el_k = el_k_save[ik]
            epdata.el_kq = el_kq_save[ikq]

            el_k = epdata.el_k
            el_kq = epdata.el_kq
            ph = epdata.ph

            # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
            check_energy_conservation_all(epdata, qpts.ngrid, model.recip_lattice, energy_conservation...) || continue

            # Compute electron-phonon coupling
            get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk, fourier_mode)
            if any(abs.(xq) .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                model.polar_eph.use && epdata_compute_eph_dipole!(epdata)
            end
            epdata_set_g2!(epdata)

            @timing "bt_push" @inbounds for imode in 1:nmodes, jb in el_kq.rng, ib in el_k.rng, sign_ph in (-1, 1)
                # Save only if the scattering satisfies energy conservation
                check_energy_conservation(el_k, el_kq, ph, ib, jb, imode, sign_ph,
                    qpts.ngrid, model.recip_lattice, energy_conservation...) || continue

                bt_nscat += 1
                bt_scat.ind_el_i[bt_nscat] = imap_el_k[ib, ik]
                bt_scat.ind_el_f[bt_nscat] = imap_el_kq[jb, ikq]
                bt_scat.ind_ph[bt_nscat] = imap_ph[imode, iq]
                bt_scat.sign_ph[bt_nscat] = sign_ph
                bt_scat.mel[bt_nscat] = epdata.g2[jb, ib, imode]
            end
        end # ik

        @timing "bt_dump" begin
            g = create_group(fid_btedata, "scattering/iq$iq")
            dump_BTData(g, bt_scat, bt_nscat)
        end

        nscat_tot += bt_nscat
    end # iq
    close(fid_btedata)
    @info "nscat_tot = $nscat_tot"
    nothing
end

function compute_electron_phonon_bte_data_outer_k(model, btedata_prefix, window_k, window_kq, kpts,
    kqpts, qpts, nband, energy_conservation, mpi_comm_k, mpi_comm_q, fourier_mode)

    if model.epmat_outer_momentum != "el"
        error("model.epmat_outer_momentum must be el")
    end

    nw = model.nw
    nmodes = model.nmodes
    nk = kpts.n
    nq = qpts.n
    nkq = kqpts.n

    mpi_isroot() && println("Calculating electron and phonon states")
    # TODO: parallelize this part
    @timing "hdf init" begin
        # Open HDF5 file for writing BTEdata
        fid_btedata = h5open("$btedata_prefix.rank$(mpi_myrank(mpi_comm_k)).h5", "w")
        #     # Write some attributes to file
        #     g = create_group(fid_btedata, "electron")
        #     write_attribute(fid_btedata["electron"], "nk", nk)
        #     write_attribute(fid_btedata["electron"], "nbandk_max", nband)
        #     fid_btedata["electron/weights"] = kpts.weights
        #     write_attribute(fid_btedata["electron"], "nelec_below_window", nelec_below_window)

        # Calculate initial (k) and final (k+q) electron states, write to HDF5 file
        mpi_isroot() && println("Calculating electron states at k")
        el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_k; fourier_mode)
        el_k_boltzmann, imap_el_k = electron_states_to_BTStates(el_k_save, kpts)
        g = create_group(fid_btedata, "initialstate_electron")
        dump_BTData(g, el_k_boltzmann)

        mpi_isroot() && println("Calculating electron states at k+q")
        el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_kq; fourier_mode)
        el_kq_boltzmann, imap_el_kq = electron_states_to_BTStates(el_kq_save, kqpts)
        g = create_group(fid_btedata, "finalstate_electron")
        dump_BTData(g, el_kq_boltzmann)

        # Write phonon states to HDF5 file
        mpi_isroot() && println("Calculating phonon states")
        ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode)
        ph_boltzmann, imap_ph = phonon_states_to_BTStates(ph_save, qpts)
        g = create_group(fid_btedata, "phonon")
        dump_BTData(g, ph_boltzmann)
    end

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epdatas = [ElPhData{Float64}(nw, nmodes, nband)]
    Threads.resize_nthreads!(epdatas)
    epobj_ekpR = WannierObject(model.epmat.irvec_next, zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    # Setup for collecting scattering processes
    @timing "bt init" begin
        max_nscat = nq * nmodes * nband^2 * 2
        bt_scat = ElPhScatteringData{Float64}(max_nscat)
    end

    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k   points = $nk")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k+q points = $nkq")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of q   points = $nq")
    flush(stdout)
    flush(stderr)

    nscat_tot = 0
    for ik in 1:nk
        if mod(ik, 100) == 0
            println("ik = $ik")
            # mpi_isroot() && println("isk = $ik")
            flush(stdout)
            flush(stderr)
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        for epdata in epdatas
            epdata.el_k = el_k
        end
        get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, el_k.u, fourier_mode)

        bt_nscat = 0

        # Threads.@threads :static for ikq in 1:nkq
        for iq in 1:nq
            tid = Threads.threadid()
            epdata = epdatas[tid]

            epdata.wtk = kpts.weights[ik]
            epdata.wtq = qpts.weights[iq]

            xq = qpts.vectors[iq]

            # Reusing k+q states: map xkq to ikq, the index of k+q point in the global list
            ikq = xk_to_ik(xk + xq, kqpts)

            # Copy saved electron and phonon states to epdata
            epdata.ph = ph_save[iq]
            epdata.el_kq = el_kq_save[ikq]

            el_k = epdata.el_k
            el_kq = epdata.el_kq
            ph = epdata.ph

            # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
            check_energy_conservation_all(epdata, qpts.ngrid, model.recip_lattice, energy_conservation...) || continue

            # Compute electron-phonon coupling
            get_eph_kR_to_kq!(epdata, epobj_ekpR, xq, fourier_mode)
            if any(abs.(xq) .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                model.polar_eph.use && epdata_compute_eph_dipole!(epdata)
            end
            epdata_set_g2!(epdata)

            @timing "bt_push" @inbounds for imode in 1:nmodes, jb in el_kq.rng, ib in el_k.rng, sign_ph in (-1, 1)
                # Save only if the scattering satisfies energy conservation
                check_energy_conservation(el_k, el_kq, ph, ib, jb, imode, sign_ph,
                    qpts.ngrid, model.recip_lattice, energy_conservation...) || continue

                bt_nscat += 1
                bt_scat.ind_el_i[bt_nscat] = imap_el_k[ib, ik]
                bt_scat.ind_el_f[bt_nscat] = imap_el_kq[jb, ikq]
                bt_scat.ind_ph[bt_nscat] = imap_ph[imode, iq]
                bt_scat.sign_ph[bt_nscat] = sign_ph
                bt_scat.mel[bt_nscat] = epdata.g2[jb, ib, imode]
            end
        end # iq

        @timing "bt_dump" begin
            g = create_group(fid_btedata, "scattering/ik$ik")
            dump_BTData(g, bt_scat, bt_nscat)
        end

        nscat_tot += bt_nscat
    end # ik
    close(fid_btedata)
    @info "nscat_tot = $nscat_tot"
    nothing
end


