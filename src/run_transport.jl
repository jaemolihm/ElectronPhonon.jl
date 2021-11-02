
using Base.Threads
using HDF5

"""
- `energy_conservation = (mode::Symbol, param::Real)`: Method to determine energy conservation. Only the scatterings that follow the energy conservation are calculated and saved.
    - `mode = :None`: Do not use energy conservation.
    - `mode = :Fixed`: Use fixed tolerence `param` for energy conservation. (Useful for fixed smearing)
    - `mode = :Linear`: Use band velocity to extrapolate energy and determine energy conservation. `param` is the maximum curvature of the band. (Useful for tetrahedron integration and adaptive smearing).
- `shift_q`: shift of the q-point grid in units of the grid size.
- `run_for_qme`: If true, calculate matrix elements for quantum master equation. If false, calculate
    for semiclassical Boltzmann equation(default: false)
- `qme_offdiag_cutoff`: Maximum interband energy difference to include the off-diagonal part. Used only if `run_for_qme`==true.
"""
function run_transport(
        model::ModelEPW,
        k_input::Union{NTuple{3,Int}, AbstractKpoints},
        q_input::Union{NTuple{3,Int}, AbstractKpoints};
        mpi_comm_k=nothing,
        mpi_comm_q=nothing,
        fourier_mode="gridopt",
        window_k=(-Inf,Inf),
        window_kq=(-Inf,Inf),
        folder,
        energy_conservation=(:None, 0.0),
        use_irr_k=true,
        shift_q=(0, 0, 0),
        average_degeneracy=false,
        run_for_qme=false,
        qme_offdiag_cutoff=EPW.electron_degen_cutoff,
        symmetry = nothing,
        kwargs...
    )
    FT = Float64
    """
    The q point grid must be a multiple of the k point grid. If so, the k+q points lie on
    the same grid as the q points.
    """

    """
    Things to implement
    - mpi_comm_k
    - mpi_comm_q
    - k_input to be a Kpoints object
    - q_input to be a Kpoints object
    - Additional sampling around q=0
    """
    mpi_comm_q !== nothing && error("mpi_comm_q not implemented")
    q_input isa Kpoints && error("q_input isa Kpoints not implemented")
    model.epmat_outer_momentum != "el" && error("model.epmat_outer_momentum must be el")

    kgrid = k_input isa AbstractKpoints ? k_input.ngrid : k_input
    qgrid = q_input isa AbstractKpoints ? q_input.ngrid : q_input
    mod.(qgrid, kgrid) == (0, 0, 0) || error("q grid must be an integer multiple of k grid.")

    if run_for_qme
        k_input isa AbstractKpoints && error("for run_for_qme==true, k_input must be a NTuple")
        q_input isa AbstractKpoints && error("for run_for_qme==true, q_input must be a NTuple")
        mpi_comm_k !== nothing && error("for run_for_qme==true, mpi_comm_k not implemented")
        kgrid != qgrid && error("for run_for_qme==true, kgrid and qgrid must be the same (otherwise not implemented)")
        use_irr_k && model.el_sym === nothing && error("for run_for_qme=true and use_irr_k=true, model.el_sym must be set. (Pass load_symmetry_operators=true to load_model).")
        shift_q != (0, 0, 0) && error("for run_for_qme==true, shift_q not implemented")
        all(window_k .≈ window_kq) || error("for run_for_qme==true, window_k and window_kq must be the same (otherwise not implemented")
    end

    if use_irr_k && symmetry === nothing
        println("use_irr_k = true, but symmetry is not passed. Set symmetry to default (maximal symmetry).")
        # For QME, one should use model.el_sym.symmetry, not model.symmetry because the symmetry
        # gauge matrix elements needs to be calculated.
        symmetry = run_for_qme ? model.el_sym.symmetry : model.symmetry
    end

    nw = model.nw

    @timing "setup kgrid" begin
        # Generate k points
        mpi_isroot() && println("Setting k-point grid")
        symmetry_k = use_irr_k ? symmetry : nothing
        kpts, iband_min_k, iband_max_k, nstates_base_k = filter_kpoints(k_input, nw,
            model.el_ham, window_k, mpi_comm_k, symmetry=symmetry_k; fourier_mode)

        # Generate k+q points
        mpi_isroot() && println("Setting k+q-point grid")
        # If k_input is a GridKpoint, set shift for kqpts so that qpts includes the Gamma point.
        shift_k = k_input isa GridKpoints ? Vec3{FT}(k_input.shift) : Vec3{FT}(0, 0, 0)
        shift_kq = shift_k .+ shift_q ./ qgrid
        kqpts, iband_min_kq, iband_max_kq, nstates_base_kq = filter_kpoints(qgrid, nw, model.el_ham, window_kq, mpi_comm_k, shift=shift_kq; fourier_mode)
        if mpi_comm_k !== nothing
            # k+q points are not distributed over mpi_comm_k in the remaining part.
            kqpts = mpi_allgather(kqpts, mpi_comm_k)
        end
    end

    iband_min = min(iband_min_k, iband_min_kq)
    iband_max = max(iband_max_k, iband_max_kq)

    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1

    qpts = add_two_kpoint_grids(kqpts, kpts, -, kqpts.ngrid)

    # Move xq inside [-0.5, 0.5]^3. This doesn't change the Fourier transform but
    # makes the long-range part more robust.
    sort!(shift_center!(qpts, (0, 0, 0)))

    if run_for_qme
        btedata_prefix = joinpath(folder, "btedata_coherence")
        compute_electron_phonon_bte_data_coherence(model, btedata_prefix, window_k, window_kq,
            GridKpoints(kpts), GridKpoints(kqpts), qpts, nband, nband_ignore, nstates_base_k, nstates_base_kq, energy_conservation,
            average_degeneracy, symmetry, mpi_comm_k, mpi_comm_q, fourier_mode, qme_offdiag_cutoff; kwargs...)
    else
        btedata_prefix = joinpath(folder, "btedata")
        compute_electron_phonon_bte_data(model, btedata_prefix, window_k, window_kq,
            kpts, kqpts, qpts, nband, nband_ignore, nstates_base_k, nstates_base_kq, energy_conservation,
            average_degeneracy, mpi_comm_k, mpi_comm_q, fourier_mode; kwargs...)
    end

    (;nband, nband_ignore, kpts, qpts, kqpts)
end

function compute_electron_phonon_bte_data(model, btedata_prefix, window_k, window_kq, kpts,
    kqpts, qpts, nband, nband_ignore, nstates_base_k, nstates_base_kq, energy_conservation,
    average_degeneracy, mpi_comm_k, mpi_comm_q, fourier_mode; kwargs...)

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
        fid_btedata = h5open("$btedata_prefix.rank$(mpi_myrank(mpi_comm_k)).h5", "w")
        #     # Write some attributes to file
        #     g = create_group(fid_btedata, "electron")
        #     write_attribute(fid_btedata["electron"], "nk", nk)
        #     write_attribute(fid_btedata["electron"], "nbandk_max", nband)
        #     fid_btedata["electron/weights"] = kpts.weights
        #     write_attribute(fid_btedata["electron"], "nelec_below_window", nelec_below_window)

        # Calculate initial (k) and final (k+q) electron states, write to HDF5 file
        mpi_isroot() && println("Calculating electron states at k")
        el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_k, nband, nband_ignore; fourier_mode)
        el_k_boltzmann, imap_el_k = electron_states_to_BTStates(el_k_save, kpts, nstates_base_k)
        g = create_group(fid_btedata, "initialstate_electron")
        dump_BTData(g, el_k_boltzmann)

        mpi_isroot() && println("Calculating electron states at k+q")
        el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_kq, nband, nband_ignore; fourier_mode)
        el_kq_boltzmann, imap_el_kq = electron_states_to_BTStates(el_kq_save, kqpts, nstates_base_kq)
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
    epdatas = [ElPhData{Float64}(nw, nmodes, nband, nband_ignore)]
    Threads.resize_nthreads!(epdatas)
    epobj_ekpR = WannierObject(model.epmat.irvec_next, zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    # Setup for collecting scattering processes
    @timing "bt init" begin
        max_nscat = nkq * nmodes * nband^2 * 2
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

            @timing "bt_push" @inbounds for imode in 1:nmodes, jb in el_kq.rng, ib in el_k.rng, sign_ph in (-1, 1)
                # Save only if the scattering satisfies energy conservation
                check_energy_conservation(el_k, el_kq, ph, ib, jb, imode, sign_ph,
                    kqpts.ngrid, model.recip_lattice, energy_conservation...) || continue

                bt_nscat += 1
                bt_scat.ind_el_i[bt_nscat] = imap_el_k[ib, ik]
                bt_scat.ind_el_f[bt_nscat] = imap_el_kq[jb, ikq]
                bt_scat.ind_ph[bt_nscat] = imap_ph[imode, iq]
                bt_scat.sign_ph[bt_nscat] = sign_ph
                bt_scat.mel[bt_nscat] = epdata.g2[jb, ib, imode]
            end
        end # ikq

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

# Check if given scattering satisfies energy conservation.
function check_energy_conservation(el_k, el_kq, ph, ib, jb, imode, sign_ph, ngrid, recip_lattice, econv_mode, econv_tol)
    if econv_mode == :None
        # Do not check energy conservation. Always return true.
        return true
    elseif econv_mode == :Fixed
        # Check if energy difference is within the range [-econv_tol, econv_tol]
        e0 = el_k.e[ib] - el_kq.e[jb] - sign_ph * ph.e[imode]
        return abs(e0) <= econv_tol
    elseif econv_mode == :Linear
        # Check if linearly-interpolated energy can be zero in the q-point grid box.
        # Use econv_tol as the maximum possible curvature of the energy difference.
        e0 = el_k.e[ib] - el_kq.e[jb] - sign_ph * ph.e[imode]
        max_curvature = econv_tol
        v0_cart = - el_kq.vdiag[jb] - sign_ph * ph.vdiag[imode]
        v0 = recip_lattice' * v0_cart
        de_max = sum(abs.(v0) ./ ngrid) / 2 + max_curvature * sum(1 ./ ngrid.^2) / 4
        return abs(e0) <= de_max
    else
        error("energy conservation mode not identified")
    end
end

# Check if energy-conserving scattering exists for all bands, modes, and sign_ph, and return true if so.
@timing "econv_all" function check_energy_conservation_all(epdata, ngrid, recip_lattice, econv_mode, econv_tol)
    el_k = epdata.el_k
    el_kq = epdata.el_kq
    ph = epdata.ph

    # If econv_mode is :None, do not check energy conservation. Always return true.
    econv_mode == :None && return true

    # Loop over all scattering processes. If there is an energy-conserving one, return true
    @inbounds for imode in 1:ph.nmodes, jb in el_kq.rng, ib in el_k.rng, sign_ph in (-1, 1)
        check_energy_conservation(el_k, el_kq, ph, ib, jb, imode, sign_ph, ngrid, recip_lattice, econv_mode, econv_tol) && return true
    end
    # If all scatterings are energy non-conserving, return false
    return false
end