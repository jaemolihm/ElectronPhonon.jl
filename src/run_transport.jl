
using HDF5

"""
Calculate electron states and write BTStates object to HDF5 file.
"""
@timing "el_states" function write_electron_BTStates(g, kpts, model, window, nband, nband_ignore)
    # TODO: MPI
    @timing "init" el_save = [ElectronState(Float64, model.nw, nband, nband_ignore) for i=1:kpts.n]
    for ik in 1:kpts.n
        xk = kpts.vectors[ik]
        el = el_save[ik]

        set_eigen!(el, model.el_ham, xk, "gridopt")
        set_window!(el, window)
        set_velocity_diag!(el, model.el_ham_R, xk, "gridopt")
    end # ik
    el_boltzmann, imap = electron_states_to_BTStates(el_save, kpts)
    dump_BTData(g, el_boltzmann)
    el_save, imap
end

"""
Calculate phonon states and write BTStates object to HDF5 file.
"""
@timing "ph_states" function write_phonon_BTStates(g, kpts, model)
    # TODO: MPI
    @timing "init" ph_save = [PhononState(Float64, model.nmodes) for i=1:kpts.n]
    for ik in 1:kpts.n
        xk = kpts.vectors[ik]
        ph = ph_save[ik]
        set_eigen!(ph, model, xk, "gridopt")
        set_velocity_diag!(ph, model, xk, "gridopt")
    end # ik
    ph_boltzmann, imap = phonon_states_to_BTStates(ph_save, kpts)
    dump_BTData(g, ph_boltzmann)
    ph_save, imap
end

function run_transport(
        model::ModelEPW,
        k_input::Union{NTuple{3,Int}, Kpoints},
        q_input::Union{NTuple{3,Int}, Kpoints};
        mpi_comm_k=nothing,
        mpi_comm_q=nothing,
        fourier_mode="gridopt",
        window_k=(-Inf,Inf),
        window_kq=(-Inf,Inf),
        folder,
        energy_conservation=(:None, 0.0),
        use_irr_k=true,
    )
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
    if mpi_comm_k !== nothing
        error("mpi_comm_k not implemented")
    end
    if mpi_comm_q !== nothing
        error("mpi_comm_q not implemented")
    end
    if k_input isa Kpoints
        error("k_input isa Kpoints not implemented")
    end
    if q_input isa Kpoints
        error("q_input isa Kpoints not implemented")
    end

    if model.epmat_outer_momentum != "el"
        throw(ArgumentError("model.epmat_outer_momentum must be el"))
    end
    if mod.(q_input, k_input) != (0, 0, 0)
        throw(ArgumentError("q grid must be an integer multiple of k grid."))
    end

    nw = model.nw
    nmodes = model.nmodes

    energy_conservation_mode, energy_conservation_tol = energy_conservation

    @timing "setup kgrid" begin
        # Generate k points
        mpi_isroot() && @info "Setting k-point grid"
        if use_irr_k
            kpts, iband_min_k, iband_max_k, nelec_below_window = setup_kgrid(k_input, nw,
                model.el_ham, window_k, mpi_comm_k, symmetry=model.symmetry)
        else
            kpts, iband_min_k, iband_max_k, nelec_below_window = setup_kgrid(k_input, nw,
                model.el_ham, window_k, mpi_comm_k)
        end

        # Generate k+q points
        mpi_isroot() && @info "Setting k+q-point grid"
        kqpts, iband_min_kq, iband_max_kq, _ = setup_kgrid(q_input, nw, model.el_ham, window_kq, mpi_comm_k)
    end

    iband_min = min(iband_min_k, iband_min_kq)
    iband_max = max(iband_max_k, iband_max_kq)

    nk = kpts.n
    nkq = kqpts.n
    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1

    # Map k and k+q points to q points
    mpi_isroot() && @info "Finding the list of q points"
    @timing "qpts" begin
        T = eltype(kpts.weights)
        xqs = Vector{Vec3{T}}()
        map_xq_int_to_iq = Dict{NTuple{3, Int}, Int}()
        map_iq_to_xq_int = Vector{NTuple{3, Int}}()
        iq = 0
        for ik in 1:nk
            xk = kpts.vectors[ik]
            for ikq in 1:nkq
                xkq = kqpts.vectors[ikq]
                xq = xkq - xk

                # Move xq inside [-0.5, 0.5]^3. This doesn't change the Fourier transform but
                # makes the long-range part more robust.
                xq = mod.(xq .+ 0.5, 1.0) .- 0.5

                # Reusing phonon states
                xq_int = round.(Int, xq .* kqpts.ngrid)
                if ! isapprox(xq, xq_int ./ kqpts.ngrid, atol=10*eps(eltype(xq)))
                    @show xq, kqpts.ngrid, xq_int, xq .- xq_int ./ kqpts.ngrid
                    error("xq is not on the grid")
                end

                # Find new q points, append to map_xq_int_to_iq and xqs
                if xq_int.data ∉ keys(map_xq_int_to_iq)
                    iq += 1
                    map_xq_int_to_iq[xq_int.data] = iq
                    push!(map_iq_to_xq_int, xq_int.data)
                    push!(xqs, xq)
                end
            end
        end
        nq = length(xqs)
        qpts = EPW.Kpoints{T}(nq, xqs, ones(T, nq) ./ prod(kqpts.ngrid), kqpts.ngrid)
        inds = EPW.sort!(qpts)
        for iq_new = 1:nq
            key = map_iq_to_xq_int[inds[iq_new]]
            map_xq_int_to_iq[key] = iq_new
        end
        # map_iq_to_xq_int is not used anymore
        map_iq_to_xq_int = nothing
    end

    mpi_isroot() && @info "Calculating electron and phonon states"
    @timing "hdf init" begin
        epdatas = [ElPhData(Float64, nw, nmodes, nband, nband_ignore) for i=1:Threads.nthreads()]

        # Open HDF5 file for writing BTEdata
        fid_btedata = h5open(joinpath(folder, "btedata.h5"), "w")
        # Write some attributes to file
        g = create_group(fid_btedata, "electron")
        write_attribute(fid_btedata["electron"], "nk", nk)
        write_attribute(fid_btedata["electron"], "nbandk_max", nband)
        fid_btedata["electron/weights"] = kpts.weights
        write_attribute(fid_btedata["electron"], "nelec_below_window", nelec_below_window)

        # Calculate initial (k) and final (k+q) electron states, write to HDF5 file
        mpi_isroot() && @info "Calculating electron states at k"
        g = create_group(fid_btedata, "initialstate_electron")
        el_k_save, imap_el_k = write_electron_BTStates(g, kpts, model, window_k, nband, nband_ignore)
        mpi_isroot() && @info "Calculating electron states at k+q"
        g = create_group(fid_btedata, "finalstate_electron")
        el_kq_save, imap_el_kq = write_electron_BTStates(g, kqpts, model, window_kq, nband, nband_ignore)

        # Write phonon states to HDF5 file
        mpi_isroot() && @info "Calculating phonon states"
        g = create_group(fid_btedata, "phonon")
        ph_save, imap_ph = write_phonon_BTStates(g, qpts, model)
    end

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epobj_ekpR = WannierObject(model.epmat.irvec_next,
    zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    # Setup for collecting scattering processes
    @timing "bt init" begin
        max_nscat = nkq * nmodes * nband^2 * 2
        bt_scat = ElPhScatteringData{Float64}(max_nscat)
    end

    mpi_isroot() && @info "Number of k   points = $nk"
    mpi_isroot() && @info "Number of k+q points = $nkq"
    mpi_isroot() && @info "Number of q   points = $nq"
    flush(stdout)
    flush(stderr)

    nscat_tot = 0
    for ik in 1:nk
        if mod(ik, 100) == 0
            mpi_isroot() && @info "ik = $ik"
            flush(stdout)
            flush(stderr)
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        for epdata in epdatas
            copyto!(epdata.el_k, el_k)
        end

        get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, EPW.get_u(el_k), fourier_mode)

        bt_nscat = 0

        # Threads.@threads :static for ikq in 1:nkq
        for ikq in 1:nkq
            tid = Threads.threadid()
            epdata = epdatas[tid]

            epdata.wtk = kpts.weights[ik]
            epdata.wtq = kqpts.weights[ikq]

            xkq = kqpts.vectors[ikq]
            xq = xkq - xk

            # Move xq inside [-0.5, 0.5]^3. This doesn't change the Fourier transform but
            # makes the long-range part more robust.
            xq = mod.(xq .+ 0.5, 1.0) .- 0.5

            # Reusing phonon states
            xq_int = round.(Int, xq .* kqpts.ngrid)
            if ! isapprox(xq, xq_int ./ kqpts.ngrid, atol=10*eps(eltype(xq)))
                @show xq, kqpts.ngrid, xq_int, xq .- xq_int ./ kqpts.ngrid
                error("xq is not on the grid")
            end

            # Index of q point in the global list
            iq = map_xq_int_to_iq[xq_int.data]
            copyto!(epdata.ph, ph_save[iq])

            # Use saved data for electron state at k+q.
            copyto!(epdata.el_kq, el_kq_save[ikq])

            el_k = epdata.el_k
            el_kq = epdata.el_kq
            ph = epdata.ph

            # Check if energy-conserving scattering exists. If not, skip the remaining parts.
            @timing "econv" if energy_conservation_mode != :None
                skip_q = true
                @inbounds for imode in 1:nmodes, jb in el_kq.rng, ib in el_k.rng, sign_ph in (-1, 1)
                    e0 = el_k.e[ib] - el_kq.e[jb] - sign_ph * ph.e[imode]
                    if energy_conservation_mode == :Fixed
                        if e0 <= energy_conservation_tol
                            skip_q = false
                            break
                        end
                    elseif energy_conservation_mode == :Linear
                        v0_cart = - el_kq.vdiag[jb] - sign_ph * ph.vdiag[imode]
                        v0 = model.recip_lattice' * v0_cart
                        de_max = sum(abs.(v0) ./ kqpts.ngrid) / 2
                        if abs(e0) <= de_max
                            skip_q = false
                            break
                        end
                    end
                end
                if skip_q
                    continue
                end
            end

            # Compute electron-phonon coupling
            get_eph_kR_to_kq!(epdata, epobj_ekpR, xq, fourier_mode)
            if any(xq .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                eph_dipole!(epdata.ep, xq, model.polar_eph, epdata.ph.u, epdata.mmat, 1)
            end
            epdata_set_g2!(epdata)

            @timing "bt_push" for imode in 1:nmodes
                for jb in el_kq.rng
                    for ib in el_k.rng
                        @inbounds for sign_ph in (-1, 1)
                            # Save only if the scattering satisfies energy conservation
                            e0 = el_k.e[ib] - el_kq.e[jb] - sign_ph * ph.e[imode]
                            if energy_conservation_mode == :Fixed
                                if e0 > energy_conservation_tol
                                    continue
                                end
                            elseif energy_conservation_mode == :Linear
                                v0_cart = - el_kq.vdiag[jb] - sign_ph * ph.vdiag[imode]
                                v0 = model.recip_lattice' * v0_cart
                                de_max = sum(abs.(v0) ./ kqpts.ngrid) / 2
                                if abs(e0) > de_max
                                    continue
                                end
                            end
                            bt_nscat += 1
                            bt_scat.ind_el_i[bt_nscat] = imap_el_k[ib, ik]
                            bt_scat.ind_el_f[bt_nscat] = imap_el_kq[jb, ikq]
                            bt_scat.ind_ph[bt_nscat] = imap_ph[imode, iq]
                            bt_scat.sign_ph[bt_nscat] = sign_ph
                            bt_scat.mel[bt_nscat] = epdata.g2[jb, ib, imode]
                        end
                    end
                end
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
