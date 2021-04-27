
using HDF5

"""
Calculate electron states and write BTStates object to HDF5 file.
"""
@timing "el_states" function write_electron_BTStates(g, kpts, model, window, nband, nband_ignore)
    # TODO: MPI
    el_save = [ElectronState(Float64, model.nw, nband, nband_ignore) for i=1:kpts.n]
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
    ph_save = [PhononState(Float64, model.nmodes) for i=1:kpts.n]
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
        window=(-Inf,Inf),
        folder,
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

    @timing "setup kgrid" begin
        # Generate k points
        kpts, iband_min_k, iband_max_k, nelec_below_window = setup_kgrid(k_input, nw,
            model.el_ham, window, mpi_comm_k)

        # Generate k+q points
        kqpts, iband_min_kq, iband_max_kq, _ = setup_kgrid(q_input, nw, model.el_ham, window, mpi_comm_k)
    end

    iband_min = min(iband_min_k, iband_min_kq)
    iband_max = max(iband_max_k, iband_max_kq)

    nk = kpts.n
    nkq = kqpts.n
    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1

    # Map k and k+q points to q points
    T = eltype(kpts.weights)
    xqs = Vector{Vec3{T}}()
    map_xq_int_to_iq = Dict{NTuple{3, Int}, Int}()
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
            if ! haskey(map_xq_int_to_iq, xq_int.data)
                iq += 1
                map_xq_int_to_iq[xq_int.data] = iq
                push!(xqs, xq)
            end
        end
    end
    nq = length(xqs)
    qpts = EPW.Kpoints{T}(nq, xqs, ones(T, nq) ./ prod(kqpts.ngrid), kqpts.ngrid)
    inds = EPW.sort!(qpts)
    for key in keys(map_xq_int_to_iq)
        map_xq_int_to_iq[key] = findfirst(inds .== map_xq_int_to_iq[key])
    end

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
    g = create_group(fid_btedata, "initialstate_electron")
    el_k_save, imap_el_k = write_electron_BTStates(g, kpts, model, window, nband, nband_ignore)
    g = create_group(fid_btedata, "finalstate_electron")
    el_kq_save, imap_el_kq = write_electron_BTStates(g, kqpts, model, window, nband, nband_ignore)

    # Write phonon states to HDF5 file
    g = create_group(fid_btedata, "phonon")
    _, imap_ph = write_phonon_BTStates(g, qpts, model)

    # Preallocate arrays for saving data for creating BTEdata


    # Dictionary to save phonon states
    ph_save = Dict{NTuple{3, Int}, PhononState{Float64}}()

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epobj_ekpR = WannierObject(model.epmat.irvec_next,
                zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    mpi_isroot() && @info "Number of k   points = $nk"
    mpi_isroot() && @info "Number of k+q points = $nkq"

    @timing "main loop" for ik in 1:nk
        if mod(ik, 100) == 0
            mpi_isroot() && @info "ik = $ik"
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        for epdata in epdatas
            copyto!(epdata.el_k, el_k)
        end

        get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, EPW.get_u(el_k), fourier_mode)

        # Setup for collecting scattering processes
        bt_nscat = 0
        bt_ind_el_i = Vector{Int}()
        bt_ind_el_f = Vector{Int}()
        bt_ind_ph = Vector{Int}()
        bt_sign_ph = Vector{Int}()
        bt_mel = Vector{Float64}()

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

            # Phonon eigenvalues and eigenstates
            if haskey(ph_save, xq_int.data)
                # Phonon eigenvalues already calculated. Copy from ph_save.
                copyto!(epdata.ph, ph_save[xq_int.data])
            else
                # Phonon eigenvalues not calculated. Calculate and save at ph_save.
                set_eigen!(epdata.ph, model, xq, fourier_mode)
                ph_save[xq_int.data] = deepcopy(epdata.ph)
            end

            # Use saved data for electron state at k+q.
            copyto!(epdata.el_kq, el_kq_save[ikq])

            # Compute electron-phonon coupling
            get_eph_kR_to_kq!(epdata, epobj_ekpR, xq, fourier_mode)
            if any(xq .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                eph_dipole!(epdata.ep, xq, model.polar_eph, ph.u, epdata.mmat, 1)
            end
            epdata_set_g2!(epdata)

            for imode in 1:nmodes
                for jb in epdata.el_kq.rng
                    for ib in epdata.el_k.rng
                        @inbounds for sign_ph in (-1, 1)
                            # if ik == 1 && ikq <= 10
                            #     @info (epdata.el_k.e[ib] - epdata.el_kq.e[jb]
                            #         - sign_ph * epdata.ph.e[imode]) / EPW.unit_to_aru(:meV)
                            # end
                            # if (epdata.el_k.e[ib] - epdata.el_kq.e[jb]
                            #     - sign_ph * epdata.ph.e[imode]) > 50.0 * EPW.unit_to_aru(:meV)
                            #     continue
                            # end
                            bt_nscat += 1
                            push!(bt_ind_el_i, imap_el_k[ib, ik])
                            push!(bt_ind_el_f, imap_el_kq[jb, ikq])
                            push!(bt_ind_ph, imap_ph[imode, iq])
                            push!(bt_sign_ph, sign_ph)
                            push!(bt_mel, epdata.g2[jb, ib, imode])
                        end
                    end
                end
            end
        end # ikq

        @views bt_scattering = ElPhScatteringData{Float64}(bt_nscat, bt_ind_el_i[1:bt_nscat],
            bt_ind_el_f[1:bt_nscat], bt_ind_ph[1:bt_nscat], bt_sign_ph[1:bt_nscat],
            bt_mel[1:bt_nscat])
        g = create_group(fid_btedata, "scattering/ik$ik")
        dump_BTData(g, bt_scattering)
    end # ik
    close(fid_btedata)
    nothing
end
