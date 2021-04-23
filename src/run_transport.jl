function run_transport(
    model::ModelEPW,
    k_input::Union{NTuple{3,Int}, Kpoints},
    kq_input::Union{NTuple{3,Int}, Kpoints};
    transport_params,
    mpi_comm_k=nothing,
    mpi_comm_q=nothing,
    fourier_mode="gridopt",
    window=(-Inf,Inf),
    )

    """
    Things to implement
    - mpi_comm_k
    - mpi_comm_q
    - k_input to be a Kpoints object
    - kq_input to be a Kpoints object
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
    if kq_input isa Kpoints
        error("kq_input isa Kpoints not implemented")
    end

    if model.epmat_outer_momentum != "el"
        throw(ArgumentError("model.epmat_outer_momentum must be el"))
    end

    compute_transport = transport_params isa TransportParams

    nw = model.nw
    nmodes = model.nmodes


    @timing "setup kgrid" begin
        # Generate k points
        kpts, iband_min_k, iband_max_k = setup_kgrid(k_input, nw, model.el_ham, window, mpi_comm_k)

        # Generate k+q points
        kqpts, iband_min_kq, iband_max_kq = setup_kgrid(kq_input, nw, model.el_ham, window, mpi_comm_k)
    end

    iband_min = min(iband_min_k, iband_min_kq)
    iband_max = max(iband_max_k, iband_max_kq)

    nk = kpts.n
    nkq = kqpts.n
    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1

    epdatas = [ElPhData(Float64, nw, nmodes, nband, nband_ignore) for i=1:Threads.nthreads()]

    if compute_transport
        transport_serta = TransportSERTA(Float64, nband, nmodes, nk,
            length(transport_params.Tlist))
    end

    # Compute and save electron state at k and k+q
    @timing "el_k el_kq" begin
        el_k_save = [ElectronState(Float64, nw, nband, nband_ignore) for ik=1:nk]
        el_kq_save = [ElectronState(Float64, nw, nband, nband_ignore) for ik=1:nkq]

        Threads.@threads :static for ik in 1:nk
            xk = kpts.vectors[ik]
            el = el_k_save[ik]

            set_eigen!(el, model.el_ham, xk, "gridopt")
            set_window!(el, window)
            set_velocity_diag!(el, model.el_ham_R, xk, "gridopt")
        end # ik

        Threads.@threads :static for ik in 1:nkq
            xk = kqpts.vectors[ik]
            el = el_kq_save[ik]

            set_eigen!(el, model.el_ham, xk, "gridopt")
            set_window!(el, window)
            set_velocity_diag!(el, model.el_ham_R, xk, "gridopt")
        end # ik
    end

    # Compute chemical potential
    @timing "chemical pot." begin
        ek_full_save = zeros(Float64, nw, nk)
        for ik in 1:nk
            ek_full_save[:, ik] .= el_k_save[ik].e_full
        end
        if compute_transport
            μ = transport_set_μ!(transport_params, ek_full_save, kpts.weights, model.volume)
        end
    end

    # omega_save = zeros(nmodes, nq)
    # ph = PhononState(Float64, nmodes)

    # # E-ph matrix in electron Wannier, phonon Bloch representation
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

        # Threads.@threads :static for ikq in 1:nkq
        for ikq in 1:nkq
            tid = Threads.threadid()
            epdata = epdatas[tid]

            xkq = kqpts.vectors[ikq]
            xq = xkq - xk

            epdata.wtk = kpts.weights[ik]
            epdata.wtq = kqpts.weights[ikq]

            # Phonon eigenvalues
            set_eigen!(epdata.ph, model, xq, fourier_mode)

            # Use saved data for electron state at k+q.
            copyto!(epdata.el_kq, el_kq_save[ikq])

            # Compute electron-phonon coupling
            get_eph_kR_to_kq!(epdata, epobj_ekpR, xq, fourier_mode)
            if any(xq .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                eph_dipole!(epdata.ep, xq, model.polar_eph, ph.u, epdata.mmat, 1)
            end
            epdata_set_g2!(epdata)

            # Now, we are done with matrix elements. All data saved in epdata.
            if compute_transport
                compute_lifetime_serta!(transport_serta, epdata, transport_params, ik)
            end
        end # ikq
    end # ik

    output = Dict()

    if compute_transport
        EPW.mpi_sum!(transport_serta.inv_τ, mpi_comm_q)
        σlist = compute_mobility_serta!(transport_params, transport_serta.inv_τ,
            el_k_save, kpts.weights, window)
        output["transport_σlist"] = σlist
    end

    output
end
