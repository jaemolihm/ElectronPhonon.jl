
using EPW.WanToBloch

function setup_kgrid(k_input, nw, el_ham, window, mpi_comm_k)
    if typeof(k_input) == EPW.Kpoints
        error("k_input as EPW.Kpoints not implemented")
    else
        if mpi_comm_k === nothing
            kpoints, iband_min, iband_max = filter_kpoints_grid(k_input...,
            nw, el_ham, window)
        else
            kpoints, iband_min, iband_max = filter_kpoints_grid(k_input...,
            nw, el_ham, window, mpi_comm_k)
        end
    end
    kpoints, iband_min, iband_max
end

"""
    run_eph_outer_loop_q(
        model::EPW.ModelEPW,
        k_input::Union{NTuple{3,Int}, EPW.Kpoints},
        q_input::Union{NTuple{3,Int}, EPW.Kpoints};
        mpi_comm_k=nothing,
        mpi_comm_q=nothing,
        fourier_mode="normal",
        window=(-Inf,Inf),
        transport_params::TransportParams,
    )
Loop over k and q points to compute e-ph related quantities.

# Arguments
- `k_input`, `q_input`: either a 3-element tuple (n1, n2, n3) or a Kpoints object.
"""
function run_eph_outer_loop_q(
        model::EPW.ModelEPW,
        k_input::Union{NTuple{3,Int}, EPW.Kpoints},
        q_input::Union{NTuple{3,Int}, EPW.Kpoints};
        mpi_comm_k=nothing,
        mpi_comm_q=nothing,
        fourier_mode="normal",
        window=(-Inf,Inf),
        transport_params::TransportParams,
    )

    # TODO: Allow k_input to be a Kpoints object
    # TODO: Allow q_input to be a Kpoints object
    # TODO: Implement mpi_comm_k

    nw = model.nw
    nmodes = model.nmodes

    if mpi_comm_k !== nothing
        error("mpi_comm_k not implemented")
    end

    # Generate k points
    kpoints, iband_min, iband_max = setup_kgrid(k_input, nw, model.el_ham, window, mpi_comm_k)

    # Generate q points
    if typeof(q_input) == EPW.Kpoints
        error("q_input as EPW.Kpoints not implemented")
    else
        qpoints_all = generate_kvec_grid(q_input..., mpi_comm_q)
    end

    qpoints = filter_qpoints(qpoints_all, kpoints, nw, model.el_ham, window)

    nk = kpoints.n
    nq = qpoints.n
    nband = iband_max - iband_min + 1

    epdatas = [ElPhData(Float64, nw, nmodes, nband) for i=1:nthreads()]
    for epdata in epdatas
        epdata.iband_offset = iband_min - 1
    end

    transport_serta = TransportSERTA(Float64, nband, nmodes, nk,
        length(transport_params.Tlist))

    # Compute and save electron matrix elements at k
    ek_full_save = zeros(Float64, nw, nk)
    uk_full_save = Array{ComplexF64,3}(undef, nw, nw, nk)
    vdiagk_save = zeros(Float64, 3, nband, nk)

    Threads.@threads :static for ik in 1:nk
        epdata = epdatas[threadid()]
        xk = kpoints.vectors[ik]

        get_el_eigen!(epdata, "k", model.el_ham, xk, fourier_mode)
        skip_k = epdata_set_window!(epdata, "k", window)
        get_el_velocity_diag!(epdata, "k", model.el_ham_R, xk, fourier_mode)

        # Save matrix elements at k for reusing
        ek_full_save[:, ik] .= epdata.ek_full
        uk_full_save[:, :, ik] .= epdata.uk_full
        vdiagk_save[:, :, ik] .= epdata.vdiagk
    end # ik

    # Compute chemical potential
    μ = transport_set_μ!(transport_params, ek_full_save, kpoints.weights, model.volume)

    omega_save = zeros(nmodes, nq)
    omegas = zeros(nmodes)
    u_ph = zeros(ComplexF64, (nmodes, nmodes))

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epobj_eRpq = WannierObject(model.el_ham.nr, model.el_ham.irvec,
                zeros(ComplexF64, (nw*nw*nmodes, model.el_ham.nr)))

    for iq in 1:nq
        if mod(iq, 100) == 0 && mpi_isroot()
            @info "iq = $iq"
        end
        xq = qpoints.vectors[iq]

        # Phonon eigenvalues
        get_ph_eigen!(omegas, u_ph, model.ph_dyn, model.mass, xq, fourier_mode)
        omega_save[:, iq] .= omegas

        get_eph_RR_to_Rq!(epobj_eRpq, model.epmat, xq, u_ph, fourier_mode)

        Threads.@threads :static for ik in 1:nk
        # for ik in 1:nk
            tid = Threads.threadid()
            epdata = epdatas[tid]
            # phself = phselfs[tid]

            # println("$tid $ik")
            xk = kpoints.vectors[ik]
            xkq = xk + xq

            epdata.wtk = kpoints.weights[ik]
            epdata.wtq = qpoints.weights[iq]
            epdata.omega .= omegas

            # Use saved data for electron eigenstate at k.
            epdata.ek_full .= @view ek_full_save[:, ik]
            epdata.uk_full .= @view uk_full_save[:, :, ik]
            epdata.vdiagk .= @view vdiagk_save[:, :, ik]

            get_el_eigen!(epdata, "k+q", model.el_ham, xkq, fourier_mode)

            # Set energy window, skip if no state is inside the window
            skip_k = epdata_set_window!(epdata, "k", window)
            skip_kq = epdata_set_window!(epdata, "k+q", window)
            if skip_k || skip_kq
                continue
            end

            get_el_velocity_diag!(epdata, "k+q", model.el_ham_R, xkq, fourier_mode)
            get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk, fourier_mode)

            # Now, we are done with matrix elements. All data saved in epdata.

            # Calculate physical quantities.
            compute_lifetime_serta!(transport_serta, epdata, transport_params, ik)
        end # ik
    end # iq
    (iband_rng=iband_min:iband_max,
    energy=ek_full_save, vel_diag=vdiagk_save,
    transport_serta=transport_serta, )
end
