
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
        elself_params=nothing::Union{Nothing,ElectronSelfEnergyParams},
        phself_params=nothing::Union{Nothing,PhononSelfEnergyParams},
        transport_params=nothing::Union{Nothing,TransportParams},
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
        elself_params=nothing::Union{Nothing,ElectronSelfEnergyParams},
        phself_params=nothing::Union{Nothing,PhononSelfEnergyParams},
        transport_params=nothing::Union{Nothing,TransportParams},
    )

    # TODO: Allow k_input to be a Kpoints object
    # TODO: Allow q_input to be a Kpoints object
    # TODO: Implement mpi_comm_k

    compute_elself = elself_params !== nothing
    compute_phself = phself_params !== nothing
    compute_transport = transport_params !== nothing

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
        if mpi_comm_q === nothing
            qpoints_all = generate_kvec_grid(q_input...)
        else
            qpoints_all = generate_kvec_grid(q_input..., mpi_comm_q)
        end
    end
    qpoints_filtered = filter_qpoints(qpoints_all, kpoints, nw, model.el_ham, window)
    qpoints = redistribute_kpoints(qpoints_filtered, mpi_comm_q)

    nk = kpoints.n
    nq = qpoints.n
    nband = iband_max - iband_min + 1

    epdatas = [ElPhData(Float64, nw, nmodes, nband) for i=1:Threads.nthreads()]
    for epdata in epdatas
        epdata.iband_offset = iband_min - 1
    end

    # Initialize data structs
    if compute_elself
        elself = ElectronSelfEnergy(Float64, nband, nmodes, nk, length(elself_params.Tlist))
    end
    if compute_phself
        phselfs = [PhononSelfEnergy(Float64, nband, nmodes, nq,
            length(elself_params.Tlist)) for i=1:Threads.nthreads()]
    end
    if compute_transport
        transport_serta = TransportSERTA(Float64, nband, nmodes, nk,
            length(transport_params.Tlist))
    end

    # Compute and save electron matrix elements at k
    ek_full_save = zeros(Float64, nw, nk)
    uk_full_save = Array{ComplexF64,3}(undef, nw, nw, nk)
    vdiagk_save = zeros(Float64, 3, nband, nk)

    Threads.@threads :static for ik in 1:nk
        epdata = epdatas[Threads.threadid()]
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
    if compute_transport
        μ = transport_set_μ!(transport_params, ek_full_save, kpoints.weights, model.volume)
    end

    omega_save = zeros(nmodes, nq)
    omegas = zeros(nmodes)
    u_ph = zeros(ComplexF64, (nmodes, nmodes))

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epobj_eRpq = WannierObject(model.el_ham.nr, model.el_ham.irvec,
                zeros(ComplexF64, (nw*nw*nmodes, model.el_ham.nr)))

    mpi_isroot() && @info "Number of q points = $nq"
    mpi_isroot() && @info "Number of k points = $nk"

    for iq in 1:nq
        if mod(iq, 100) == 0 && mpi_isroot()
            mpi_isroot() && @info "iq = $iq"
        end
        xq = qpoints.vectors[iq]

        # Phonon eigenvalues
        get_ph_eigen!(omegas, u_ph, model, xq, fourier_mode=fourier_mode)
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
            if any(xq .> 1.0e-8) && model.use_polar_dipole
                epdata_set_bmat!(epdata)
                eph_dipole!(epdata.ep, xq, model.polar_eph, u_ph, epdata.bmat, 1)
            end
            epdata_set_g2!(epdata)

            # Now, we are done with matrix elements. All data saved in epdata.

            # Calculate physical quantities.
            if compute_elself
                compute_electron_selfen!(elself, epdata, elself_params, ik)
            end
            if compute_phself
                compute_phonon_selfen!(phselfs[tid], epdata, phself_params, iq)
            end
            if compute_transport
                compute_lifetime_serta!(transport_serta, epdata, transport_params, ik)
            end
        end # ik
    end # iq

    output = Dict()

    output["ek"] = ek_full_save
    output["iband_min"] = iband_min
    output["iband_max"] = iband_max
    output["omega"] = EPW.mpi_gather(omega_save, mpi_comm_q)

    # Post-process computed data, write to file if asked to.
    if compute_elself
        EPW.mpi_sum!(elself.imsigma, mpi_comm_q)
        # Average over degenerate states
        el_imsigma_avg = similar(elself.imsigma)
        ek_partial = view(ek_full_save, iband_min:iband_max, :)
        @views for iT in 1:size(elself.imsigma, 3)
            average_degeneracy!(el_imsigma_avg[:,:,iT], elself.imsigma[:,:,iT], ek_partial)
        end

        output["elself_imsigma"] = el_imsigma_avg
    end

    if compute_phself
        ph_imsigma = sum([phself.imsigma for phself in phselfs]) .* phself_params.spin_degeneracy
        ph_imsigma_avg = similar(ph_imsigma)
        @views for iT in 1:size(ph_imsigma, 3)
            average_degeneracy!(ph_imsigma_avg[:,:,iT], ph_imsigma[:,:,iT], omega_save)
        end
        output["phself_imsigma"] = EPW.mpi_gather(ph_imsigma_avg, mpi_comm_q)
    end

    if compute_transport
        EPW.mpi_sum!(transport_serta.inv_τ, mpi_comm_q)
        σlist = compute_mobility_serta!(transport_params, transport_serta.inv_τ,
            ek_full_save[iband_min:iband_max, :], vdiagk_save, kpoints.weights, window)
        output["transport_σlist"] = σlist
    end

    output
end
