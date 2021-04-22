
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
        phspec_params=nothing::Union{Nothing,PhononSpectralParams},
        transport_params=nothing::Union{Nothing,TransportParams},
    )

    # TODO: Allow k_input to be a Kpoints object
    # TODO: Allow q_input to be a Kpoints object
    # TODO: Implement mpi_comm_k

    compute_elself = elself_params !== nothing
    compute_phself = phself_params !== nothing
    compute_phspec = phspec_params !== nothing
    compute_transport = transport_params !== nothing

    nw = model.nw
    nmodes = model.nmodes

    if mpi_comm_k !== nothing
        error("mpi_comm_k not implemented")
    end

    # Generate k points
    kpoints, iband_min, iband_max = setup_kgrid(k_input, nw, model.el_ham, window, mpi_comm_k)

    # Generate q points
    if q_input isa EPW.Kpoints
        if mpi_comm_q !== nothing
            error("q_input as EPW.Kpoints with mpi_comm_q not implemented")
        end
        qpoints_all = q_input
    elseif q_input isa NTuple{3,Int}
        if mpi_comm_q === nothing
            qpoints_all = generate_kvec_grid(q_input...)
        else
            qpoints_all = generate_kvec_grid(q_input..., mpi_comm_q)
        end
    else
        error("type of q_input is wrong")
    end
    qpoints_filtered = filter_qpoints(qpoints_all, kpoints, nw, model.el_ham, window)
    qpoints = redistribute_kpoints(qpoints_filtered, mpi_comm_q)

    nk = kpoints.n
    nq = qpoints.n
    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1

    epdatas = [ElPhData(Float64, nw, nmodes, nband, nband_ignore) for i=1:Threads.nthreads()]

    # Initialize data structs
    if compute_elself
        elself = ElectronSelfEnergy(Float64, nband, nmodes, nk, length(elself_params.Tlist))
    end
    if compute_phself
        phselfs = [PhononSelfEnergy(Float64, nband, nmodes, nq,
            length(phself_params.Tlist)) for i=1:Threads.nthreads()]
    end
    if compute_phspec
        phspecs = [PhononSpectralData(phspec_params, nmodes, nq) for i=1:Threads.nthreads()]
    end
    if compute_transport
        transport_serta = TransportSERTA(Float64, nband, nmodes, nk,
            length(transport_params.Tlist))
    end

    # Compute and save electron state at k
    el_k_save = [ElectronState(Float64, nw, nband, nband_ignore) for ik=1:nk]
    ek_full_save = zeros(Float64, nw, nk)

    Threads.@threads :static for ik in 1:nk
        xk = kpoints.vectors[ik]
        el_k = el_k_save[ik]

        set_eigen!(el_k, model.el_ham, xk, "gridopt")
        set_window!(el_k, window)
        set_velocity_diag!(el_k, model.el_ham_R, xk, "gridopt")
        ek_full_save[:, ik] .= el_k.e_full
    end # ik

    # Compute chemical potential
    if compute_transport
        μ = transport_set_μ!(transport_params, ek_full_save, kpoints.weights, model.volume)
    end

    omega_save = zeros(nmodes, nq)
    ph = PhononState(Float64, nmodes)

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epobj_eRpq = WannierObject(model.el_ham.irvec,
                zeros(ComplexF64, (nw*nw*nmodes, model.el_ham.nr)))

    mpi_isroot() && @info "Number of q points = $nq"
    mpi_isroot() && @info "Number of k points = $nk"

    for iq in 1:nq
        if mod(iq, 100) == 0 && mpi_isroot()
            mpi_isroot() && @info "iq = $iq"
        end
        xq = qpoints.vectors[iq]

        # Phonon eigenvalues
        set_eigen!(ph, model, xq, fourier_mode)
        omega_save[:, iq] .= ph.e

        get_eph_RR_to_Rq!(epobj_eRpq, model.epmat, xq, ph.u, fourier_mode)

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
            copyto!(epdata.ph, ph)

            # Use saved data for electron state at k.
            copyto!(epdata.el_k, el_k_save[ik])

            # Compute electron state at k+q.
            set_eigen!(epdata.el_kq, model.el_ham, xkq, fourier_mode)

            # Set energy window, skip if no state is inside the window
            skip_kq = set_window!(epdata.el_kq, window)
            if skip_kq
                continue
            end

            set_velocity_diag!(epdata.el_kq, model.el_ham_R, xkq, fourier_mode)
            get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk, fourier_mode)
            if any(xq .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                eph_dipole!(epdata.ep, xq, model.polar_eph, ph.u, epdata.mmat, 1)
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
            if compute_phspec
                compute_phonon_spectral!(phspecs[tid], epdata, phspec_params, iq)
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

    if compute_phspec
        # TODO: MPI k, MPI q, threading are not tested.
        ph_selfen_dynamic = sum([phspec.selfen for phspec in phspecs]) .* phspec_params.degeneracy
        ph_selfen_static = sum([phspec.selfen_static for phspec in phspecs]) .* phspec_params.degeneracy

        # calculate selfen_non_adiabatic(ω) = selfen_dynamic(ω) - selfen_static
        ph_selfen_non_adiabatic = copy(ph_selfen_dynamic)
        for arr in eachslice(ph_selfen_non_adiabatic, dims=1)
            arr .-= ph_selfen_static
        end

        ph_green = calculate_phonon_green(phspec_params.ωlist, omega_save, ph_selfen_non_adiabatic)
        output["ph_green"] = ph_green
        output["ph_selfen_dynamic"] = ph_selfen_dynamic
        output["ph_selfen_static"] = ph_selfen_static
    end

    if compute_transport
        EPW.mpi_sum!(transport_serta.inv_τ, mpi_comm_q)
        σlist = compute_mobility_serta!(transport_params, transport_serta.inv_τ,
            el_k_save, kpoints.weights, window)
        output["transport_σlist"] = σlist
    end

    output
end
