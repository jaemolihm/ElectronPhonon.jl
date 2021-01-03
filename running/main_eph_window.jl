using PrettyPrint
# using PyPlot
using StaticArrays
using BenchmarkTools
using NPZ
using PyCall
using Profile
using LinearAlgebra
using Base.Threads
using Distributed
using Revise
BLAS.set_num_threads(1)

# Execute MPI
using MPIClusterManagers
manager = MPIManager(np=3)
addprocs(manager)

@everywhere begin
    using LinearAlgebra
    using Base.Threads
    using Revise
    push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
    using EPW
    using EPW.WanToBloch

    # Fourier transform electron-phonon matrix from (Re, Rp) -> (Re, q)
    function fourier_eph(model::EPW.ModelEPW, kpoints::EPW.Kpoints, qpoints::EPW.Kpoints;
            fourier_mode="normal",
            window=(-Inf,Inf),
            iband_min=1,
            iband_max=model.nw,)

        nw = model.nw
        nmodes = model.nmodes
        nk = kpoints.n
        nq = qpoints.n
        nband = iband_max - iband_min + 1

        elself = ElectronSelfEnergy(Float64, nband, nmodes, nk)
        phselfs = [PhononSelfEnergy(Float64, nband, nmodes, nq) for i=1:nthreads()]

        epdatas = [ElPhData(Float64, nw, nmodes, nband) for i=1:nthreads()]
        for epdata in epdatas
            epdata.iband_offset = iband_min - 1
        end

        # Compute and save electron matrix elements at k
        ek_full_save = zeros(Float64, nw, nk)
        uk_full_save = Array{ComplexF64,3}(undef, nw, nw, nk)
        vdiagk_save = zeros(Float64, 3, nband, nk)

        Threads.@threads :static for ik in 1:nk
        # for ik in 1:nk
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

        omega_save = zeros(nmodes, nq)
        omegas = zeros(nmodes)
        u_ph = zeros(ComplexF64, (nmodes, nmodes))
        dynq = zeros(ComplexF64, (nmodes, nmodes))
        epmat_q_tmp = Array{ComplexF64,3}(undef, nw*nw, nmodes, model.nr_el)

        # E-ph matrix in electron Wannier, phonon Bloch representation
        epobj_q = WannierObject(model.el_ham.nr, model.el_ham.irvec,
                    zeros(ComplexF64, (nw*nw*nmodes, model.el_ham.nr)))
        epmat_re_q_op_r = zeros(ComplexF64, nw*nw, nmodes, model.nr_el)


        for iq in 1:nq
            if mod(iq, 100) == 0 && mpi_isroot()
                @info "iq = $iq"
            end
            xq = qpoints.vectors[iq]

            # Phonon eigenvalues
            get_fourier!(dynq, model.ph_dyn, xq, mode=fourier_mode)
            omegas .= solve_eigen_ph!(u_ph, dynq, model.mass)
            omega_save[:, iq] = omegas

            # Transform e-ph matrix (Re, Rp) -> (Re, q)
            get_fourier!(epmat_q_tmp, model.epmat, xq, mode=fourier_mode)
            epmat_re_q_op_r .= 0
            @views for jmode in 1:nmodes, imode in 1:nmodes
                epmat_re_q_op_r[:, jmode, :] .+= (epmat_q_tmp[:, imode, :]
                                                .* u_ph[imode, jmode])
            end
            epmat_re_q = reshape(epmat_re_q_op_r, (nw*nw*nmodes, model.nr_el))
            update_op_r!(epobj_q, epmat_re_q)

            epmatf_wans = [zeros(ComplexF64, (nw, nw, nmodes)) for i=1:nthreads()]

            Threads.@threads :static for ik in 1:nk
            # for ik in 1:nk
                tid = Threads.threadid()
                epdata = epdatas[tid]
                epmatf_wan = epmatf_wans[tid]
                phself = phselfs[tid]

                # println("$tid $ik")
                xk = kpoints.vectors[ik]
                xkq = xk + xq

                epdata.wtk = kpoints.weights[ik]
                epdata.wtq = qpoints.weights[iq]
                epdata.omega .= omegas

                # Electron eigenstate at k. Use saved data.
                epdata.ek_full .= @view ek_full_save[:, ik]
                epdata.uk_full .= @view uk_full_save[:, :, ik]
                epdata.vdiagk .= @view vdiagk_save[:, :, ik]

                # Electron eigenstate at k+q
                get_el_eigen!(epdata, "k+q", model.el_ham, xkq, fourier_mode)

                # Set energy window, skip if no state is inside the window
                skip_k = epdata_set_window!(epdata, "k", window)
                skip_kq = epdata_set_window!(epdata, "k+q", window)
                if skip_k || skip_kq
                    continue
                end

                # Compute band velocity at k+q
                get_el_velocity_diag!(epdata, "k+q", model.el_ham_R, xkq, fourier_mode)

                # Transform e-ph matrix (Re, q) -> (k, q)
                get_fourier!(epmatf_wan, epobj_q, xk, mode=fourier_mode)

                # Rotate e-ph matrix from electron Wannier to BLoch
                apply_gauge_matrix!(epdata.ep, epmatf_wan, epdata, "k+q", "k", nmodes)

                # Compute g2 = |ep|^2 / omega
                epdata_set_g2!(epdata)

                # Now, we are done with matrix elements. All data saved in epdata.

                # Calculate physical quantities.
                efermi = 5.40 * unit_to_aru(:eV)
                temperature = 300.0 * unit_to_aru(:K)
                degaussw = 0.2 * unit_to_aru(:eV)

                compute_electron_selfen!(elself, epdata, ik;
                    efermi=efermi, temperature=temperature, smear=degaussw)
                compute_phonon_selfen!(phself, epdata, iq;
                    efermi=efermi, temperature=temperature, smear=degaussw)
            end # ik
        end # iq

        ph_imsigma = sum([phself.imsigma for phself in phselfs])

        # Average over degenerate states
        el_imsigma_avg = average_degeneracy(elself.imsigma, ek_full_save[iband_min:iband_max, :])
        ph_imsigma_avg = average_degeneracy(ph_imsigma, omega_save)

        (ek=ek_full_save, omega=omega_save,
        el_imsigma=el_imsigma_avg, ph_imsigma=ph_imsigma_avg,)
    end
end # everywhere

@everywhere begin
    folder = "/home/jmlim/julia_epw/silicon_nk6"
    window = (-Inf, Inf)

    folder = "/home/jmlim/julia_epw/silicon_nk6_window"
    window_max = 6.4 * unit_to_aru(:eV)
    window_min = 4.4 * unit_to_aru(:eV)
    window = (window_min, window_max)
end

@mpi_do manager begin
    using MPI
    using NPZ
    using EPW
    import EPW: mpi_split_iterator, mpi_bcast, mpi_gather, mpi_sum!
    world_comm = EPW.mpi_world_comm()

    # model = load_model(folder)
    model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")

    nkf = [12, 12, 12]
    nqf = [10, 10, 10]

    # Do not distribute k points
    kpoints, ib_min, ib_max = filter_kpoints_grid(nkf..., model.nw, model.el_ham, window)
    nband = ib_max - ib_min + 1

    # Distribute q points
    qpoints = generate_kvec_grid(nqf..., world_comm)
    qpoints = filter_qpoints(qpoints, kpoints, model.nw, model.el_ham, window)

    # Electron-phonon coupling
    @time output = fourier_eph(model, kpoints, qpoints,
        fourier_mode="gridopt",
        window=window,
        iband_min=ib_min,
        iband_max=ib_max,
    )

    ek_all = output.ek
    omega_all = mpi_gather(output.omega, world_comm)
    mpi_sum!(output.el_imsigma, world_comm)
    el_imsigma_all = output.el_imsigma
    ph_imsigma_all = mpi_gather(output.ph_imsigma, world_comm)

    if mpi_isroot()
        ph_imsigma_all .*= 2 # Spin factor
        npzwrite(joinpath(folder, "eig_kk.npy"), ek_all[ib_min:ib_max, :])
        npzwrite(joinpath(folder, "eig_phonon.npy"), omega_all)
        npzwrite(joinpath(folder, "imsigma_el.npy"), el_imsigma_all)
        npzwrite(joinpath(folder, "imsigma_ph.npy"), ph_imsigma_all)
    end
end

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

Profile.clear()
@profile output = fourier_eph(model, kpoints, qpoints, fourier_mode="gridopt",
    window=window, iband_min=ib_min, iband_max=ib_max,)
@profile fourier_eph(model, kpoints, qpoints, "gridopt")
@profile fourier_eph(model, kpoints, qpoints, "normal")
@profile fourier_eph(model, kk, qq, "gridopt")
Juno.profiler()
