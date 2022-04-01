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
    using SharedArrays
    using Base.Threads
    using Revise
    push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
    using EPW
    using EPW.Diagonalize

    function get_eigenvalues_el(model::EPW.ModelEPW, kpoints::EPW.Kpoints, fourier_mode="normal")
        nw = model.nw
        nk = kpoints.n
        eigenvalues = zeros(nw, nk)
        hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]
        phases = [zeros(ComplexF64, model.el_ham.nr) for i=1:nthreads()]

        Threads.@threads :static for ik in 1:nk
            xk = kpoints.vectors[ik]
            phase = phases[threadid()]
            hk = hks[threadid()]

            # v1: Using phase argument: good if phase is reused for many operators
            @inbounds for (ir, r) in enumerate(model.el_ham.irvec)
                phase[ir] = cispi(2 * dot(r, xk))
            end
            get_fourier!(hk, model.el_ham, xk, phase, mode=fourier_mode)

            # # v2: Not using phase argument
            # get_fourier!(hk, model.el_ham, xk, mode=fourier_mode)

            eigenvalues[:, ik] = solve_eigen_el_valueonly!(hk)
        end
        return eigenvalues
    end

    function get_eigenvalues_ph(model::EPW.ModelEPW, kpoints::EPW.Kpoints, fourier_mode="normal")
        nmodes = model.nmodes
        nk = kpoints.n
        eigenvalues = zeros(nmodes, nk)
        dynq = zeros(ComplexF64, (nmodes, nmodes))

        for ik in 1:nk
            xk = kpoints.vectors[ik]
            get_fourier!(dynq, model.ph_dyn, xk, mode=fourier_mode)
            eigenvalues[:, ik] = solve_eigen_ph_valueonly!(dynq)
        end
        return eigenvalues
    end

    # Fourier transform electron-phonon matrix from (Re, Rp) -> (Re, q)
    function fourier_eph(model::EPW.ModelEPW, kpoints::EPW.Kpoints,
            qpoints::EPW.Kpoints, fourier_mode="normal")
        nw = model.nw
        nmodes = model.nmodes
        nk = kpoints.n
        nq = qpoints.n

        elself = ElectronSelfEnergy{Float64}(1:nw, nk)
        phself = PhononSelfEnergy{Float64}(nmodes, nq)

        epdatas = [ElPhData(nw, nmodes) for i=1:nthreads()]

        # Compute electron eigenvectors at k
        ek_save = zeros(Float64, nw, nk)
        uk_save = Array{ComplexF64,3}(undef, nw, nw, nk)
        hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]

        Threads.@threads :static for ik in 1:nk
        # for ik in 1:nk
            hk = hks[Threads.threadid()]
            xk = kpoints.vectors[ik]
            get_fourier!(hk, model.el_ham, xk, mode=fourier_mode)
            @views ek_save[:, ik] = solve_eigen_el!(uk_save[:, :, ik], hk)
        end # ik

        omega_save = zeros(nmodes, nq)
        omegas = zeros(nmodes)
        u_ph = zeros(ComplexF64, (nmodes, nmodes))
        dynq = zeros(ComplexF64, (nmodes, nmodes))
        epmat_q_tmp = Array{ComplexF64,3}(undef, nw*nw, nmodes, model.nr_el)

        # E-ph matrix in electron Wannier, phonon Bloch representation
        epobj_q = WannierObject(model.el_ham.nr, model.el_ham.irvec,
                    zeros(ComplexF64, (nw*nw*nmodes, model.el_ham.nr)))

        for iq in 1:nq
            if mod(iq, 10) == 0 && mpi_isroot()
                @info "iq = $iq"
            end
            xq = qpoints.vectors[iq]

            # Phonon eigenvalues
            get_fourier!(dynq, model.ph_dyn, xq, mode=fourier_mode)
            omegas .= solve_eigen_ph!(u_ph, dynq, model.mass)
            omega_save[:, iq] = omegas

            # Transform e-ph matrix (Re, Rp) -> (Re, q)
            get_fourier!(epmat_q_tmp, model.epmat, xq, mode=fourier_mode)
            epmat_re_q_3 = zero(epmat_q_tmp)
            @views for jmode in 1:nmodes, imode in 1:nmodes
                epmat_re_q_3[:, jmode, :] .+= (epmat_q_tmp[:, imode, :]
                                             .* u_ph[imode, jmode])
            end
            epmat_re_q = reshape(epmat_re_q_3, (nw*nw*nmodes, model.nr_el))
            update_op_r!(epobj_q, epmat_re_q)

            epmatf_wans = [zeros(ComplexF64, (nw, nw, nmodes)) for i=1:nthreads()]

            Threads.@threads :static for ik in 1:nk
            # for ik in 1:nk
                tid = Threads.threadid()
                epdata = epdatas[tid]
                epmatf_wan = epmatf_wans[tid]

                # println("$tid $ik")
                xk = kpoints.vectors[ik]
                xkq = xk + xq

                epdata.wtk = kpoints.weights[ik]
                epdata.wtq = qpoints.weights[iq]
                epdata.omega .= omegas

                # Electron eigenstate at k. Use saved data.
                epdata.ek_full .= @view ek_save[:, ik]
                epdata.uk_full .= @view uk_save[:, :, ik]

                # Electron eigenstate at k+q
                hkq = epdata.buffer
                get_fourier!(hkq, model.el_ham, xkq, mode=fourier_mode)
                epdata.ekq_full .= solve_eigen_el!(epdata.ukq_full, hkq)

                # Transform e-ph matrix (Re, q) -> (k, q)
                get_fourier!(epmatf_wan, epobj_q, xk, mode=fourier_mode)

                # Rotate e-ph matrix from electron Wannier to BLoch
                apply_gauge_matrix!(epdata.ep, epmatf_wan, epdata, "k+q", "k", nmodes)

                # Compute g2 = |ep|^2 / omega
                set_g2!(epdata)

                # Now, we are done with matrix elements. All data saved in epdata.

                # Now calculate physical quantities.
                efermi = 6.10 * unit_to_aru(:eV)
                temperature = 300.0 * unit_to_aru(:K)
                degaussw = 0.50 * unit_to_aru(:eV)

                compute_electron_selfen!(elself, epdata, ik;
                    efermi=efermi, temperature=temperature, smear=degaussw)
                compute_phonon_selfen!(phself, epdata, iq;
                    efermi=efermi, temperature=temperature, smear=degaussw)
            end # ik
        end # iq

        # Average over degenerate states
        el_imsigma_avg = average_degeneracy(elself.imsigma, ek_save)
        ph_imsigma_avg = average_degeneracy(phself.imsigma, omega_save)

        (ek=ek_save, omega=omega_save,
        el_imsigma=el_imsigma_avg, ph_imsigma=ph_imsigma_avg,)
    end
end # everywhere

@everywhere folder = "/home/jmlim/julia_epw/silicon_nk6"

@mpi_do manager begin
    using MPI
    using NPZ
    import EPW: mpi_split_iterator, mpi_bcast, mpi_gather
    world_comm = EPW.mpi_world_comm()

    # Read model from file
    # model = load_model(folder)
    model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")

    # Electron eigenvalues
    # Distribute k points
    nkf = [10, 10, 10]
    range = mpi_split_iterator(1:prod(nkf), world_comm)
    kpoints = generate_kvec_grid(nkf..., range)
    # Compute eigenvalues
    @time eig_kk = get_eigenvalues_el(model, kpoints, "gridopt")
    # Gather and write eigenvalues to file
    eig_kk_all = mpi_gather(eig_kk, world_comm)
    if mpi_isroot()
        npzwrite(joinpath(folder, "eig_kk.npy"), eig_kk_all)
    end

    # Phonon eigenvalues
    # Distribute q points
    nqf = [5, 5, 5]
    range = mpi_split_iterator(1:prod(nqf), world_comm)
    qpoints = generate_kvec_grid(nqf..., range)
    # Compute eigenvalues
    @time eig_phonon = get_eigenvalues_ph(model, qpoints, "gridopt")
    # Gather and write eigenvalues to file
    eig_phonon_all = mpi_gather(eig_phonon, world_comm)
    if mpi_isroot()
        npzwrite(joinpath(folder, "eig_phonon.npy"), eig_phonon_all)
    end
end

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

@mpi_do manager begin
    using MPI
    using NPZ
    using EPW
    import EPW: mpi_split_iterator, mpi_bcast, mpi_gather, mpi_sum!
    world_comm = EPW.mpi_world_comm()

    # model = load_model(folder)
    model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")

    # Do not distribute k points
    nkf = [10, 10, 10]
    kvecs = generate_kvec_grid(nkf...)

    # Distribute q points
    nqf = [5, 5, 5]
    range = mpi_split_iterator(1:prod(nqf), world_comm)
    qvecs = generate_kvec_grid(nqf..., range)

    # Electron-phonon coupling
    @time output = fourier_eph(model, kvecs, qvecs, "gridopt")

    ek_all = output.ek
    omega_all = mpi_gather(output.omega, world_comm)
    mpi_sum!(output.el_imsigma, world_comm)
    el_imsigma_all = output.el_imsigma
    ph_imsigma_all = mpi_gather(output.ph_imsigma, world_comm)

    if mpi_isroot()
        npzwrite(joinpath(folder, "eig_kk.npy"), ek_all)
        npzwrite(joinpath(folder, "eig_phonon.npy"), omega_all)
        npzwrite(joinpath(folder, "imsigma_el.npy"), el_imsigma_all)
        npzwrite(joinpath(folder, "imsigma_ph.npy"), ph_imsigma_all)
    end
end

model = load_model(folder)
model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")
kpoints = generate_kvec_grid(10, 10, 10)
qpoints = generate_kvec_grid(5, 5, 5)
@everywhere EPW.reset_timer!(EPW.timer)
@time output = fourier_eph(model, kpoints, qpoints, "gridopt")
EPW.print_timer(EPW.timer)
@spawnat 2 EPW.print_timer(EPW.timer)


npzwrite(joinpath(folder, "eig_kk.npy"), output.ek)
npzwrite(joinpath(folder, "eig_phonon.npy"), output.omega)
npzwrite(joinpath(folder, "imsigma_el.npy"), output.el_imsigma)
npzwrite(joinpath(folder, "imsigma_ph.npy"), output.ph_imsigma)

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

Profile.clear()
@profile fourier_eph(model, kvecs, qvecs, "gridopt")
@profile fourier_eph(model, kvecs, qvecs, "normal")
@profile fourier_eph(model, kk, qq, "gridopt")
Juno.profiler()



@mpi_do manager begin
    # model = load_model(folder)

    window_max = 6.7 * unit_to_aru(:eV)
    window_min = 5.5 * unit_to_aru(:eV)
    window = (window_min, window_max)

    k = filter_kpoints_grid(10, 10, 10, model.nw, model.el_ham, window, EPW.mpi_world_comm())
    @show k.n
end
