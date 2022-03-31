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
            iband_max=model.nw,
            temperature
        )

        nw = model.nw
        nmodes = model.nmodes
        nk = kpoints.n
        nq = qpoints.n
        nband = iband_max - iband_min + 1

        elself = ElectronSelfEnergy(Float64, iband_min:iband_max, nmodes, nk)
        phselfs = [PhononSelfEnergy(Float64, nband, nmodes, nq) for i=1:nthreads()]

        epdatas = [ElPhData(nw, nmodes; nband) for i=1:nthreads()]
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
            epdata_set_window!(epdata, "k", window)
            get_el_velocity_diag!(epdata, "k", model.el_ham_R, xk, fourier_mode)

            # Save matrix elements at k for reusing
            ek_full_save[:, ik] .= epdata.ek_full
            uk_full_save[:, :, ik] .= epdata.uk_full
            vdiagk_save[:, :, ik] .= epdata.vdiagk
        end # ik

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
                phself = phselfs[tid]

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
                epdata_set_window!(c, "k", window)
                epdata_set_window!(epdata, "k+q", window)
                length(epdata.el_k.rng) == 0 && continue
                length(epdata.el_kq.rng) == 0 && continue

                get_el_velocity_diag!(epdata, "k+q", model.el_ham_R, xkq, fourier_mode)
                get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk, fourier_mode)

                # Now, we are done with matrix elements. All data saved in epdata.

                # Calculate physical quantities.
                efermi = 5.40 * unit_to_aru(:eV)
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

    model = load_model(folder)
    # model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")

    nkf = [12, 12, 12]
    nqf = [10, 10, 10]

    # Do not distribute k points
    kpoints, ib_min, ib_max = filter_kpoints_grid(nkf..., model.nw, model.el_ham, window)

    # Distribute q points
    qpoints = generate_kvec_grid(nqf..., world_comm)
    qpoints = filter_qpoints(qpoints, kpoints, model.nw, model.el_ham, window)

    # Electron-phonon coupling
    @time output = fourier_eph(model, kpoints, qpoints,
        fourier_mode="gridopt",
        window=window,
        iband_min=ib_min,
        iband_max=ib_max,
        temperature=300 * unit_to_aru(:K)
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
