using PrettyPrint
# using PyPlot
using StaticArrays
using BenchmarkTools
using NPZ
using PyCall
using Profile
using LinearAlgebra
using SharedArrays
using Base.Threads
using MPIClusterManagers
using Distributed
using Revise
BLAS.set_num_threads(1)

# Execute MPI
manager=MPIManager(np=4)
addprocs(manager)

@everywhere begin
    using Revise
    push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
    using EPW
    using EPW.Diagonalize
end

folder = "/home/jmlim/julia_epw/silicon_nk2"
folder = "/home/jmlim/julia_epw/silicon_nk6"
# epmat(Re, Rp) in memory
model = load_model_from_epw(folder)
# epmat(Re, Rp) in disk
model = load_model_from_epw(folder, true, "/home/jmlim/julia_epw/tmp")

# pprintln(model)

nkf = [10, 10, 10]
kvecs = generate_kvec_grid(nkf)

nqf = [5, 5, 5]
qvecs = generate_kvec_grid(nqf)

qvecs_mini = generate_kvec_grid([1, 1, 1])
kvecs_mini = generate_kvec_grid([1, 1, 4])

function get_eigenvalues_el(model, kvecs, fourier_mode::String="normal")
    nw = model.nw
    nk = length(kvecs)
    eigenvalues = zeros(nw, nk)
    hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]
    phases = [zeros(ComplexF64, model.el_ham.nr) for i=1:nthreads()]

    @threads :static for ik in 1:nk
        xk = kvecs[ik]
        phase = phases[threadid()]
        hk = hks[threadid()]

        # v1: Using phase argument: good if phase is reused for many operators
        @inbounds for (ir, r) in enumerate(model.el_ham.irvec)
            phase[ir] = cis(dot(r, 2pi*xk))
        end
        get_fourier!(hk, model.el_ham, xk, phase, mode=fourier_mode)

        # # v2: Not using phase argument
        # get_fourier!(hk, model.el_ham, xk, mode=fourier_mode)

        eigenvalues[:, ik] = solve_eigen_el_valueonly!(hk)
    end
    return eigenvalues
end

function get_eigenvalues_ph(model, kvecs, fourier_mode::String="normal")
    nmodes = model.nmodes
    nk = length(kvecs)
    eigenvalues = zeros(nmodes, nk)
    dynq = zeros(ComplexF64, (nmodes, nmodes))

    for ik in 1:nk
        xk = kvecs[ik]
        get_fourier!(dynq, model.ph_dyn, xk, mode=fourier_mode)
        eigenvalues[:, ik] = solve_eigen_ph_valueonly!(dynq)
    end
    return eigenvalues
end

eig_kk = get_eigenvalues_el(model, kvecs, "gridopt")
npzwrite(joinpath(folder, "eig_kk.npy"), eig_kk)

eig_phonon = get_eigenvalues_ph(model, qvecs, "gridopt")
npzwrite(joinpath(folder, "eig_phonon.npy"), eig_phonon)

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

# Fourier transform electron-phonon matrix from (Re, Rp) -> (Re, q)
function fourier_eph(model::ModelEPW, kvecs::Vector{Vec3{Float64}},
        qvecs::Vector{Vec3{Float64}}, fourier_mode::String="normal")
    nw = model.nw
    nmodes = model.nmodes

    nk = length(kvecs)
    nq = length(qvecs)
    wtk = ones(Float64, nk) / nk
    wtq = ones(Float64, nq) / nq

    elself = ElectronSelfEnergy(Float64, nw, nmodes, nk)
    phself = PhononSelfEnergy(Float64, nw, nmodes, nq)

    epdatas = [ElPhData(Float64, nw, nmodes) for i=1:nthreads()]

    # Compute electron eigenvectors at k
    ek_save = zeros(Float64, nw, nk)
    uk_save = Array{ComplexF64,3}(undef, nw, nw, nk)
    hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]

    @threads :static for ik in 1:nk
    # for ik in 1:nk
        hk = hks[Threads.threadid()]
        xk = kvecs[ik]
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
        if mod(iq, 10) == 0
            @info "iq = $iq"
        end
        xq = qvecs[iq]

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
        nocc_qs = [zeros(Float64, nmodes,) for i=1:nthreads()]
        focc_kqs = [zeros(Float64, nw,) for i=1:nthreads()]

        Threads.@threads :static for ik in 1:nk
        # for ik in 1:nk
            tid = Threads.threadid()
            epdata = epdatas[tid]
            epmatf_wan = epmatf_wans[tid]

            # println("$tid $ik")
            xk = kvecs[ik]
            xkq = xk + xq

            epdata.wtq = wtq[iq]
            epdata.wtk = wtk[ik]
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

    npzwrite(joinpath(folder, "eig_kk.npy"), ek_save)
    npzwrite(joinpath(folder, "eig_phonon.npy"), omega_save)
    npzwrite(joinpath(folder, "imsigma_el.npy"), el_imsigma_avg)
    npzwrite(joinpath(folder, "imsigma_ph.npy"), ph_imsigma_avg)
end

fourier_eph(model, kvecs_mini, qvecs_mini, "gridopt")
@time fourier_eph(model, kvecs, qvecs, "gridopt")
@time fourier_eph(model, kvecs, qvecs, "normal")

@code_warntype fourier_eph(model, kvecs, qvecs, "gridopt")

kk = generate_kvec_grid([50, 50, 50])
qq = generate_kvec_grid([1, 1, 1])
@time fourier_eph(model, kk, qq, "gridopt")
@time fourier_eph(model, kk, qq, "normal")
@btime fourier_eph($model, $kk, $qq, "gridopt")

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

Profile.clear()
@profile fourier_eph(model, kvecs, qvecs, "gridopt")
@profile fourier_eph(model, kvecs, qvecs, "normal")
@profile fourier_eph(model, kk, qq, "gridopt")
Juno.profiler()







@mpi_do manager begin
    using MPI
    using NPZ
    using LinearAlgebra
    using Base.Threads
    using EPW
    import EPW: mpi_split_iterator, mpi_bcast, mpi_gather
    world_comm = MPI.COMM_WORLD

    # Read model from file
    folder = "/home/jmlim/julia_epw/silicon_nk2"
    if mpi_isroot(world_comm)
        model = load_model_from_epw(folder)
    else
        model = nothing
    end
    model = mpi_bcast(model, world_comm)

    function get_eigenvalues_el(model, kvecs, fourier_mode::String="normal")
        nw = model.nw
        nk = length(kvecs)
        eigenvalues = zeros(nw, nk)
        hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]
        phases = [zeros(ComplexF64, model.el_ham.nr) for i=1:nthreads()]

        Threads.@threads :static for ik in 1:nk
            xk = kvecs[ik]
            phase = phases[threadid()]
            hk = hks[threadid()]

            # v1: Using phase argument: good if phase is reused for many operators
            @inbounds for (ir, r) in enumerate(model.el_ham.irvec)
                phase[ir] = cis(dot(r, 2pi*xk))
            end
            get_fourier!(hk, model.el_ham, xk, phase, mode=fourier_mode)

            # # v2: Not using phase argument
            # get_fourier!(hk, model.el_ham, xk, mode=fourier_mode)

            eigenvalues[:, ik] = solve_eigen_el_valueonly!(hk)
        end
        return eigenvalues
    end

    function get_eigenvalues_ph(model, kvecs, fourier_mode::String="normal")
        nmodes = model.nmodes
        nk = length(kvecs)
        eigenvalues = zeros(nmodes, nk)
        dynq = zeros(ComplexF64, (nmodes, nmodes))

        for ik in 1:nk
            xk = kvecs[ik]
            get_fourier!(dynq, model.ph_dyn, xk, mode=fourier_mode)
            eigenvalues[:, ik] = solve_eigen_ph_valueonly!(dynq)
        end
        return eigenvalues
    end

    # Distribute k vectors
    nkf = [10, 10, 10]
    range = mpi_split_iterator(1:prod(nkf), world_comm)
    kvecs = generate_kvec_grid(nkf..., range)

    nqf = [5, 5, 5]
    range = mpi_split_iterator(1:prod(nqf), world_comm)
    qvecs = generate_kvec_grid(nqf..., range)

    eig_kk = get_eigenvalues_el(model, kvecs, "gridopt")
    eig_kk_all = mpi_gather(eig_kk, world_comm)
    if mpi_isroot()
        npzwrite(joinpath(folder, "eig_kk.npy"), eig_kk_all)
    end

    eig_phonon = get_eigenvalues_ph(model, qvecs, "gridopt")
    eig_phonon_all = mpi_gather(eig_phonon, world_comm)
    if mpi_isroot()
        npzwrite(joinpath(folder, "eig_phonon.npy"), eig_phonon_all)
    end
end

folder = "/home/jmlim/julia_epw/silicon_nk2"
testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"
