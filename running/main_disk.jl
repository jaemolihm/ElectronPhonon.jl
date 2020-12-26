using PrettyPrint
# using PyPlot
using FortranFiles
using StaticArrays
using BenchmarkTools
using NPZ
using PyCall
using Profile
using Parameters
using LinearAlgebra
using SharedArrays
using Base.Threads
using Revise
BLAS.set_num_threads(1)

push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
using EPW
using EPW.Diagonalize

"Data in coarse real-space grid, read from EPW"
Base.@kwdef struct ModelEPW1{WannType <: AbstractWannierObject{Float64}}
    nw::Int
    nmodes::Int
    mass::Array{Float64,1}

    nr_el::Int

    el_ham::WannierObject{Float64}
    # TODO: Use Hermiticity of hk

    ph_dyn::WannierObject{Float64}
    # TODO: Use real-valuedness of dyn_r
    # TODO: Use Hermiticity of dyn_q

    "electron-phonon coupling matrix in electron and phonon Wannier representation"
    epmat::WannType
end
ModelEPW = ModelEPW1 # WARNING, only while developing.

"""
epmat_on_disk
    If true, write epmat to file and read at each get_fourier! call.
    If false, load epmat to memory.
tmpdir
    Directory to write temporary binary files for epmat_on_disk=false calse.
"""
function load_epwdata(folder::String, epmat_on_disk::Bool=false, tmpdir=nothing)
    if epmat_on_disk && tmpdir == nothing
        error("If epmat_on_disk is true, tmpdir must be provided.")
    end
    # Read binary data written by EPW and create ModelEPW object
    f = FortranFile(joinpath(folder, "epw_data_julia.bin"), "r")
    nw = convert(Int, read(f, Int32))
    nmodes = convert(Int, read(f, Int32))
    mass = read(f, (Float64, nmodes))

    # Electron Hamiltonian
    nrr_k = convert(Int, read(f, Int32))
    irvec_k = convert.(Int, read(f, (Int32, 3, nrr_k)))
    ham_r = read(f, (ComplexF64, nw, nw, nrr_k))

    # Phonon dynamical matrix
    nr_ph = convert(Int, read(f, Int32))
    irvec_ph = convert.(Int, read(f, (Int32, 3, nr_ph)))
    dyn_r_real = read(f, (Float64, nmodes, nmodes, nr_ph))
    dyn_r = complex(dyn_r_real)

    # Electron-phonon matrix
    nr_ep = convert(Int, read(f, Int32))
    irvec_ep = convert.(Int, read(f, (Int32, 3, nr_ep)))

    nr_el = nrr_k
    irvec_el = irvec_k

    # Transform R vectors from matrix to vector of StaticVectors
    irvec_el = reinterpret(Vec3{Int}, irvec_el)[:]
    irvec_ph = reinterpret(Vec3{Int}, irvec_ph)[:]
    irvec_ep = reinterpret(Vec3{Int}, irvec_ep)[:]

    # Sort R vectors using R[3], and then R[2], and then R[1].
    # This is useful for gridopt.
    ind_el = sortperm(irvec_el, by=x->reverse(x))
    ind_ph = sortperm(irvec_ph, by=x->reverse(x))
    ind_ep = sortperm(irvec_ep, by=x->reverse(x))

    irvec_el = irvec_el[ind_el]
    irvec_ph = irvec_ph[ind_ph]
    irvec_ep = irvec_ep[ind_ep]

    ham_r = ham_r[:, :, ind_el]
    dyn_r = dyn_r[:, :, ind_ph]

    # Electron-phonon coupling
    if epmat_on_disk
        empat_filename = "tmp_epmat.bin"

        # epmat stays on disk. Read each epmat for each ir and write to file.
        size_column =  sizeof(ComplexF64)*nw^2*nrr_k*nmodes # Size of one column of data

        # Open file to write
        filename = joinpath(tmpdir, empat_filename)
        rm(filename, force=true)
        fw = open(filename, "w")

        for ir in 1:nr_ep
            # Read from Fortran file
            epmat_re_rp_read_ir = read(f, (ComplexF64, nw, nw, nr_el, nmodes))
            epmat_re_rp_ir = permutedims(epmat_re_rp_read_ir, [1, 2, 4, 3])

            # Shuffle irvec_el
            epmat_re_rp_ir = epmat_re_rp_ir[:, :, :, ind_el]

            # Write to file, as index epmat[:, :, :, :, ir_new]
            ir_new = findfirst(ind_ep .== ir) # Index of R point after sorting
            seek(fw, size_column * (ir_new - 1))
            write(fw, epmat_re_rp_ir)
        end
        close(fw)
    else
        # Read epmat to memory
        epmat_re_rp = zeros(ComplexF64, nw, nw, nmodes, nr_el, nr_ep)
        for ir in 1:nr_ep
            epmat_re_rp_read_ir = read(f, (ComplexF64, nw, nw, nr_el, nmodes))
            epmat_re_rp[:,:,:,:,ir] = permutedims(epmat_re_rp_read_ir, [1, 2, 4, 3])
        end
        epmat_re_rp = epmat_re_rp[:, :, :, ind_el, ind_ep]
    end
    close(f)

    # Reshape real-space matrix elements into 2-dimensional matrices
    # First index: all other indices
    # Second index: R vectors
    ham_r = reshape(ham_r, (nw*nw, nrr_k))
    dyn_r = reshape(dyn_r, (nmodes*nmodes, nr_ph))
    el_ham = WannierObject(nr_el, irvec_el, ham_r)
    ph_dyn = WannierObject(nr_ph, irvec_ph, dyn_r)

    if epmat_on_disk
        epmat = DiskWannierObject1(Float64, "epmat", nr_ep, irvec_ep, nw*nw*nmodes*nr_el,
            tmpdir, empat_filename)
    else
        epmat_re_rp = reshape(epmat_re_rp, (nw*nw*nmodes*nrr_k, nr_ep))
        epmat = WannierObject(nr_ep, irvec_ep, epmat_re_rp)
    end

    model = ModelEPW(nw=nw, nmodes=nmodes, mass=mass,
        nr_el=nr_el,
        el_ham=el_ham, ph_dyn=ph_dyn, epmat=epmat
    )

    model
end

folder = "/home/jmlim/julia_epw/silicon_nk2"
folder = "/home/jmlim/julia_epw/silicon_nk6"
model = load_epwdata(folder)

model = load_epwdata(folder, true, "/home/jmlim/julia_epw/tmp")

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
