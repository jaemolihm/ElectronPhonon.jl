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
Base.@kwdef struct ModelEPW1
    nw::Int
    nmodes::Int
    mass::Array{Float64,1}

    el_ham::WannierObject
    # TODO: Use Hermiticity of hk

    ph_dyn::WannierObject
    # TODO: Use real-valuedness of dyn_r
    # TODO: Use Hermiticity of dyn_q

    "electron-phonon coupling matrix in electron and phonon Wannier representation"
    epmat::WannierObject
end
ModelEPW = ModelEPW1 # WARNING, only while developing.

function load_epwdata(folder::String)
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
    epmat_re_rp_read = read(f, (ComplexF64, nw, nw, nrr_k, nmodes, nr_ep))
    epmat_re_rp = permutedims(epmat_re_rp_read, [1, 2, 4, 3, 5])
    close(f)

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
    epmat_re_rp = epmat_re_rp[:, :, :, ind_el, ind_ep]

    # Reshape real-space matrix elements into 2-dimensional matrices
    # First index: all other indices
    # Second index: R vectors
    ham_r = reshape(ham_r, (nw*nw, nrr_k))
    dyn_r = reshape(dyn_r, (nmodes*nmodes, nr_ph))
    epmat_re_rp = reshape(epmat_re_rp, (nw*nw*nmodes*nrr_k, nr_ep))

    el_ham = WannierObject(nr_el, irvec_el, ham_r)
    ph_dyn = WannierObject(nr_ph, irvec_ph, dyn_r)
    epmat = WannierObject(nr_ep, irvec_ep, epmat_re_rp)

    model = ModelEPW(nw, nmodes, mass, el_ham, ph_dyn, epmat)

    model
end

folder = "/home/jmlim/julia_epw/silicon_nk2"
folder = "/home/jmlim/julia_epw/silicon_nk6"
model = load_epwdata(folder)

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

        # # v1: Using phase argument: good if phase is reused for many operators
        # @inbounds for (ir, r) in enumerate(model.el_ham.irvec)
        #     phase[ir] = cis(dot(r, 2pi*xk))
        # end
        # get_fourier!(hk, model.el_ham, xk, phase, mode=fourier_mode)

        # v2: Not using phase argument
        get_fourier!(hk, model.el_ham, xk, mode=fourier_mode)

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
    wtq = ones(Float64, nq) / nq
    imsigma_save = zeros(Float64, nw, nk)

    epdatas = [initialize_elphdata(Float64, nw, nmodes) for i=1:nthreads()]

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
    epmat_q_tmp = Array{ComplexF64,3}(undef, nw*nw, nmodes, model.el_ham.nr)

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
        epmat_re_q = reshape(epmat_re_q_3, (nw*nw*nmodes, model.el_ham.nr))
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

            # Completed calculating matrix elements and saved in epdata.

            # Now calculate physical quantities.
            efermi = 6.10 * unit_to_aru(:eV)
            temperature = 300.0 * unit_to_aru(:K)
            degaussw = 0.50 * unit_to_aru(:eV)
            omega_acoustic = 6.1992E-04 * unit_to_aru(:eV)
            inv_degaussw = 1 / degaussw

            nocc_q = nocc_qs[tid]
            focc_kq = focc_kqs[tid]
            nocc_q .= occ_boson.(epdata.omega ./ temperature)
            focc_kq .= occ_fermion.((epdata.ekq_full .- efermi) ./ temperature)

            # Calculate imaginary part of electron self-energy
            for imode in 1:nmodes
                omega = epdata.omega[imode]
                if (omega < omega_acoustic)
                    continue
                end
                @views epdata.g2[:, :, imode] .= abs.(epdata.ep[:, :, imode]).^2 ./ (2 * omega)
                # TODO: Move 1/(2*omega) to where epmat_re_q is defined.

                @inbounds for ib in 1:nw, jb in 1:nw
                    delta_e1 = epdata.ek_full[ib] - (epdata.ekq_full[jb] - omega)
                    delta_e2 = epdata.ek_full[ib] - (epdata.ekq_full[jb] + omega)
                    delta1 = gaussian(delta_e1 * inv_degaussw) * inv_degaussw
                    delta2 = gaussian(delta_e2 * inv_degaussw) * inv_degaussw
                    fcoeff1 = nocc_q[imode] + focc_kq[jb]
                    fcoeff2 = nocc_q[imode] + 1.0 - focc_kq[jb]
                    imsigma_save[ib, ik] += (epdata.g2[jb, ib, imode] * wtq[iq]
                        * pi * (fcoeff1 * delta1 + fcoeff2 * delta2))
                end
            end
        end # ik
    end # iq

    # Average over degenerate states
    degeneracy_cutoff = 1.e-6
    imsigma_save_copy = copy(imsigma_save)
    Threads.@threads :static for ik in 1:nk
    # for ik in 1:nk
        for ib in 1:nw
            iblist_degen = abs.(ek_save[:, ik] .- ek_save[ib, ik]) .< degeneracy_cutoff
            imsigma_save[ib, ik] = mean(imsigma_save_copy[iblist_degen, ik])
        end # ib
    end # ik

    npzwrite(joinpath(folder, "eig_kk.npy"), ek_save)
    npzwrite(joinpath(folder, "eig_phonon.npy"), omega_save)
    npzwrite(joinpath(folder, "imsigma_el.npy"), imsigma_save)
end

fourier_eph(model, kvecs_mini, qvecs_mini, "gridopt")
@time fourier_eph(model, kvecs, qvecs, "gridopt")
@time fourier_eph(model, kvecs, qvecs, "normal")

@code_warntype fourier_eph(model, kvecs, qvecs, "gridopt")

kk = generate_kvec_grid([100, 100, 100])
qq = generate_kvec_grid([1, 1, 1])
@time fourier_eph(model, kk, qq, "gridopt")
@btime fourier_eph($model, $kk, $qq, "gridopt")

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

Profile.clear()
@profile fourier_eph(model, kvecs, qvecs, "gridopt")
@profile fourier_eph(model, kvecs, qvecs, "normal")
@profile fourier_eph(model, kk, qq, "gridopt")
Juno.profiler()
