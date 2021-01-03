using PrettyPrint
# using PyPlot
# using FortranFiles
using StaticArrays
using BenchmarkTools
using NPZ
using PyCall
using Profile
# using Parameters
using LinearAlgebra
using Base.Threads
using Revise
BLAS.set_num_threads(1)

push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
using EPW
using EPW.WanToBloch

function get_eigenvalues_el(model::EPW.ModelEPW, kpoints::EPW.Kpoints,
        fourier_mode="normal")
    nw = model.nw
    nk = kpoints.n
    eigenvalues = zeros(nw, nk)
    hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]
    phases = [zeros(ComplexF64, model.el_ham.nr) for i=1:nthreads()]

    @threads :static for ik in 1:nk
        xk = kpoints.vectors[ik]
        phase = phases[threadid()]
        hk = hks[threadid()]

        # # v1: Using phase argument: good if phase is reused for many operators
        # @inbounds for (ir, r) in enumerate(model.el_ham.irvec)
        #     phase[ir] = cis(dot(r, 2pi*xk))
        # end
        # get_fourier!(hk, model.el_ham, xk, phase, mode=fourier_mode)
        # eigenvalues[:, ik] = solve_eigen_el_valueonly!(hk)

        # # v2: Not using phase argument
        # get_fourier!(hk, model.el_ham, xk, mode=fourier_mode)
        # eigenvalues[:, ik] = solve_eigen_el_valueonly!(hk)

        # v3: Wrapper
        @views get_el_eigen_valueonly!(eigenvalues[:, ik], nw, model.el_ham, xk, fourier_mode)
    end
    return eigenvalues
end

function get_eigenvalues_ph(model::EPW.ModelEPW, kpoints::EPW.Kpoints,
        fourier_mode::String="normal")
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
        qpoints::EPW.Kpoints, fourier_mode::String="normal")
    nw = model.nw
    nmodes = model.nmodes

    nk = kpoints.n
    nq = qpoints.n
    wtk = ones(Float64, nk) / nk
    wtq = ones(Float64, nq) / nq

    elself = ElectronSelfEnergy(Float64, nw, nmodes, nk)
    phselfs = [PhononSelfEnergy(Float64, nw, nmodes, nq) for i=1:nthreads()]

    epdatas = [ElPhData(Float64, nw, nmodes) for i=1:nthreads()]

    # Compute electron eigenvectors at k
    ek_save = zeros(Float64, nw, nk)
    uk_save = Array{ComplexF64,3}(undef, nw, nw, nk)

    @views @threads :static for ik in 1:nk
    # for ik in 1:nk
        xk = kpoints.vectors[ik]
        get_el_eigen!(ek_save[:, ik], uk_save[:, :, ik], nw, model.el_ham, xk, fourier_mode)
    end # ik

    omega_save = zeros(nmodes, nq)
    omegas = zeros(nmodes)
    u_ph = zeros(ComplexF64, (nmodes, nmodes))

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epobj_eRpq = WannierObject(model.el_ham.nr, model.el_ham.irvec,
                zeros(ComplexF64, (nw*nw*nmodes, model.el_ham.nr)))

    for iq in 1:nq
        if mod(iq, 10) == 0
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

            epdata.wtq = wtq[iq]
            epdata.wtk = wtk[ik]
            epdata.omega .= omegas

            # Use saved data for electron eigenstate at k.
            epdata.ek_full .= @view ek_save[:, ik]
            epdata.uk_full .= @view uk_save[:, :, ik]

            get_el_eigen!(epdata, "k+q", model.el_ham, xkq, fourier_mode)

            # Set energy window
            skip_k = epdata_set_window!(epdata, "k")
            skip_kq = epdata_set_window!(epdata, "k+q")
            if skip_k || skip_kq
                continue
            end

            get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk, fourier_mode)

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

    ph_imsigma = sum([phself.imsigma for phself in phselfs])

    # Average over degenerate states
    imsigma_avg = average_degeneracy(elself.imsigma, ek_save)

    npzwrite(joinpath(folder, "eig_kk.npy"), ek_save)
    npzwrite(joinpath(folder, "eig_phonon.npy"), omega_save)
    npzwrite(joinpath(folder, "imsigma_el.npy"), imsigma_avg)
end

folder = "/home/jmlim/julia_epw/silicon_nk2"
folder = "/home/jmlim/julia_epw/silicon_nk6"
folder = "/home/jmlim/julia_epw/silicon_nk6_window"
model = load_model(folder)
model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")

kpoints = generate_kvec_grid(10, 10, 10)
qpoints = generate_kvec_grid(5, 5, 5)

qpoints_mini = generate_kvec_grid(1, 1, 1)
kpoints_mini = generate_kvec_grid(1, 1, 10)

window_max = 6.7 * unit_to_aru(:eV)
window_min = 5.5 * unit_to_aru(:eV)
window = (window_min, window_max)
window = (-Inf, Inf)

kpoints = filter_kpoints_grid(10, 10, 10, model.nw, model.el_ham, window)

# Test electron and phonon eigenvalues
eig_kk = get_eigenvalues_el(model, kpoints, "gridopt")
eig_phonon = get_eigenvalues_ph(model, qpoints, "gridopt")

npzwrite(joinpath(folder, "eig_kk.npy"), eig_kk)
npzwrite(joinpath(folder, "eig_phonon.npy"), eig_phonon)

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

# Test electron-phonon coupling
fourier_eph(model, kpoints_mini, qpoints_mini, "gridopt")
@time fourier_eph(model, kpoints, qpoints, "gridopt")
@time fourier_eph(model, kpoints, qpoints, "normal")

@code_warntype fourier_eph(model, kpoints, qpoints, "gridopt")

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

Profile.clear()
@profile fourier_eph(model, kpoints, qpoints, "gridopt")
@profile fourier_eph(model, kpoints, qpoints, "normal")
@profile fourier_eph(model, kk, qq, "gridopt")
Juno.profiler()
