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
        dynq[:, :] .*= sqrt.(model.mass)
        dynq[:, :] .*= sqrt.(model.mass)'
        if model.use_polar_dipole
            dynmat_dipole!(dynq, xk, model.recip_lattice, model.volume, model.atom_pos, model.alat, model.polar_phonon, 1)
        end
        dynq[:, :] ./= sqrt.(model.mass)
        dynq[:, :] ./= sqrt.(model.mass)'
        eigenvalues[:, ik] = solve_eigen_ph_valueonly!(dynq)
    end
    return eigenvalues
end

folder = "/home/jmlim/julia_epw/silicon_nk2"
folder = "/home/jmlim/julia_epw/silicon_nk6"
folder = "/home/jmlim/julia_epw/silicon_nk6_window"
folder = "/home/jmlim/julia_epw/cubicBN_nk6"
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

# Test electron and phonon eigenvalues
eig_kk = get_eigenvalues_el(model, kpoints, "gridopt")
eig_phonon = get_eigenvalues_ph(model, qpoints, "gridopt")

npzwrite(joinpath(folder, "eig_kk.npy"), eig_kk)
npzwrite(joinpath(folder, "eig_phonon.npy"), eig_phonon)

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

nklist = (10, 10, 10)
nqlist = (5, 5, 5)

elself_params = ElectronSelfEnergyParams(
    μ = 6.10 * unit_to_aru(:eV),
    Tlist = [300.0 * unit_to_aru(:K)],
    smearing = 500.0 * unit_to_aru(:meV)
)

phself_params = PhononSelfEnergyParams(
    μ = 6.10 * unit_to_aru(:eV),
    Tlist = [300.0 * unit_to_aru(:K)],
    smearing = 500.0 * unit_to_aru(:meV)
)

# Run electron-phonon coupling
@time output = EPW.run_eph_outer_loop_q(
    model, nklist, nqlist,
    fourier_mode="gridopt",
    window=window,
    elself_params=elself_params,
    phself_params=phself_params,
)

npzwrite(joinpath(folder, "eig_kk.npy"), output["ek"])
npzwrite(joinpath(folder, "eig_phonon.npy"), output["omega"])
npzwrite(joinpath(folder, "imsigma_el.npy"), output["elself_imsigma"])
npzwrite(joinpath(folder, "imsigma_ph.npy"), output["phself_imsigma"])

testscript = joinpath(folder, "test.py")
py"exec(open($testscript).read())"

Profile.clear()
@profile output = EPW.run_eph_outer_loop_q(
    model, nklist, nqlist,
    fourier_mode="gridopt",
    window=window,
    elself_params=elself_params,
)
Juno.profiler()
