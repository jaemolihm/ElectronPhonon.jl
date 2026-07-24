# Benchmark: electron band energies over a k-grid, CPU vs GPU.
#
# Uses the SAME batched drivers on both backends — only the backend of `op_r` differs:
#   * get_el_eigen_valueonly_batched  — eigenvalues only
#   * get_el_eigen_batched            — eigenvalues AND eigenvectors
#
# Reports eigenvalue agreement (GPU vs CPU) and the wall time of each backend.
#
# Run in an environment with both ElectronPhonon (this gpu branch) and CUDA:
#   julia --project=<env> benchmark/bench_el_eigen_gpu.jl

using ElectronPhonon
using ElectronPhonon: to_device
using CUDA
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Load the Pb model
# ---------------------------------------------------------------------------
const PB_FOLDER = "/mnt/home/jlihm/ceph/superconductivity/Pb/tutorial/1_epw/"
const PB_OUTDIR = "temp"          # holds temp/pb.epmatwp
const PB_PREFIX = "pb"

model = ElectronPhonon.load_model_from_epw_new(PB_FOLDER, PB_OUTDIR, PB_PREFIX;
                                               epmat_outer_momentum = "el")
nw = model.nw
@printf "Model: nw=%d, el_ham ndata=%d, nr=%d\n" nw model.el_ham.ndata model.el_ham.nr
@printf "CUDA functional: %s\n" CUDA.functional()

# k-points via kpoints_grid
nk = 16
kpts = kpoints_grid((nk, nk, nk)).vectors
@printf "Number of k-points: %d\n\n" length(kpts)

ham_cpu = model.el_ham
ham_gpu = to_device(ElectronPhonon.gpu_backend(), model.el_ham)

# ---------------------------------------------------------------------------
# Benchmark one driver on both backends.
#   driver : ham -> result    (run on ham_cpu and ham_gpu)
#   eigvals_of : result -> (nw, nk) eigenvalue array (identity, or `first` for (E, U))
# ---------------------------------------------------------------------------
function bench(name, driver, eigvals_of)
    nrep = 5
    Ec = eigvals_of(driver(ham_cpu))                                       # warmup + reference
    t_cpu = minimum(@elapsed(driver(ham_cpu)) for _ in 1:nrep)

    driver(ham_gpu)                                                        # warmup
    Eg = eigvals_of(driver(ham_gpu))
    t_gpu = minimum(CUDA.@elapsed(CUDA.@sync driver(ham_gpu)) for _ in 1:nrep)

    d = maximum(abs.(sort(Array(Eg), dims=1) .- sort(Ec, dims=1)))
    @printf "  %-10s  max|Δ|=%.2e   CPU %7.3f ms   GPU %7.3f ms   speedup x%.2f\n" name d t_cpu*1e3 t_gpu*1e3 t_cpu/t_gpu
end

println("Band eigenvalues over $(length(kpts)) k-points:")
bench("valueonly", ham -> get_el_eigen_valueonly_batched(get_interpolator(ham; fourier_mode="batched"), kpts), E -> E);
bench("eigen",     ham -> get_el_eigen_batched(get_interpolator(ham; fourier_mode="batched"), kpts),           EU -> first(EU));

# NOTE: For Pb (nw=4) the operators are tiny; the GPU advantage grows with band count and
# k-count, and is largest when eigenvectors are needed and for the e-ph matrix (Phase 2).
