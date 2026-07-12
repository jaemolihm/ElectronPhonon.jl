# Benchmark: full e-ph calculator loop (run_eph_over_k_and_kq) with the BoltzmannCalculator
# (BTE transport scattering), CPU vs GPU, over a k/k+q grid. Same driver as the EliashbergCalculator
# benchmark, but the calculator folds the temperature-dependent occupation physics into the
# scattering-out (Sₒ) / scattering-in (Sᵢ) matrices on the device via the shared
# `bte_scattering_increments` core, so CPU and GPU compute the same scattering (to round-off).
#
#   CPU : fourier_mode="gridopt", capped to 12 threads (nchunks_threads=12)
#   GPU : use_gpu=true, device-native batched calculator + device-resident Sₒ/Sᵢ scatter
#
# Methodology (see GPU_PROGRESS.md): benchmark with a realistic *energy window* around E_F
# (production transport uses an fsthick window). For large grids (a run > ~60 s) warm up on a
# coarse grid, then time the main grid ONCE. Pass `gpu` as the 4th arg to skip the CPU side.
#
# Requires ElectronPhonon (this branch) + CUDA in the environment. BoltzmannCalculator lives in
# ElectronPhonon itself (no MigdalEliashberg needed), e.g.:
#   julia --project=/mnt/home/jlihm/EPjl/gpuenv-stage benchmark/bench_bte_gpu.jl
#
# CLI args (all optional, positional):
#   1: grids           e.g. "24,36"          (default "8,16")
#   2: fsthick_eV      half-window in eV; "inf"/<0 = full band   (default 0.3)
#   3: ef_eV           Fermi level in eV      (default 11.682221647, Pb fine-mesh E_F)
#   4: "gpu"           skip the CPU side (GPU-only; for large grids)

using ElectronPhonon
using CUDA
using Printf
const EP = ElectronPhonon

CUDA.allowscalar(false)   # any scalar fallback on the device is then a hard error

const PB_FOLDER = "/mnt/home/jlihm/ceph/superconductivity/Pb/tutorial/1_epw/"

# ---- CLI ----
grids   = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [8, 16]
fsthick = (length(ARGS) >= 2 && (ARGS[2] == "inf" || parse(Float64, ARGS[2]) < 0)) ? Inf :
          (length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.3)
ef_eV   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 11.682221647
gpu_only = length(ARGS) >= 4 && lowercase(ARGS[4]) == "gpu"

ef = ef_eV * EP.unit_to_aru(:eV)
window = isfinite(fsthick) ? (ef - fsthick * EP.unit_to_aru(:eV), ef + fsthick * EP.unit_to_aru(:eV)) :
         (-Inf, Inf)

model = EP.load_model_from_epw_new(PB_FOLDER, "temp", "pb"; epmat_outer_momentum="el")
@printf "Model: nw=%d nmodes=%d   CUDA functional: %s\n" model.nw model.nmodes CUDA.functional()
@printf "Window: %s   grids: %s   %s\n\n" (isfinite(fsthick) ? "E_F ± $fsthick eV" : "full band") grids (gpu_only ? "(GPU only)" : "(CPU vs GPU)")

# FermiDirac occupation + Gaussian smearing (the configuration the BTE calculator supports). μ is
# fixed to E_F so the setup skips the μ solve; the timing is dominated by the e-ph loop regardless.
const K = EP.unit_to_aru(:K)
const meV = EP.unit_to_aru(:meV)
occ() = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 14.0, μlist = ef,
    volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac)

newcalc() = BoltzmannCalculator{Float64}(; occ = occ(),
    smearing = [(:Gaussian, 50.0 * meV)], occupation_method = 5)

run_cpu(g, win) = (c = newcalc(); EP.run_eph_over_k_and_kq(model, (g,g,g), (g,g,g); calculators=[c],
    symmetry=nothing, window_k=win, window_kq=win,
    fourier_mode="gridopt", nchunks_threads=12, progress_print_step=10^9); c)

run_gpu(g, win) = (c = newcalc(); EP.run_eph_over_k_and_kq(model, (g,g,g), (g,g,g); calculators=[c],
    symmetry=nothing, window_k=win, window_kq=win,
    use_gpu=true, progress_print_step=10^9); c)

# Max relative error of the SERTA scattering-out Sₒ (γ_{nk}) between two runs.
function relerr_So(cc, cg)
    a = stack(cc.Sₒ); b = stack(cg.Sₒ)
    denom = maximum(abs, a)
    denom == 0 ? 0.0 : maximum(abs, a .- b) / denom
end

function main()
    # Warm up compilation on a coarse 4^3 grid, full band so the coarse grid is never empty.
    gpu_only || run_cpu(4, (-Inf, Inf))
    run_gpu(4, (-Inf, Inf))

    for g in grids
        tg = @elapsed (cg = run_gpu(g, window))
        n_i = length(cg.Sₒ[1])
        if gpu_only
            @printf "%2d^3 : n_i=%-7d  GPU %8.2f s\n" g n_i tg
        else
            tc = @elapsed (cc = run_cpu(g, window))
            @printf "%2d^3 : n_i=%-7d  CPU %8.2f s   GPU %8.2f s   speedup %5.2fx   relerr(Sₒ)=%.1e\n" g n_i tc tg tc/tg relerr_So(cc, cg)
        end
    end
end

main()
