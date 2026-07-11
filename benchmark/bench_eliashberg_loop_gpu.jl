# Benchmark: full e-ph calculator loop (run_eph_over_k_and_kq) with the MigdalEliashberg
# EliashbergCalculator, CPU vs GPU, over a k/k+q grid. This is the end-to-end driver the GPU
# work targets — it exercises RR->kR, kR->kq, and the device-native g2 scatter / host-streaming.
#
#   CPU : fourier_mode="gridopt", capped to 12 threads (nchunks_threads=12)
#   GPU : use_gpu=true, device-native batched calculator + host-streamed g2
#
# Methodology (see GPU_PROGRESS.md): benchmark with a realistic *energy window* around E_F
# (production runs use an fsthick window). For large grids (a run > ~60 s) warm up on a coarse
# grid, then time the main grid ONCE. CPU scales ~grid^6 like the GPU, so for big grids pass
# `gpu` as the 4th arg to skip the CPU side (GPU-only timing).
#
# Requires ElectronPhonon (this gpu branch) + CUDA + MigdalEliashberg in the environment, e.g.:
#   julia --project=/mnt/home/jlihm/EPjl/gpuenv benchmark/bench_eliashberg_loop_gpu.jl
#
# CLI args (all optional, positional):
#   1: grids           e.g. "24,36"          (default "8,16")
#   2: fsthick_eV      half-window in eV; "inf"/<0 = full band   (default 0.3)
#   3: ef_eV           Fermi level in eV      (default 11.682221647, Pb fine-mesh E_F)
#   4: "gpu"           skip the CPU side (GPU-only; for large grids)

using ElectronPhonon
using CUDA
using MigdalEliashberg
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

newcalc() = EliashbergCalculator{Float64}(; nmodes = model.nmodes)

run_cpu(g, win) = (c = newcalc(); EP.run_eph_over_k_and_kq(model, (g,g,g), (g,g,g); calculators=[c],
    symmetry=nothing, window_k=win, window_kq=win,
    fourier_mode="gridopt", nchunks_threads=12, progress_print_step=10^9); c)

run_gpu(g, win) = (c = newcalc(); EP.run_eph_over_k_and_kq(model, (g,g,g), (g,g,g); calculators=[c],
    symmetry=nothing, window_k=win, window_kq=win,
    use_gpu=true, progress_print_step=10^9); c)

function main()
    # Warm up compilation on a coarse 4^3 grid. Use full band (-Inf,Inf) so the coarse grid is
    # never empty (a narrow window can keep zero states at 4^3); it compiles the same methods.
    gpu_only || run_cpu(4, (-Inf, Inf))
    run_gpu(4, (-Inf, Inf))

    for g in grids
        # Single timed main run per backend (methodology for large grids); GPU correctness is
        # checked against the CPU result when available. g2 layout is [ν, i, f] (mode-fastest).
        tg = @elapsed (cg = run_gpu(g, window))
        if gpu_only
            @printf "%2d^3 : n_i=%-7d  GPU %8.2f s\n" g size(cg.g2,2) tg
        else
            tc = @elapsed (cc = run_cpu(g, window))
            relerr = maximum(abs, cc.g2 .- cg.g2) / maximum(abs, cc.g2)
            @printf "%2d^3 : n_i=%-7d  CPU %8.2f s   GPU %8.2f s   speedup %5.2fx   relerr(g2)=%.1e  ωq=%s\n" g size(cc.g2,2) tc tg tc/tg relerr (cc.ωq == cg.ωq)
        end
    end
end

main()
