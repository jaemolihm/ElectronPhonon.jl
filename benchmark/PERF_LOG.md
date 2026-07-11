# GPU performance log

Append-only record of GPU benchmark timings, one entry per pre-commit gate (see
`../MERGE_PLAN.md` → "Pre-commit gate"). Purpose: track performance across the merge so a
later stage that regresses timing is caught. Always record the **hardware** and versions —
numbers are only comparable within the same row of hardware.

Scripts: `bench_el_eigen_gpu.jl` (batched electron eigensolve), `bench_eliashberg_loop_gpu.jl`
(end-to-end e-ph calculator loop, CPU vs GPU). Both use the local Pb model (nw=4, nmodes=3);
see `README.md`. Small-grid CPU times are JIT/warmup-noisy — treat the largest grid as the
stable metric.

---

## 2026-07-11 — `56e58ad` (gpu-1-foundation, Stage 1 baseline)

**Hardware:** NVIDIA RTX A6000 (48 GB, driver 580.142), host `ccqlin059`.
**Software:** Julia 1.11.7, CUDA.jl runtime 13.2.0.

**Eigensolve** (`bench_el_eigen_gpu.jl`, 4096 k-points, nw=4):

| driver | CPU | GPU | speedup | max\|Δ(E)\| |
|-|-|-|-|-|
| valueonly | 11.0 ms | 24.5 ms | 0.45× | 3.8e-15 |
| eigen (E+U) | 14.2 ms | 24.1 ms | 0.59× | 3.8e-15 |

GPU slower here: for Pb (nw=4) the per-k matrices are tiny; the GPU advantage grows with band
count. Eigenvalues agree with the CPU to machine precision.

**End-to-end e-ph loop** (`bench_eliashberg_loop_gpu.jl`, EliashbergCalculator, window E_F±0.3 eV;
CPU = 12-thread `gridopt` path):

| grid | n_i | CPU | GPU | speedup |
|-|-|-|-|-|
| 16³ | 432  | 1.16 s | 0.71 s | 1.6× |
| 24³ | 1716 | 15.1 s | 3.55 s | 4.2× |
| 32³ | 4052 | 78.8 s | 11.2 s | 7.0× |

The GPU advantage grows with grid size (the loop is launch-bound at small `n_i`, compute-bound
at large `n_i`). **32³ (GPU 11 s) is the reliable signal**; the 16³ GPU time (<1 s) is near
timing noise and should not be trended on its own.

**Caveat — the bench's `relerr(g2)=0.76` / `ωq` mismatch is NOT a correctness regression.**
`run_cpu` uses the EPW-degeneracy-gauge-fixed per-k eigensolver; `run_gpu` uses the batched
Jacobi (`eigen_batched`, no gauge-fixing). For Pb's degenerate bands the two pick different
eigenvector gauges, so per-band `g2 = |ep|²/(2ω)` differs while gauge-independent quantities
match: the sorted `ωq` value set agrees to 2.7e-9, eigenvalues agree to 3.8e-15, and the
Eliashberg gaps were previously validated CPU-vs-GPU to 1e-5 meV. The in-repo loop test
(`check_eph_batched`, identical fixed eigenvectors on both backends) matches to 1e-9. The
bench's element-wise `g2` comparison is simply not a valid check across two eigensolvers for a
degenerate system.

---

## 2026-07-11 — `71f90e0` (gpu-2b-shared-opts, Stage 2b: 3 consolidated GPU opts)

**Hardware:** NVIDIA RTX A6000 (48 GB), host `ccqlin059`.
**Software:** Julia 1.11.7, CUDA.jl functional. GPU tests: 81/81 green.

Stage 2b adds three GPU optimizations on top of Stage 2a: fused e-ph rotation kernel
parallelized over (band, band, q); k-side window projection of the batched interpolation
(the k side carries only `nbk_max` in-window bands, not all `nw`); and `@inbounds` on the
per-batch device iq gathers.

**End-to-end e-ph loop** (`bench_eliashberg_loop_gpu.jl`, EliashbergCalculator, window E_F±0.3 eV):

| grid | n_i | CPU | GPU | speedup |
|-|-|-|-|-|
| 16³ | 432  | 0.58 s | 0.55 s | 1.05× |
| 24³ | 1716 | 7.12 s | 2.94 s | 2.42× |
| 32³ | 4052 | 41.1 s | 8.83 s | 4.66× |

**No GPU regression — the reliable 32³ GPU signal improved 11.2 s → 8.83 s (1.27×) vs the Stage 1
baseline row above.** The 24³ GPU also improved (3.55 → 2.94 s). The window-projection win is
modest here because Pb has `nw = 4` (little to project away); the large speedups it targets
(2.9–4.9× measured in EP `15d2570` on TaAs `nw = 32`, ±0.1 eV) need a wide-band, narrow-window
system. The **CPU** times dropped vs the baseline row too (e.g. 78.8 → 41.1 s at 32³), which
reflects the current MigdalEliashberg working-tree state rather than a GPU change — so the
cross-row *speedup ratio* is not directly comparable; trend the GPU wall-time column instead.

`relerr(g2)=0.76` / `ωq=false`: same degenerate-gauge artifact documented in the Stage 1 entry
above (two eigensolvers pick different gauges for Pb's degenerate bands), not a correctness
regression — the 81/81 GPU tests use gauge-invariant / fixed-eigenvector checks.
