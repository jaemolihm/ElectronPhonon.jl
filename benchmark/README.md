# Benchmarks

Development scripts for timing the GPU e-ph / eigensolve paths against their CPU counterparts.
They are **not** part of the test suite and are not run in CI.

Unlike the tests (which pull a small model from `Artifacts.toml`), these scripts load a full Pb
model from a **hardcoded local path** and select the Julia environment with a hardcoded
`--project`. Both are specific to the author's machine — edit `PB_FOLDER` (and the `--project`
in the usage comment) to point at your own model and environment before running.

| script | measures |
|-|-|
| `bench_el_eigen_gpu.jl` | batched electron eigenvalues / eigenvectors, CPU vs GPU |
| `bench_eph_gpu.jl` | e-ph Wannier→Bloch interpolation + gauge rotation over a k/q grid |
| `bench_eliashberg_loop_gpu.jl` | full device-resident e-ph loop as driven by a calculator |
| `bench_bte_gpu.jl` | full GPU BTE transport pass (`BoltzmannCalculator`), CPU vs GPU |

## GPU BTE profile (recorded so it is not re-litigated)

`CUDA.@profile` of a full GPU BTE pass — Pb, `nk = 24`, ±2 eV window, `occupation_method = 5`,
RTX A6000 — GPU busy 12.0 s (67% of the trace). Device-side kernel breakdown:

| % of GPU time | kernel | what it is |
|-:|-|-|
| 27.7% | complex ZGEMM (`z884gemm`) | Wannier→Bloch interpolation — **the bottleneck** |
| 20.8% | `_bte_window_accumulate_kernel` | the BTE Sₒ/Sᵢ scatter |
| 7.8% | `_fused_eph_rot_kernel` | e-ph gauge rotation |
| ~4.4% | H2D/D2H memcpy | incl. per-tile Sᵢ streaming (~2.2%) |
| 0.5% | eigensolve (`heevj`) | k+q window filter only |

Takeaways for anyone tempted to re-optimize:

- **The scatter kernel is ~21% of GPU time here, NOT ~0.6%.** An older commit note quoted 0.6%;
  that was a much narrower window. The scatter's share grows with window width / in-window state
  count, so it is a legitimate target at realistic transport windows — but the interpolation ZGEMM
  (28%) is the larger one.
- **Sᵢ device-memory residency is not a speed knob.** Sᵢ is always streamed to the host one tile per
  outer-k batch (block-resident); this is within ~2% of a single whole-Sᵢ copy even at 1.1 GB Sᵢ
  (~40 separate D2H copies) because the total bytes moved are identical. There is deliberately no
  full-device-resident Sᵢ path — do not add one for speed.
- **The `CUDA.@atomic` on Sₒ is ~5% of the scatter kernel (~1% of GPU time)** in the worst case
  (all bands in-window). Left as-is on purpose; a block/warp reduction before one atomic per
  `(i, iocc)` would recover most of it if the scatter ever becomes the bottleneck.
