# GPU acceleration of Wannier→Bloch Fourier interpolation

This document is the implementation plan for running the Wannier→Bloch Fourier
interpolation of ElectronPhonon.jl on NVIDIA GPUs via CUDA.jl, following the
DFTK GPU strategy (<https://docs.dftk.org/stable/tricks/gpu/>).

## Scope and decisions

**In scope (this effort):**

- `get_fourier!` (normal and batched modes) — `src/wannier/WannierInterpolator.jl`,
  `src/wannier/batched_interpolator.jl`
- `get_el_eigen!` — `src/wannier_to_bloch.jl` (validation vehicle for Phase 1)
- `get_eph_RR_to_kR!` and `get_eph_kR_to_kq!` — `src/wannier_to_bloch.jl` (Phase 2)

**Out of scope (for now):** `gridopt` / `batched-gridopt` modes, `DiskWannierObject`,
MPI, phonon dynamical-matrix dipole terms, and the LAPACK eigensolves
(`solve_eigen_*`) beyond what Phase 1 requires.

**Decisions (agreed):**

- **Precision: Float64** everywhere. Matches the CPU path exactly so correctness can
  be validated with `Array(gpu) ≈ cpu`. (The dev workstation RTX A6000 has poor FP64
  throughput, ~1/32 of FP32; this is a dev-speed concern only, not correctness. FP32 is
  a possible later optimization, not part of this plan.)
- **Integration: package extension.** CUDA is a `[weakdeps]` dependency; all GPU code
  lives in `ext/ElectronPhononCUDAExt.jl`. The base package loads and runs on CPU-only
  machines and in CI without CUDA installed.
- **Assumption:** the full `op_r` array fits in GPU memory (no streaming / chunking).

## Hardware and how to run

- **Develop and test on the workstation.** `ccqlin059` has an NVIDIA RTX A6000 (48 GB),
  which is sufficient for all correctness work and small benchmarks. **No Slurm job is
  needed for development.**
- **Do NOT `module load cuda`.** CUDA.jl ships its own CUDA toolkit artifacts and only
  needs the system driver. Loading the `cuda` module risks the ABI mismatch the site
  policy warns about. Just add `CUDA` to the project.
- A Slurm GPU node (`salloc -p gpu -C a100 -G1`) is only needed later, for large-scale
  benchmarking on A100/H100.

## Why these functions are a good GPU fit

Stripped of buffers and wrappers, the interpolation is dense linear algebra plus a phase
vector:

| Function | GPU primitive |
|---|---|
| `get_fourier!` (normal) | `phase = cispi.(...)` (length `nr`) + GEMV `op_r(ndata×nr)·phase` |
| `get_fourier!` (batched) | phase batch `(nr×B)` + GEMM `op_r(ndata×nr)·phase_batch(nr×B)` |
| `get_el_eigen!` | `get_fourier!` → `hk(nw×nw)` + Hermitian eigensolve |
| `get_eph_RR_to_kR!` | `get_fourier!` + loop of small GEMMs `(nw×nw)·(nw×nband)` → **batched GEMM** |
| `get_eph_kR_to_kq!` | `get_fourier!` + two large reshaped GEMMs (already cuBLAS-ready) |

cuBLAS provides `mul!` for `CuArray` automatically, so most of the work is making the
code array-type-generic and keeping the data on the device.

## DFTK strategy, mapped to this codebase

1. **Generic array type.** Code is written against `AbstractArray`; the concrete array
   type (`Array` vs `CuArray`) flows through the structs. `mul!` and broadcast
   (`.=`, `cispi.`) dispatch to CUDA.jl. No algorithm is duplicated for the GPU.
2. **No scalar indexing.** `CUDA.allowscalar(false)` is set in the extension/tests to
   catch any host-style indexing that would silently kill performance.
3. **Move data to the device once.** A `to_device` / `cu` converter moves `op_r` (and the
   interpolation buffers) to the GPU at setup; per-k-point work stays on the device.
4. **Isolate CUDA in an extension.** Base package has no CUDA dependency.

## Key design points

### Revised design — generic array type + batched API

**Phase 1 (per-k `get_el_eigen!`) shipped and validated, but proved the wrong shape for
GPUs on small systems** (×0.09 on Pb — see results below). Issuing thousands of tiny per-k
GEMV + eigensolve kernels is launch-bound. The fix is twofold and was agreed with the user:

1. **Parameterize `WannierObject` over its array type** (bite the bullet), so the same code
   runs on CPU and GPU.
2. **Make the GPU API batched** — process *all* k-points with a few big kernels.

#### 1. `WannierObject{T}` → `WannierObject{T, AT<:AbstractMatrix{Complex{T}}}`

`op_r::AT` may now be a `CuMatrix`. This is far less invasive than first feared:

- `WannierObject{FT}` is just the abstract alias `WannierObject{FT,AT} where AT`, so every
  existing dispatch signature and `Model` field annotation (`el_ham::WannierObject{FT}`, …)
  keeps compiling.
- The only required fix is the inner `@kwdef` constructor (`WannierObject{T}(…)` →
  `WannierObject{T,typeof(op_r)}(…)`).
- `Model`'s `WannierObject{FT}` fields become abstractly typed, but that is **harmless**:
  they are read once at setup to build interpolators, never in hot loops, and the
  interpolator captures the concrete type from then on.

`to_device(obj::WannierObject)` then returns a genuine **device `WannierObject`**
(`op_r` on the GPU; `irvec` stays on the host).

#### 2. Generalized `BatchedWannierInterpolator` (reuse, not duplication)

Rather than add a separate primitive, the existing `BatchedWannierInterpolator`
(`src/wannier/batched_interpolator.jl`) was **generalized** so it is the one batching
mechanism for both CPU and GPU:

- Its buffer fields (`cached_results`, `phase_batch`, `rdotk`, `out`, …) now follow the
  backend of `parent.op_r` (allocated via `similar`; host fallback for `DiskWannierObject`).
- Its phase computation was switched from a host scalar loop to a **GEMM + broadcast**,
  which has no scalar indexing and therefore runs on any backend (and is faster on the CPU
  too):

  ```julia
  # irvec_mat :: (nr × 3) real, on backend (built once in the constructor)
  # kmat      :: (3 × batch) real, on backend
  mul!(rdotk, irvec_mat, kmat)               # (nr × batch) real   GEMM
  phase_batch .= cispi.(2 .* rdotk)          # (nr × batch) complex broadcast
  mul!(cached_results, op_r, phase_batch)    # (ndata × batch) GEMM  → op_k for all k
  ```

- The per-k query API (`register_kpoints!` + sequential `get_fourier!`) used by the
  calculators is **unchanged**. A new whole-batch entry point `get_fourier_batched!(out,
  itp, xk_list)` returns the entire `(ndata, nk)` result on the backend — this is what the
  GPU path uses, avoiding any per-k device→host copy.

Everything dispatches through `mul!`, broadcasting, `similar`, and `copyto!`, which CUDA.jl
provides for `CuArray`. No CUDA code lives in the base package.

#### 3. Batched Hermitian eigensolve + band-eigenvalue drivers

Diagonalization lives in `src/wannier_to_bloch_gpu.jl` (`src/wannier/` is kept to pure
Fourier). Two batched eigensolves over a stack `Hk :: (nw, nw, nk)`:

- `eigvals_batched(Hk) -> W` — eigenvalues only.
- `eigen_batched(Hk) -> (W, V)` — eigenvalues + eigenvectors.

CPU methods loop over LAPACK `syev!`; the CUDA extension provides
`CUSOLVER.heevjBatched!('N'/'V', …)` methods (the batched Jacobi solver). The often-quoted
`nw ≤ 32` figure is a *performance* characteristic, not a correctness bound — verified on
cuSOLVER 13.3 to solve correctly to `nw=256` (agreeing with LAPACK to ~1e-11), so no size
guard is imposed (older cuSOLVER versions may raise their own error for large `nw`).

The drivers mirror the per-k API names:

| per-k (host, `wannier_to_bloch.jl`) | batched (`wannier_to_bloch_gpu.jl`) |
|---|---|
| `get_el_eigen_valueonly!` (values) | `get_el_eigen_valueonly_batched` |
| `get_el_eigen!` (values + vectors) | `get_el_eigen_batched` |

Each driver interpolates `H(k)` for all k with `BatchedWannierInterpolator` +
`get_fourier_batched!`, then calls the matching batched eigensolve. Results are on
`op_r`'s backend. (The batched eigenvectors carry no EPW degeneracy gauge-fixing, so for
degenerate bands they may differ from `get_el_eigen!` by a gauge.)

> The Phase-1 extension-local `CuNormalWannierInterpolator` is **removed** — it is superseded
> by the generic device `WannierObject` + the generalized `BatchedWannierInterpolator`.

### e-ph rotations (Phase 2)

`get_eph_RR_to_kR!`'s `(ir_ep, imode)` loop of small `(nw×nw)·(nw×nband)` GEMMs is the only
non-trivial piece. Rather than a `CUBLAS.gemm_strided_batched!` (extension-only), it is
recast as a **single** GEMM that works on any backend:
`out[iw,ib,b] = Σ_jw g[iw,jw,b] uk[jw,ib]` becomes `permutedims(g,(2,1,3))`, then
`transpose(uk) * g`, then `permutedims` back — one cuBLAS/BLAS call, no batched-GEMM
primitive. `get_eph_kR_to_kq!` is already two reshaped `mul!` calls and is reused verbatim.

For the **per-k / per-q** drivers this needs **no new extension code**: the device Fourier
reuses `get_fourier_batched!` and the single shared `uk` makes the rotation one generic GEMM.

For the **list-batched** drivers (process many k or many q at once — the form that wins on
the GPU), each k/q carries its *own* rotation matrix (`uk(k)`, `ukq(q)`, `u_ph(q)`). A stack
of independent GEMMs with distinct operands is not an ordinary GEMM, so it uses
`batched_gemm!(transA, transB, A, B, C)` — a `mul!` loop on the CPU, and
`CUBLAS.gemm_strided_batched!` in the extension on the GPU. This is the *only* e-ph-related
piece of extension code.

All four drivers live in `wannier_to_bloch_gpu.jl` and run on the CPU or GPU by the backend
of the parent `op_r`:
- `get_eph_RR_to_kR_batched!(epobj, itp, xk, uk)` / `get_eph_kR_to_kq_batched!(ep_kq, itp, xq, u_ph, ukq)` — single k/q.
- `get_eph_RR_to_kR_batched!(ep_all, itp, ks, uks)` / `get_eph_kR_to_kq_batched!(ep_kq_all, itp, qs, u_phs, ukqs)` — list-batched.

## Phased plan

### Phase 0 — scaffolding (no algorithm changes) ✅ DONE

- [x] Add `CUDA` to `[weakdeps]` + `[extensions]` in `Project.toml`.
- [x] Declare `function to_device end` (+ export) in the base package.
- [x] Create `ext/ElectronPhononCUDAExt.jl` with `CuNormalWannierInterpolator` + `to_device`.

### Phase 1 — `get_fourier!` + `get_el_eigen!`  ⟵ CHECKPOINT (user reviews) ✅ DONE

- [x] `CuNormalWannierInterpolator` holds device `op_r` + host/device buffers (extension).
- [x] `to_device(::WannierObject)` copies `op_r` to the GPU.
- [x] `get_fourier!`: host phase compute + `copyto!` to device + cuBLAS GEMV.
- [x] `get_el_eigen!`: device Fourier → host copy of `hk` → existing host eigensolve.
- [x] Correctness validated against CPU (synthetic + real Pb model).
- [ ] TODO: move the validations into `test/test_wannier.jl`, guarded by `CUDA.functional()`.
- **Stopped here for review before Phase 2.**

### Phase 1.5 — generic array type + batched GPU eigenvalues ✅ DONE

- [x] Parameterize `WannierObject{T}` → `{T,AT}`; fix the inner constructor.
- [x] `to_device(::WannierObject)` returns a device `WannierObject`.
- [x] Generalize `BatchedWannierInterpolator` (backend-generic buffers + GEMM phase +
      `get_fourier_batched!`); per-k calculator API unchanged.
- [x] Batched eigensolves + drivers in `src/wannier_to_bloch_gpu.jl`
      (`eigvals_batched`/`eigen_batched`, `get_el_eigen_valueonly_batched`/`get_el_eigen_batched`).
- [x] Extension: `eigvals_batched`/`eigen_batched` (`CuArray`) via `heevjBatched!`; removed
      `CuNormalWannierInterpolator`.
- [x] `test/test_wannier.jl` passes (126/126).
- [x] Benchmark (CPU vs GPU, same batched driver), for eigenvalues-only and +eigenvectors.

**Benchmark** (RTX A6000, Pb `nw=4`, 16³ = 4096 k-points). The `get_el_eigen[_valueonly]_batched`
driver runs on both backends — only `op_r`'s backend differs. GPU eigenvalues agree with CPU
to `max|Δ| ≈ 4e-15`:

| quantity | CPU | GPU | speedup |
|---|---|---|---|
| valueonly | 11.6 ms | 8.6 ms | 1.34× |
| + eigenvectors | 14.4 ms | 8.8 ms | 1.64× |

The GPU advantage is larger when eigenvectors are needed (4096 eigenvector solves are cheap
batched on the GPU). For reference, the earlier *per-k* GPU `get_el_eigen!` was ×0.09 vs CPU
— batching is the whole story. Produced by `benchmark/bench_el_eigen_gpu.jl`.

### Phase 2 — e-ph interpolation ✅ DONE

- [x] Per-k/q drivers `get_eph_RR_to_kR_batched!` / `get_eph_kR_to_kq_batched!`: device Fourier
      (`get_fourier_batched!`) + rotations as generic GEMMs. Backend-generic.
- [x] List-batched drivers (over k / over q) + `batched_gemm!` primitive (CPU loop / GPU
      `CUBLAS.gemm_strided_batched!`) for the per-k/q-distinct rotation matrices.
- [x] Validated vs the **independent per-k/q** `get_eph_*!` reference, every batch element,
      on CPU (always-run testset) and GPU (`test/test_gpu.jl`): CPU exact (`~1e-16`), GPU `~1e-13`.
- [x] List-batched drivers are **full-band only** (no energy window) — by design, confirmed OK.

**Benchmark** (RTX A6000, Pb `nw=4 nmodes=3 nr_ep=43`, 64 k / 64 q):

| op | per-pt GPU | **batched GPU** | batched CPU | batching speedup |
|---|---|---|---|---|
| RR→kR (64 k) | 12.3 ms | **0.61 ms** | 3.11 ms | ×20 (GPU 5× over CPU) |
| kR→kq (64 q) | 11.8 ms | **0.18 ms** | 0.12 ms | ×65 (CPU ties — tiny `nw=4`) |

Batching collapses thousands of per-point kernel launches into a few large ones. RR→kR moves
the large e-ph operator (`nw²·nmodes·nr_ep × nr_el`) and is a clear GPU win; kR→kq's matrices
are tiny for Pb so the CPU is competitive — the GPU pulls ahead for larger `nw`/`nband`/`nmodes`.
Produced by `benchmark/bench_eph_gpu.jl`.

### Phase 3 — calculator integration ✅ DONE (first step)

Wired the batched GPU e-ph drivers into the calculator loop `run_eph_over_k_and_kq`
(`src/calculator/run_eph_over_k_and_kq.jl`), validated with the `EliashbergCalculator`
(`MigdalEliashberg.jl/src/calculator.jl`).

- [x] New entry point via a `use_gpu` keyword on `run_eph_over_k_and_kq` (+ `q_batch_size`),
      branching to `_loop_eph_over_k_and_kq_gpu`. The CPU loop and per-k/q CPU e-ph functions
      are unchanged. The new loop is backend-generic (only calls `to_device` + the batched
      drivers; no CUDA in the base package).
- [x] Minimal scope asserted off: no polar/long-range (ϵs = 1, no dipole), full bands (no
      energy window → `nband == nw`), commensurate grids (`precompute_ph`), and no covariant
      derivative / screening / symmetry / MPI / nontrivial energy conservation.
- [x] Restructured as in the plan below: `to_device(model.epmat)` once; per `ik` a single-k
      `get_eph_RR_to_kR_batched!`; the inner `ikq` loop batched (chunked by `q_batch_size`)
      through one `get_eph_kR_to_kq_batched!`; then a cheap host loop writing the slice into
      `epdata.ep`, `epdata_set_g2!`, and `run_calculator!` unchanged.
- [x] Shared device interpolators + per-(k,q) staging allocated once outside the `ik` loop and
      reused across all (k, q). (The batched drivers still allocate their internal scratch per
      call — recycled by CUDA's pool; pushing that into the drivers is a follow-up.)
- [x] Validated: `EliashbergCalculator` through the CPU and GPU paths on the Pb model, full-BZ
      commensurate grids, `symmetry=nothing` — `calc.g2`/`calc.ωq` agree (see results). GPU-guarded
      test in `test/test_gpu.jl` (skips when CUDA unavailable or the Pb data dir is absent).

**Validation & benchmark** (RTX A6000, Pb `nw=4 nmodes=3`, full BZ, CPU path on 32 threads).
GPU vs CPU `calc.g2` agree to `max|Δ| ≈ 3–4e-11` (relative `~1e-15`; cuBLAS reordering only) and
`calc.ωq` is bit-identical (same precomputed phonon states). End-to-end wall time (warm):

| grid | nk = nkq | CPU (32 thr) | GPU | speedup |
|---|---|---|---|---|
| 4³ | 64  | 108 ms  | 47 ms   | 2.3× |
| 6³ | 216 | 398 ms  | 246 ms  | 1.6× |
| 8³ | 512 | 1991 ms | 1364 ms | 1.5× |

The GPU win is real but modest for Pb because its matrices are tiny (`nw=4, nmodes=3`), so the
serial host-side tail (`run_calculator!` writing g2 + building the per-q rotation stacks, both
O(nk·nkq)) dominates as the grid grows — consistent with the Phase-2 finding that the small-`nw`
kR→kq rotation is CPU-competitive. The advantage grows with `nw`/`nband`/`nmodes`. Chunking
(`q_batch_size`) was verified correct for partial final chunks (`q_batch_size=7`, `=1`).

**Follow-ups:** (1) benchmark full in-place drivers (workspace-backed scratch) to see if buffer
reuse beats CUDA's pool here; (2) parallelize / vectorize the host tail; (3) Phase 4 below.

**First step — minimal scope (assert/disable the rest):**
- Ignore long-range / polar: assert `model.polar_phonon` and `model.polar_eph` are
  nothing/disabled; skip `set_eph_dipole_coeff!` / `epdata_compute_eph_dipole!` (ϵs = 1).
- Full bands, no energy window (`window_* = (-Inf, Inf)`) → uniform shapes for batching.
- Disable (assert) `covariant_derivative_of_g`, `screening_params`,
  `symmetry`/`el_kq_from_unfolding`, nontrivial `energy_conservation`, MPI. Single node.
- Add a **new** entry point (e.g. `run_eph_over_k_and_kq_gpu` or a `use_gpu` branch); do not
  change the CPU loop or the per-k/q CPU e-ph functions.

**Restructuring (where the GPU win + buffer reuse live):**
1. `to_device(model.epmat)` once; build device batched interpolators.
2. Per outer `ik`: GPU `get_eph_RR_to_kR_batched!` (single k) → device `ep_ekpR(k)`.
3. Batch the inner `ikq` loop: stack `ukqs (nw,nw,nkq)`, `u_phs (nmodes,nmodes,nkq)`,
   `qs = kqpts .- xk`; one `get_eph_kR_to_kq_batched!` → `ep_kq_all (nw,nw,nmodes,nkq)`; copy to host.
4. Cheap host loop over `ikq`: write the slice into `epdata.ep`, run `epdata_set_g2!`, call
   `run_calculator!(calc, epdata, ik, iq, ikq)` unchanged (calculator API untouched).
5. Shared GPU **workspace** allocated once outside the `ik` loop and reused across all (k, q)
   — this is the buffer-reuse follow-up; it belongs here, not in the standalone drivers.

**Validation:** run an `EliashbergCalculator` through the CPU and GPU paths on a small
commensurate full-BZ grid for the Pb model and assert `calc.g2` / `calc.ωq` match to ~1e-9;
benchmark end-to-end. GPU-guarded test.

Phase 4 (later): energy window (`nband < nband_bound`) and long-range/polar on the GPU.

## Testing

```julia
# from the worktree
julia --project=. -e 'using Pkg; Pkg.develop(...); Pkg.add("CUDA")'
julia --project=. test/test_wannier.jl   # Phase 1
julia --project=. test/test_epmat.jl     # Phase 2
```

All GPU tests are guarded with `if CUDA.functional()` so the suite still passes on
CPU-only machines.

## Files changed

- `Project.toml` — `[weakdeps]`, `[extensions]`, `[compat] CUDA`.
- `src/wannier/WannierObject.jl` — array-type parameter `AT`; constructor fix.
- `src/wannier/WannierInterpolator.jl` — declare/export `to_device`.
- `src/wannier/batched_interpolator.jl` — generalized over backend (buffers via `similar`,
  GEMM phase); new `get_fourier_batched!`. Per-k API unchanged. (Pure Fourier only.)
- `src/wannier_to_bloch_gpu.jl` — **new**; `eigvals_batched`/`eigen_batched` (CPU), the
  `get_el_eigen[_valueonly]_batched` drivers, the e-ph drivers
  `get_eph_RR_to_kR_batched!` / `get_eph_kR_to_kq_batched!` (per-k/q and list-batched), and
  the `batched_gemm!` primitive. Included after `wannier_to_bloch.jl`. All backend-generic.
- `ext/ElectronPhononCUDAExt.jl` — `to_device(::WannierObject)`,
  `eigvals_batched`/`eigen_batched`, and `batched_gemm!` (`CUBLAS.gemm_strided_batched!`).
- `src/calculator/run_eph_over_k_and_kq.jl` — **Phase 3**: `use_gpu` / `q_batch_size` keywords on
  `run_eph_over_k_and_kq`; new backend-generic `_loop_eph_over_k_and_kq_gpu`. CPU path unchanged.
- `benchmark/bench_el_eigen_gpu.jl` — CPU-vs-GPU band-eigenvalue benchmark.
- `benchmark/bench_eph_gpu.jl` — e-ph benchmark, per-(k,q) vs list-batched, CPU vs GPU.
- `test/test_gpu.jl` — GPU-guarded tests incl. e-ph drivers and the `use_gpu` calculator loop
  (skips when CUDA unavailable or the Pb data dir is absent); wired into `runtests.jl`.

## Design notes & follow-ups

### Full bands (no energy window) on the GPU — intended

The list-batched drivers (and the GPU path generally) are **full-band only**: every k-point
in a batch shares the same `nband`, and `ep_ekpR_all` / `ep_kq_all` are sized exactly with no
`nband_bound` slack. This is a deliberate choice for GPU calculations, which keep **all
bands** rather than applying a per-k energy window:

- On the GPU the extra bands are cheap, and batching wants uniform shapes across k/q anyway —
  a per-k variable `nband` would break the single large GEMMs that make the GPU fast.
- Energy windowing mainly saves work when only a few bands matter; that optimization stays on
  the CPU per-k path (`get_eph_RR_to_kR!` etc.), which keeps its `nband < nband_bound` support.

So when wiring the GPU path into a full calculation, pass full `nband` (= `nw` for the
electron side); no window handling is needed.

### Buffer reuse (open optimization)

Each batched driver currently allocates scratch arrays per call and frees them — e.g.
`get_eph_RR_to_kR_batched!` over 64 k allocates ~8 MiB of scratch (`g`, the `permutedims`
copies `gp`/`out`, and the GEMM result `C`); `get_eph_kR_to_kq_batched!` allocates `g` + `tmp`
on each of its ~`nk` calls; `get_fourier_batched!` allocates `kmat` per chunk; and every
`permutedims` allocates a fresh array.

**Buffer reuse** = allocate these once and write into them on every call (in-place
`permutedims!` / `mul!` / broadcast), either via a small workspace struct or by storing them
in the interpolator (as `BatchedWannierInterpolator` already does for
`phase_batch`/`rdotk`/`cached_results`).

- Bigger payoff on the **CPU** (avoids GC pauses); more **modest on the GPU**, where CUDA's
  memory pool already recycles device buffers — there the win is mainly dropping the two
  `permutedims` copies (memory-bound kernels).
- **Recommended timing:** do this together with wiring the batched drivers into the
  `run_eph_*` calculator loop, where a workspace allocated once at loop setup is reused across
  all (k, q) and the end-to-end effect is measurable. Doing it on the standalone drivers in
  isolation complicates their API for a benefit that can't be measured outside a real loop.

## Git workflow

All GPU work is done on branch `gpu-interpolation` in the worktree
`/mnt/home/jlihm/EPjl/ElectronPhonon.jl-gpu`, separate from `main`.
