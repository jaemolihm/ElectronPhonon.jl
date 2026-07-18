# GPU acceleration of Wannier→Bloch Fourier interpolation

Runs the Wannier→Bloch Fourier interpolation and the e-ph calculator loop of
ElectronPhonon.jl on NVIDIA GPUs via CUDA.jl. The base package has no CUDA dependency; all
device-specific code lives in a package extension and the same source runs on CPU or GPU
depending on the backend of the data.

## Scope

On the GPU:

- `get_fourier!` (normal and batched) — `src/wannier/WannierInterpolator.jl`,
  `src/wannier/batched_interpolator.jl`.
- Batched band eigenvalues/eigenvectors — `src/wannier_to_bloch_batched.jl`.
- e-ph interpolation `get_eph_RR_to_kR!` / `get_eph_kR_to_kq!` (per-k/q and list-batched
  forms) — `src/wannier_to_bloch_batched.jl`.
- The e-ph calculator loop `run_eph_over_k_and_kq` via a `use_gpu` branch, including a
  device-native calculator hook.

Deliberately **not** on the GPU (stays on the CPU per-k path):

- `gridopt` / `batched-gridopt` interpolation and `DiskWannierObject`.
- Per-k energy windowing — the GPU path is full-band only (see below).
- Long-range/polar (dipole) e-ph terms, screening, covariant derivatives, symmetry /
  k+q-from-unfolding, and nontrivial energy conservation. The GPU loop asserts these off.

## Decisions

- **Float64 everywhere.** Matches the CPU path exactly so correctness is checked with
  `Array(gpu) ≈ cpu`. FP32/mixed precision is not used (accuracy requirement).
- **Package extension.** CUDA is a `[weakdeps]` dependency; all GPU code is in
  `ext/ElectronPhononCUDAExt.jl`. The base package loads and runs on CPU-only machines and in
  CI without CUDA installed.
- **`op_r` fits in GPU memory.** No streaming/chunking of the Wannier operator itself.
- **Full-band interpolation on the GPU, with energy windows supported.** The batched drivers
  interpolate all `nw` bands (uniform shapes keep the GEMMs large), but energy windows *are*
  supported: the outer-k loop projects the k side onto a contiguous in-window eigenvector window
  and applies the window in the calculator scatter (out-of-window states map to `imap == 0`); the
  outer-q loop masks out-of-window eigenvector columns so those matrix elements are exactly 0. See
  the design note.
- **The GPU path is fully GPU — no silent fallback.** The GPU calculator loop requires every
  calculator to support the batched device payload (`supports(calc, ElPhDataOuterKBatched) = true`
  for the outer-k loop, `supports(calc, ElPhDataOuterQBatched) = true` for the outer-q loop). It
  must not silently fall back to the per-`(k,q)` host `run_calculator!(calc, ::ElPhDataPoint, ctx)`
  path, which would be a hard-to-spot performance cliff — a calculator that forgets to opt in should
  fail loudly instead.

## Strategy

The interpolation is dense linear algebra plus a phase vector, which maps directly onto
cuBLAS `mul!` and broadcasting. The approach is therefore to make the code array-type-generic
rather than to duplicate kernels:

1. **Generic array type.** Code is written against `AbstractArray`; the concrete type
   (`Array` vs `CuArray`) flows through the structs. `mul!`, broadcast, `similar`, `copyto!`
   dispatch to CUDA.jl. No algorithm is duplicated for the GPU.
2. **No scalar indexing.** `CUDA.allowscalar(false)` in the extension/tests turns any
   host-style device indexing (which would silently kill performance) into a hard error.
3. **Move data to the device once.** `to_device` moves `op_r` and the interpolation buffers
   to the GPU at setup; per-k work stays on the device.
4. **Batch.** Process many k/q with a few large kernels rather than thousands of tiny per-point
   launches — this is what makes the GPU win.

## Design

### `WannierObject{T}` → `WannierObject{T, AT<:AbstractMatrix{Complex{T}}}`

`op_r::AT` may be a `CuMatrix`. `WannierObject{FT}` remains an abstract alias
(`WannierObject{FT,AT} where AT`), so existing dispatch signatures and `Model` field
annotations keep compiling; only the inner `@kwdef` constructor needed a fix. The abstractly
typed `Model` fields are read once at setup to build interpolators, never in hot loops.
`to_device(obj::WannierObject)` returns a device `WannierObject` (`op_r` on the GPU, `irvec`
on the host).

### Generalized `BatchedWannierInterpolator` (one mechanism, both backends)

The existing `BatchedWannierInterpolator` (`src/wannier/batched_interpolator.jl`) is the single
batching mechanism for CPU and GPU:

- Buffer fields (`cached_results`, `phase_batch`, `rdotk`, `out`, …) follow the backend of
  `parent.op_r` (allocated via `similar`; host fallback for `DiskWannierObject`).
- The phase computation is a GEMM + broadcast (no scalar indexing), so it runs on any backend
  and is faster on the CPU too:

  ```julia
  # irvec_mat :: (nr × 3) real, on backend (built once in the constructor)
  # xkmat     :: (3 × batch) real, on backend
  mul!(rdotk, irvec_mat, xkmat)              # (nr × batch) real    GEMM
  phase_batch .= cispi.(2 .* rdotk)          # (nr × batch) complex broadcast
  mul!(cached_results, op_r, phase_batch)    # (ndata × batch) GEMM → op_k for all k
  ```

- The per-k query API (`register_kpoints!` + sequential `get_fourier!`) used by the calculators
  is unchanged. A new whole-batch entry point `get_fourier_batched!(out, itp, xk_list)` returns
  the entire `(ndata, nk)` result on the backend — the GPU path uses this, avoiding any per-k
  device→host copy.

### Batched Hermitian eigensolve + band-eigenvalue drivers

Diagonalization lives in `src/wannier_to_bloch_batched.jl` (keeping `src/wannier/` pure Fourier).
Two batched eigensolves over a stack `Hk :: (nw, nw, nk)`: `eigvals_batched` (values) and
`eigen_batched` (values + vectors). CPU methods loop over LAPACK `syev!`; the extension uses
`CUSOLVER.heevjBatched!` (batched Jacobi). The `nw ≤ 32` figure often quoted for that solver is
a *performance* characteristic, not a correctness bound (verified correct to `nw=256`,
agreeing with LAPACK to ~1e-11), so no size guard is imposed.

Drivers `get_el_eigen_valueonly_batched` / `get_el_eigen_batched` mirror the per-k API names:
each interpolates `H(k)` for all k with `get_fourier_batched!`, then calls the matching
eigensolve. The batched eigenvectors carry no EPW degeneracy gauge-fixing, so for degenerate
bands they may differ from `get_el_eigen!` by a gauge.

### e-ph rotations

`get_eph_RR_to_kR!`'s loop of small `(nw×nw)·(nw×nband)` GEMMs is recast as a single generic
GEMM (`permutedims` → `transpose(uk) * g` → `permutedims` back), needing no extension code.
`get_eph_kR_to_kq!` is already two reshaped `mul!` calls and is reused verbatim.

The **list-batched** drivers (many k or many q at once — the form that wins on the GPU) give
each k/q its own rotation matrix, which is a stack of independent GEMMs with distinct operands.
This uses `batched_gemm!(transA, transB, A, B, C)` — a `mul!` loop on the CPU and
`CUBLAS.gemm_strided_batched!` in the extension. This is the only e-ph-related extension code.

All four drivers (`get_eph_RR_to_kR_batched!` / `get_eph_kR_to_kq_batched!`, each in single and
list-batched forms) live in `wannier_to_bloch_batched.jl` and pick CPU/GPU by the backend of
`parent.op_r`.

### Calculator integration

The calculator interface is unified around **payload dispatch**: `run_calculator!(calc, payload,
ctx::LoopContext)` has one method per payload type. The full spec is in the docstrings of
`src/calculator/AbstractCalculator.jl`, and `docs/writing_a_calculator.md` is the tutorial (start
there for a CPU-only calculator). This section covers only the GPU side.

`run_eph_over_k_and_kq` / `run_eph_over_q_and_k` take a `use_gpu` keyword (with `nq_batch_max` /
`nk_outer_batch_max` for outer-k, `nk_batch_max` for outer-q) that branches to a backend-generic
device loop. `use_gpu` is the only user-facing switch; below the driver entry a `backend` object
(`CPUBackend()` / `GPUBackend(proto)`) is resolved once, uploaded `to_device(model.epmat)` once,
and carried in `LoopContext`. The CPU loop and per-(k,q) CPU e-ph functions are unchanged. The GPU
outer-k loop processes a batch of outer k with one list-batched `get_eph_RR_to_kR_batched!` and
batches the inner `ikq` loop (in batches of `nq_batch_max`) through one `get_eph_kR_to_kq_batched!`.

To run on the GPU a calculator opts into a **device-native batched payload** so the e-ph matrix for
a whole batch stays on the device and the reduction/scatter happens there. Each loop has its own
payload, named for which momentum is the outer loop and which is batched on the inner axis:

- **Outer-k loop (`run_eph_over_k_and_kq`, outer k / inner k+q batched).** A calculator MUST
  (1) declare `supports(calc, ::Type{ElPhDataOuterKBatched}) = true` and (2) implement
  `run_calculator!(calc, p::ElPhDataOuterKBatched, ctx)`. The payload `p` carries the batch's
  `ep_kq` / `g2` / `ωq` on the device plus `ik`, `ikqs`, `ibandk_offset`. The loop calls it once
  per `(k, {k+q})` batch (outer k is still serial: one `OuterIteration` bracket + payload per k).
- **Outer-q loop (`run_eph_over_q_and_k`, outer q / inner k batched).** A calculator MUST
  (1) declare `supports(calc, ::Type{ElPhDataOuterQBatched}) = true` and (2) implement
  `run_calculator!(calc, p::ElPhDataOuterQBatched, ctx)`. The payload carries `ep_kq`, k/k+q
  energies and eigenvectors, `wtk`, `xks`, `iq` on the device. The per-q device accumulator is
  bracketed by `calculator_begin!/end!(calc, OuterIteration(), ctx)` (the same brackets the CPU
  loop uses — the batched path is selected by `ctx.mode`, and the device buffer is allocated via
  `ctx.backend`), and the per-k device scratch is declared via
  `eph_batched_bytes_per_point(calc, ElPhDataOuterQBatched)` for the loop's memory-adaptive batch
  sizing. The k side is streamed per k-batch (host-staged, no whole-grid device stack), and the
  payload is trimmed to the batch's actual width — a consumer reads its own size from any field
  (e.g. `size(ep_kq, 4)`) and never sees a padded tail (the outer-k convention).
- Both loops fold their device-buffer byte accounting into `src/calculator/eph_device_staging.jl`:
  `_outer_{k,q}_staging_bytes(…)` return the loop's `(per_point, committed)` device-byte counts, and
  `plan_batch(backend, per_point, committed, cap; …)` turns those into the memory-adaptive batch
  width (committed-vs-free check + 30% headroom). `estimate_device_memory(model; nk, nkq, batch
  kwargs…)` calls the same byte functions to report committed + per-point bytes ahead of a run so
  batch sizes can be picked (and the drivers print them at `verbosity > 0`). These counts cover the
  driver's own device buffers (which scale with the grid/batch); actual device usage starts
  **~100–150 MB higher** because of a fixed CUDA library context/workspace floor (cuBLAS etc.)
  allocated lazily on the first in-loop kernel launch — inherent overhead, not a per-run buffer, so
  treat it as a fixed additive constant on top of the estimate. (A calculator's own device output,
  e.g. a full-band `(nw·nk)²` coupling array, is sized separately by the calculator, not by this
  estimate.)
- Either path is rejected with an error if a calculator does not opt in — there is no silent
  fallback to the per-`(k,q)` host path.
- A calculator implements these backend-generically (only `alloc(ctx.backend, …)` /
  `similar`/`copyto!`/broadcast/scatter-assignment) and adds no CUDA dependency of its own.

**Distinguishing the per-point host loop from the device-batched loop.** `LoopContext` carries a
loop *mode* (`PointMode` / `BatchedMode`) that names the loop shape independently of the backend.
Hooks a calculator shares between the two loop shapes — most commonly the `OuterIteration` brackets,
which the batched outer-k loop *also* fires per k — must key off the mode, not the backend: either
dispatch (`ctx::LoopContext{<:AbstractBackend, PointMode}` vs `{<:AbstractBackend, BatchedMode}`, as
`BoltzmannCalculator` / `EliashbergCalculator` do) or branch at runtime on `ctx.mode isa BatchedMode`
(as the outer-q calculators do). Dispatching these on `LoopContext{CPUBackend}` / `{<:GPUBackend}`
would misfire, because the batched loop's per-iteration bracket runs on a device backend but must not
trigger the per-point reduction. `ctx.backend` stays reserved for allocation / `free_bytes` /
`synchronize`.

### Full-band interpolation on the GPU, with energy windows (design note)

The list-batched drivers interpolate full bands: every k in a batch shares the same `nband`, and
the e-ph stacks are sized with no `nband_bound` slack. A per-k variable `nband` would break the
single large GEMMs that make the GPU fast, and the extra bands are cheap on the GPU. Energy windows
are nonetheless supported without shrinking the dense GEMMs:

- **Outer-k**: the k side is rotated by only an `nbandk_max`-wide *contiguous* window of eigenvector
  columns per k (`ibandk_offset` positions it over that k's in-window range); the window itself is
  applied in the calculator scatter, where out-of-window states map to `imap == 0` and are skipped.
- **Outer-q**: out-of-window k+q eigenvector columns are masked to zero, so those matrix elements
  are exactly 0 — reproducing the CPU's `for m in el_kq.rng` window loop.

Full-band runs are the special case `nbandk_max = nw`, `ibandk_offset = 0` (shapes/results
unchanged). No window handling is needed in the calculator beyond addressing its imaps.

## Abandoned (tried, decided against)

- **Per-k GPU interpolation.** An early extension-local `CuNormalWannierInterpolator` issued
  thousands of tiny per-k GEMV + eigensolve launches and was launch-bound (slower than CPU on small
  systems). Removed and superseded by the generic device `WannierObject` + the generalized
  `BatchedWannierInterpolator`. Not planned to return.

## Deferred (may do later)

- **In-place workspace drivers** (workspace-backed scratch instead of per-call `similar()`) were
  benchmarked and validated bit-identical, but the gain is small on the GPU (CUDA's pool already
  recycles device buffers), so it is deferred. Best done together with the calculator loop, where
  one workspace allocated at loop setup is reused across all (k, q).
- **Energy window and long-range/polar on the GPU** — left on the CPU per-k path for now.
- **MPI / multi-GPU** for the GPU loop — not in this foundation.
- **Backend as a type parameter instead of a `use_gpu` keyword (future).** The backend is
  currently selected by the `use_gpu` keyword + `to_device`. A cleaner long-term design might
  carry it as a type (e.g. dispatch the loop on the array backend). Note a `ModelGPU` that puts
  the whole `Model` on the device is *not* obviously right: `Model` is large, and one may want
  it resident on the CPU while only the calculation runs on the GPU.
- **Keep `el_kq_save` (k+q electron states) on the device (future).** The GPU loop already hoists
  every k+q eigenvector onto the device once (`ukqs_all_dev`); keeping the `el_kq_save` states
  themselves device-resident would avoid the host staging entirely, but needs a states refactor.
- **A dense-grid `Kpoints` type with integer-hash lookup (future).** The q-index lookup in the GPU
  loop special-cases a "full grid" (iq == hash+1) vs a fallback `Dict` (`GridKpoints` uses a Dict
  to allow huge `ngrid` with few points). A dedicated type for the common case — `ngrid` small
  enough that a `prod(ngrid)` array fits — could carry an `is_full` field and use a simple integer
  hash, replacing the special-casing here.
- **A QR-based batched eigensolve (future).** The batched eigensolve uses `CUSOLVER.heevjBatched!`
  (Jacobi). A QR-based `HEEV` (e.g. via cuSolverDx) may be faster/more accurate for the small
  matrices here; worth evaluating, but not in this PR.
- **Parametrize `Model` over its `WannierObject` array type (future).** Widening `WannierObject`
  to `WannierObject{T, AT}` turned `WannierObject{FT}` into a `UnionAll`, so `Model`'s Wannier
  fields (`el_ham`, `el_pos`, …) are currently pinned to the concrete host type
  `HostWannierObject{FT} = WannierObject{FT, Matrix{Complex{FT}}}` to stay type-stable. That
  hard-codes host storage; if a device-resident `Model` is ever wanted, add a per-field (or a
  shared) array-type parameter to `Model` instead of the host pin. Tied to the backend-as-a-type
  item above.

## Conventions

- **Device arrays use a `_dev` suffix** (e.g. `epmat_dev`, `ukqs_all_dev`), not a `gpu_` prefix.
  Host copies of device results drop the suffix (e.g. `E = Array(E_dev)`).

## Files

- `Project.toml` — `[weakdeps]`, `[extensions]`, `[compat] CUDA`.
- `src/wannier/WannierObject.jl` — array-type parameter `AT`; constructor fix.
- `src/wannier/WannierInterpolator.jl` — declare/export `to_device`.
- `src/wannier/batched_interpolator.jl` — backend-generic buffers + GEMM phase; new
  `get_fourier_batched!`. Per-k API unchanged. Pure Fourier only.
- `src/wannier_to_bloch_batched.jl` — **new**; `eigvals_batched`/`eigen_batched` (CPU), the
  `get_el_eigen[_valueonly]_batched` and e-ph drivers (per-k/q and list-batched), and the
  `batched_gemm!` primitive. Included after `wannier_to_bloch.jl`. All backend-generic.
- `ext/ElectronPhononCUDAExt.jl` — `to_device(::WannierObject)`, `eigvals_batched`/
  `eigen_batched` (`heevjBatched!`), `batched_gemm!` (`gemm_strided_batched!`), and the fused
  rotation / window-scatter kernels.
- `src/calculator/run_eph_over_k_and_kq.jl` — `use_gpu` / `nq_batch_max` / `nk_outer_batch_max`
  keywords; backend-generic `_loop_eph_over_k_and_kq_gpu` and the device-native calculator payload.
  CPU path unchanged.
- `benchmark/bench_el_eigen_gpu.jl`, `benchmark/bench_eph_gpu.jl`,
  `benchmark/bench_eliashberg_loop_gpu.jl` — CPU-vs-GPU benchmarks.
- `test/test_gpu.jl` — GPU-guarded tests (skip when CUDA is unavailable or the Pb data dir is
  absent); wired into `runtests.jl`.

## Testing

GPU tests are guarded with `CUDA.functional()`, so the suite passes on CPU-only machines. The
e-ph drivers and calculator loop are also validated on the CPU (always-run testset) by comparing
the batched path against the independent per-k/q reference.
