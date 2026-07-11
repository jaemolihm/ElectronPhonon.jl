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
- **Full-band on the GPU.** No per-k energy window; see the design note.
- **The GPU path is fully GPU — no silent fallback.** The GPU calculator loop requires every
  calculator to implement the batched device hook (`allow_eph_batched`). It must not silently
  fall back to the per-`(k,q)` host `run_calculator!` path, which would be a hard-to-spot
  performance cliff — a calculator that forgets to opt in should fail loudly instead.

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

`run_eph_over_k_and_kq` gains a `use_gpu` keyword (with `q_batch_size` / `k_batch_size`) that
branches to a backend-generic `_loop_eph_over_k_and_kq_gpu`. The CPU loop and per-k/q CPU e-ph
functions are unchanged. The GPU loop: `to_device(model.epmat)` once; processes a tile of
outer-k with one list-batched `get_eph_RR_to_kR_batched!`; batches the inner `ikq` loop (chunked
by `q_batch_size`) through one `get_eph_kR_to_kq_batched!`.

A calculator can opt into a **device-native hook** so the e-ph matrix for a whole `(k, {k+q})`
chunk stays on the device and the reduction/scatter happens there:

- **To run on the GPU, a calculator must (1) define `allow_eph_batched(calc) = true` and (2)
  implement `run_calculator_batched!(calc, ep_kq, ωq, ik, ikqs)`.** The GPU path is batched-only:
  the loop keeps `ep_kq` on the device and calls the batched hook once per chunk. A calculator that
  does not opt in is rejected with an error — there is no silent fallback to the per-`(k,q)` host
  path.
- A calculator can implement this backend-generically (only `similar`/`copyto!`/broadcast/
  scatter-assignment) and add no CUDA dependency of its own.

### Full bands on the GPU (design note)

The list-batched drivers and the GPU loop are full-band only: every k in a batch shares the same
`nband`, and `ep_ekpR_all` / `ep_kq_all` are sized with no `nband_bound` slack. A per-k variable
`nband` would break the single large GEMMs that make the GPU fast, and the extra bands are cheap
on the GPU. Energy windowing — which mainly helps when only a few bands matter — stays on the CPU
per-k path (`get_eph_RR_to_kR!` etc. keep their `nband < nband_bound` support). When wiring the
GPU path into a calculation, pass full `nband` (= `nw` for the electron side); no window handling
is needed.

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
- **A memory-estimate helper (future).** A utility that reports the device memory a run needs
  (given the model and `q_batch_size` / `k_batch_size`) would make it easier to pick chunk sizes
  and to fail early instead of OOM-ing mid-run.
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
- `src/calculator/run_eph_over_k_and_kq.jl` — `use_gpu` / `q_batch_size` / `k_batch_size`
  keywords; backend-generic `_loop_eph_over_k_and_kq_gpu` and the device-native calculator hook.
  CPU path unchanged.
- `benchmark/bench_el_eigen_gpu.jl`, `benchmark/bench_eph_gpu.jl`,
  `benchmark/bench_eliashberg_loop_gpu.jl` — CPU-vs-GPU benchmarks.
- `test/test_gpu.jl` — GPU-guarded tests (skip when CUDA is unavailable or the Pb data dir is
  absent); wired into `runtests.jl`.

## Testing

GPU tests are guarded with `CUDA.functional()`, so the suite passes on CPU-only machines. The
e-ph drivers and calculator loop are also validated on the CPU (always-run testset) by comparing
the batched path against the independent per-k/q reference.
