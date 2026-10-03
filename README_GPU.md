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
- The e-ph calculator loop `run_eph_over_k_and_kq` via a `batched` branch, including a
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
- **Energy windows as box storage.** The state containers hold each point's in-window bands in its
  first columns (a box of the largest window's width), so every batched GEMM keeps one uniform shape
  and the calculators skip the box padding through their index maps. See the design note.
- **The GPU path is fully GPU — no silent fallback.** On every backend the drivers hand the
  calculators device-resident `EPBlock`s; a calculator that does not support the loop order fails
  at the driver entry, and nothing falls back to a per-`(k,q)` host loop.

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

`get_interpolator(obj; fourier_mode="batched", backend, batch_size, nk_hint)`
(`src/wannier/batched_interpolator.jl`) is the single batching mechanism for CPU and GPU. It is the
only public constructor: the `BatchedFourierCore` engine it composes is internal.

- **`backend` says where the buffers live**, and it must be the backend `obj.op_r` is already on —
  the constructor checks that and names the array if it is not, rather than running a silent mixed
  host/device broadcast. It defaults to `CPUBackend()`, so a GPU caller passes it explicitly:

  ```julia
  ham = ElectronPhonon.to_device(backend, model.el_ham)
  itp = get_interpolator(ham; fourier_mode="batched", backend, nk_hint = kpts.n)
  ```

- **The batched modes need an in-memory `op_r`**, so a `DiskWannierObject` is served by the per-k
  `"normal"` / `"gridopt"` modes only; `"batched"` / `"batched-gridopt"` with one raise an
  `ArgumentError` naming those two. `"batched-gridopt"` is host-only and rejects a `GPUBackend`.

- **`batch_size` is the block width**, and its default is keyed on the backend: a fixed 32 on
  `CPUBackend` (the balance for the sequential per-k query API), and on a `GPUBackend` a byte
  budget, `clamp(fld(GPU_FOURIER_BATCH_BYTES, 16*(nr + ndata)), 1, nk_hint)` with a 1 GiB constant.
  **Pass `nk_hint = <your k-count>`** whenever you know it: it caps the scratch at the grid size, so
  a 50-point run allocates 50 columns rather than a budget-sized buffer. Unlike a bare
  `batch_size = kpts.n` it is a `min` against the budget, so it stays bounded however large the grid
  — that is what keeps a 7.6 M-point q list from asking for a 75 GB phase buffer.

- The phase computation is a single fused broadcast (no scalar indexing, no `nr × batch` real
  scratch), so it runs on any backend and is faster on the CPU too:

  ```julia
  # irvec_mat :: (nr × 3) real, on backend (built once in the constructor)
  # xkmat     :: (3 × nk)  real, on backend (staged once per call by get_fourier_batched!)
  build_fourier_phase!(phase, irvec_mat, xkmat)   # (nr × nk) complex, one broadcast
  mul!(out, op_r, phase)                          # (ndata × nk) GEMM → op_k for all k
  ```

  `build_fourier_phase!` is stateless and writes into a caller-owned destination, so a caller whose
  k-list is loop-invariant can build one phase matrix and reuse it — which is what the GPU outer-k
  e-ph loop does (next section).

- The per-k query API (`register_kpoints!` + sequential `get_fourier!`) used by the calculators
  is unchanged; its `cached_results` buffer is allocated on the first `register_kpoints!`, so a
  whole-batch caller never pays for it. The whole-batch entry point
  `get_fourier_batched!(out, itp, xk_list)` returns the entire `(ndata, nk)` result on the backend —
  the GPU path uses this, avoiding any per-k device→host copy. This is the only layer that stages a
  host k list; it also accepts an already-staged `(3 × nk)` device matrix, while the drivers above
  it (`wannier_to_bloch_batched.jl`) take a `Vector{Vec3}` and nothing else.

### Batched Hermitian eigensolve + band-eigenvalue drivers

Diagonalization lives in `src/wannier_to_bloch_batched.jl` (keeping `src/wannier/` pure Fourier).
Two batched eigensolves over a stack `Hk :: (nw, nw, nk)`: `eigvals_batched` (values) and
`eigen_batched` (values + vectors). CPU methods loop over LAPACK `syev!`; the extension uses
`CUSOLVER.heevjBatched!` (batched Jacobi). The `nw ≤ 32` figure often quoted for that solver is
a *performance* characteristic, not a correctness bound (verified correct to `nw=256`,
agreeing with LAPACK to ~1e-11), so no size guard is imposed. Accuracy matches LAPACK for
eigenvalues *and* eigenvectors (relative residual ≤ 3e-15 for `nw ≤ 16`); CUDA.jl runs the Jacobi
sweeps at `tol = eps(Float64)`.

`eigvals_batched` / `eigen_batched` chunk the batch at `heevj_batch_max` (2^16 for small `nw`).
Two reasons: the solver reports its workspace size as a 32-bit int, and that workspace is ~16 kB
per matrix and is cached on the cuSOLVER handle for the lifetime of the process — neither GC nor
`CUDA.reclaim()` returns it, so an oversized chunk permanently parks memory the device-resident
e-ph tiles need. Chunking is exact (results are batch-position independent).

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

The list-batched kernels (`eph_rotate_kR_batched!`, `get_eph_kR_to_kq_batched!`,
`eph_apply_rotations_rqkq!`) live in `wannier_to_bloch_batched.jl` and run on the backend of their
arrays.

### Calculator integration

The calculators receive one `EPBlock` per (outer point, inner tile): `run_calculator!(calc,
block::EPBlock{OuterKLoop|OuterQLoop}, ctx::LoopContext)`, one method per loop order, on every
backend. The full spec is in the docstrings of `src/calculator/AbstractCalculator.jl`, and
`docs/writing_a_calculator.md` is the tutorial. This section covers the device side.

`run_eph_over_k_and_kq` / `run_eph_over_q_and_k` are one loop, `_run_eph` in
`src/calculator/run_eph.jl`, with an engine per order in `src/calculator/eph_engine.jl`
(`OuterKEngine`, `OuterQEngine`). They take the backend and the two widths:

- `backend :: AbstractBackend = CPUBackend()` — where arrays live. Pass
  `backend = ElectronPhonon.gpu_backend()` for a GPU run; `model.epmat` is uploaded once per run and
  the backend is carried in `LoopContext` as `ctx.backend`.
- `n_outer_batch` — outer points per stage-1 batch and per calculator bracket; `n_inner_tile` —
  inner points per block, sized to free device memory on a GPU.

The loop holds no device-specific code, so it runs on host arrays on a `CPUBackend`, where the
inner points are split into `nchunks_threads` thread chunks, each with its own tile buffers
(`eng.tiles[chunk]`); a device run uses one chunk on the calling task. A CPU run does **not** cover
the CUDA kernels (`_bte_window_accumulate_kernel!`, `_window_scatter_kernel!`, the fused rotation
kernels, `CUBLAS.gemm_strided_batched!`, the cuSOLVER batched eigensolve): those need the GPU box.

Each order is two stages. **Outer k:** stage 1 is one batched Fourier transform over R_e
and one `eph_rotate_kR_batched!` over the outer batch, which stores the kR intermediate in the
**k+q convention**
(`g̃(k, R_p) = conj(exp(2πi R_p·x_k)) · g(k, R_p)`, folded in via `additional_phase`), so the
stage-2 Fourier phase `exp(2πi R_p·x_{k+q})` of a k+q tile is the same for every k of the batch:
the loop is `k-batch -> k+q tile (phase built once) -> k -> block`, and stage 2 is one
`get_eph_kR_to_kq_batched!` per block with the phonon basis fused into its right rotation.
**Outer q:** stage 1 is `g(R_e, q)` for the outer batch on the device (one GEMM over the q batch,
then the phonon-basis rotation as a batched GEMM), stage 2 one Fourier transform over R_e and one
`eph_apply_rotations_rqkq!` per `(q, k tile)` after the k+q states of the tile are solved
(`compute_electron_states_batched!`, into the leading `maximum(nband)` columns of the tile buffers, so the block's k+q box is its widest
window). The model's `epmat_outer_momentum` must match the order (`"el"` for outer k, `"ph"` for
outer q), so stage 1 always contracts the column R of `epmat`. The polar
dipole term is added on the block for both orders (`finish_ep!`).

A block is `ep` `(nband_max_kq, nband_max_k, nmodes, nb)` with the states, phonons, weights and
indices of its pairs; the side shared by the block has extent 1 (see `EPBlock`). Under outer k a
k's k+q tiles are not contiguous (the tile loop is outside the k loop), so a per-k reduction is
done in the brackets around the outer batch (`ctx.batch`).

Memory: `engine_bytes` counts the engine's buffers next to their `alloc` calls, each calculator
adds its `eph_batched_bytes_per_point` triple `(; persistent, per_outer, per_pair)`, and
`plan_batch(backend, per_point, committed, cap; …)` turns the sum into the inner-tile width
(committed-vs-free check + 30% headroom) before the engine is built. `estimate_device_memory(model;
nk, nkq, …)` reports the same counts ahead of a run. Actual device usage starts **~100-150 MB
higher** because of a fixed CUDA library context/workspace floor (cuBLAS etc.) allocated lazily on
the first kernel launch.

A calculator implements its block method backend-generically (only `alloc(ctx.backend, …)` /
`similar`/`copyto!`/broadcast/scatter-assignment) and adds no CUDA dependency of its own.

### Full-band interpolation on the GPU, with energy windows (design note)

The list-batched drivers need one uniform band width per batch: a per-k variable `nband` would break
the single large GEMMs that make the GPU fast. The state containers therefore store a **box**: each
point's in-window bands in its first `nband[j]` columns (physical band `iband_offset[j] + n`), padded
to the container's `nband_max`. Every e-ph matrix inherits the box on both electron sides, so a
narrow window shrinks every per-(k, q) object by `nw / nband_max` on each side. The padding is
undefined; the calculators build their state-index maps in box coordinates (`_indmap_to_device(backend,
states)` on the selection the containers were built from, 0 on the padding), so the existing scatter
kernels skip it with no window code. Array bounds cannot catch a read of the band padding, so every
reader is bounded by `nband` or by such a map. On the point axis every array a block hands over has
exactly the block's points, so a reader takes its extents from the arrays themselves and a size
mismatch between them is detectable; the scatter kernels check the sizes of their inputs on entry,
since neither their `@inbounds` host loops nor the device kernels check bounds.
Full-band runs are the special case `nband_max = nw`, `iband_offset = 0`.

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
- **Long-range/polar in the outer-k loop** — refused there for now (the outer-q loop adds it).
- **MPI / multi-GPU** for the GPU loop — not in this foundation.
- **Backend as a type parameter instead of a backend object (future).** The `use_gpu` keyword is
  gone: the backend is now the user-facing `backend::AbstractBackend` argument, and the loop shape
  is the separate `batched` keyword. Renaming `GPUBackend(proto)` to a DFTK-style `GPU{AT}` was
  considered and **decided against**: `similar(proto, ...)` propagates CUDA.jl 6's memory-type
  parameter where a bare type constructor would not, and nothing dispatches on `AT`. Note a
  `ModelGPU` that puts the whole `Model` on the device is *not* obviously right either: `Model` is
  large, and one may want it resident on the CPU while only the calculation runs on the GPU.
- **A dense-grid `Kpoints` type with integer-hash lookup (future).** The q-index lookup in the GPU
  loop special-cases a "full grid" (iq == hash+1) vs a fallback `Dict` (`GridKpoints` uses a Dict
  to allow huge `ngrid` with few points). A dedicated type for the common case — `ngrid` small
  enough that a `prod(ngrid)` array fits — could carry an `is_full` field and use a simple integer
  hash, replacing the special-casing here.
- **A QR-based batched eigensolve (future).** The batched eigensolve uses `CUSOLVER.heevjBatched!`
  (Jacobi). A QR-based `HEEV` (e.g. via cuSolverDx) may be faster for the small matrices here;
  worth evaluating, but not in this PR. Accuracy is not a motivation — Jacobi is already at
  machine precision. Note that `cusolverDnXsyevBatched` is *not* the answer: it is 2–3× faster per
  matrix but needs ~1.05 MB of workspace per matrix (65× heevj), which caps one call at ~72k
  matrices on an 80 GB A100 and competes with the device-resident e-ph tiles.
- **Parametrize `Model` over its `WannierObject` array type (future).** Widening `WannierObject`
  to `WannierObject{T, AT}` turned `WannierObject{FT}` into a `UnionAll`, so `Model`'s Wannier
  fields (`el_ham`, `el_pos`, …) are currently pinned to the concrete host type
  `HostWannierObject{FT} = WannierObject{FT, Matrix{Complex{FT}}}` to stay type-stable. That
  hard-codes host storage; if a device-resident `Model` is ever wanted, add a per-field (or a
  shared) array-type parameter to `Model` instead of the host pin. Tied to the backend-as-a-type
  item above.

## Conventions

- **Device arrays use a `_dev` suffix** (e.g. `epmat_dev`, `wtkq_dev`), not a `gpu_` prefix.
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
- `src/calculator/run_eph.jl` — both drivers as one `_run_eph` (entry checks, state containers,
  `plan_batch`, brackets) with the two loop bodies; `src/calculator/eph_engine.jl` — the
  `OuterKEngine` / `OuterQEngine` stages and `engine_bytes`. Backend-generic; `backend`,
  `n_outer_batch`, `n_inner_tile` and `nchunks_threads` select placement and widths.
- `benchmark/bench_el_eigen_gpu.jl`, `benchmark/bench_eph_gpu.jl`,
  `benchmark/bench_eliashberg_loop_gpu.jl` — CPU-vs-GPU benchmarks.
- `test/test_gpu.jl` — GPU-guarded tests (skip when CUDA is unavailable or the Pb data dir is
  absent); wired into `runtests.jl`.

## Testing

GPU tests are guarded with `CUDA.functional()`, so the suite passes on CPU-only machines. The
e-ph drivers and calculator loop are also validated on the CPU (always-run testset) by comparing
the batched path against the independent per-k/q reference.
