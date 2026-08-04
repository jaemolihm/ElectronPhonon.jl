# TODO

## Code style

- [x] Global dash sweep: every Unicode minus `−` (U+2212) and en-dash `–` (U+2013) replaced with the
  ASCII hyphen-minus `-` (U+002D) across the source, tests, benchmarks, examples, and maintained docs
  of ElectronPhonon.jl and MigdalEliashberg.jl. (Working notes / progress docs are left as-is.) The
  rule is documented in the EPjl-developer and EPjl-reviewer agent definitions.

## GPU

- [ ] Rename `GPUBackend(arr::GPUArray)` -> `GPU{GPUArray}` (as in DFTK)? Investigate whether/how this
  is expressible in DFTK. See `src/architecture.jl`:
  https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/architecture.jl and follow the pattern in
  https://docs.dftk.org/stable/developer/gpu_computations/ .

- [x] Remove all the `nk_batch+1:nk_batch_max` dummy padding fills — done at both sites
  (`run_eph_over_k_and_kq.jl`, `run_eph_over_q_and_k.jl`). The batched interpolation/eigensolve/kernels
  are handed contiguous width-`nk_batch` trailing-prefix views of the max-width staging buffers, so a
  partial final batch computes only its own columns. `get_eph_Rq_to_kq_batched!` now asserts its
  workspace sizes with `>=` and prefix-views them internally, as its sibling
  `get_eph_kR_to_kq_batched!` already did.

- [ ] JML should review `TiledDeviceOutput` (the Sᵢ tiling machinery: `tile_begin!` / `tile_download!` /
  `tile_offset` / `tile_length`, used by `BoltzmannCalculator`'s batched path).

- [x] Unify the per-point/batched e-ph loop signatures — done for both driver families. Each pair
  takes the identical positional list of what BOTH need (outer-k: `(model, kpts, qpts, kqpts,
  el_k_save, el_kq_save, ph_save, precompute_ph, backend)`; outer-q: `(model, kpts, qpts, el_k_save,
  ph_save, eph_buffers, backend)`) and its own path-specific data as individual keyword arguments
  (per-point: the host interpolators/channels, resp. the precomputed k+q states; batched:
  `epmat_dev`, resp. `el_ham_dev`). `backend` is shared positional #9 rather than a batched-only
  kwarg, so the per-point loops no longer hardcode `CPUBackend()` (`run_eph_over_k_and_q` too, which
  has no batched twin).

- [x] ~~Reconsider whether `backend` should be built inside `_loop_eph_over_k_and_kq_batched` rather
  than in `_setup_eph_over_k_and_kq`.~~ Subsumed: `backend` is now a user-facing driver keyword, so it
  is built by the caller and nothing inside the package resolves it. (The item's stated rationale was
  also wrong: `gpu_backend()` builds an EMPTY prototype, `GPUBackend(CuArray{ComplexF64}(undef, 0))`
  — it never wrapped `epmat_dev.op_r`.)

- [x] ~~Clean up `calculator_begin!` for `BoltzmannCalculator`~~ — done, but not as written. There
  were no CPU no-op brackets to restrict: the `OuterIterationBatch` brackets were already correctly
  keyed on the loop MODE (`LoopContext{<:AbstractBackend, BatchedMode}`) and are backend-agnostic,
  which is exactly right now that batched-on-`CPUBackend` is a supported configuration. The real
  defect was the flag: `calc.on_gpu = backend isa GPUBackend` inferred the loop shape from the
  backend, which builds the wrong buffers under CPU+batched. It is now `calc.batched = mode isa
  BatchedMode`, from the positional `mode::LoopMode` the drivers pass to `setup_calculator!`.

- [x] Clean up the `LoopContext` construction at the batch/per-k scope — done. The `BatchedMode`
  convenience constructor is keyword-only (`batch` / `outer_index` / required `n_batch_max`), so
  argument 3 is never positional there, and the per-k context is `with_outer_index(ctx_batch, ik)`.
