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

- [ ] Remove all the `nk_batch+1:nk_batch_max` dummy padding fills (e.g.
  `src/calculator/run_eph_over_k_and_kq.jl` ~L702) by passing the actual `nk_batch` through to the
  batched interpolation/kernels instead of padding partial tails with duplicated valid data to run on
  dense `nk_batch_max`-sized arrays.

- [ ] JML should review `TiledDeviceOutput` (the Sᵢ tiling machinery: `tile_begin!` / `tile_download!` /
  `tile_offset` / `tile_length`, used by `BoltzmannCalculator`'s batched path).

- [ ] Unify the CPU/GPU e-ph loop signatures (`_loop_eph_over_k_and_kq` vs
  `_loop_eph_over_k_and_kq_gpu`, `src/calculator/run_eph_over_k_and_kq.jl`). Decided direction (JML,
  PR #9): keep **two functions with different names**, but give them the **identical positional
  argument list** and push the path-specific data into **keyword arguments** — CPU kwargs
  `(epstates, ep_ekpRs, epmat, ep_ekpR_obj, dyn_threads, epmat_R, epobj_ekpR_R, ep_ekpR_Rs)`, GPU
  kwargs `(epmat_dev, backend)`. Constraint (`_setup`/`_loop` Core.Box rule): the CPU `_loop` must
  destructure any NamedTuple into locals at the top before `@threads`, never index it inside the
  threaded closure.

- [ ] Reconsider whether `backend` should be built inside `_loop_eph_over_k_and_kq_gpu` rather than in
  `_setup_eph_over_k_and_kq`. It currently lives in `_setup` because `_setup_calculators!` runs there
  and passes `backend` into `setup_calculator!`, and because `backend` wraps the once-uploaded
  `epmat_dev.op_r` as its allocation prototype (the loop reuses that device object). Moving it into
  the loop would re-plumb where `setup_calculator!` gets its backend.

- [ ] Clean up the `LoopContext` construction at the batch/per-k scope
  (`src/calculator/run_eph_over_k_and_kq.jl` ~L713/L728). The batch-scope `ctx_batch` and per-k
  `ctx_k` are built from positional constructors that are disambiguated by whether the argument is an
  `Integer` outer index (`ik`) or a `UnitRange` batch (`iks_batch`) — flagged as flaky in review.
  Consider a clearer, explicitly-named construction API for the two scopes.
