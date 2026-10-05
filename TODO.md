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

- [ ] JML should review `TiledDeviceOutput` (the Sᵢ tiling machinery: `tile_begin!` / `tile_download!` /
  `tile_offset` / `tile_length`, used by `BoltzmannCalculator`'s batched path).

- [x] ~~Reconsider whether `backend` should be built inside `_loop_eph_over_k_and_kq_batched` rather
  than in `_setup_eph_over_k_and_kq`.~~ Subsumed: `backend` is now a user-facing driver keyword, so it
  is built by the caller and nothing inside the package resolves it. (The item's stated rationale was
  also wrong: `gpu_backend()` builds an EMPTY prototype, `GPUBackend(CuArray{ComplexF64}(undef, 0))`
  — it never wrapped `epmat_dev.op_r`.)

- [ ] Device phonon builder for polar models. On a GPU backend `compute_phonon_states_batched`
  solves only `e` and `u` of a non-polar model on the device (Fourier transform of the short-range
  dynamical matrix plus the batched eigensolve); a polar model or any of `vdiag` /
  `eph_dipole_coeff` / `eph_r_coeff` is built on the host and copied. A device build needs the
  long-range dynamical-matrix term and device versions of the velocity and dipole-coefficient
  kernels.
