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
