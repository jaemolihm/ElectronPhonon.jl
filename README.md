# ElectronPhonon.jl

[![CI](https://github.com/jaemolihm/ElectronPhonon.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jaemolihm/ElectronPhonon.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/jaemolihm/ElectronPhonon.jl/graph/badge.svg?token=P7BO11SX2C)](https://codecov.io/gh/jaemolihm/ElectronPhonon.jl)

Julia implementation of electron-phonon coupling using Wannier functions

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jaemolihm/ElectronPhonon.jl.git")
```

## Documentation

- [Writing your own calculator](docs/writing_a_calculator.md) — how to implement an
  `AbstractCalculator` to compute a custom property during an e-ph driver pass (with a runnable
  minimal example and the threading contract).
- [GPU acceleration](README_GPU.md) — the CUDA package-extension path and the device-native
  calculator interface.


## Development

The package precompiles the EPW loader `load_model_from_epw_new`, which makes its first call fast
but lengthens every precompilation of the package. While editing the source, switch that workload
off in your development environment by adding to its `LocalPreferences.toml` (next to its
`Project.toml`):
```toml
[ElectronPhonon]
precompile_workload = false
```
or, with `Preferences` in that environment,
`using Preferences, ElectronPhonon; set_preferences!(ElectronPhonon, "precompile_workload" => false; force = true)`.
Users and CI keep the workload on.
