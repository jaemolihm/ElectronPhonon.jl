# ElectronPhonon.jl

[![CI](https://github.com/jaemolihm/ElectronPhonon.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jaemolihm/ElectronPhonon.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/jaemolihm/ElectronPhonon.jl/graph/badge.svg?token=P7BO11SX2C)](https://codecov.io/gh/jaemolihm/ElectronPhonon.jl)

Julia implementation of electron-phonon coupling using Wannier functions

## Installation

To install ElectronPhonon.jl, first install the TetrahedronIntegration.jl dependency, then install ElectronPhonon.jl:

```julia
using Pkg
Pkg.add(url="https://github.com/jaemolihm/TetrahedronIntegration.jl.git")
Pkg.add(url="https://github.com/jaemolihm/ElectronPhonon.jl.git")
```

## Documentation

- [Writing your own calculator](docs/writing_a_calculator.md) — how to implement an
  `AbstractCalculator` to compute a custom property during an e-ph driver pass (with a runnable
  minimal example and the threading contract).
- [GPU acceleration](README_GPU.md) — the CUDA package-extension path and the device-native
  calculator interface.

