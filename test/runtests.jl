using ElectronPhonon
using Test

include("common_models_from_artifacts.jl")

@time @testset "ElectronPhonon.jl" begin
    # Basic tests
    include("test_occupation.jl")
    include("test_smearing.jl")
    include("test_kpoints.jl")
    include("test_symmetry.jl")
    include("test_wannier.jl")
    include("test_gpu.jl")  # skips gracefully when CUDA is unavailable
    include("test_diagonalize.jl")
    include("test_iterativesolvers.jl")
    # include("test_symmetry_operator.jl")
    # include("test_unfold.jl")
    include("test_velocity.jl")
    include("test_epmat.jl")
    include("test_ElectronState.jl")
    include("test_plot_bandstructure.jl")
    # include("test_check_model_symmetry.jl")
    include("test_screening.jl")
    include("test_calculator_guide.jl")  # runs the docs/writing_a_calculator.md example
    include("test_calculator_contract.jl")  # supports trait + driver fail-early checks

    # Integration tests
    include("test_cubicBN_eigenvalues.jl")
    include("test_cubicBN_spectral.jl")
    include("test_cubicBN_selfen.jl")
    # include("test_transport.jl")

    # Boltzmann routines
    include("boltzmann/test_hdf5.jl")
    include("boltzmann/test_QMEVector.jl")
    include("boltzmann/test_gpu_boltzmann_calculator.jl")  # GPU BTE scatter; GPU part skips w/o CUDA
    # include("boltzmann/test_QMEModel.jl")
    # include("boltzmann/test_covariant_derivative.jl")
    # include("boltzmann/test_el_constant_rta.jl")
    # include("boltzmann/test_el_transport_semiconductor.jl")
    # include("boltzmann/test_el_transport_metal.jl")
    # include("boltzmann/test_el_master_equation.jl")
    # include("boltzmann/test_el_transport_screening.jl")
    # include("boltzmann/test_el_ac_conductivity.jl")
    # include("boltzmann/test_el_hall_conductivity.jl")
    # include("boltzmann/test_el_transport_finite_efield.jl")
end
