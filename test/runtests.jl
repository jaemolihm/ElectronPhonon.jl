using EPW
using Test

@testset "EPW.jl" begin
    include("test_occupation.jl")
    include("test_kpoints.jl")
    include("test_symmetry.jl")
    include("test_fourier.jl")
    include("test_diagonalize.jl")
    include("test_iterativesolvers.jl")
    include("test_symmetry_operator.jl")
    include("test_unfold.jl")
    include("test_velocity.jl")
    include("test_epmat.jl")
    include("test_ElectronState.jl")
    include("test_plot_bandstructure.jl")
    include("test_check_model_symmetry.jl")

    # Integration tests
    include("test_cubicBN_eigenvalues.jl")
    include("test_cubicBN_spectral.jl")
    include("test_cubicBN_selfen.jl")
    include("test_transport.jl")

    # Boltzmann routines
    include("boltzmann/test_hdf5.jl")
    include("boltzmann/test_QMEVector.jl")
    include("boltzmann/test_QMEModel.jl")
    include("boltzmann/test_covariant_derivative.jl")
    include("boltzmann/test_el_constant_rta.jl")
    include("boltzmann/test_el_transport_semiconductor.jl")
    include("boltzmann/test_el_transport_metal.jl")
    include("boltzmann/test_el_master_equation.jl")
    include("boltzmann/test_el_ac_conductivity.jl")
    include("boltzmann/test_el_hall_conductivity.jl")
    include("boltzmann/test_el_transport_finite_efield.jl")
end
