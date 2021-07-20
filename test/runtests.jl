using EPW
using Test

@testset "EPW.jl" begin
    include("test_occupation.jl")
    include("test_kpoints.jl")
    include("test_symmetry.jl")
    include("test_fourier.jl")
    include("test_diagonalize.jl")

    # Integration tests
    include("test_cubicBN_eigenvalues.jl")
    include("test_cubicBN_spectral.jl")
    include("test_cubicBN_selfen.jl")
    include("test_cubicBN_transport.jl")

    # Boltzmann routines
    include("boltzmann/test_hdf5.jl")
    include("boltzmann/test_mobility.jl")
end
