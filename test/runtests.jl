using EPW
using Test

@testset "EPW.jl" begin
    include("test_occupation.jl")
    include("test_fourier.jl")
    include("test_diagonalize.jl")

    # Integration tests
    include("test_cubicBN_selfen.jl")
    include("test_cubicBN_transport.jl")
end
