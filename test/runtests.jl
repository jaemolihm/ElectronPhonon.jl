using EPW
using Test

@testset "EPW.jl" begin
    include("test_fourier.jl")
    include("test_diagonalize.jl")

    # Integration tests
    include("test_cubicBN.jl")
end
