using EPW
using Test

@testset "EPW.jl" begin
    include("test_fourier.jl")
    include("test_diagonalize.jl")
end
