using EPW
using Test
using Random
using LinearAlgebra
using EPW.Diagonalize

@testset "diagonalize" begin
    Random.seed!(123)
    A = randn(ComplexF64, 5, 5)
    A .+= Adjoint(A)

    F = eigen(A)

    @test F.values ≈ solve_eigen_el_valueonly!(copy(A))

    vectors = similar(A)
    @test F.values ≈ solve_eigen_el!(vectors, copy(A))
    @test vectors .* F.values' ≈ A * vectors
end
