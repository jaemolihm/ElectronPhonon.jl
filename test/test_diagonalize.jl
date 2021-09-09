using EPW
using Test
using Random
using LinearAlgebra

@testset "diagonalize" begin
    Random.seed!(123)
    A = randn(ComplexF64, 5, 5)
    A .+= Adjoint(A)

    F = eigen(A)

    values = zero(F.values)
    @test F.values ≈ solve_eigen_el_valueonly!(values, copy(A))
    @test F.values ≈ values

    vectors = similar(A)
    values = zero(F.values)
    @test F.values ≈ solve_eigen_el!(values, vectors, copy(A))[1]
    @test F.values ≈ values
    @test vectors .* F.values' ≈ A * vectors
end
