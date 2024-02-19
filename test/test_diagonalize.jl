using ElectronPhonon
using Test
using Random
using LinearAlgebra

@testset "diagonalize" begin
    Random.seed!(123)
    A = randn(ComplexF64, 5, 5)
    A .+= A'
    A_copy = copy(A)

    F = eigen(A)

    values = zero(F.values)
    @test F.values ≈ solve_eigen_el_valueonly!(values, copy(A))
    @test F.values ≈ values

    values = zero(F.values)
    solve_eigen_ph_valueonly!(values, copy(A))
    @test F.values ≈ values.^2 .* sign.(values)

    vectors = similar(A)
    values = zero(F.values)
    @test F.values ≈ solve_eigen_el!(values, vectors, A)[1]
    @test F.values ≈ values
    @test vectors .* F.values' ≈ A * vectors
    @test A ≈ A_copy

    mass = abs.(rand(5)) .+ 1
    vectors = similar(A)
    values = zero(F.values)
    solve_eigen_ph!(values, vectors, A, mass)
    @test F.values ≈ values.^2 .* sign.(values)
    @test (vectors .* sqrt.(mass)) .* F.values' ≈ A * (vectors .* sqrt.(mass))
    @test A ≈ A_copy
end
