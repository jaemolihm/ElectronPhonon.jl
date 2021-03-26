using Test
using EPW

@testset "Smearing functions" begin
    using EPW: occ_fermion, occ_fermion_derivative
    e = 0.04
    T = 1.5
    δ = 1e-8

    @test EPW.occ_fermion(-Inf, T) == 1
    @test EPW.occ_fermion(Inf, T) == 0
    @test EPW.occ_boson(Inf, T) == 0

    @test abs((EPW.occ_fermion(e+δ, T) - EPW.occ_fermion(e-δ, T)) / (2 * δ)
             - EPW.occ_fermion_derivative(e, T)) < 1e-8
end