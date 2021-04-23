using Test
using Random
using EPW

@testset "Smearing functions" begin
    using EPW: occ_fermion, occ_fermion_derivative
    e = 0.04
    T = 0.02
    δ = 1e-8

    @test EPW.occ_fermion(-Inf, T) == 1
    @test EPW.occ_fermion(Inf, T) == 0
    @test EPW.occ_boson(Inf, T) == 0

    @test abs((EPW.occ_fermion(e+δ, T) - EPW.occ_fermion(e-δ, T)) / (2 * δ)
             - EPW.occ_fermion_derivative(e, T)) < 1e-8
end

@testset "Chemical potential" begin
    # Setup energy and weights
    Random.seed!(123)
    nband = 4
    nk = 5
    energy = zeros(nband, nk)
    for ik = 1:nk
        energy[1:2, ik] .= sort(rand(2)) # valence bands
        energy[3:4, ik] .= sort(rand(2)) .+ 3.0 # condunction band
    end
    weights = rand(nk)
    weights /= sum(weights)

    T = 0.02
    nband_valence = 2

    @test EPW.compute_ncarrier(Inf, T, energy, weights) ≈ nband
    @test EPW.compute_ncarrier(-Inf, T, energy, weights) ≈ 0

    for ncarrier in [1e-3, -1e-3]
        μ = EPW.find_chemical_potential(ncarrier, T, energy, weights, nband_valence)
        @test EPW.compute_ncarrier(μ, T, energy, weights) ≈ ncarrier + nband_valence
    end
end
