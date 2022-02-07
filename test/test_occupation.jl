using Test
using Random
using EPW

@testset "Smearing functions" begin
    e = 0.04
    T = 0.02
    δ = 1e-8

    @test occ_fermion(-Inf, T) == 1
    @test occ_fermion(Inf, T) == 0
    @test occ_boson(Inf, T) == 0
    @test occ_boson(T, T) ≈ 1 / (exp(1) - 1)
    @test occ_fermion(0, T) ≈ 1 / 2
    @test occ_fermion(T, T) ≈ 1 / (exp(1) + 1)

    @test abs((occ_fermion(e+δ, T) - occ_fermion(e-δ, T)) / (2 * δ)
             - occ_fermion_derivative(e, T)) < 1e-8
end

@testset "Chemical potential" begin
    # Setup energy and weights
    Random.seed!(123)
    nband = 4
    nk = 5
    energy = rand(nband, nk)
    energy[1:2, :] .-= 2.0 # Valence bands
    energy[3:4, :] .+= 2.0 # Conduction bands
    weights = rand(nk)
    weights /= sum(weights)

    T = 0.25
    nband_valence = 2

    @test EPW.compute_ncarrier(Inf, T, energy, weights) ≈ nband
    @test EPW.compute_ncarrier(-Inf, T, energy, weights) ≈ 0

    for ncarrier in [1e-3, -1e-3]
        # General function
        μ = EPW.find_chemical_potential(ncarrier + nband_valence, T, energy, weights)
        @test EPW.compute_ncarrier(μ, T, energy, weights) ≈ ncarrier + nband_valence

        # Semiconductor-specific function
        energy_e = vec(energy[3:4, :])
        weights_e = repeat(weights, inner=2)
        energy_h = vec(energy[1:2, :])
        weights_h = repeat(weights, inner=2)

        n_e = EPW.compute_ncarrier(μ, T, energy_e, weights_e)
        n_h = EPW.compute_ncarrier_hole(μ, T, energy_h, weights_h)
        @test n_e - n_h ≈ ncarrier

        μ_s = EPW.find_chemical_potential_semiconductor(ncarrier, T, energy_e, weights_e,
                                                        energy_h, weights_h)
        @test μ_s ≈ μ
    end
end
