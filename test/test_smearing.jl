using Test
using ElectronPhonon
const EP = ElectronPhonon

@testset "SmearingType" begin
    η = 0.01

    # Gaussian: normalized exp(-(Δe/η)^2) / (√π η)
    g = SmearingType(:Gaussian, η)
    @test g(0.0) ≈ 1 / (sqrt(π) * η)
    @test g(0.02) ≈ exp(-(0.02 / η)^2) / (sqrt(π) * η)
    @test g(-0.02) ≈ g(0.02)                       # even

    # Lorentzian: (η/π) / (Δe^2 + η^2)
    l = SmearingType(:Lorentzian, η)
    @test l(0.0) ≈ η / π / η^2
    @test l(0.02) ≈ (η / π) / (0.02^2 + η^2)
    @test l(-0.02) ≈ l(0.02)                       # even

    # Unit-integral normalization (numeric quadrature). The Lorentzian's algebraic tails truncate
    # slowly (∫ over ±R misses O(η/R)), so this is a loose sanity check, not a precision test.
    for δ in (g, l)
        xs = range(-5.0, 5.0; length = 400001)
        integ = sum(δ, xs) * step(xs)
        @test integ ≈ 1.0 rtol = 3e-3
    end

    # Invalid method errors clearly.
    @test_throws ArgumentError SmearingType(:Nonexistent, η)

    # isbitstype is required to store a SmearingType in a CuArray and evaluate it in a GPU kernel.
    @test isbitstype(SmearingType{Float64})
    @test isbitstype(SmearingType{Float32})

    # FT is preserved and evaluation stays in that type.
    @test SmearingType(:Gaussian, 0.01f0) isa SmearingType{Float32}
    @test SmearingType(:Gaussian, 0.01f0)(0.0f0) isa Float32
end
