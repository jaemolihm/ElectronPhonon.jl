using Test

@testset "epsilon_lindhard" begin
    using LinearAlgebra
    using EPW: LindhardScreeningParams, epsilon_lindhard

    screening_params = LindhardScreeningParams(
        degeneracy = 2,
        m_eff = 0.4,
        nlist = [-1.5e-6],
        ϵM = 4.5 * Mat3(I(3)),
        smearing = 0.05 * unit_to_aru(:eV)
    )
    @test epsilon_lindhard(Vec3(0., 0., 0.0), 0.0, screening_params) == [1]
    @test epsilon_lindhard(Vec3(0., 0., 0.1), 100.0, screening_params) ≈ [1.0] atol=1e-6
    @test epsilon_lindhard(Vec3(0., 0., 0.1), 0.0, screening_params) ≈ [1.0727349778159407]
    @test epsilon_lindhard(Vec3(0., 0., 0.1), 0.5, screening_params) ≈ [0.9998319286612122 + 2.478812334205997e-6im]

    screening_params = LindhardScreeningParams(
        degeneracy = 8,
        m_eff = -0.3,
        nlist = [1e-4, 1e-3],
        ϵM = 6.0 * Mat3(I(3)),
        smearing = 0.1 * unit_to_aru(:eV)
    )
    @test epsilon_lindhard(Vec3(0., 0., 0.0), 0.0, screening_params) == [1, 1]
    @test epsilon_lindhard(Vec3(0., 0., 0.1), 100.0, screening_params) ≈ [1.0, 1.0] atol=1e-5
    @test epsilon_lindhard(Vec3(0., 0., 0.1), 0.0, screening_params) ≈ [2.6547449254840867, 5.428052540389297]
    @test epsilon_lindhard(Vec3(0., 0., 0.1), 0.5, screening_params) ≈ [0.9886875820784919 + 0.00033713836570004523im,  0.8830538663440538 + 0.0036052347045188826im]
end