using IterativeSolvers
using ElectronPhonon
using Test

@testset "iterative solvers" begin
    # Test resetting iterable struct implementation of iterative solver
    N = 5
    reltol = 1e-8
    A = I(N) .+ 0.1 .* rand(N, N)
    b = zeros(N)
    x = zeros(N)
    g = IterativeSolvers.gmres_iterable!(x, A, b)

    b = rand(N)
    x = rand(N)
    x_direct = gmres(A, b; reltol)
    ElectronPhonon.reset_gmres_iterable!(g, x, b; reltol)
    for _ in g; end
    @test g.x ≈ x_direct
end
