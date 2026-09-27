using Test
using LinearAlgebra
using StaticArrays
using ElectronPhonon

@testset "box_quadratic_minimum and Polar2D Glist" begin
    using ElectronPhonon: box_quadratic_minimum, get_Glist, Polar2D, Structure, Vec3, Mat3

    # Diagonal A: min = Σ a_i dist(0, [l_i, u_i])², boxes on both sides of and around 0.
    dist0(l, u) = max(l, 0.0, -u)
    for (a, l) in ((SVector(0.7, 3.1), SVector(-2.5, 0.2)), (SVector(0.7, 3.1, 1.9), SVector(1.5, -0.5, -3.5)))
        u = l .+ 1
        @test box_quadratic_minimum(SMatrix{length(a), length(a)}(Diagonal(a)), l, u) ≈ sum(a .* dist0.(l, u) .^ 2)
    end

    # A box containing the origin gives exactly 0.
    A3 = @SMatrix [4.0 1.5 -0.7; 1.5 2.0 0.3; -0.7 0.3 1.0]
    @test box_quadratic_minimum(A3, SVector(-0.3, -0.5, -0.1), SVector(0.7, 0.5, 0.9)) == 0

    # Non-diagonal A: the exact minimum is below a dense sampling of the box (up to rounding, since the
    # grid can hit the minimizer), and close to it.
    A2 = @SMatrix [2.0 -1.3; -1.3 1.5]
    for (A, l, n) in ((A2, SVector(0.5, 1.5), 400), (A3, SVector(-1.5, 0.5, 1.5), 60))
        m = box_quadratic_minimum(A, l, l .+ 1)
        ms = minimum(x -> x' * A * x, (l .+ SVector(ci.I) ./ n for ci in CartesianIndices(ntuple(_ -> 0:n, length(l)))))
        @test m <= ms * (1 + 1e-12)
        @test ms <= m * (1 + 1e-2)
    end

    # Polar2D on a rectangular cell, where |B x|² is diagonal: the analytic Glist, in loop order.
    a1, a2, L = 5.0, 8.0, 20.0
    cell = Structure(1.0, Mat3([a1 0 0; 0 a2 0; 0 0 30.0]), [1.0], [Vec3(0.0, 0, 0)], ["X"]; compute_symmetry = false)
    nxs = (4, 6, 0)
    thr = (2 * 15.0 / L)^2
    Glist_ref = [Vec3{Int}(ci.I) for ci in CartesianIndices((-nxs[1]:nxs[1], -nxs[2]:nxs[2], 0:0))
                 if (2π / a1)^2 * dist0(ci[1] - 0.5, ci[1] + 0.5)^2 + (2π / a2)^2 * dist0(ci[2] - 0.5, ci[2] + 0.5)^2 < thr]
    @test get_Glist(Polar2D(15.0, L), cell, nxs, nothing) == Glist_ref
    @test 0 < length(Glist_ref) < prod(2 .* nxs[1:2] .+ 1)  # the cutoff truncates the box
end
