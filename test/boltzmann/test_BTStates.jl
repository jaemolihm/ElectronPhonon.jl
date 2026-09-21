using Test
using ElectronPhonon

# `states_index_map`'s optional `symmetry` argument: each state must also be reachable through
# every point of the star of its k-point. Model-free -- the states are built by hand on an
# IBZ-reduced grid, whose stars tile the full grid exactly once. Two point groups, because they
# constrain the transform differently: fcc (Oh, centrosymmetric) pins the collapse of the rotation
# part, zincblende (Td, non-centrosymmetric) pins the sign of the time-reversal part, whose star
# points are not reachable by any rotation alone.
@testset "states_index_map with symmetry" begin
    using ElectronPhonon: Vec3, BTStates, kpoints_grid, symmetry_operations, apply_symop,
        states_index_map, CI

    lattice = [0.0 2.0 2.0; 2.0 0.0 2.0; 2.0 2.0 0.0]
    ngrid = (4, 4, 4)
    nband = 2
    grid_key(xk) = CI(mod.(round.(Int, xk .* ngrid), ngrid)...)

    for atoms in ([(:Pb, [Vec3(0.0, 0.0, 0.0)])],
                  [(:B, [Vec3(0.0, 0.0, 0.0)]), (:N, [Vec3(0.25, 0.25, 0.25)])])
        symmetry = symmetry_operations(lattice, atoms)
        kpts = kpoints_grid(ngrid; symmetry)
        n = kpts.n * nband
        xks = repeat(kpts.vectors, inner = nband)
        iband = repeat(1:nband, kpts.n)
        states = BTStates{Float64}(; n, nk = kpts.n, nband, ngrid, xks, iband,
            e = zeros(n), vdiag = zeros(Vec3{Float64}, n),
            k_weight = repeat(kpts.weights, inner = nband))

        # Without symmetry only the irreducible points are keyed.
        map_nosym = states_index_map(states)
        @test length(map_nosym) == kpts.n
        @test all(map_nosym[grid_key(xks[i])][iband[i]] == i for i in 1:n)

        # With symmetry every star point resolves to the state at the irreducible representative,
        # and the stars together cover the full grid.
        index_map = states_index_map(states, symmetry)
        @test all(index_map[grid_key(apply_symop(symop, xks[i], :momentum))][iband[i]] == i
                  for i in 1:n, symop in symmetry)
        @test length(index_map) == prod(ngrid)
    end
end
