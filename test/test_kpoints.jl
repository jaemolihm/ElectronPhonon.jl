using Test
using Random
using EPW
using EPW: Kpoints

@testset "kpoints: grid" begin
    nk1 = 2
    nk2 = 3
    nk3 = 4
    kpts = generate_kvec_grid(nk1, nk2, nk3)
    @test kpts.n == nk1 * nk2 * nk3
    @test sum(kpts.weights) ≈ 1
    @test kpts.vectors[19] ≈ [1/2, 1/3, 2/4]

    # Test get_filtered_kpoints
    ik_keep = zeros(Bool, nk1 * nk2 * nk3)
    ik_keep[7] = true
    ik_keep[11] = true
    kpts_filtered = EPW.get_filtered_kpoints(kpts, ik_keep)
    @test kpts_filtered.n == sum(ik_keep)
    @test kpts_filtered.vectors[1] ≈ kpts.vectors[7]
    @test kpts_filtered.vectors[2] ≈ kpts.vectors[11]
    @test kpts_filtered.weights[1] ≈ 1 / (nk1 * nk2 * nk3)
    @test kpts_filtered.weights[2] ≈ 1 / (nk1 * nk2 * nk3)

    # Test kpoints_create_subgrid
    kpts = generate_kvec_grid(2, 2, 2)
    ik_keep = zeros(Bool, kpts.n)
    ik_keep[7:8] .= true
    kpts = EPW.get_filtered_kpoints(kpts, ik_keep)
    kpts2 = EPW.kpoints_create_subgrid(kpts, (2, 3, 4))
    @test kpts2.n == 48
    @test kpts2.vectors[1]  ≈ kpts.vectors[1] .+ (-1/4, -2/6, -3/8) ./ (2, 2, 2)
    @test kpts2.vectors[19] ≈ kpts.vectors[1] .+ ( 1/4,  0/6,  1/8) ./ (2, 2, 2)
end

@testset "kpoints: array, sort" begin
    xks = zeros(3, 4)
    xks[:, 1] = [0.2, 0.2, 0.9]
    xks[:, 2] = [0.0, 0.9, 0.9]
    xks[:, 3] = [0.2, 0.1, 0.1]
    xks[:, 4] = [0.2, 0.1, 0.0]
    inds = sortperm(xks[1, :] .* 100 .+ xks[2, :] .* 10 .+ xks[3, :])

    kpts = Kpoints(xks)
    @test kpts.n == size(xks, 2)
    @test sum(kpts.weights) ≈ 1
    @test kpts.ngrid == (0, 0, 0)

    kpts.weights .= [1, 2, 3, 4]
    EPW.sort!(kpts)
    @test kpts.weights ≈ inds
    for i in 1:size(xks, 2)
        @test kpts.vectors[i] ≈ xks[:, inds[i]]
    end
end
