using Test
using Random
using ElectronPhonon

@testset "kpoints: grid" begin
    using ElectronPhonon: get_filtered_kpoints, kpoints_grid, kpoints_create_subgrid, kpoints_grid_range

    nk1 = 2
    nk2 = 3
    nk3 = 4
    kpts = kpoints_grid((nk1, nk2, nk3))
    @test kpts.n == nk1 * nk2 * nk3
    @test sum(kpts.weights) ≈ 1
    @test kpts.vectors[19] ≈ [1/2, 1/3, 2/4]

    kpts_split = split_kpoints(kpts, 5)
    @test sum([k.n for k in kpts_split]) == kpts.n
    @test all([k isa Kpoints for k in kpts_split])

    # kpoints_grid_range weights each point by the global BZ fraction 1/prod(ngrid), NOT
    # 1/length(rng): a sub-range (as used per MPI rank) must carry the global fraction so the
    # union of sub-ranges still sums to 1.
    ngrid = (nk1, nk2, nk3)
    nk = prod(ngrid)
    kpts_full = kpoints_grid_range(ngrid, 1:nk)
    @test all(kpts_full.weights .≈ 1 / nk)
    kpts_sub = kpoints_grid_range(ngrid, 3:9)  # a strict sub-range
    @test all(kpts_sub.weights .≈ 1 / nk)      # global BZ fraction, not 1/7
    subranges = [1:5, 6:12, 13:nk]
    @test sum(sum(kpoints_grid_range(ngrid, r).weights) for r in subranges) ≈ 1

    # Test get_filtered_kpoints
    ik_keep = zeros(Bool, nk1 * nk2 * nk3)
    ik_keep[7] = true
    ik_keep[11] = true
    kpts_filtered = get_filtered_kpoints(kpts, ik_keep)
    @test kpts_filtered.n == sum(ik_keep)
    @test kpts_filtered.vectors[1] ≈ kpts.vectors[7]
    @test kpts_filtered.vectors[2] ≈ kpts.vectors[11]
    @test kpts_filtered.weights[1] ≈ 1 / (nk1 * nk2 * nk3)
    @test kpts_filtered.weights[2] ≈ 1 / (nk1 * nk2 * nk3)

    # Test kpoints_create_subgrid
    kpts = kpoints_grid((2, 2, 2))
    ik_keep = zeros(Bool, kpts.n)
    ik_keep[7:8] .= true
    kpts = get_filtered_kpoints(kpts, ik_keep)
    kpts2 = kpoints_create_subgrid(kpts, (2, 3, 4))
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
    sort!(kpts)
    @test kpts.weights ≈ inds
    for i in 1:size(xks, 2)
        @test kpts.vectors[i] ≈ xks[:, inds[i]]
    end

    kpts1 = Kpoints(xks)
    kpts2 = Kpoints(xks)
    center = (0, 1//2, 1)
    shift_center!(kpts2, center)
    for k in kpts2.vectors
        @test all(abs.(k .- center) .< 0.5 + 1e-8)
    end
    for dk in kpts2.vectors .- kpts1.vectors
        @test all(abs.(dk - round.(Int, dk)) .< 1e-8)
    end
end

@testset "kpoints: GridKpoints" begin
    using ElectronPhonon: get_filtered_kpoints, kpoints_grid

    Random.seed!(111)
    N = 3
    shift = [0, 1//2, 1//2] ./ N
    kpts = kpoints_grid((N, N, N); shift)
    @test kpts.vectors[1] ≈ Vec3(0, 1//2N, 1//2N)

    kpts = get_filtered_kpoints(kpts, rand(Bool, kpts.n))
    gridkpts = GridKpoints(kpts)
    @test gridkpts.ngrid == (N, N, N)
    @test gridkpts.shift ≈ shift
    @test all(xk_to_ik.(gridkpts.vectors, Ref(gridkpts)) .== 1:gridkpts.n)

    # test mixed order
    inds = randperm(kpts.n)
    kpts_mix = Kpoints(kpts.n, kpts.vectors[inds], kpts.weights[inds], kpts.ngrid)
    gridkpts_mix = GridKpoints(kpts_mix)
    @test all(xk_to_ik.(gridkpts_mix.vectors, Ref(gridkpts_mix)) .== 1:gridkpts_mix.n)

    # test sorting GridKpoints
    sort!(gridkpts_mix)
    @test all(gridkpts_mix.vectors .≈ gridkpts.vectors)
    @test all(xk_to_ik.(gridkpts_mix.vectors, Ref(gridkpts_mix)) .== 1:gridkpts_mix.n)

    kpts_split = split_kpoints(gridkpts, 5)
    @test sum([k.n for k in kpts_split]) == gridkpts.n
    @test all([k isa GridKpoints for k in kpts_split])
end

@testset "kpoints: combine_kpoint_grids" begin
    using ElectronPhonon: kpoints_grid, combine_kpoint_grids, add_two_kpoint_grids

    # Check that op(k, q) for all (k, q) pairs is found in the combined grid, and that
    # the combined points are folded into [-0.5, 0.5)^3 and sorted in grid order.
    function test_combine(kpts, qpts, op, ngrid_kq)
        kqpts = combine_kpoint_grids(kpts, qpts, op, ngrid_kq)
        @test kqpts.ngrid == ngrid_kq

        # All op(k, q) can be found, and the combined grid contains no other points.
        found = Set{Int}()
        for xk in kpts.vectors, xq in qpts.vectors
            ikq = xk_to_ik(op(xk, xq), kqpts)
            @test ikq !== nothing
            push!(found, ikq)
        end
        @test length(found) == kqpts.n

        # Points are folded into [-0.5, 0.5)^3.
        for xkq in kqpts.vectors
            @test all(-0.5 - 1e-8 .≤ xkq .< 0.5 + 1e-8)
        end

        # Points are sorted in grid order, and the hash table is consistent.
        @test sortperm(kqpts) == 1:kqpts.n
        @test all(xk_to_ik.(kqpts.vectors, Ref(kqpts)) .== 1:kqpts.n)

        # Weights are uniform.
        @test all(kqpts.weights .≈ 1 / prod(ngrid_kq))

        kqpts
    end

    # Equal grids, no shift.
    kpts = GridKpoints(kpoints_grid((3, 3, 3)))
    qpts = GridKpoints(kpoints_grid((3, 3, 3)))
    test_combine(kpts, qpts, +, (3, 3, 3))
    test_combine(kpts, qpts, -, (3, 3, 3))

    # Commensurate grids (qpts denser); result lives on the denser grid.
    kpts = GridKpoints(kpoints_grid((2, 2, 2)))
    qpts = GridKpoints(kpoints_grid((4, 4, 4)))
    test_combine(kpts, qpts, +, (4, 4, 4))
    test_combine(kpts, qpts, -, (4, 4, 4))

    # Shifted q grid, and plain Kpoints input (converted to GridKpoints internally).
    shift = (0, 1//2, 1//2) ./ 3
    kpts = kpoints_grid((3, 3, 3))            # plain Kpoints, no shift
    qpts = kpoints_grid((3, 3, 3); shift)     # plain Kpoints, shifted
    kqpts = test_combine(kpts, qpts, +, (3, 3, 3))
    @test kqpts.shift ≈ Vec3(shift)           # shift of combined grid is shift_k + shift_q

    # Deprecated alias warns and still returns the same result.
    kpts = GridKpoints(kpoints_grid((2, 2, 2)))
    qpts = GridKpoints(kpoints_grid((2, 2, 2)))
    deprecated = @test_deprecated add_two_kpoint_grids(kpts, qpts, +, (2, 2, 2))
    @test deprecated == combine_kpoint_grids(kpts, qpts, +, (2, 2, 2))

    # ngrid_kq must be divisible by both input grids.
    kpts = GridKpoints(kpoints_grid((2, 2, 2)))
    qpts = GridKpoints(kpoints_grid((3, 3, 3)))
    @test_throws ArgumentError combine_kpoint_grids(kpts, qpts, +, (3, 3, 3))  # not divisible by kpts
    @test_throws ArgumentError combine_kpoint_grids(kpts, qpts, +, (2, 2, 2))  # not divisible by qpts
    @test combine_kpoint_grids(kpts, qpts, +, (6, 6, 6)) isa GridKpoints       # divisible by both

    # Overflow guard: prod(ngrid) must fit in Int (checked in the GridKpoints constructor).
    @test GridKpoints(kpoints_grid((1, 1, 1)), (1000, 1000, 1000)) isa GridKpoints
    @test_throws ArgumentError GridKpoints(kpoints_grid((1, 1, 1)), (10^7, 10^7, 10^7))
    # combine_kpoint_grids routes through that constructor, so the guard applies to ngrid_kq.
    single = GridKpoints(kpoints_grid((1, 1, 1)))
    @test_throws ArgumentError combine_kpoint_grids(single, single, +, (10^7, 10^7, 10^7))
end

@testset "kpoints: mpi_scatter return type" begin
    using ElectronPhonon: kpoints_grid, mpi_scatter, mpi_gather_and_scatter
    MPI = ElectronPhonon.MPI
    MPI.Initialized() || MPI.Init()
    # COMM_SELF exercises only the single-rank path (scatter keeps all points on this rank).
    # TODO: actually test with a real multi-rank MPI communicator (np > 1), where scatter/gather
    # split points across ranks — e.g. under mpiexec, or a spawned child comm.
    comm = MPI.COMM_SELF

    # Empty GridKpoints placeholder (the non-root receive side before mpi_scatter).
    empty_grid = GridKpoints{Float64}()
    @test empty_grid isa GridKpoints
    @test empty_grid.n == 0

    # mpi_scatter of a GridKpoints preserves the GridKpoints type (rebuilds the per-rank hash).
    gridk = GridKpoints(kpoints_grid((2, 2, 3)))
    scattered = mpi_scatter(gridk, comm)
    @test scattered isa GridKpoints
    @test scattered.n == gridk.n
    @test scattered.vectors ≈ gridk.vectors
    @test all(xk_to_ik.(scattered.vectors, Ref(scattered)) .== 1:scattered.n)

    # The load-balancing gather+scatter (used by filter_kpoints under MPI) also stays GridKpoints.
    balanced = mpi_gather_and_scatter(gridk, comm)
    @test balanced isa GridKpoints
    @test balanced.n == gridk.n

    # Contrast: a plain Kpoints scatters back to a plain Kpoints.
    plaink = kpoints_grid((2, 2, 3))
    @test plaink isa Kpoints
    @test mpi_scatter(plaink, comm) isa Kpoints
    @test mpi_gather_and_scatter(plaink, comm) isa Kpoints
end
