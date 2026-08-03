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

@testset "kpoints: dense inverse index" begin
    using ElectronPhonon: get_filtered_kpoints, kpoints_grid, combine_kpoint_grids, unfold_kpoints,
        fold_kpoints, _hash_xk, _ik_from_hash, _use_dense_index

    Random.seed!(222)
    lattice = 2.0 * [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    atoms = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)

    # Reference `hash => ik` map, built independently of the *builder* the production index uses:
    # the packing `(c1*ng2 + c2)*ng3 + c3` is re-derived here instead of calling `_hash_xk`, so a
    # change to the packing (including a transposed table layout, which the non-cubic grid below
    # makes visible) is caught. The coordinate normalization is deliberately the same expression as
    # `_hash_xk`'s, so this does NOT independently check normalization. Ascending `ik` with plain
    # assignment, so a repeated point resolves to the last one — the documented tie-break.
    function reference_hash_to_ik(kpts)
        ng1, ng2, ng3 = kpts.ngrid
        ref = Dict{Int,Int}()
        for (ik, xk) in enumerate(kpts.vectors)
            c1, c2, c3 = mod.(round.(Int, (xk - kpts.shift) .* kpts.ngrid), kpts.ngrid)
            ref[(c1 * ng2 + c2) * ng3 + c3] = ik
        end
        ref
    end

    # The index must agree with the reference over the WHOLE hash domain, hits and misses alike.
    # `dense` pins which of the two indices is expected to be the live one, so a future change to
    # the gate cannot silently move these grids onto the other path.
    function test_index_equivalence(kpts; dense = true)
        @test isempty(kpts._dense_hash_to_ik) == !dense
        @test isempty(kpts._xk_hash_to_ik) == dense
        ref = reference_hash_to_ik(kpts)
        @test count(h -> _ik_from_hash(kpts, h) != get(ref, h, 0), 0:prod(kpts.ngrid)-1) == 0
        @test all(xk_to_ik.(kpts.vectors, Ref(kpts)) .== 1:kpts.n)
    end

    N = 4
    full = GridKpoints(kpoints_grid((N, N, N)))
    shifted = GridKpoints(kpoints_grid((N, N, N); shift = (1//2, 1//2, 1//2) ./ N))
    subset = GridKpoints(get_filtered_kpoints(kpoints_grid((N, N, N)), rand(Bool, N^3)))
    @test subset.n > 0   # the sweep below assumes a non-empty selection
    irr = GridKpoints(kpoints_grid((N, N, N); symmetry))
    combined = combine_kpoint_grids(GridKpoints(kpoints_grid((2, 2, 2))), full, +, (N, N, N))
    unfolded, _ = unfold_kpoints(irr, symmetry)
    folded, _ = fold_kpoints(full, symmetry)
    for kpts in (full, shifted, subset, irr, combined, unfolded, folded)
        test_index_equivalence(kpts)
    end

    # An empty GridKpoints has NEITHER index; it is the one case where "exactly one" does not hold.
    # (The `GridKpoints{T}()` placeholder carries ngrid == (0,0,0), on which `_hash_xk` divides by
    # zero, so lookups are only meaningful on the empty-with-a-real-grid form.)
    @test isempty(GridKpoints{Float64}()._dense_hash_to_ik)
    @test isempty(GridKpoints{Float64}()._xk_hash_to_ik)
    let empty_kpts = GridKpoints(Kpoints{Float64}(0, Vec3{Float64}[], Float64[], (N, N, N)))
        @test empty_kpts.n == 0
        @test isempty(empty_kpts._dense_hash_to_ik) && isempty(empty_kpts._xk_hash_to_ik)
        @test xk_to_ik(Vec3(0.0, 0.0, 0.0), empty_kpts) === nothing
    end

    # Gate: dense for a grid as dense as it gets, Dict for a huge grid holding few points.
    @test !isempty(full._dense_hash_to_ik)
    sparse_kpts = GridKpoints(kpoints_grid((2, 2, 2)), (1000, 1000, 1000))
    @test !_use_dense_index(sparse_kpts.n, sparse_kpts.ngrid)
    @test isempty(sparse_kpts._dense_hash_to_ik)
    @test !isempty(sparse_kpts._xk_hash_to_ik)   # the Dict is the index here, not dead weight
    @test all(xk_to_ik.(sparse_kpts.vectors, Ref(sparse_kpts)) .== 1:sparse_kpts.n)
    # A grid node that holds no k point must miss on the Dict fallback (the branch every
    # multigrid/AMR grid takes).
    @test xk_to_ik(Vec3(1/1000, 0.0, 0.0), sparse_kpts) === nothing

    # The Dict fallback must satisfy the same whole-domain equivalence, and so must the paths that
    # only it exercises: `unfold_kpoints`/`fold_kpoints` hand their own Dict to the constructor
    # (discarded on a dense grid, so untested there) and `sort!` has a separate Dict branch. The
    # `index = :dict` override reaches the fallback on these same small grids, instead of depending
    # on a grid shape sparse enough for the policy to pick the Dict by itself (`sparse_kpts` above
    # is the one that pins the policy). Shift 0 on `irr`, as `unfold_kpoints` requires.
    for kpts in (full, shifted, subset, irr)
        test_index_equivalence(GridKpoints(Kpoints(kpts), kpts.ngrid; index = :dict); dense = false)
    end

    dict_kpts = GridKpoints(Kpoints(irr), irr.ngrid; index = :dict)
    dict_unfolded, _ = unfold_kpoints(dict_kpts, symmetry; index = :dict)
    test_index_equivalence(dict_unfolded; dense = false)          # `() -> sk_hash_dict` hand-through
    dict_folded, _ = fold_kpoints(dict_unfolded, symmetry; index = :dict)
    test_index_equivalence(dict_folded; dense = false)            # `() -> hash_dict_irr` hand-through

    # Both branches of `sort!`'s index renumbering.
    for index in (:dense, :dict)
        inds = randperm(full.n)
        mixed = GridKpoints(Kpoints(full.n, full.vectors[inds], full.weights[inds], full.ngrid),
                            full.ngrid; index)
        sort!(mixed)
        test_index_equivalence(mixed; dense = index === :dense)
        @test sortperm(mixed) == 1:mixed.n
    end

    # Gate policy at its boundaries. Small grids are always dense; a grid sparser than 8 nodes per
    # point is not; and the size cap is waived only when the Dict would be the bigger of the two.
    @test _use_dense_index(1, (100, 100, 100))            # below the node floor
    @test _use_dense_index(10^6, (200, 200, 200))         # 8 nodes per point, under the cap
    @test !_use_dense_index(10^6 - 1, (200, 200, 200))    # just over 8 nodes per point
    # The 256 MB cap is a local of `_use_dense_index` in src/common/kpoints.jl, so it is repeated
    # here rather than imported; keep the two in sync.
    dense_index_max_bytes = 256 * 1024^2
    nnode_over_cap = 2 * dense_index_max_bytes ÷ sizeof(Int)
    @test !_use_dense_index(nnode_over_cap ÷ 8, (nnode_over_cap, 1, 1))  # cap binds
    @test _use_dense_index(nnode_over_cap ÷ 4, (nnode_over_cap, 1, 1))   # cap waived: Dict is bigger

    # The `index` override wins over the policy in both directions, and rejects a bad value.
    @test _use_dense_index(1, (1000, 1000, 1000); index = :dense)
    @test !_use_dense_index(10^6, (100, 100, 100); index = :dict)
    @test_throws ArgumentError _use_dense_index(1, (4, 4, 4); index = :nonsense)

    # Layout: the table is a (ng3, ng2, ng1) array indexed by the integer grid coordinates, so that
    # its linear index — how `_ik_from_hash` reads it — is exactly `hash + 1`. Flipping the dims
    # would silently return wrong indices. The grid must have three distinct sizes, or the wrong
    # dims would still be the right total length.
    noncubic = GridKpoints(kpoints_grid((2, 3, 4); shift = (1//4, 1//6, 1//8)))
    test_index_equivalence(noncubic)
    ng1, ng2, ng3 = noncubic.ngrid
    table_3d = noncubic._dense_hash_to_ik
    @test table_3d isa Array{Int,3}
    @test size(table_3d) == (ng3, ng2, ng1)
    for xk in noncubic.vectors
        c = mod.(round.(Int, (xk - noncubic.shift) .* noncubic.ngrid), noncubic.ngrid)
        @test table_3d[c[3]+1, c[2]+1, c[1]+1] == xk_to_ik(xk, noncubic)
        # The cartesian slot and the linear `hash + 1` slot are the same slot.
        @test LinearIndices(table_3d)[c[3]+1, c[2]+1, c[1]+1] == _hash_xk(xk, noncubic) + 1
    end

    # A duplicated point resolves to the last index, matching the reference's tie-break.
    dup_vectors = push!(copy(full.vectors), full.vectors[3])
    dup = GridKpoints(Kpoints(length(dup_vectors), dup_vectors, ones(length(dup_vectors)) / N^3, (N, N, N)))
    @test _ik_from_hash(dup, _hash_xk(dup.vectors[3], dup)) == dup.n
    # (not `test_index_equivalence`: the point-to-index identity it asserts cannot hold here)
    let ref = reference_hash_to_ik(dup)
        @test all(_ik_from_hash(dup, h) == get(ref, h, 0) for h in 0:prod(dup.ngrid)-1)
    end

    # shift_center! folds by integer lattice vectors, which leaves every hash unchanged.
    centered = GridKpoints(kpoints_grid((N, N, N)))
    shift_center!(centered, (0, 0, 0))
    test_index_equivalence(centered)

    # sort! renumbers the points and must renumber the live index (dense branch here, Dict branch
    # covered above).
    inds = randperm(full.n)
    mixed = GridKpoints(Kpoints(full.n, full.vectors[inds], full.weights[inds], full.ngrid))
    sort!(mixed)
    test_index_equivalence(mixed)
    @test mixed.vectors ≈ full.vectors
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

    # Non-cubic ngrid, so a transposed table layout or a coordinate swap cannot pass by accident.
    kpts = GridKpoints(kpoints_grid((3, 4, 5)))
    qpts = GridKpoints(kpoints_grid((3, 4, 5)))
    test_combine(kpts, qpts, +, (3, 4, 5))
    test_combine(kpts, qpts, -, (3, 4, 5))

    # Grids of different density in each direction, combined on the finer common grid.
    kpts = GridKpoints(kpoints_grid((2, 4, 6)))
    qpts = GridKpoints(kpoints_grid((1, 2, 3)))
    test_combine(kpts, qpts, +, (2, 4, 6))
    test_combine(kpts, qpts, -, (2, 4, 6))

    # Subsets of a grid (what a windowed driver run actually passes), on a non-cubic grid.
    let g = kpoints_grid((4, 5, 6))
        Random.seed!(2026)
        sel = sort(randperm(g.n)[1:(g.n ÷ 3)])
        sub = GridKpoints(Kpoints(length(sel), g.vectors[sel], g.weights[sel], g.ngrid), g.ngrid)
        test_combine(sub, GridKpoints(g), -, (4, 5, 6))
    end

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
    combined = combine_kpoint_grids(kpts, qpts, +, (2, 2, 2))
    # `GridKpoints` has no `==` (the index maps are derived caches), so compare the fields.
    @test deprecated.n == combined.n
    @test deprecated.vectors == combined.vectors
    @test deprecated.weights == combined.weights
    @test deprecated.ngrid == combined.ngrid
    @test deprecated.shift == combined.shift

    # ngrid_kq must be divisible by both input grids.
    kpts = GridKpoints(kpoints_grid((2, 2, 2)))
    qpts = GridKpoints(kpoints_grid((3, 3, 3)))
    @test_throws ArgumentError combine_kpoint_grids(kpts, qpts, +, (3, 3, 3))  # not divisible by kpts
    @test_throws ArgumentError combine_kpoint_grids(kpts, qpts, +, (2, 2, 2))  # not divisible by qpts
    @test combine_kpoint_grids(kpts, qpts, +, (6, 6, 6)) isa GridKpoints       # divisible by both

    # The dense and the Dict de-duplication (selected by the size of the dense table) must return
    # the same SET of points: the dense path sweeps its table, so it emits them in grid order, while
    # the Dict path has no table to sweep and emits them in order of first appearance.
    # `combine_kpoint_grids` sorts the result either way. Only the dense one is reachable at these
    # grid sizes, so call both directly on the same integer grid coordinates.
    using ElectronPhonon: _combine_kq_dedup_dense, _combine_kq_dedup_dict
    Random.seed!(333)
    ngrid_kq = (4, 6, 5)
    for sgn in (1, -1)
        # Coordinates repeat across pairs, so both paths must hit their "already seen" branch.
        # Both routines require their inputs already reduced into 0:ng-1.
        BkS = [Vec3(mod.(rand(-6:6, 3), ngrid_kq)) for _ in 1:20]
        BqS = [Vec3(mod.(rand(-6:6, 3), ngrid_kq)) for _ in 1:20]
        shift_kq = Vec3(0.0, 1/12, 0.0)
        xkqs_dense = _combine_kq_dedup_dense(BkS, BqS, sgn, ngrid_kq, shift_kq)
        xkqs_dict = _combine_kq_dedup_dict(BkS, BqS, sgn, ngrid_kq, shift_kq)
        @test 0 < length(xkqs_dense) < length(BkS) * length(BqS)  # de-duplication happened
        @test sort(xkqs_dense) == sort(xkqs_dict)
        @test issorted(xkqs_dense; by = x -> round.(Int, (x - shift_kq).data .* ngrid_kq))
    end

    # Both paths fold with `_wrap_reduced` and write their table under `@inbounds`, so an unreduced
    # coordinate would be memory corruption rather than a wrong answer. The precondition is checked
    # once per list at entry; without that check this call segfaults.
    let ngrid_kq = (4, 6, 5), shift_kq = Vec3(0.0, 0.0, 0.0)
        raw = [Vec3(7, -3, 11)]
        ok = [Vec3(1, 2, 3)]
        @test_throws ArgumentError _combine_kq_dedup_dense(raw, ok, -1, ngrid_kq, shift_kq)
        @test_throws ArgumentError _combine_kq_dedup_dense(ok, raw, -1, ngrid_kq, shift_kq)
        @test_throws ArgumentError _combine_kq_dedup_dict(raw, ok, 1, ngrid_kq, shift_kq)
    end

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
