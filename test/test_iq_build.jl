using Test
using ElectronPhonon
using Random
using ElectronPhonon: Vec3, _grid_coords_reduced, _wrap_reduced, _fill_iqs_serial!

# The per-(k-strip, q-tile) `iq` index build of the GPU outer-k loop
# (`_loop_eph_over_k_and_kq_gpu`). The loop hashes integer grid coordinates instead of calling
# `xk_to_ik` per pair, so the whole correctness story is "the fast hash agrees with `xk_to_ik`".
# CPU-only: the build is host arithmetic and the loop it feeds is never reached on `CPUBackend`.

# Everything the builder needs, assembled the way the loop's prologue assembles it. The outer k mesh
# is always unshifted — the loop's `xks_int` does not subtract a shift, so a shifted k would not
# land on integer grid coordinates at all. `shift` shifts the k+q mesh, which is what gives `qpts`
# a nonzero shift for the k+q side to subtract.
function _iq_build_fixture(ngrid; shift = (0, 0, 0))
    kpts  = GridKpoints(kpoints_grid(ngrid))
    kqpts = GridKpoints(kpoints_grid(ngrid; shift))
    qpts  = ElectronPhonon.combine_kpoint_grids(kqpts, kpts, -, ngrid)
    xkqs_int = Matrix{Int}(undef, 3, kqpts.n)
    xks_int  = Matrix{Int}(undef, 3, kpts.n)
    for ikq in 1:kqpts.n
        xkqs_int[:, ikq] .= _grid_coords_reduced(kqpts.vectors[ikq], qpts.ngrid, qpts.shift)
    end
    for ik in 1:kpts.n
        xks_int[:, ik] .= _grid_coords_reduced(kpts.vectors[ik], qpts.ngrid, zero(Vec3{Float64}))
    end
    (; kpts, kqpts, qpts, xkqs_int, xks_int)
end

# `xk_to_ik` on the actual difference vector — the independent oracle. It goes through the Float64
# `_hash_xk` path, so agreeing with it also pins the integer shortcut's rounding.
function _check_against_oracle(f, iks, nq)
    stage = fill(-7, nq, length(iks))
    _fill_iqs_serial!(stage, f.qpts, f.xkqs_int, f.xks_int, iks, 1, 1, nq)
    ok = true
    for (kloc, ik) in enumerate(iks), j in 1:nq
        ok &= stage[j, kloc] == xk_to_ik(f.kqpts.vectors[j] - f.kpts.vectors[ik], f.qpts)
    end
    (ok, stage)
end

@testset "iq build" begin
    @testset "serial builder == xk_to_ik oracle" begin
        f = _iq_build_fixture((4, 4, 4))
        ok, stage = _check_against_oracle(f, 1:f.kpts.n, f.kqpts.n)
        @test ok
        @test all(1 .<= stage .<= f.qpts.n)   # every pair resolved; no sentinel left behind
    end

    @testset "non-cubic ngrid" begin
        # Three distinct sizes, so a transposed `(c1*ng2 + c2)*ng3 + c3` packing cannot pass by
        # accident (the same reason `test_kpoints.jl` pins the dense table's layout on (2,3,4)).
        f = _iq_build_fixture((3, 4, 5))
        @test f.qpts.ngrid == (3, 4, 5)
        ok, _ = _check_against_oracle(f, 1:f.kpts.n, f.kqpts.n)
        @test ok
    end

    @testset "shifted grid" begin
        # The q-grid shift is subtracted on the k+q side only (`_grid_coords_reduced(..., qpts.shift)`
        # for k+q, a zero shift for k), because q = x_{k+q} - x_k - shift.
        f = _iq_build_fixture((3, 4, 5); shift = (1//2, 0, 1//2))
        @test iszero(f.kpts.shift) && !iszero(f.qpts.shift)
        ok, _ = _check_against_oracle(f, 1:f.kpts.n, f.kqpts.n)
        @test ok
    end

    @testset "[0,ng) reduction is exact" begin
        # `mod(a - b, ng) == _wrap_reduced(mod(a,ng) - mod(b,ng), ng)`, and `_wrap_reduced` also
        # folds a SUM of two reduced coordinates (what `combine_kpoint_grids` would need).
        Random.seed!(20260802)
        for ng in (1, 2, 3, 5, 17, 150)
            for _ in 1:400
                a, b = rand(-3ng:3ng), rand(-3ng:3ng)
                @test _wrap_reduced(mod(a, ng) - mod(b, ng), ng) == mod(a - b, ng)
                @test _wrap_reduced(mod(a, ng) + mod(b, ng), ng) == mod(a + b, ng)
            end
        end
        # And the array-level reduction it relies on.
        for ng in ((3, 4, 5), (150, 150, 150))
            b = Vec3(rand(-1000:1000, 3))
            @test _grid_coords_reduced(b, ng) == Vec3(mod.(b.data, ng))
        end
    end

    @testset "partial final tile and strip" begin
        # The loop stages a whole kR->kq strip at once into an (nq_batch_max, nk_b) matrix. A
        # partial final q-tile writes only the leading `nq` rows, and a partial final strip only the
        # leading `nks` columns; everything else must be left alone (it is copied to the device
        # along with the rest but never read).
        f = _iq_build_fixture((4, 4, 4))
        nq_max, nk_b = f.kqpts.n, 3
        for (nq, nks) in ((nq_max, nk_b), (7, nk_b), (nq_max, 2), (7, 1))
            stage = fill(-7, nq_max, nk_b)
            _fill_iqs_serial!(stage, f.qpts, f.xkqs_int, f.xks_int, 2:(1 + nks), 1, 1, nq)
            @test all(stage[1:nq, 1:nks] .>= 1)
            @test all(stage[(nq + 1):end, :] .== -7)
            @test all(stage[:, (nks + 1):end] .== -7)
        end
    end

    @testset "per-k staging == per-strip staging" begin
        # `kloc0` lets the builder fill one column at a time; the loop uses that only as the
        # `nk_b == 1` degenerate case, but the two must agree column for column.
        f = _iq_build_fixture((3, 4, 5))
        nq, iks = f.kqpts.n, 3:6
        strip = fill(-7, nq, length(iks))
        perk  = fill(-7, nq, length(iks))
        _fill_iqs_serial!(strip, f.qpts, f.xkqs_int, f.xks_int, iks, 1, 1, nq)
        for (kloc, ik) in enumerate(iks)
            _fill_iqs_serial!(perk, f.qpts, f.xkqs_int, f.xks_int, ik:ik, kloc, 1, nq)
        end
        @test strip == perk
    end

    @testset "q-tile offset" begin
        # `qstart` shifts which k+q the tile covers; tiling must reproduce the untiled build.
        f = _iq_build_fixture((4, 4, 4))
        nkq, iks = f.kqpts.n, 5:7
        whole = fill(-7, nkq, length(iks))
        _fill_iqs_serial!(whole, f.qpts, f.xkqs_int, f.xks_int, iks, 1, 1, nkq)
        tiled = fill(-7, nkq, length(iks))
        qstart = 1
        while qstart <= nkq
            nq = min(25, nkq - qstart + 1)
            part = fill(-7, nq, length(iks))
            _fill_iqs_serial!(part, f.qpts, f.xkqs_int, f.xks_int, iks, 1, qstart, nq)
            tiled[qstart:(qstart + nq - 1), :] .= part
            qstart += nq
        end
        @test whole == tiled
    end

    @testset "a q point missing from qpts is rejected" begin
        # `iq == 0` is unreachable for a `qpts` built by `combine_kpoint_grids` from the same lists
        # the loop iterates (it is the complete deduplicated difference set), so the guard is
        # provoked with a `qpts` holding only one of those q points.
        f = _iq_build_fixture((4, 4, 4))
        one_q = GridKpoints(Kpoints(1, f.qpts.vectors[2:2], f.qpts.weights[2:2], f.qpts.ngrid),
                            f.qpts.ngrid)
        stage = fill(-7, f.kqpts.n, 1)
        @test_throws ArgumentError _fill_iqs_serial!(stage, one_q, f.xkqs_int, f.xks_int,
                                                     1:1, 1, 1, f.kqpts.n)
    end
end
