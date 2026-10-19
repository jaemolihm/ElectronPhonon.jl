@testsnippet IqBuild begin
    using Random
    using ElectronPhonon: _pair_grid_coords, _wrap_reduced, _fill_iqs!, q_index_table

    # Everything the build needs, assembled the way the `OuterKEngine` constructor assembles it.
    # `shift_k` and `shift_kq` shift the k and k+q meshes; `qpts` then carries their difference.
    function _iq_build_fixture(ngrid; shift_k = (0, 0, 0), shift_kq = (0, 0, 0))
        kpts  = GridKpoints(kpoints_grid(ngrid; shift = shift_k))
        kqpts = GridKpoints(kpoints_grid(ngrid; shift = shift_kq))
        qpts  = ElectronPhonon.combine_kpoint_grids(kqpts, kpts, -, ngrid)
        ck, ckq = _pair_grid_coords(kpts, kqpts, qpts)
        (; kpts, kqpts, qpts, ck, ckq)
    end

    # `xk_to_ik` on the actual difference vector — the independent oracle. It goes through the Float64
    # `_hash_xk` path, so agreeing with it also pins the integer shortcut's rounding.
    function _check_against_oracle(f, iks, nq)
        iqs = fill(-7, nq)
        ok = true
        for ik in iks
            _fill_iqs!(iqs, f.qpts, f.ck, f.ckq, ik, 1, nq)
            for j in 1:nq
                ok &= iqs[j] == xk_to_ik(f.kqpts.vectors[j] - f.kpts.vectors[ik], f.qpts)
            end
            ok &= all(1 .<= iqs .<= f.qpts.n)   # every pair resolved; no sentinel left behind
        end
        ok
    end
end

# The per-(k, k+q tile) `iq` index build of the outer-k loop (`_fill_iqs!`). The loop hashes
# integer grid coordinates instead of calling `xk_to_ik` per pair, so the whole correctness story is
# "the fast hash agrees with `xk_to_ik`". The build is pure host arithmetic, so this file tests the
# hash in isolation and needs no backend at all; the loop runs end-to-end elsewhere.

@testitem "iq build" setup = [IqBuild] begin
    @testset "build == xk_to_ik oracle" begin
        f = _iq_build_fixture((4, 4, 4))
        @test _check_against_oracle(f, 1:f.kpts.n, f.kqpts.n)
    end

    @testset "non-cubic ngrid" begin
        # Three distinct sizes, so a transposed `(c1*ng2 + c2)*ng3 + c3` packing cannot pass by
        # accident (the same reason `test_kpoints.jl` pins the dense table's layout on (2,3,4)).
        f = _iq_build_fixture((3, 4, 5))
        @test f.qpts.ngrid == (3, 4, 5)
        @test _check_against_oracle(f, 1:f.kpts.n, f.kqpts.n)
    end

    @testset "shifted grids" begin
        # Half-step shifts, `1/(2 ng)`: a shift of 1/2 would fold back onto the unshifted grid on an
        # even `ng`. The k+q mesh alone, the k mesh alone, both by the same shift (unshifted q grid,
        # every k on a shifted node: the case an unshifted k reduction gets wrong), and both apart.
        ngrid = (4, 6, 2)
        half = 1 ./ (2 .* ngrid)
        for (shift_k, shift_kq) in (((0, 0, 0), half), (half, (0, 0, 0)), (half, half),
                                    ((0, half[2], half[3]), (half[1], 0, half[3])))
            f = _iq_build_fixture(ngrid; shift_k, shift_kq)
            @test f.qpts.shift ≈ mod.(f.kqpts.shift - f.kpts.shift, 1 ./ ngrid)
            @test _check_against_oracle(f, 1:f.kpts.n, f.kqpts.n)
        end
    end

    @testset "q grid shift inconsistent with the k and k+q shifts" begin
        # A q grid whose shift is not the k+q shift minus the k shift holds no pair's q.
        f = _iq_build_fixture((4, 4, 4); shift_k = (1/8, 0, 0))
        qpts_unshifted = GridKpoints(kpoints_grid((4, 4, 4)))
        @test_throws ArgumentError _pair_grid_coords(f.kpts, f.kqpts, qpts_unshifted)
        @test_throws ArgumentError q_index_table(f.kpts, f.kqpts, qpts_unshifted)
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
        # Outside (-ng, 2ng) one fold cannot reduce into 0:ng-1, so the precondition is checked
        # (and the pair loops elide the check with `@inbounds` once they have established it).
        @test_throws ArgumentError _wrap_reduced(-8, 4)
        @test_throws ArgumentError _wrap_reduced(8, 4)
        @test _wrap_reduced(-3, 4) == 1 && _wrap_reduced(7, 4) == 3
    end

    @testset "partial final tile" begin
        # The loop stages into an `n_inner_tile`-long buffer and a partial final q-tile writes only
        # the leading `nq` entries; the tail must be left alone (the H2D copies only `1:nq`).
        f = _iq_build_fixture((4, 4, 4))
        nq_max = f.kqpts.n
        for nq in (nq_max, 7, 1)
            iqs = fill(-7, nq_max)
            _fill_iqs!(iqs, f.qpts, f.ck, f.ckq, 2, 1, nq)
            @test all(iqs[1:nq] .>= 1)
            @test all(iqs[(nq + 1):end] .== -7)
        end
    end

    @testset "q-tile offset" begin
        # `qstart` shifts which k+q the tile covers; tiling must reproduce the untiled build.
        f = _iq_build_fixture((4, 4, 4))
        nkq, ik = f.kqpts.n, 5
        whole = fill(-7, nkq)
        _fill_iqs!(whole, f.qpts, f.ck, f.ckq, ik, 1, nkq)
        tiled = fill(-7, nkq)
        qstart = 1
        while qstart <= nkq
            nq = min(25, nkq - qstart + 1)
            part = fill(-7, nq)
            _fill_iqs!(part, f.qpts, f.ck, f.ckq, ik, qstart, nq)
            tiled[qstart:(qstart + nq - 1)] .= part
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
        iqs = fill(-7, f.kqpts.n)
        @test_throws ArgumentError _fill_iqs!(iqs, one_q, f.ck, f.ckq, 1, 1, f.kqpts.n)
        # The same miss in the threaded table build, raised after the threaded loop.
        @test_throws ArgumentError q_index_table(f.kpts, f.kqpts, one_q)
    end

    @testset "q_index_table" begin
        # The whole (k, k+q) table on shifted non-cubic grids, with an outer subset of the k grid
        # (the IBZ shape: n_k < n_kq), against a k-vector search of `qpts` for q = x_{k+q} - x_k.
        f = _iq_build_fixture((4, 3, 5); shift_k = (1/8, 1/6, 0), shift_kq = (1//2, 0, 1//2))
        iks_sub = 3:4:f.kpts.n
        kpts = GridKpoints(Kpoints(length(iks_sub), f.kpts.vectors[iks_sub], f.kpts.weights[iks_sub],
                                   f.kpts.ngrid), f.kpts.ngrid)
        iq_kk = q_index_table(kpts, f.kqpts, f.qpts)
        @test iq_kk isa Matrix{Int32} && size(iq_kk) == (kpts.n, f.kqpts.n)
        iq_ref = [findfirst(xq -> all(abs.(mod.(xq - (xkq - xk) .+ 1/2, 1) .- 1/2) .< 1e-10),
                            f.qpts.vectors)
                  for xk in kpts.vectors, xkq in f.kqpts.vectors]
        @test iq_kk == iq_ref
    end
end
