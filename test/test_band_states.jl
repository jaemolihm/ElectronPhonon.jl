using Test
using ElectronPhonon

# The `(k, band)`-selection machinery: per-state indexing, `state_index` in all three forms,
# `filter_states`, and the two symmetry-star lookups. Model-free — the states are built by hand.
@testset "BandStates: selection utilities" begin
    using ElectronPhonon: Vec3, GridKpoints, Kpoints, FilteredBandStates, BandStates,
        filter_states, state_index, state_index_in_star, state_indices_full_star,
        state_weights, ind_range_for_k_range, band_range, symmetry_operations, apply_symop,
        find_unfolding_indices

    # fcc lattice (cubic point group, 96 operations with time reversal) on a 4×4×4 grid.
    lattice = [0.0 2.0 2.0; 2.0 0.0 2.0; 2.0 2.0 0.0]
    symmetry = symmetry_operations(lattice, [(:Pb, [Vec3(0.0, 0.0, 0.0)])])

    ng = (4, 4, 4)
    kv = vec([Vec3((i-1)/4, (j-1)/4, (k-1)/4) for k in 1:4, j in 1:4, i in 1:4])
    nk = length(kv)
    kpts = GridKpoints(Kpoints(nk, kv, fill(1/nk, nk), ng))

    # Bands 3:4 everywhere, band 4 dropped on every 7th k so the selection is not rectangular.
    iks = Int[]; ibands = Int[]
    for ik in 1:nk, b in 3:4
        (b == 4 && ik % 7 == 0) && continue
        push!(iks, ik); push!(ibands, b)
    end
    sel = FilteredBandStates(kpts, iks, ibands; nw = 8, nstates_base = 2.0)
    es = [0.1 * i for i in 1:sel.n]
    vs = [Vec3(0.0, 0.0, 1.0 * i) for i in 1:sel.n]
    bs = BandStates(kpts, sel.iks, sel.ibands, es; nw = 8, v = vs, nstates_base = 2.0)

    @testset "getindex / iterate" begin
        st = sel[5]
        @test keys(st) == (:ik, :iband, :xk, :weight)   # no `e` on the lean selection
        @test st.ik == sel.iks[5] && st.iband == sel.ibands[5]
        @test st.xk == kpts.vectors[sel.iks[5]]
        stb = bs[5]
        @test keys(stb) == (:ik, :iband, :xk, :e, :weight)
        @test stb.e == es[5]
        for s in (sel, bs)
            @test eltype(s) == typeof(s[5])
            @test isbitstype(eltype(s))     # non-allocating in a hot loop
            @test length(collect(s)) == s.n
            @test eltype(collect(s)) == eltype(s)
        end
    end

    @testset "state_index item form" begin
        # `state_index(dst, src[i])` locates a state of one selection in another, via `xk`.
        @test all(state_index(sel, sel[i]) == i for i in 1:sel.n)
        @test all(state_index(bs, sel[i]) == i for i in 1:sel.n)
        @test all(state_index(sel, bs[i]) == i for i in 1:bs.n)
    end

    @testset "filter_states" begin
        keep = [12, 3, 3, 40, 41, 7]    # unsorted, with a duplicate
        for s in (sel, bs)
            sub = filter_states(s, keep)
            want = sort(unique(keep); by = i -> (s.iks[i], s.ibands[i]))
            @test typeof(sub).name === typeof(s).name       # same concrete type
            @test sub.n == length(want)
            @test [sub[j].xk for j in 1:sub.n] == [s[i].xk for i in want]
            @test sub.ibands == s.ibands[want]
            @test state_weights(sub) == state_weights(s)[want]
            @test sub.nw == s.nw && sub.nstates_base == s.nstates_base
            @test sub.kpts.ngrid == s.kpts.ngrid                 # stays commensurate
            @test sub.kpts.n == length(unique(s.iks[want]))      # bare k-points dropped
            @test all(state_index(sub, s[i]) != 0 for i in want)
            # k-major order, which `ind_range_for_k_range` requires
            @test ind_range_for_k_range(sub, 1, sub.kpts.n) == 1:sub.n
            @test band_range(sub) ⊆ band_range(s)
        end
        want = sort(unique(keep); by = i -> (bs.iks[i], bs.ibands[i]))
        subb = filter_states(bs, keep)
        @test subb.es == bs.es[want]
        @test subb.vs == bs.vs[want]
        @test filter_states(bs, 1:bs.n).es == bs.es      # full subset is the identity
        @test filter_states(sel, Int[]).n == 0
        # `vs` is empty when velocities were never computed; the subset must stay empty, not throw
        bs_novel = BandStates(kpts, bs.iks, bs.ibands, bs.es; nw = bs.nw)
        @test isempty(bs_novel.vs) && isempty(filter_states(bs_novel, keep).vs)
    end

    @testset "mpi_allgather (COMM_SELF)" begin
        using ElectronPhonon: mpi_allgather
        import MPI
        MPI.Initialized() || MPI.Init()
        # One rank: the concatenation is the input, so this pins the bookkeeping (k-offset,
        # field-by-field round-trip) without needing a multi-rank launcher.
        for src in (bs, BandStates(kpts, bs.iks, bs.ibands, bs.es; nw = bs.nw))   # with and w/o vs
            g = mpi_allgather(src, MPI.COMM_SELF)
            @test g.n == src.n
            @test g.iks == src.iks && g.ibands == src.ibands
            @test g.es == src.es
            @test state_weights(g) == state_weights(src)
            @test state_xks(g) == state_xks(src)
            @test g.kpts.n == src.kpts.n && g.kpts.ngrid == src.kpts.ngrid
            @test g.kpts.weights == src.kpts.weights      # per-k weights survive the gather
            @test (g.nw, g.nstates_base) == (src.nw, src.nstates_base)
            @test g.vs == src.vs                          # carried iff present
            @test all(state_index(g, src[i]) == i for i in 1:src.n)
        end
    end

    @testset "gather_band_states" begin
        using ElectronPhonon: gather_band_states
        import MPI
        MPI.Initialized() || MPI.Init()
        # One rank: the global set is this rank's set, so the whole block starts at offset 0.
        g, offset, counts = gather_band_states(bs, MPI.COMM_SELF)
        @test (offset, counts) == (0, [bs.n])
        @test g.n == bs.n && g.es == bs.es
        # The documented convention: the local states sit at `offset+1 : offset+n`.
        @test g.es[offset+1 : offset+bs.n] == bs.es
        @test all(state_index(g, bs[i]) == i + offset for i in 1:bs.n)

        # Serial path: the input object itself, not a copy.
        gs, offset_s, counts_s = gather_band_states(bs, nothing)
        @test gs === bs
        @test (offset_s, counts_s) == (0, [bs.n])
    end

    @testset "symmetry-star lookups" begin
        xk = Vec3(0.25, 0.0, 0.0)
        J = state_indices_full_star(sel, xk, 3, symmetry)
        star = unique([apply_symop(S, xk, :momentum) for S in symmetry])
        @test J == sort(unique(filter(!iszero, [state_index(sel, Sk, 3) for Sk in star])))
        @test !isempty(J) && issorted(J) && allunique(J)
        @test all(sel.ibands[j] == 3 for j in J)
        @test isempty(state_indices_full_star(sel, xk, 7, symmetry))   # band not selected
        # Item form: a state of one selection, looked up in another.
        i = state_index(sel, xk, 3)
        @test state_indices_full_star(sel, sel[i], symmetry) == J

        @test state_index_in_star(sel, xk, 3, symmetry) == state_index(sel, xk, 3)  # exact hit

        # A k-point absent from the selection, but with a star image present: the representative
        # chosen on one grid need not be the one chosen on another.
        ikdrop = 7
        sub = filter_states(sel, [i for i in 1:sel.n if sel.iks[i] != ikdrop])
        xk_gone = kpts.vectors[ikdrop]
        b = sel.ibands[findfirst(==(ikdrop), sel.iks)]
        @test state_index(sub, xk_gone, b) == 0
        j = state_index_in_star(sub, xk_gone, b, symmetry)
        @test j != 0 && sub.ibands[j] == b
        modone(v) = mod.(v .+ 1e-9, 1.0)     # k-vectors identified mod a reciprocal lattice vector
        @test any(modone(apply_symop(S, xk_gone, :momentum)) ≈ modone(sub[j].xk) for S in symmetry)
        @test state_index_in_star(sub, xk_gone, 9, symmetry) == 0   # no image carries band 9
    end

    # The two unfolding maps against an independent reference: the O(nsym·n_f·n_i) scan of `el_i`
    # inside the symmetry loop that the grid-hash lookups replace. The group argument in the
    # docstring is what makes the two agree, so the reference is the claim's only real check.
    @testset "find_unfolding_indices" begin
        function scan_reference(el_i, el_f, symmetry)
            xks_i, xks_f = state_xks(el_i), state_xks(el_f)
            map(1:el_f.n) do f
                for S in (symmetry === nothing ? (nothing,) : symmetry), j in 1:el_i.n
                    el_i.ibands[j] == el_f.ibands[f] || continue
                    Ski = symmetry === nothing ? xks_i[j] : apply_symop(S, xks_i[j], :momentum)
                    dk = Ski - xks_f[f]
                    all(abs.(dk .- round.(dk)) .< 1e-10) && return j
                end
                error("no representative for inner state $f")
            end
        end

        @testset "no symmetry" begin
            @test find_unfolding_indices(bs, bs, nothing) == collect(1:bs.n)
            @test find_unfolding_indices(bs, bs, nothing) == scan_reference(bs, bs, nothing)

            # A permuted inner set is resolved, not assumed away: the map is derived from
            # `(k, band)`, so it comes out as the permutation.
            rev = BandStates(kpts, reverse(bs.iks), reverse(bs.ibands), reverse(es);
                nw = 8, nstates_base = 2.0)
            @test find_unfolding_indices(bs, rev, nothing) == [bs.n + 1 - f for f in 1:bs.n]
            @test find_unfolding_indices(bs, rev, nothing) == scan_reference(bs, rev, nothing)

            # Energies play no part, so the match survives a shift in them.
            shifted = BandStates(kpts, bs.iks, bs.ibands, es .+ 1; nw = 8, nstates_base = 2.0)
            @test find_unfolding_indices(shifted, bs, nothing) == collect(1:bs.n)

            # An inner state with no outer partner errors, whether its k or its band is missing.
            @test_throws ErrorException find_unfolding_indices(
                filter_states(bs, 1:bs.n-1), bs, nothing)
            otherband = BandStates(kpts, bs.iks, fill(9, bs.n), es; nw = 9, nstates_base = 2.0)
            @test_throws ErrorException find_unfolding_indices(bs, otherband, nothing)
        end

        @testset "with symmetry" begin
            # An IBZ outer set: one state per (star, band), which is the case the answer is
            # unique in and the one an IBZ reduction hands the function.
            seen = Set{Tuple{Int,Int}}()
            keep = Int[]
            for i in 1:bs.n
                star = state_indices_full_star(bs, bs[i].xk, bs.ibands[i], symmetry)
                key = (minimum(star), bs.ibands[i])
                key in seen || (push!(seen, key); push!(keep, i))
            end
            ibz = filter_states(bs, keep)
            @test 0 < ibz.n < bs.n                      # the reduction is not a no-op

            fti = find_unfolding_indices(ibz, bs, symmetry)
            @test fti == scan_reference(ibz, bs, symmetry)
            @test length(fti) == bs.n && all(j -> 1 <= j <= ibz.n, fti)
            # Every inner state gets an outer partner of its own band, in its own star.
            @test all(ibz.ibands[fti[f]] == bs.ibands[f] for f in 1:bs.n)
            @test all(fti[f] in state_indices_full_star(ibz, bs[f].xk, bs.ibands[f], symmetry)
                      for f in 1:bs.n)

            # An inner state whose star is absent from the outer set errors.
            @test_throws ErrorException find_unfolding_indices(
                filter_states(ibz, 2:ibz.n), bs, symmetry)
        end

        # The `(k, band)` lookup is `GridKpoints`' integer-grid hash, so plain `Kpoints` has no
        # `state_index` method to reach.
        plain = BandStates(Kpoints(nk, kv, fill(1/nk, nk), ng), bs.iks, bs.ibands, es;
            nw = 8, nstates_base = 2.0)
        @test_throws MethodError find_unfolding_indices(plain, bs, nothing)
        @test_throws MethodError find_unfolding_indices(plain, bs, symmetry)

        # The `::Nothing` members of the star-lookup family are the star `{xk}`.
        @test state_index_in_star(bs, bs[3].xk, bs.ibands[3], nothing) == 3
        @test state_index_in_star(bs, bs[3].xk, 9, nothing) == 0
        @test state_indices_full_star(bs, bs[3].xk, bs.ibands[3], nothing) == [3]
        @test state_indices_full_star(bs, bs[3].xk, 9, nothing) == Int[]
        @test state_indices_full_star(bs, bs[3], nothing) == [3]
    end
end
