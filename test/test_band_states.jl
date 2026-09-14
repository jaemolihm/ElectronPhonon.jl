using Test
using ElectronPhonon

# The `(k, band)`-selection machinery: per-state indexing, `state_index` in all three forms,
# `filter_states`, and the two symmetry-star lookups. Model-free — the states are built by hand.
@testset "BandStates: selection utilities" begin
    using ElectronPhonon: Vec3, GridKpoints, Kpoints, FilteredBandStates, BandStates,
        filter_states, state_index, state_index_in_star, state_indices_full_star,
        state_weights, ind_range_for_k_range, band_range, symmetry_operations, apply_symop

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
    end

    @testset "symmetry-star lookups" begin
        xk = Vec3(0.25, 0.0, 0.0)
        J = state_indices_full_star(sel, xk, 3, symmetry)
        star = unique([apply_symop(S, xk, :momentum) for S in symmetry])
        @test J == sort(unique(filter(!iszero, [state_index(sel, Sk, 3) for Sk in star])))
        @test !isempty(J) && issorted(J) && allunique(J)
        @test all(sel.ibands[j] == 3 for j in J)
        @test isempty(state_indices_full_star(sel, xk, 7, symmetry))   # band not selected

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
end
