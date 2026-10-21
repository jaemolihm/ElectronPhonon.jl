# Linear Hall conductivity of the BandStates BTE (`solve_electron_bte_hall`) on 2D Holstein:
# ε = -2t (cos(2π kx) + cos(2π ky)), v = 2 t a sin(2π k), ∂_α v_b = δ_αb 2 t a² cos(2π k_b).
# References are computed in the same run.

@testsnippet BteHall begin
    using LinearAlgebra
    using StaticArrays
    using ElectronPhonon: BandStates, filter_electron_states, unfold_band_states,
        compute_electron_states, holstein_model
    const EP = ElectronPhonon

    const t_hop = 0.1
    const alat = 5.0
    const model = holstein_model(; t = t_hop, ω₀ = 0.02, g = 0.01, alat, dimension = 2, verbose = false)
    model.el_velocity_mode = :BerryConnection

    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

    # el_i (IBZ if `symmetry`) and el_f (its unfolding) on an nk × nk × 1 grid, without e-ph.
    function hall_states(nk, window, symmetry)
        sel = filter_electron_states((nk, nk, 1), model, window; symmetry)
        sel_f = symmetry === nothing ? sel : unfold_band_states(sel, symmetry)
        mk(s) = BandStates(compute_electron_states(model, s,
            ["eigenvalue", "eigenvector", "velocity_diagonal"]), s)
        mk(sel), mk(sel_f)
    end

    # Band-bottom window and constant τ⁻¹ = 0.01, no scattering in: CRTA of one nearly parabolic band.
    # `holes = true`: window at the band top, the band is the valence band (nelec = 2) and nlist < 0.
    function crta_hall(nk, kT_over_t, order; nlist, symmetry = model.symmetry, holes = false)
        T = kT_over_t * t_hop
        window = holes ? (4t_hop - 15T, 4t_hop + 1e-9) : (-4t_hop - 1e-9, -4t_hop + 15T)
        el_i, el_f = hall_states(nk, window, symmetry)
        occ = ElectronOccupationParams(; Tlist = [T], nlist, volume = model.volume,
            nelec = holes ? 2 : 0, spin_degeneracy = 2, type = :Semiconductor)
        EP.bte_compute_μ!(occ, el_i; do_print = false)
        inv_τ = fill(0.01, el_i.n, 1)
        Sᵢ = [zeros(el_i.n, el_f.n)]
        out = EP.solve_electron_bte(el_i, el_f, Sᵢ, inv_τ, occ, symmetry)
        h = EP.solve_electron_bte_hall(out, el_i, el_f, Sᵢ, inv_τ, occ, model, symmetry; order)
        (; el_i, el_f, Sᵢ, inv_τ, occ, out, h)
    end

    # Full-band selection, group-invariant but momentum-dependent Sᵢ[i, f] ∝ w_f (1 + cos 2πΔkx + cos 2πΔky)
    # (so that ∑_f Sᵢ (U X)_f ≠ 0, which the e-ph Holstein model does not give), τ⁻¹ a function of ε.
    function synthetic_case(symmetry; nk = 8, rtol = 1e-13)
        el_i, el_f = hall_states(nk, (-Inf, Inf), symmetry)
        occ = ElectronOccupationParams(; Tlist = [0.05], nlist = 0.0, μlist = -0.1,
            volume = model.volume, nelec = 0, spin_degeneracy = 2, type = :Metal)
        inv_τ = reshape(0.02 .+ 0.01 .* el_i.es .^ 2, :, 1)
        xi, xf = EP.state_xks(el_i), EP.state_xks(el_f)
        Sᵢ = [[0.3 * 0.02 * el_f.weights[f] *
               (1 + cos(2π * (xi[i][1] - xf[f][1])) + cos(2π * (xi[i][2] - xf[f][2]))) *
               (el_i.ibands[i] == el_f.ibands[f]) for i in 1:el_i.n, f in 1:el_f.n]]
        out = EP.solve_electron_bte(el_i, el_f, Sᵢ, inv_τ, occ, symmetry; rtol)
        h = EP.solve_electron_bte_hall(out, el_i, el_f, Sᵢ, inv_τ, occ, model, symmetry; order = 2, rtol)
        (; el_i, el_f, Sᵢ, inv_τ, occ, out, h)
    end
end

@testitem "BTE Hall: finite-difference stencil" begin
    using LinearAlgebra
    max_order = 5

    alat = 0.8
    recip_lattice = Mat3(alat * [-1 1 -1; -1 1 1; 1 1 -1])
    ngrid = 10
    bvecs, bvecs_cart, wbs = finite_difference_vectors(recip_lattice, ngrid)
    @test length(bvecs) == 8
    @test bvecs_cart ≈ Ref(recip_lattice) .* bvecs
    @test all(round.(Int, b .* ngrid) ≈ b .* ngrid for b in bvecs)
    @test sum([b_cart * b_cart' .* wb for (b_cart, wb) in zip(bvecs_cart, wbs)]) ≈ I(3)

    # Higher-order: test ∑_b wb b^(2*i) = 0 for i = 2, ..., order.
    for order in 2:max_order
        bvecs, bvecs_cart, wbs = finite_difference_vectors(recip_lattice, ngrid; order)
        for i in 2:order
            @test sum([b_cart[1]^(2i) .* wb for (b_cart, wb) in zip(bvecs_cart, wbs)]) ≈ 0 atol=1e-13
        end
    end

    a = 2π
    c = 0.3π
    recip_lattice = Mat3([a 0 0; 0 a 0; 0 0 c])
    ngrid = (4, 5, 2)
    bvecs, bvecs_cart, wbs = finite_difference_vectors(recip_lattice, ngrid)
    @test length(bvecs) == 10
    @test bvecs_cart ≈ Ref(recip_lattice) .* bvecs
    @test all(round.(Int, b .* ngrid) ≈ b .* ngrid for b in bvecs)
    @test sum([b_cart * b_cart' .* wb for (b_cart, wb) in zip(bvecs_cart, wbs)]) ≈ I(3)
    # These values have been verified using Wannier90.
    @test wbs ≈ vcat(fill.([0.6304429204412128, -0.0886560356870456, 0.2026423672846756], [2, 2, 6])...)
    @test bvecs ≈ [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, -0.2, 0.0], [0.0, 0.2, 0.0],
                   [0.0, -0.2, -1.0], [0.0, 0.2, -1.0], [-0.25, 0.0, 0.0], [0.25, 0.0, 0.0],
                   [0.0, -0.2, 1.0], [0.0, 0.2, 1.0]]

    for order in 2:max_order
        bvecs, bvecs_cart, wbs = finite_difference_vectors(recip_lattice, ngrid; order)
        for i in 2:order
            @test sum([b_cart[1]^(2i) .* wb for (b_cart, wb) in zip(bvecs_cart, wbs)]) ≈ 0 atol=1e-10
        end
    end
end

@testitem "BTE Hall: gradient matrix and stencil closure" setup = [BteHall] begin
    # D v against the analytic ∂_α v_b: the error ratio between nk and 2nk is 4 (order 1), 16 (order 2).
    function gradient_error(nk, order)
        el_i, el_f = quiet(() -> hall_states(nk, (-Inf, Inf), model.symmetry))
        stencil = quiet(() -> EP.finite_difference_vectors(model.recip_lattice, el_i.kpts.ngrid; order, dimension = model.dimension))
        EP._assert_stencil_closed(stencil, model.symmetry)
        D = EP._finite_difference_gradient_matrices(el_i, el_f, stencil)
        ks = 2π .* EP.state_xks(el_i)
        err = 0.0
        for α in 1:3
            dv = D[α] * el_f.vs
            for i in 1:el_i.n, b in 1:3
                exact = (α == b && b <= 2) ? 2t_hop * alat^2 * cos(ks[i][b]) : 0.0
                err = max(err, abs(dv[i][b] - exact))
            end
        end
        err
    end
    for (order, ratio) in ((1, 4), (2, 16))
        errs = [gradient_error(nk, order) for nk in (8, 16, 32)]
        @test all(isapprox.(errs[1:2] ./ errs[2:3], ratio; rtol = 0.15))
    end

    # The stencil of the 3D Holstein (cubic, 96 operations) is closed too.
    model3 = holstein_model(; t = t_hop, ω₀ = 0.02, g = 0.01, alat, dimension = 3, verbose = false)
    EP._assert_stencil_closed(quiet(() -> EP.finite_difference_vectors(model3.recip_lattice, (6, 6, 6); order = 2, dimension = 3)),
        model3.symmetry)
    # A stencil missing a vector of a shell is not.
    (; bvecs_cart, wbs) = quiet(() -> EP.finite_difference_vectors(model3.recip_lattice, (6, 6, 6)))
    @test_throws ErrorException EP._assert_stencil_closed(
        (; bvecs_cart = bvecs_cart[1:end-1], wbs = wbs[1:end-1]), model3.symmetry)
end

@testitem "BTE Hall: stencil of a 2D model whose cell has c = 100a" setup = [BteHall] begin
    # |b_z| = 2π/(100a) coincides with the length of the in-plane shell at nk = 100 and, as a
    # multiple, at nk = 200. Imposing completeness along z then merged a z vector into an in-plane shell
    # (weight 1.5e-12 on the nearest neighbours at nk = 200, duplicates at order 2). With
    # `dimension = 2` the stencil is the in-plane one at every nk.
    for nk in (100, 200), order in (1, 2)
        stencil = quiet(() -> EP.finite_difference_vectors(model.recip_lattice, (nk, nk, 1);
            order, dimension = model.dimension))
        steps = [Tuple(round.(Int, b .* (nk, nk, 1))) for b in stencil.bvecs]
        @test length(unique(steps)) == length(steps)
        @test all(step -> step[3] == 0, steps)
        @test count(step -> abs(step[1]) + abs(step[2]) == 1, steps) == 4
        @test count(step -> abs(step[1]) + abs(step[2]) == 2 && 0 in step, steps) == (order == 2 ? 4 : 0)
        @test length(steps) == 4 * order
        EP._assert_stencil_closed(stencil, model.symmetry)
    end

    # Order 1 gradient of v at the interior of a band-bottom window: the error falls by 4 from nk = 100 to 200.
    function gradient_error(nk)
        el_i, el_f = quiet(() -> hall_states(nk, (-4t_hop - 1e-9, -4t_hop + 0.05), model.symmetry))
        stencil = quiet(() -> EP.finite_difference_vectors(model.recip_lattice, el_i.kpts.ngrid;
            order = 1, dimension = model.dimension))
        D = EP._finite_difference_gradient_matrices(el_i, el_f, stencil)
        ks = 2π .* EP.state_xks(el_i)
        interior = findall(<(-4t_hop + 0.025), el_i.es)
        err = 0.0
        for α in 1:3
            dv = D[α] * el_f.vs
            for i in interior, b in 1:3
                exact = (α == b && b <= 2) ? 2t_hop * alat^2 * cos(ks[i][b]) : 0.0
                err = max(err, abs(dv[i][b] - exact))
            end
        end
        err
    end
    @test gradient_error(100) / gradient_error(200) ≈ 4 rtol = 0.15
end

@testitem "BTE Hall: rank-3 symmetrize" setup = [BteHall] begin
    # The Levi-Civita tensor ε[a, b, c] is invariant under proper rotations and
    # changes sign under improper ones, so it is invariant when `axial = true` and killed otherwise.
    levi = zeros(3, 3, 3)
    for (a, b, c) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
        levi[a, b, c] = 1
        levi[b, a, c] = -1
    end
    sym3 = holstein_model(; t = t_hop, ω₀ = 0.02, g = 0.01, alat, dimension = 3, verbose = false).symmetry
    @test EP.symmetrize_array(levi, sym3; order = 3, axial = true) ≈ levi
    @test norm(EP.symmetrize_array(levi, sym3; order = 3)) < 1e-14

    # Averaging a random tensor gives a group-invariant one (idempotent), equal to the explicit sum.
    A = randn(3, 3, 3, 2)
    As = EP.symmetrize_array(A, model.symmetry; order = 3, axial = true)
    @test EP.symmetrize_array(As, model.symmetry; order = 3, axial = true) ≈ As
    explicit = zeros(3, 3, 3)
    for (S, inv_op) in zip(model.symmetry.Scart, model.symmetry.is_inv), a in 1:3, b in 1:3, c in 1:3,
            a2 in 1:3, b2 in 1:3, c2 in 1:3
        explicit[a, b, c] += (inv_op ? -1 : 1) * S[a2, a] * S[b2, b] * S[c2, c] * A[a2, b2, c2, 1]
    end
    @test As[:, :, :, 1] ≈ explicit ./ model.symmetry.nsym
end

@testitem "BTE Hall: CRTA Hall factor of a parabolic band" setup = [BteHall] begin
    # r_hall[1, 2, 3] = +1, r_hall[2, 1, 3] = -1 for a parabolic band with constant τ; the
    # tight-binding band is parabolic up to O(kT/t). The error falls from order 1 to 2.
    errors = map((1, 2)) do order
        r = quiet(() -> crta_hall(128, 0.02, order; nlist = 4e-4))
        for r_hall in (r.h.r_hall_serta, r.h.r_hall)   # S_in = 0: SERTA == full
            @test r_hall[1, 2, 3, 1] ≈ 1 rtol = 0.02
            @test r_hall[2, 1, 3, 1] ≈ -1 rtol = 0.02
            @test abs(r_hall[1, 1, 3, 1]) < 1e-10
            # 2D model: entries with a or b = 3 are NaN, the in-plane block is finite.
            @test all(isnan, r_hall[3, :, :, 1]) && all(isnan, r_hall[:, 3, :, 1])
            @test all(isfinite, r_hall[1:2, 1:2, :, 1])
        end
        abs(r.h.r_hall_serta[1, 2, 3, 1] - 1)
    end
    @test errors[2] < errors[1]

    # Holes (band top, nlist < 0): the same signs, so that r_hall[1, 2, 3] = r_H > 0.
    r = quiet(() -> crta_hall(128, 0.02, 2; nlist = -4e-4, holes = true))
    @test r.h.r_hall_serta[1, 2, 3, 1] ≈ 1 rtol = 0.02
    @test r.h.r_hall_serta[2, 1, 3, 1] ≈ -1 rtol = 0.02
end

@testitem "BTE Hall: CRTA σ_hall against the analytic derivative" setup = [BteHall] begin
    # F = τ (-f') v, ∂_α F_b = τ [-f'' v_α v_b - f' ∂_α v_b], X = τ (v_c1 ∂_c2 F_b - v_c2 ∂_c1 F_b).
    function relerr(nk, order)
        c = quiet(() -> crta_hall(nk, 0.05, order; nlist = 4e-3))
        (; el_i, h, occ) = c
        T, μ, τ = occ.Tlist[1], occ.μlist[1], 1 / 0.01
        ks = 2π .* EP.state_xks(el_i)
        X = map(1:el_i.n) do i
            ε, v = el_i.es[i], el_i.vs[i]
            f = EP.occ_fermion(ε - μ, T)
            fp = -f * (1 - f) / T
            fpp = -(1 - 2f) * fp / T
            ∂F(α, b) = τ * (-fpp * v[α] * v[b] -
                fp * ((α == b && b <= 2) ? 2t_hop * alat^2 * cos(ks[i][b]) : 0.0))
            SVector{9}(ntuple(9) do k
                b, cc = mod1(k, 3), (k - 1) ÷ 3 + 1
                c1, c2 = mod1(cc + 1, 3), mod1(cc + 2, 3)
                τ * (v[c1] * ∂F(c2, b) - v[c2] * ∂F(c1, b))
            end)
        end
        σ_exact = EP._hall_conductivity(X, el_i, occ, model.symmetry)
        norm(h.σ_hall_serta[:, :, :, 1] .- σ_exact) / norm(σ_exact)
    end
    e1, e2 = relerr(64, 1), relerr(64, 2)
    @test e1 < 5e-3
    @test e2 < e1 / 4
    @test relerr(128, 1) < e1 / 3     # O(b²) in order 1
end

@testitem "BTE Hall: IBZ equals full BZ with a nontrivial Sᵢ" setup = [BteHall] begin
    a = quiet(() -> synthetic_case(model.symmetry))
    b = quiet(() -> synthetic_case(nothing))
    @test a.el_i.n < b.el_i.n
    for key in (:σ_serta, :σ, :σ_hall_serta, :σ_hall)
        @test getfield(a.h, key) ≈ getfield(b.h, key) rtol = 1e-10
    end
    # The Sᵢ term matters: the full Hall conductivity differs from SERTA.
    @test norm(b.h.σ_hall .- b.h.σ_hall_serta) > 1e-2 * norm(b.h.σ_hall)

    # Field level: the IBZ solution unfolded by U⁽²⁾ reproduces the full-BZ solution on el_f; a map
    # without det R or without the time-reversal sign does not.
    function unfolded_field(c, symmetry, law)
        (; el_i, el_f, Sᵢ, inv_τ, out) = c
        stencil = quiet(() -> EP.finite_difference_vectors(model.recip_lattice, el_i.kpts.ngrid; order = 2, dimension = model.dimension))
        grad = EP._finite_difference_gradient_matrices(el_i, el_f, stencil)
        map_1 = EP.vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry)
        map_2 = EP.vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry, law)
        source = EP._hall_source(grad, map_1 * out.δf_i[:, 1], el_i.vs, inv_τ[:, 1])
        map_2 * EP._solve_bte(source, Sᵢ[1], map_2, inv_τ[:, 1]; rtol = 1e-13).δf
    end
    X_full = quiet(() -> unfolded_field(b, nothing, EP._hall_occupation_law))
    perm = [EP.state_index(b.el_f, a.el_f[f]) for f in 1:a.el_f.n]
    relmiss(law) = maximum(norm.(quiet(() -> unfolded_field(a, model.symmetry, law)) .- X_full[perm])) /
                   maximum(norm.(X_full))
    @test relmiss(EP._hall_occupation_law) < 1e-10
    @test relmiss((is_tr, Scart) -> (is_tr ? -1 : 1) * kron(Scart, Scart)) > 1e-3
    @test relmiss((is_tr, Scart) -> sign(det(Scart)) * kron(Scart, Scart)) > 1e-3
end

@testitem "BTE Hall: input checks" setup = [BteHall] begin
    c = quiet(() -> crta_hall(16, 0.5, 1; nlist = 0.02))
    (; el_i, el_f, Sᵢ, inv_τ, occ, out) = c
    sym = model.symmetry
    hall(out, el_i, el_f, occ, symmetry) =
        quiet(() -> EP.solve_electron_bte_hall(out, el_i, el_f, Sᵢ, inv_τ, occ, model, symmetry))
    @test hall(out, el_i, el_f, occ, sym).σ_hall isa Array

    # out_bte of other temperatures / chemical potentials / symmetry
    occ_T = ElectronOccupationParams(; Tlist = [2 * occ.Tlist[1]], nlist = occ.nlist, μlist = occ.μlist,
        volume = occ.volume, nelec = 0, spin_degeneracy = 2, type = :Semiconductor)
    @test_throws ArgumentError hall(out, el_i, el_f, occ_T, sym)
    @test_throws ArgumentError hall(out, el_i, el_f, occ, nothing)

    # multigrid: an el_f state with a coarse weight
    el_f_mg = deepcopy(el_f)
    el_f_mg.weights[1] *= 2
    @test_throws "multigrids are not supported" hall(out, el_i, el_f_mg, occ, sym)

    # shifted grid
    sel_shift = filter_electron_states((8, 8, 1), model, (-Inf, Inf); shift = (0.5 / 8, 0, 0))
    el_shift = BandStates(compute_electron_states(model, sel_shift,
        ["eigenvalue", "eigenvector", "velocity_diagonal"]), sel_shift)
    @test_throws "shifted grids" hall(out, el_shift, el_shift, occ, nothing)

    # window_kq smaller than window_k: unfolded el_i states missing from el_f
    full = quiet(() -> synthetic_case(nothing))
    el_narrow = hall_states(8, (-4t_hop - 1e-9, -3t_hop), nothing)[1]
    @test_throws "window_kq must contain window_k" quiet(() -> EP.solve_electron_bte_hall(full.out, full.el_i, el_narrow,
        full.Sᵢ, full.inv_τ, full.occ, model, nothing))
end

@testitem "BTE Hall: BoltzmannCalculator end to end" setup = [BteHall] begin
    # Plumbing only: with Holstein e-ph, Sᵢ U⁽²⁾ X = 0 (Sᵢ depends on the energies only).
    function run_hall(symmetry)
        occ = ElectronOccupationParams(; Tlist = [0.04], nlist = 1.0, μlist = -0.1,
            model.volume, nelec = 0, spin_degeneracy = 2, type = :Metal)
        calc = BoltzmannCalculator{Float64}(; occ,
            smearing_list = [SmearingType(:Gaussian, 0.05)], occupation_method = 5)
        EP.run_eph_over_k_and_kq(model, (6, 6, 1), (6, 6, 1); calculators = [calc], symmetry,
            window_k = (-Inf, Inf), window_kq = (-Inf, Inf), progress_print_step = 10^9, verbosity = 0)
        Sₒ = stack(calc.Sₒ)
        out = EP.solve_electron_bte(calc.el_i, calc.el_f, calc.Sᵢ, Sₒ, occ, symmetry; rtol = 1e-13)
        EP.solve_electron_bte_hall(out, calc.el_i, calc.el_f, calc.Sᵢ, Sₒ, occ, model, symmetry;
            order = 2, rtol = 1e-13)
    end
    a = quiet(() -> run_hall(model.symmetry))
    b = quiet(() -> run_hall(nothing))
    @test norm(b.σ_hall) > 0
    # kT = 0.04 Ry and μ off the band center (a particle-hole symmetric μ = 0 gives σ_hall = 0)
    # keep τ⁻¹ in 1.1e-3 ... 6.9e-3 Ry, so Sᵢ X_odd = 0 holds to round-off.
    @test a.σ_hall_serta ≈ b.σ_hall_serta rtol = 1e-10
    @test a.σ_hall ≈ b.σ_hall rtol = 1e-10
    @test b.σ_hall ≈ b.σ_hall_serta rtol = 1e-10     # exact for Holstein: Sᵢ X_odd = 0
end
