# Null mode of the BTE operator `A = I - D⁻¹ Sᵢ U` (`_bte_null_mode`, bordered solve in `_solve_bte`) on 2D Holstein
# states with a synthetic kernel that obeys the pairwise detailed balance of the F5 occupation form.

@testsnippet BteNullMode begin
    using LinearAlgebra
    using StaticArrays
    using ElectronPhonon: BandStates, filter_electron_states, unfold_band_states,
        compute_electron_states, holstein_model, Symmetry, Mat3, Vec3
    const EP = ElectronPhonon

    const model = holstein_model(; t = 0.1, ω₀ = 0.02, g = 0.01, alat = 5.0, dimension = 2, verbose = false)
    model.el_velocity_mode = :BerryConnection
    const T_null, μ_null = 0.05, -0.1

    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

    # Full-band el_i (IBZ if `symmetry`) and el_f (its unfolding) on an nk × nk × 1 grid.
    function null_mode_states(nk, symmetry)
        sel = filter_electron_states((nk, nk, 1), model, (-Inf, Inf); symmetry)
        sel_f = symmetry === nothing ? sel : unfold_band_states(sel, symmetry)
        mk(s) = BandStates(compute_electron_states(model, s,
            ["eigenvalue", "eigenvector", "velocity_diagonal"]), s)
        mk(sel), mk(sel_f)
    end

    # Kernel `Sᵢ[i, j] = w_j c_ij √(f′_i / f′_j)`, `τ⁻¹_i = ∑_j w_j c_ij √(f′_j / f′_i)` with `f′ = -∂f/∂ε`: then
    # `Sᵢ f′ = D f′` pair by pair, i.e. `A (f′ e_α) = 0` for an identity unfolding. `c` is positive and
    # invariant under the group; `asym` adds a term odd in k_x of the initial state (not group invariant):
    # it breaks the k ↔ -k parity, so the right-hand side acquires a component along the left null vector.
    function f5_case(symmetry; nk = 8, asym = 0.0, kw...)
        el_i, el_f = null_mode_states(nk, symmetry)
        occ = ElectronOccupationParams(; Tlist = [T_null], nlist = 0.0, μlist = μ_null,
            volume = model.volume, nelec = 0, spin_degeneracy = 2, type = :Metal)
        fp_i = -EP.occ_fermion_derivative.(el_i.es .- μ_null, T_null)
        fp_f = -EP.occ_fermion_derivative.(el_f.es .- μ_null, T_null)
        xi, xf = EP.state_xks(el_i), EP.state_xks(el_f)
        C = [0.02 * (2.5 + cos(2π * (xi[i][1] - xf[f][1])) + cos(2π * (xi[i][2] - xf[f][2])) +
                     0.2 * (el_i.es[i] - el_f.es[f]) + asym * sin(2π * xi[i][1]))
             for i in 1:el_i.n, f in 1:el_f.n] .* (el_i.ibands .== el_f.ibands')
        Sᵢ = [C .* el_f.weights' .* sqrt.(fp_i ./ fp_f')]
        inv_τ = reshape(vec(sum(C .* el_f.weights' .* sqrt.(fp_f' ./ fp_i), dims = 2)), :, 1)
        out = quiet(() -> EP.solve_electron_bte(el_i, el_f, Sᵢ, inv_τ, occ, symmetry; kw...))
        (; el_i, el_f, Sᵢ, inv_τ, occ, out, symmetry)
    end

    # `_solve_bte` on the SERTA right-hand side of `case`, with the null mode of the symmetry or without it.
    function raw_solve(case, bordered; solver = :gmres, max_iter = 500, rtol = 1e-10)
        (; el_i, el_f, Sᵢ, inv_τ, occ, symmetry) = case
        rhs = @. -EP.occ_fermion_derivative(el_i.es - μ_null, T_null) / inv_τ[:, 1] * el_i.vs
        map_i_to_f = EP.vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry)
        obs = δf -> EP.symmetrize(EP.occupation_to_conductivity(δf, el_i, occ), symmetry)
        null_mode = bordered ? EP._bte_null_mode(el_i, symmetry, μ_null, T_null) : nothing
        quiet(() -> EP._solve_bte(rhs, Sᵢ[1], map_i_to_f, view(inv_τ, :, 1); solver, max_iter, rtol,
                                  observable = obs, null_mode))
    end

    # Particle-number moment `∑ᵢ wᵢ δfᵢ[α]` of a solution.
    number_moment(case, δf) = sum(case.el_i.weights[i] .* δf[i] for i in 1:case.el_i.n)
end

@testitem "BTE null mode: criterion of the surviving directions" setup = [BteNullMode] begin
    nullmode(symmetry, args...) = EP._bte_null_mode(first(null_mode_states(4, nothing)), symmetry, μ_null, T_null, args...)
    id3, Cz = Mat3{Float64}(I), Mat3{Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1)   # rotation by π/2 about z
    mk(Scarts, is_tr) = Symmetry(length(Scarts), fill(Mat3{Int}(I), length(Scarts)),
        fill(Vec3(0.0, 0.0, 0.0), length(Scarts)), Scarts, fill(Vec3(0.0, 0.0, 0.0), length(Scarts)),
        fill(false, length(Scarts)), is_tr)

    # symmetry = nothing: every component keeps its null mode.
    @test nullmode(nothing).dirs == Matrix(I, 3, 3)
    @test size(nullmode(nothing, EP._hall_transport_mapping).dirs) == (9, 9)
    # Identity only, no time reversal: the same.
    @test nullmode(mk([id3], [false])).dirs == Matrix(I, 3, 3)
    # Time reversal in the group excludes the mode (even in k against the field odd in k).
    @test nullmode(mk([id3, id3], [false, true])) === nothing
    @test nullmode(model.symmetry) === nothing
    # Polar group without time reversal: the rotation axis is an invariant polar vector.
    d = nullmode(mk([id3, Cz, Cz * Cz, Cz^3], fill(false, 4))).dirs
    @test size(d) == (3, 1) && abs(d[3, 1]) ≈ 1 && norm(d[1:2, 1]) < 1e-12
    # Inversion removes it; the Hall field (one polar and one axial index, the map carries det R) is also odd under it.
    @test nullmode(mk([id3, -id3], [false, false])) === nothing
    @test nullmode(mk([id3, -id3], [false, false]), EP._hall_transport_mapping) === nothing
    # Hall field under the rotation about z: span(X_xx + X_yy, X_xy - X_yx, X_zz) is invariant.
    @test size(nullmode(mk([id3, Cz, Cz * Cz, Cz^3], fill(false, 4)), EP._hall_transport_mapping).dirs) == (9, 3)
    # The model's spatial group (D4h-like, inversion included) keeps no mode even without time reversal.
    spatial = findall(!, model.symmetry.is_tr)
    @test nullmode(mk(model.symmetry.Scart[spatial], fill(false, length(spatial)))) === nothing
end

@testitem "BTE null mode: full BZ against the IBZ and a dense bordered solve" setup = [BteNullMode] begin
    # Parity-symmetric kernel: the system is consistent, the full-BZ solve equals the IBZ one, λ = 0.
    ibz, full = f5_case(model.symmetry), f5_case(nothing)
    @test ibz.el_i.n < full.el_i.n
    @test full.out.σ ≈ ibz.out.σ rtol = 1e-10
    @test norm(full.out.λ) < 1e-10
    @test ibz.out.λ == zeros(3, 1)          # the IBZ path (time reversal in the map) is not bordered

    # Parity-broken kernel: A x = b is inconsistent along the null mode.
    case = f5_case(nothing; asym = 0.05)
    @test abs(case.out.λ[1]) > 0.1          # λ_x measures the inconsistency; λ_y = λ_z = 0 here
    @test abs(case.out.λ[2]) < 1e-10 && case.out.λ[3] == 0
    # Old path (no border): GMRES stops unconverged and σ is off by 1.3 %.
    old = raw_solve(case, false)
    @test !old.converged
    @test abs(old.obs[1, 1] - full.out.σ[1, 1, 1]) / full.out.σ[1, 1, 1] > 5e-3
    # Bordered GMRES converges and moves σ only at the size of the parity breaking (3.6e-4 here).
    @test case.out.σ[1, 1, 1] ≈ full.out.σ[1, 1, 1] rtol = 5e-3

    # Independent reference: dense solve of [A f′; wᵀ 0] per Cartesian component.
    (; el_i, Sᵢ, inv_τ) = case
    A = I - Sᵢ[1] ./ inv_τ[:, 1]
    fp = -EP.occ_fermion_derivative.(el_i.es .- μ_null, T_null)
    rhs = @. fp / inv_τ[:, 1] * el_i.vs
    for α in 1:2
        b = [getindex.(rhs, α); 0.0]
        sol = [A fp; el_i.weights' 0.0] \ b
        @test norm(sol[1:end-1] - getindex.(case.out.δf_i[:, 1], α)) / norm(sol[1:end-1]) < 1e-8
        @test sol[end] ≈ case.out.λ[α, 1] rtol = 1e-6 atol = 1e-9
    end
end

@testitem "BTE null mode: fixed point and polar group without time reversal" setup = [BteNullMode] begin
    case = f5_case(nothing; asym = 0.05)
    gm = raw_solve(case, true)
    fp = raw_solve(case, true; solver = :fixed_point)
    @test gm.converged && fp.converged
    @test fp.δf ≈ gm.δf rtol = 1e-6
    @test fp.λ ≈ gm.λ rtol = 1e-6
    @test abs(number_moment(case, gm.δf)[1]) < 1e-10 * sum(abs, getindex.(gm.δf, 1)) / case.el_i.n
    @test abs(number_moment(case, fp.δf)[1]) < 1e-10 * sum(abs, getindex.(fp.δf, 1)) / case.el_i.n
    # Without the border the fixed point drifts along f′ (the iteration has the eigenvalue 1).
    fp_old = raw_solve(case, false; solver = :fixed_point)
    @test abs(number_moment(case, fp_old.δf)[1]) > 1e-3 * sum(abs, getindex.(fp_old.δf, 1)) / case.el_i.n

    # The C4 rotations (no time reversal, z axis invariant): IBZ equals full BZ, the mode e_z is bordered with λ = 0.
    idx = findall(i -> !model.symmetry.is_tr[i] && model.symmetry.Scart[i][3, 3] == 1 &&
                       det(model.symmetry.Scart[i]) > 0, 1:model.symmetry.nsym)
    s = model.symmetry
    c4 = Symmetry(length(idx), s.S[idx], s.τ[idx], s.Scart[idx], s.τcart[idx], s.is_inv[idx], s.is_tr[idx])
    @test length(idx) == 4 && !c4.time_reversal
    polar, full = f5_case(c4), f5_case(nothing)
    @test size(EP._bte_null_mode(polar.el_i, c4, μ_null, T_null).dirs, 2) == 1
    @test polar.out.σ ≈ full.out.σ rtol = 1e-10
    @test norm(polar.out.λ) < 1e-10
end
