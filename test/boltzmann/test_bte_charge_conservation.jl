# Charge-conservation projection of the BTE solve (`charge_conservation_projection`, `_solve_bte`) on Holstein
# states with a synthetic kernel that obeys the pairwise detailed balance of the F5 occupation form, so that
# `A = I - D⁻¹ Sᵢ U` has the null mode `f′ ⊗ v` (a uniform shift of μ).

@testsnippet BteChargeConservation begin
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
    function charge_test_states(nk, symmetry; nkz = 1, mdl = model)
        sel = filter_electron_states((nk, nk, nkz), mdl, (-Inf, Inf); symmetry)
        sel_f = symmetry === nothing ? sel : unfold_band_states(sel, symmetry)
        mk(s) = BandStates(compute_electron_states(mdl, s,
            ["eigenvalue", "eigenvector", "velocity_diagonal"]), s)
        mk(sel), mk(sel_f)
    end

    # Kernel `Sᵢ[i, j] = w_j c_ij √(f′_i / f′_j)`, `τ⁻¹_i = ∑_j w_j c_ij √(f′_j / f′_i)` with `f′ = -∂f/∂ε`: then
    # `Sᵢ f′ = D f′` pair by pair, i.e. `A (f′ e_α) = 0` for an identity unfolding. `c` is positive and
    # invariant under the group; `asym` adds a term odd in k_x of the initial state (not group invariant):
    # it breaks the k ↔ -k parity, so the right-hand side acquires a component along the left null vector.
    # `asym_z` adds a term odd in k_z (invariant under rotations about z). `balanced = false` drops the
    # `√(f′_i / f′_j)` factors: no detailed balance, so the system is nonsingular.
    function f5_case(symmetry; nk = 8, nkz = 1, mdl = model, asym = 0.0, asym_z = 0.0, balanced = true, kw...)
        el_i, el_f = charge_test_states(nk, symmetry; nkz, mdl)
        occ = ElectronOccupationParams(; Tlist = [T_null], nlist = 0.0, μlist = μ_null,
            volume = model.volume, nelec = 0, spin_degeneracy = 2, type = :Metal)
        fp_i = -EP.occ_fermion_derivative.(el_i.es .- μ_null, T_null)
        fp_f = -EP.occ_fermion_derivative.(el_f.es .- μ_null, T_null)
        ratio = balanced ? sqrt.(fp_i ./ fp_f') : ones(length(fp_i), length(fp_f))
        ratio_inv = balanced ? sqrt.(fp_f' ./ fp_i) : ones(length(fp_i), length(fp_f))
        xi, xf = EP.state_xks(el_i), EP.state_xks(el_f)
        C = [0.02 * (2.5 + cos(2π * (xi[i][1] - xf[f][1])) + cos(2π * (xi[i][2] - xf[f][2])) +
                     0.2 * (el_i.es[i] - el_f.es[f]) + asym * sin(2π * xi[i][1]) +
                     (mdl.dimension == 3 ? cos(2π * (xi[i][3] - xf[f][3])) : 0.0) + asym_z * sin(2π * xi[i][3]))
             for i in 1:el_i.n, f in 1:el_f.n] .* (el_i.ibands .== el_f.ibands')
        Sᵢ = [C .* el_f.weights' .* ratio]
        inv_τ = reshape(vec(sum(C .* el_f.weights' .* ratio_inv, dims = 2)), :, 1)
        out = quiet(() -> EP.solve_electron_bte(el_i, el_f, Sᵢ, inv_τ, occ, symmetry; kw...))
        (; el_i, el_f, Sᵢ, inv_τ, occ, out, symmetry)
    end

    # `_solve_bte` on the SERTA right-hand side of `case`, with or without the charge-conservation constraint.
    function raw_solve(case, enforce; solver = :gmres, max_iter = 500, rtol = 1e-10)
        (; el_i, el_f, Sᵢ, inv_τ, occ, symmetry) = case
        rhs = @. -EP.occ_fermion_derivative(el_i.es - μ_null, T_null) / inv_τ[:, 1] * el_i.vs
        map_i_to_f = EP.vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry)
        obs = δf -> EP.symmetrize(EP.occupation_to_conductivity(δf, el_i, occ), symmetry)
        P_charge = EP.charge_conservation_projection(el_i, symmetry, μ_null, T_null; enforce)
        quiet(() -> EP._solve_bte(rhs, Sᵢ[1], map_i_to_f, view(inv_τ, :, 1); solver, max_iter, rtol,
                                  observable = obs, P_charge))
    end

    # Particle-number moment `∑ᵢ wᵢ δfᵢ[α]` of a solution.
    number_moment(case, δf) = sum(case.el_i.weights[i] .* δf[i] for i in 1:case.el_i.n)
end

@testitem "BTE charge conservation: constraint directions" setup = [BteChargeConservation] begin
    dirs(symmetry, args...; kw...) = EP.charge_conservation_projection(first(charge_test_states(4, nothing)), symmetry,
        μ_null, T_null, args...; kw...).dirs
    none3 = zeros(3, 0)
    id3, Cz = Mat3{Float64}(I), Mat3{Float64}(0, 1, 0, -1, 0, 0, 0, 0, 1)   # rotation by π/2 about z
    mk(Scarts, is_tr) = Symmetry(length(Scarts), fill(Mat3{Int}(I), length(Scarts)),
        fill(Vec3(0.0, 0.0, 0.0), length(Scarts)), Scarts, fill(Vec3(0.0, 0.0, 0.0), length(Scarts)),
        fill(false, length(Scarts)), is_tr)

    # symmetry = nothing: P = I, one constraint per component.
    @test dirs(nothing) == Matrix(I, 3, 3)
    @test size(dirs(nothing, EP._hall_transport_mapping)) == (9, 9)
    # enforce = false: no constraint directions.
    @test dirs(nothing; enforce = false) == none3
    @test size(dirs(nothing, EP._hall_transport_mapping; enforce = false)) == (9, 0)
    # Identity only, no time reversal: the same as no symmetry.
    @test dirs(mk([id3], [false])) == Matrix(I, 3, 3)
    # Time reversal in the group: P = 0, the constraint holds identically (no directions).
    @test dirs(mk([id3, id3], [false, true])) == none3
    @test dirs(model.symmetry) == none3
    # Polar group without time reversal: the rotation axis is an invariant polar vector.
    d = dirs(mk([id3, Cz, Cz * Cz, Cz^3], fill(false, 4)))
    @test size(d) == (3, 1) && abs(d[3, 1]) ≈ 1 && norm(d[1:2, 1]) < 1e-12
    # Inversion removes it; the Hall field (one polar and one axial index, the map carries det R) is also odd under it.
    @test dirs(mk([id3, -id3], [false, false])) == none3
    @test size(dirs(mk([id3, -id3], [false, false]), EP._hall_transport_mapping)) == (9, 0)
    # Hall field under the rotation about z: span(X_xx + X_yy, X_xy - X_yx, X_zz) is invariant.
    @test size(dirs(mk([id3, Cz, Cz * Cz, Cz^3], fill(false, 4)), EP._hall_transport_mapping)) == (9, 3)
    # The model's spatial group (D4h-like, inversion included) keeps no mode even without time reversal.
    spatial = findall(!, model.symmetry.is_tr)
    @test dirs(mk(model.symmetry.Scart[spatial], fill(false, length(spatial)))) == none3
end

@testitem "BTE charge conservation: projector properties" setup = [BteChargeConservation] begin
    el_i = first(charge_test_states(4, nothing))
    P = EP.charge_conservation_projection(el_i, nothing, μ_null, T_null)
    nx = 3 * el_i.n
    @test size(P) == (nx, nx) && size(P.B, 2) == 3
    x = randn(nx)
    @test P * (P * x) ≈ P * x rtol = 1e-12                         # Π² = Π
    @test norm(P.U' * (P * x)) < 1e-12 * norm(x)                   # UᵀΠ = 0: charge conserved
    @test all(norm(P * b) < 1e-12 * norm(b) for b in eachcol(P.B))  # Π B = 0: μ shift removed
    # The removed part is f′ ⊗ δμ, and chemical_potential_shift returns that δμ.
    fprime = -EP.occ_fermion_derivative.(el_i.es .- μ_null, T_null)
    δμ = [0.3, -1.2, 0.7]
    r = collect(reinterpret(Float64, [fprime[i] * Vec3(δμ...) for i in 1:el_i.n]))
    @test EP.chemical_potential_shift(P, r) ≈ δμ rtol = 1e-12
    @test norm(P * r) < 1e-12 * norm(r)
    # enforce = false: the identity, δμ = 0.
    P_off = EP.charge_conservation_projection(el_i, nothing, μ_null, T_null; enforce = false)
    @test P_off * x == x && EP.chemical_potential_shift(P_off, r) == zeros(3)
end

@testitem "BTE charge conservation: full BZ against the IBZ and a dense constrained solve" setup = [BteChargeConservation] begin
    # Parity-symmetric kernel: the system is consistent, the full-BZ solve equals the IBZ one, δμ = 0.
    ibz, full = f5_case(model.symmetry), f5_case(nothing)
    @test ibz.el_i.n < full.el_i.n
    @test full.out.σ ≈ ibz.out.σ rtol = 1e-10
    @test norm(full.out.δμ) < 1e-10
    @test ibz.out.δμ == zeros(3, 1)          # time reversal in the map: no constraint directions
    # In the IBZ the constraint holds identically: enforcing it or not gives the same solution.
    ibz_off = f5_case(model.symmetry; enforce_charge_conservation = false)
    @test ibz_off.out.σ == ibz.out.σ && ibz_off.out.δf_i == ibz.out.δf_i

    # Parity-broken kernel: A x = b is inconsistent along the null mode.
    case = f5_case(nothing; asym = 0.05)
    @test abs(case.out.δμ[1]) > 0.1          # δμ_x measures the inconsistency; δμ_y = δμ_z = 0 here
    @test abs(case.out.δμ[2]) < 1e-10 && case.out.δμ[3] == 0
    # Without the constraint: GMRES stops unconverged and σ is off (by 0.2–1.3 %: the size depends on where
    # GMRES stalls, which varies with BLAS and threads).
    old = raw_solve(case, false)
    @test !old.converged
    @test abs(old.obs[1, 1] - full.out.σ[1, 1, 1]) / full.out.σ[1, 1, 1] > 1e-4
    off = f5_case(nothing; asym = 0.05, enforce_charge_conservation = false)
    @test off.out.σ[:, :, 1] ≈ old.obs rtol = 1e-12
    @test off.out.δμ == zeros(3, 1)
    # Constrained GMRES converges and moves σ only at the size of the parity breaking (3.6e-4 here).
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
        @test sol[end] ≈ case.out.δμ[α, 1] rtol = 1e-6 atol = 1e-9
    end
end

@testitem "BTE charge conservation: fixed point and polar group without time reversal" setup = [BteChargeConservation] begin
    case = f5_case(nothing; asym = 0.05)
    gm = raw_solve(case, true)
    fp = raw_solve(case, true; solver = :fixed_point)
    @test gm.converged && fp.converged
    @test fp.δf ≈ gm.δf rtol = 1e-6
    @test fp.δμ ≈ gm.δμ rtol = 1e-6
    @test abs(number_moment(case, gm.δf)[1]) < 1e-10 * sum(abs, getindex.(gm.δf, 1)) / case.el_i.n
    @test abs(number_moment(case, fp.δf)[1]) < 1e-10 * sum(abs, getindex.(fp.δf, 1)) / case.el_i.n
    # Without the constraint the fixed point drifts along f′ (the iteration has the eigenvalue 1).
    fp_old = raw_solve(case, false; solver = :fixed_point)
    @test abs(number_moment(case, fp_old.δf)[1]) > 1e-3 * sum(abs, getindex.(fp_old.δf, 1)) / case.el_i.n

    # The C4 rotations (no time reversal, z axis invariant): IBZ equals full BZ, one constraint along e_z, δμ = 0.
    idx = findall(i -> !model.symmetry.is_tr[i] && model.symmetry.Scart[i][3, 3] == 1 &&
                       det(model.symmetry.Scart[i]) > 0, 1:model.symmetry.nsym)
    s = model.symmetry
    c4 = Symmetry(length(idx), s.S[idx], s.τ[idx], s.Scart[idx], s.τcart[idx], s.is_inv[idx], s.is_tr[idx])
    @test length(idx) == 4 && !c4.time_reversal
    polar, full = f5_case(c4), f5_case(nothing)
    @test size(EP.charge_conservation_projection(polar.el_i, c4, μ_null, T_null).dirs, 2) == 1
    @test polar.out.σ ≈ full.out.σ rtol = 1e-10
    @test norm(polar.out.δμ) < 1e-10
end

@testitem "BTE charge conservation: polar group without time reversal on a 3D model, drive along the axis" setup = [BteChargeConservation] begin
    mdl = holstein_model(; t = 0.1, ω₀ = 0.02, g = 0.01, alat = 5.0, dimension = 3, verbose = false)
    mdl.el_velocity_mode = :BerryConnection
    s = mdl.symmetry
    idx = findall(i -> !s.is_tr[i] && s.Scart[i][3, 3] == 1 && det(s.Scart[i]) > 0, 1:s.nsym)
    c4 = Symmetry(length(idx), s.S[idx], s.τ[idx], s.Scart[idx], s.τcart[idx], s.is_inv[idx], s.is_tr[idx])
    @test length(idx) == 4 && !c4.time_reversal
    # The kernel is odd in k_z (invariant under C4 about z), so rhs_z is nonzero and inconsistent: δμ_z != 0.
    polar = f5_case(c4; nk = 4, nkz = 4, mdl, asym_z = 0.05)
    full = f5_case(nothing; nk = 4, nkz = 4, mdl, asym_z = 0.05)
    @test polar.el_i.n < full.el_i.n
    @test abs(full.out.δμ[3]) > 1e-3
    @test polar.out.δμ[3] ≈ full.out.δμ[3] rtol = 1e-6
    @test polar.out.σ ≈ full.out.σ rtol = 1e-6
    # The unconstrained solve on the same system is off.
    old = raw_solve(polar, false)
    cons = raw_solve(polar, true)
    @test cons.converged
    @test abs(old.obs[3, 3] - cons.obs[3, 3]) > 1e-6 * abs(cons.obs[3, 3]) || !old.converged
end

@testitem "BTE charge conservation: kernel without detailed balance" setup = [BteChargeConservation] begin
    # Nonsingular system (no f′ null mode): the constraint still imposes Σᵢ wᵢ δfᵢ = 0 and returns δμ.
    case = f5_case(nothing; asym = 0.05, balanced = false)
    gm = raw_solve(case, true)
    fp = raw_solve(case, true; solver = :fixed_point)
    @test gm.converged && fp.converged
    @test fp.δf ≈ gm.δf rtol = 1e-6
    @test fp.δμ ≈ gm.δμ rtol = 1e-5 atol = 1e-9
    @test abs(number_moment(case, gm.δf)[1]) < 1e-10 * sum(abs, getindex.(gm.δf, 1)) / case.el_i.n
    # Dense reference of the constrained system for the x component.
    (; el_i, Sᵢ, inv_τ) = case
    A = I - Sᵢ[1] ./ inv_τ[:, 1]
    fprime = -EP.occ_fermion_derivative.(el_i.es .- μ_null, T_null)
    rhs = @. fprime / inv_τ[:, 1] * el_i.vs
    sol = [A fprime; el_i.weights' 0.0] \ [getindex.(rhs, 1); 0.0]
    @test norm(sol[1:end-1] - getindex.(gm.δf, 1)) / norm(sol[1:end-1]) < 1e-8
    @test sol[end] ≈ gm.δμ[1] rtol = 1e-6 atol = 1e-9
end

@testitem "BTE charge conservation: Hall solve" setup = [BteChargeConservation] begin
    hall(case; kw...) = quiet(() -> EP.solve_electron_bte_hall(case.out, case.el_i, case.el_f, case.Sᵢ, case.inv_τ,
        case.occ, model, case.symmetry; order = 2, max_iter = 500, kw...))
    # Parity-symmetric kernel: the Hall source is consistent, δμ_hall = 0 and the keyword changes nothing.
    sym = f5_case(nothing)
    @test norm(hall(sym).δμ_hall) < 1e-10
    # Parity-broken kernel: the Hall source is inconsistent along the null mode; the constraint absorbs it in
    # δμ_hall (34 here) and σ_hall without it is off (3.7 % here; the size depends on where GMRES stalls).
    case = f5_case(nothing; asym = 0.05)
    on, off = hall(case), hall(case; enforce_charge_conservation = false)
    @test norm(on.δμ_hall) > 1
    @test off.δμ_hall == zeros(9, 1)
    @test norm(on.σ_hall .- off.σ_hall) > 1e-4 * norm(on.σ_hall)
    # IBZ (time reversal in the map): no constraint rows.
    ibz = f5_case(model.symmetry)
    @test hall(ibz).σ_hall == hall(ibz; enforce_charge_conservation = false).σ_hall
    @test hall(ibz).δμ_hall == zeros(9, 1)
end
