@testsnippet Symmetry begin
    using StaticArrays
    using LinearAlgebra
    using Random
end

@testitem "symmetry" setup = [Common, Symmetry] begin
    # Silicon structure
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["Si" => [ones(3)/8, -ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)
    @test symmetry.nsym == 96
    @test symmetry.time_reversal == true
    @test sum(symmetry.is_tr) == 48
    @test sum(symmetry.is_inv) == 48
    @test sum(symmetry.is_inv .& symmetry.is_tr) == 24

    @test all(inv(symop) * symop ≈ one(symop) for symop in symmetry)
    @test all(symop * inv(symop) ≈ one(symop) for symop in symmetry)
    ElectronPhonon.check_group(symmetry)

    kpts = kpoints_grid((6, 6, 6); symmetry)
    @test kpts.n == 16
    kpts = kpoints_grid((6, 6, 6); symmetry, ignore_time_reversal=true)
    @test kpts.n == 16
    kpts = kpoints_grid((7, 7, 7); symmetry)
    @test kpts.n == 20

    # Cubic Boron Nitride structure
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)
    @test symmetry.nsym == 48
    @test symmetry.time_reversal == true
    @test sum(symmetry.is_tr) == 24
    @test sum(symmetry.is_inv) == 24
    @test sum(symmetry.is_inv .& symmetry.is_tr) == 12

    @test all(inv(symop) * symop ≈ one(symop) for symop in symmetry)
    @test all(symop * inv(symop) ≈ one(symop) for symop in symmetry)
    ElectronPhonon.check_group(symmetry)

    kpts = kpoints_grid((6, 6, 6); symmetry)
    @test kpts.n == 16
    kpts = kpoints_grid((6, 6, 6); symmetry, ignore_time_reversal=true)
    @test kpts.n == 22
    kpts = kpoints_grid((7, 7, 7); symmetry)
    @test kpts.n == 20

    # Cubic Boron Nitride structure imported from file.
    model = _load_model_from_artifacts("cubicBN"; load_epmat = false)
    @test model.symmetry.nsym == 48
    @test model.symmetry.time_reversal == true
    @test length(model.symmetry.is_tr) == 48
    @test count(model.symmetry.is_tr) == 24

    symmetry = model.symmetry
    @test all(inv(symop) * symop ≈ one(symop) for symop in symmetry)
    @test all(symop * inv(symop) ≈ one(symop) for symop in symmetry)
    ElectronPhonon.check_group(symmetry)

    kpts = kpoints_grid((6, 6, 6); symmetry)
    @test kpts.n == 16
    kpts = kpoints_grid((7, 7, 7); symmetry)
    @test kpts.n == 20

    # Pb (fcc, one atom)
    lattice = 4.9 / 2 * [[0 1 1.];
                         [1 0 1.];
                         [1 1 0.]]
    symmetry = symmetry_operations(lattice, ["Pb" => [zeros(3)]])
    @test symmetry.nsym == 96
    kpts = kpoints_grid((6, 6, 6); symmetry)
    @test kpts.n == 16
    @test kpts.vectors[2] == Vec3(0, 0, 1/6)
    @test kpts.weights[2] ≈ 8 / 6^3
    kpts = kpoints_grid((7, 7, 7); symmetry, ignore_time_reversal=true)
    @test kpts.n == 20
    # A grid that breaks the cubic symmetry
    @test_throws "breaks the symmetry" kpoints_grid((6, 6, 4); symmetry)
    @test_throws "breaks the symmetry" kpoints_grid((1, 2, 1); symmetry)

    # Simple tetragonal, non-uniform grids
    symmetry = symmetry_operations([3.0 0 0; 0 3.0 0; 0 0 5.0], ["X" => [zeros(3)]])
    @test symmetry.nsym == 32
    @test kpoints_grid((6, 6, 4); symmetry).n == 30
    @test kpoints_grid((5, 5, 7); symmetry).n == 24
end

@testitem "small group of q" setup = [Common, Symmetry] begin
    # Silicon (diamond) structure: full cubic point group + time reversal.
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["Si" => [ones(3)/8, -ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)
    @test symmetry.nsym == 96

    # Every kept op must map xq onto xq modulo a reciprocal lattice vector.
    fixes_q(sym, xq) = all(sym) do s
        sxq = s.is_tr ? -s.S * xq : s.S * xq
        isapprox(sxq - xq, round.(sxq - xq); atol=sqrt(eps(Float64)))
    end

    # Γ point: the whole group is the little group.
    xq = Vec3(0., 0., 0.)
    small, inds = symmetry_small_group_of_q(symmetry, xq)
    @test small.nsym == 48
    @test small.time_reversal == false
    @test all(.!small.is_tr)
    small_tr, inds_tr = symmetry_small_group_of_q(symmetry, xq; keep_trs=true)
    @test small_tr.nsym == 96
    @test small_tr.time_reversal == true

    # General low-symmetry q: only the identity survives without time reversal.
    # With TR the little group is {(I, is_tr=false), (-I, is_tr=true)}, which contains
    # a time-reversal op but is NOT a grey group, so time_reversal must be false.
    xq = Vec3(0.13, 0.27, 0.41)
    small, inds = symmetry_small_group_of_q(symmetry, xq)
    @test small.nsym == 1
    @test isone(small[1])
    small_tr, inds_tr = symmetry_small_group_of_q(symmetry, xq; keep_trs=true)
    @test small_tr.nsym == 2
    @test any(small_tr.is_tr)
    @test small_tr.time_reversal == false

    # Zone-boundary (TRIM) q: requires the mod-G comparison (S*q = q - G with G ≠ 0).
    # At a TRIM point the little group is a grey group, so time_reversal is true.
    xq = Vec3(0.5, 0.0, 0.5)
    small, inds = symmetry_small_group_of_q(symmetry, xq)
    @test small.nsym == 16
    # `mod_G = false` keeps only S * xq == xq, the operations that commute with the gauge shift exp(iπ xq·R)
    small_exact, inds_exact = symmetry_small_group_of_q(symmetry, xq; mod_G = false)
    @test small_exact.nsym == 8
    @test inds_exact == [i for i in inds if symmetry.S[i] * xq == xq]
    @test symmetry_small_group_of_q(symmetry, Vec3(0.0, 0.0, 0.0); mod_G = false)[1].nsym == 48
    small_tr, inds_tr = symmetry_small_group_of_q(symmetry, xq; keep_trs=true)
    @test small_tr.nsym == 32
    @test small_tr.time_reversal == true
    # At least one kept op shifts q by a nonzero reciprocal lattice vector.
    @test any(s -> !isapprox(s.is_tr ? -s.S * xq : s.S * xq, xq), small_tr)

    # Returned indices are consistent and the result is a genuine subgroup.
    for xq in (Vec3(0., 0., 0.), Vec3(0.5, 0., 0.5), Vec3(0.13, 0.27, 0.41))
        for keep_trs in (false, true)
            local small, inds
            small, inds = symmetry_small_group_of_q(symmetry, xq; keep_trs)
            @test small.S == symmetry.S[inds]
            @test small.is_tr == symmetry.is_tr[inds]
            @test fixes_q(small, xq)
            @test symmetry_is_subset(small, symmetry)
            ElectronPhonon.check_group(small)
            keep_trs || @test all(.!small.is_tr)
        end
    end
end

@testitem "fit_onsite_symmetry_operators" setup = [Common, Symmetry, CubicSP] begin
    model = cubic_sp_model()
    nw = model.nw
    symmetry = model.symmetry
    @test symmetry.nsym == 96

    el_sym = fit_onsite_symmetry_operators(model)
    @test el_sym.symmetry === symmetry
    @test length(el_sym.operators) == symmetry.nsym
    D_of(isym) = real(reshape(el_sym.operators[isym].op_r, nw, nw))
    @test all(op -> op.ndata == nw^2 && op.irvec == [Vec3(0, 0, 0)], el_sym.operators)
    ops = findall(!, symmetry.is_tr)
    @test length(ops) == 48

    # Independent residual of H(S_R R) D = D H(R) against the model's own H(R), without the fitter's code
    blocks = reshape(model.el_ham.op_r, nw, nw, :)
    irindex = Dict(R => ir for (ir, R) in enumerate(model.el_ham.irvec))
    hscale = maximum(ir -> norm(blocks[:, :, ir]), axes(blocks, 3))
    resid = maximum(ops) do isym
        SR = round.(Int, inv(symmetry.S[isym])')
        maximum(ir -> norm(blocks[:, :, irindex[SR * model.el_ham.irvec[ir]]] * D_of(isym) -
                           D_of(isym) * blocks[:, :, ir]) / hscale, eachindex(model.el_ham.irvec))
    end
    @test resid < 1e-13

    # The fit reproduces the analytic D = blockdiag(1, Scart) up to the representation-fixing sign
    # character, is orthogonal, closes into a representation, and time-reversal copies share it.
    @test maximum(isym -> min(norm(D_of(isym) - analytic_D(symmetry.Scart[isym])),
                              norm(D_of(isym) + analytic_D(symmetry.Scart[isym]))), ops) < 1e-12
    @test maximum(isym -> norm(D_of(isym)' * D_of(isym) - I), ops) < 1e-13
    op_of = Dict(Tuple(symmetry.S[isym]) => isym for isym in ops)
    @test maximum(norm(D_of(a) * D_of(b) - D_of(op_of[Tuple(symmetry.S[a] * symmetry.S[b])])) for a in ops, b in ops) < 1e-13
    @test all(isym -> D_of(isym) == D_of(op_of[Tuple(symmetry.S[isym])]), findall(symmetry.is_tr))

    # A subgroup: operators for a little group only
    small = symmetry_small_group_of_q(symmetry, Vec3(0.5, 0.0, 0.5); mod_G = false)[1]
    @test fit_onsite_symmetry_operators(model; symmetry = small).symmetry === small

    # Outside the envelope: H(R) breaking the cubic symmetry (one hopping scaled), Wannier centre off
    # the atom.
    @test_throws "not symmetric under an on-site D" fit_onsite_symmetry_operators(cubic_sp_model(perturb = true))
    model_off = cubic_sp_model()
    model_off.wann_centers[1] = Vec3(0.3 * model.alat, 0.0, 0.0)
    @test_throws "off itself" fit_onsite_symmetry_operators(model_off)
    # s and p decoupled: the commutant is two-dimensional, D_S is not fixed by H(R) alone
    @test_throws "not unique" fit_onsite_symmetry_operators(cubic_sp_model(decoupled = true))
end

@testitem "symmetrize" setup = [Common, Symmetry] begin
    Random.seed!(123)

    lattice = 2.0 * [[0 1 1.]; [1 0 1.]; [1 1 0.]];
    atoms = ["A" => [ones(3)/8], "B" => [-ones(3)/9]];
    symmetry = symmetry_operations(lattice, atoms);

    # Scalar
    @test symmetrize(1.0, symmetry) == 1.0
    @test symmetrize(1.0, symmetry, tr_odd=true, axial=false) == 0.0
    @test symmetrize(1.0, symmetry, tr_odd=false, axial=true) == 0.0
    @test symmetrize(1.0, symmetry, tr_odd=true, axial=true) == 0.0

    # Vector
    v = Vec3(1., 2., 3.)
    @test symmetrize(v, symmetry) ≈ Vec3(2., 2., 2.)
    @test symmetrize(v, symmetry, tr_odd=true, axial=false) ≈ zero(v)
    @test symmetrize(v, symmetry, tr_odd=false, axial=true) ≈ zero(v)
    @test symmetrize(v, symmetry, tr_odd=true, axial=true) ≈ zero(v)

    # Matrix
    m = Mat3(rand(3, 3))
    m_sym = symmetrize(m, symmetry)
    for i = 1:3, j = 1:3
        if i == j
            @test m_sym[i, j] ≈ tr(m) / 3
        else
            @test m_sym[i, j] ≈ (sum(m) - tr(m)) / 6
        end
    end
    m_sym = symmetrize(m, symmetry, tr_odd=false, axial=true)
    val_asym = (m[1, 2] + m[2, 3] + m[3, 1] - m[2, 1] - m[3, 2] - m[1, 3]) / 6
    for i = 1:3, j = 1:3
        if i == j
            @test m_sym[i, j] ≈ 0 atol=eps(1.0)
        elseif j == mod1(i + 1, 3)
            @test m_sym[i, j] ≈ val_asym
        else
            @test m_sym[i, j] ≈ -val_asym
        end
    end
    @test symmetrize(m, symmetry, tr_odd=true, axial=false) ≈ zero(m) atol=eps(Float64)
    @test symmetrize(m, symmetry, tr_odd=true, axial=true) ≈ zero(m) atol=eps(Float64)

    # Array
    arr = rand(3, 2)
    @test symmetrize_array(arr, symmetry, order=0) ≈ symmetrize.(arr, Ref(symmetry))
    @test all(symmetrize_array(arr, symmetry, order=1)[:, 1] .≈ sum(arr[:, 1]) / 3)
    @test all(symmetrize_array(arr, symmetry, order=1)[:, 2] .≈ sum(arr[:, 2]) / 3)

    # Axial vector
    Ss = [Mat3{Int}(I(3)), Mat3{Int}(-I(3))]
    symmetry = Symmetry(Ss, zeros(Vec3{Float64}, 2), true, lattice)
    v = Vec3(rand(3))
    @test symmetrize(v, symmetry) ≈ zero(v)
    @test symmetrize(v, symmetry, axial=true) ≈ v
end

@testitem "symmetry subset" setup = [Common, Symmetry] begin
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    lattice_low = 2.0 * [[0 1 1.];
                         [1 0 1.];
                         [1 1 0.1]]
    atoms = ["Si" => [ones(3)/8, -ones(3)/8]]
    atoms_low = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetry_high = symmetry_operations(lattice, atoms)

    symmetry_low1 = symmetry_operations(lattice, atoms_low)
    symmetry_low2 = symmetry_operations(lattice_low, atoms)

    @test symmetry_is_subset(symmetry_high, symmetry_high) == true
    @test symmetry_is_subset(symmetry_low1, symmetry_high) == true
    @test symmetry_is_subset(symmetry_low2, symmetry_high) == true
    @test symmetry_is_subset(symmetry_high, symmetry_low1) == false
    @test symmetry_is_subset(symmetry_low1, symmetry_low2) == false
    @test symmetry_is_subset(symmetry_low2, symmetry_low1) == false
end
