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

@testitem "fit_onsite_symmetry_operators epmat check" setup = [Common, Symmetry, CubicSP] begin
    using ElectronPhonon: Symmetry, Model
    without_epmat(m) = Model(; (k => getfield(m, k) for k in fieldnames(Model) if k != :epmat)...,
                             epmat = nothing)
    @testset "Cartesian modes, epmat_outer_momentum = $outer" for outer in ("el", "ph")
        model = cubic_sp_model(; cartesian_phonon = true, epmat_outer_momentum = outer)
        nw, nr = model.nw, length(model.epmat.irvec)
        @test model.nmodes == 3 && model.epmat_outer_momentum == outer
        # The law holds with Re != 0 and Rp != 0 entries
        @test fit_onsite_symmetry_operators(model) isa ElectronPhonon.SymmetryOperators

        # (Re, Rp) block of the epmat as (m, n, mode, Re, Rp), shared memory with model.epmat.op_r
        g_view(m) = outer == "el" ? permutedims(reshape(m.epmat.op_r, nw, nw, 3, nr, nr), (1, 2, 3, 5, 4)) :
                                    reshape(m.epmat.op_r, nw, nw, 3, nr, nr)
        irvec = model.epmat.irvec
        iRx, i0 = findfirst(==(Vec3(1, 0, 0)), irvec), findfirst(==(Vec3(0, 0, 0)), irvec)
        # Negative control: one entry g_ss,x(Re = a_x, Rp = 0) scaled by 1.3 breaks the law, not the H fit
        model_bad = cubic_sp_model(; cartesian_phonon = true, epmat_outer_momentum = outer)
        if outer == "el"
            reshape(model_bad.epmat.op_r, nw, nw, 3, nr, nr)[1, 1, 1, i0, iRx] *= 1.3
        else
            reshape(model_bad.epmat.op_r, nw, nw, 3, nr, nr)[1, 1, 1, iRx, i0] *= 1.3
        end
        @test g_view(model_bad)[1, 1, 1, iRx, i0] ≈ 1.3 * g_view(model)[1, 1, 1, iRx, i0]
        err = try fit_onsite_symmetry_operators(model_bad); nothing catch e e end
        @test err isa ErrorException && occursin("epmat", err.msg)
        # The H fit alone still passes on it, so it is the epmat check that fails
        @test fit_onsite_symmetry_operators(without_epmat(model_bad)) isa ElectronPhonon.SymmetryOperators

        # A second negative: x and y modes exchanged
        model_perm = cubic_sp_model(; cartesian_phonon = true, epmat_outer_momentum = outer)
        g5 = reshape(model_perm.epmat.op_r, nw, nw, 3, nr, nr)
        g5 .= g5[:, :, [2, 1, 3], :, :]
        @test_throws "epmat" fit_onsite_symmetry_operators(model_perm)

        # A zeroed block whose images stay nonzero fails too
        model_missing = cubic_sp_model(; cartesian_phonon = true, epmat_outer_momentum = outer)
        g5_missing = reshape(model_missing.epmat.op_r, nw, nw, 3, nr, nr)
        outer == "el" ? (g5_missing[:, :, :, i0, iRx] .= 0) : (g5_missing[:, :, :, iRx, i0] .= 0)
        @test_throws "epmat" fit_onsite_symmetry_operators(model_missing)
    end

    @testset "missing image is a zero block" begin
        # Drop Re = -a_x from a vector epmat: the sources at Re = +a_x stay nonzero while their image under
        # the operations that flip x is missing (a zero block), so the law fails; with all sources zero
        # (a zero epmat) nothing fails whatever images are missing
        model = cubic_sp_model(; cartesian_phonon = true)
        nr = length(model.epmat.irvec)
        iRmx = findfirst(==(Vec3(-1, 0, 0)), model.epmat.irvec)
        model_trunc = cubic_sp_model(; cartesian_phonon = true)
        keep = setdiff(1:nr, [iRmx])
        g5 = reshape(model_trunc.epmat.op_r, model.nw, model.nw, 3, nr, nr)
        epmat_trunc = ElectronPhonon.WannierObject(model.epmat.irvec[keep],
            reshape(g5[:, :, :, :, keep], model.nw^2 * 3 * nr, length(keep));
            irvec_next = model.epmat.irvec_next)
        model_trunc.epmat = epmat_trunc
        @test_throws "epmat" fit_onsite_symmetry_operators(model_trunc)
        # An all-zero epmat has nothing to violate, whatever images are missing
        epmat_zero = ElectronPhonon.WannierObject(model.epmat.irvec[keep],
            zeros(ComplexF64, model.nw^2 * 3 * nr, length(keep)); irvec_next = model.epmat.irvec_next)
        model_trunc.epmat = epmat_zero
        @test fit_onsite_symmetry_operators(model_trunc) isa ElectronPhonon.SymmetryOperators
    end

    @testset "invariant mode (nmodes != 3 nat)" begin
        model = cubic_sp_model(; g_s = 0.02)
        @test model.nmodes == 1 && maximum(abs, model.epmat.op_r) > 0
        @test fit_onsite_symmetry_operators(model) isa ElectronPhonon.SymmetryOperators
        # Negative control: one p-orbital coupling changed breaks the invariance
        model_bad = cubic_sp_model(; g_s = 0.02)
        reshape(model_bad.epmat.op_r, model.nw, model.nw)[2, 2] *= 1.5
        @test_throws "epmat" fit_onsite_symmetry_operators(model_bad)
    end

    @testset "no epmat in memory" begin
        @test_logs (:info, r"epmat not loaded") fit_onsite_symmetry_operators(without_epmat(cubic_sp_model()))
    end

    @testset "preconditions of the law" begin
        # Scart inconsistent with S_R (operation 2 given the rotation of operation 3)
        model = cubic_sp_model(; cartesian_phonon = true)
        sym = model.symmetry
        Scart_bad = copy(sym.Scart)
        Scart_bad[2] = Scart_bad[3]
        sym_bad = Symmetry(sym.nsym, sym.S, sym.τ, Scart_bad, sym.τcart, sym.is_inv, sym.is_tr)
        @test_throws "Scart" fit_onsite_symmetry_operators(model; symmetry = sym_bad)
        # An atom the operations do not map onto an atom
        model_off = cubic_sp_model(; cartesian_phonon = true)
        model_off.structure.atom_pos[1] = Vec3(0.3, 0.0, 0.0)
        @test_throws "off every atom" fit_onsite_symmetry_operators(model_off)
    end
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
