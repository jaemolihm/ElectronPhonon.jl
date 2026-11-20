@testitem "fit_onsite_symmetry_operators and pair unfold on a hexagonal s+p lattice" setup = [Common, SPLattice] begin
    using LinearAlgebra
    using ElectronPhonon: fold_kpoints, symmetry_small_group_of_q, get_interpolator, get_fourier!, Vec3, xk_to_ik
    model = sp_lattice_model()
    nw = model.nw
    symmetry = symmetry_small_group_of_q(model.symmetry, Vec3(0.0, 0.0, 0.0))[1]      # spatial operations
    @test symmetry.nsym == 24

    # The lattice is not orthogonal: S_R = S^(-T) differs from S for most operations, so a mix-up of S, S' and
    # S^(-T) in the fit or in the epmat check cannot pass
    S_R(S) = round.(Int, inv(S)')
    @test count(S -> S_R(S) != S, symmetry.S) >= 12

    # The fit, including the epmat check (hexagonal, nat = 1), reproduces the analytic D up to the sign character
    el_sym = fit_onsite_symmetry_operators(model; symmetry)
    D_of(isym) = real(reshape(el_sym.operators[isym].op_r, nw, nw))
    @test maximum(isym -> min(norm(D_of(isym) - analytic_D_sp(symmetry.Scart[isym])),
                              norm(D_of(isym) + analytic_D_sp(symmetry.Scart[isym]))), 1:symmetry.nsym) < 1e-12
    model.el_sym = el_sym

    # Pair-folded cache against the plain eigensolve
    kpts = GridKpoints(kpoints_grid((4, 4, 4)))
    kpts_irr, map = fold_kpoints(kpts, symmetry; pair_k_and_mk = true)
    @test kpts_irr.n < kpts.n
    eig = unfold_eigenpairs(electron_eigenpairs(model, kpts_irr; fourier_mode = "normal"), kpts, map, symmetry, model)
    eig_ref = electron_eigenpairs(model, kpts; fourier_mode = "normal")
    @test maximum(abs, eig.e_full - eig_ref.e_full) < 1e-12
    # u(Sk) = D_S u(k) with the analytic D (up to its sign), and every column an eigenvector of H(k)
    u_rep(ik) = eig.u_full[:, :, xk_to_ik(kpts_irr.vectors[first(map[ik])], kpts)]
    @test all(1:kpts.n) do ik
        D = analytic_D_sp(symmetry.Scart[last(map[ik])])
        isapprox(eig.u_full[:, :, ik], D * u_rep(ik); atol = 1e-12) || isapprox(eig.u_full[:, :, ik], -D * u_rep(ik); atol = 1e-12)
    end
    ham = get_interpolator(model.el_ham; fourier_mode = "normal")
    H = zeros(ComplexF64, nw, nw)
    resid = maximum(1:kpts.n) do ik
        get_fourier!(H, ham, kpts.vectors[ik])
        norm(H * eig.u_full[:, :, ik] - eig.u_full[:, :, ik] * Diagonal(eig.e_full[:, ik]))
    end
    @test resid < 1e-12
end

@testitem "fit_onsite_symmetry_operators epmat check with an atom permutation and L_kappa != 0" setup = [Common, SPLattice] begin
    using LinearAlgebra
    using ElectronPhonon: symmetry_small_group_of_q, Vec3
    # A at the origin carries the orbitals, the orbital-free B atoms sit at +u and at -u + a1 (inversion maps
    # one onto the other with a lattice vector L = -a1 left over), u = 0.3 a x
    atoms = [Vec3(0.0, 0.0, 0.0), Vec3(0.3, 0.0, 0.0), Vec3(0.7, 0.0, 0.0)]
    model = sp_lattice_model(; atoms)
    nw, nat = model.nw, 3
    symmetry = symmetry_small_group_of_q(model.symmetry, Vec3(0.0, 0.0, 0.0))[1]
    @test symmetry.nsym == 8 && model.nmodes == 9

    # The fixture reaches the atom-map branches: some operation swaps the B atoms, with L_kappa != 0
    frac = [model.lattice \ (model.alat * τ) for τ in model.atom_pos]
    atom_images(S) = let W = round.(Int, inv(S)')
        [findfirst(κ2 -> all(x -> isapprox(x, round(x); atol = 1e-8), W * frac[κ] - frac[κ2]), 1:nat) for κ in 1:nat]
    end
    swaps = count(S -> atom_images(S) != 1:nat, symmetry.S)
    shifted = count(symmetry.S) do S
        W, images = round.(Int, inv(S)'), atom_images(S)
        any(κ -> !(W * frac[κ] ≈ frac[images[κ]]), 1:nat)
    end
    @test swaps == 4 && shifted == 4

    el_sym = fit_onsite_symmetry_operators(model; symmetry)
    @test el_sym isa ElectronPhonon.SymmetryOperators

    # Negative controls: an entry of one B atom's mode scaled, and the x and y modes of one B atom exchanged
    g5(m) = reshape(m.epmat.op_r, nw, nw, 9, length(m.epmat.irvec_next), length(m.epmat.irvec))
    ip0 = findfirst(==(Vec3(0, 0, 0)), model.epmat.irvec_next)
    ie1 = findfirst(==(Vec3(1, 0, 0)), model.epmat.irvec)
    model_bad = sp_lattice_model(; atoms)
    g5(model_bad)[1, 1, 4, ip0, ie1] *= 1.3
    @test_throws "epmat" fit_onsite_symmetry_operators(model_bad; symmetry)
    model_xy = sp_lattice_model(; atoms)
    g5(model_xy) .= g5(model_xy)[:, :, [1:3; 5; 4; 6; 7:9], :, :]
    @test_throws "epmat" fit_onsite_symmetry_operators(model_xy; symmetry)
end
