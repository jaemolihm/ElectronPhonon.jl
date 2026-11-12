# A @testmodule, not a @testsnippet: evaluated once per process. `sp_holstein_model` (simple cubic, s + p_x,
# p_y, p_z Wannier orbitals on one atom, nearest-neighbour Slater-Koster hopping) with test-only variants, whose
# on-site symmetry operators are known analytically, D_S = blockdiag(1, Scart). It is the fixture of
# `fit_onsite_symmetry_operators` and of `electron_eigenpairs(...; symmetry)`.
@testmodule CubicSP begin
    using ElectronPhonon
    using ElectronPhonon: Vec3, WannierObject, sp_holstein_model, wannier_object_multiply_R
    using LinearAlgebra

    export cubic_sp_model, analytic_D

    const ALAT = 5.0
    const E_S, E_P = 0.0, 0.5               # on-site energies (Ry)
    const T_SS, V_PPSIGMA, V_PPPI, V_SP = -0.10, 0.12, -0.04, 0.07

    # Analytic on-site operator of a spatial operation: s is invariant, p transforms as a vector.
    analytic_D(Scart) = [1.0 zeros(1, 3); zeros(3, 1) Matrix(Scart)]

    """
        cubic_sp_model(; xQ = Vec3(0, 0, 0), perturb = false, decoupled = false) -> Model

    `sp_holstein_model` with the Hamiltonian modified as `H'(R) = exp(iπ xQ·R) H(R)`, i.e. `H(k + Q/2)`.
    `perturb` multiplies the s-s hopping along x by 1.3, which breaks the cubic symmetry.
    `decoupled` switches off the s-p hopping, which leaves s and p separately symmetric (a commutant
    of dimension 2).
    """
    function cubic_sp_model(; xQ = Vec3(0.0, 0.0, 0.0), perturb = false, decoupled = false)
        model = sp_holstein_model(; ε_s = E_S, ε_p = E_P, t_ss = T_SS, V_sp = decoupled ? 0.0 : V_SP,
            V_ppσ = V_PPSIGMA, V_ppπ = V_PPPI, ω₀ = 0.01, g_s = 0.0, alat = ALAT, verbose = false)
        irvec = model.el_ham.irvec
        ham = reshape(copy(model.el_ham.op_r), model.nw, model.nw, :)
        for (ir, R) in enumerate(irvec)
            perturb && !iszero(R) && findfirst(!iszero, R) == 1 && (ham[1, 1, ir] *= 1.3)
            ham[:, :, ir] .*= cispi(dot(xQ, R))
        end
        model.el_ham = WannierObject(irvec, reshape(ham, model.nw^2, :))
        model.el_ham_R = wannier_object_multiply_R(model.el_ham, model.structure.lattice)
        model.el_vel = WannierObject(irvec, copy(model.el_ham_R.op_r))
        model
    end
end
