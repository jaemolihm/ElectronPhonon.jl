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
    of dimension 2). `g_s` is the on-site orbital-density coupling of the single (A1) phonon mode.

    `cartesian_phonon` replaces that mode by the three Cartesian displacements of the atom at the origin
    (`nmodes = 3`, Einstein phonon) and the epmat by one that is a vector under O_h, with `Re != 0` and
    `Rp != 0` entries (`R = ±a_i`, `R_α` the Cartesian component of the lattice vector, `g` in Ry):
        g[s,s,α](Re = R, Rp = 0)   =  LAMBDA R_α,    g[s,s,α](Re = R, Rp = R) = -LAMBDA R_α,
        g[s,p_β,α](0, 0) = g[p_β,s,α](0, 0) = C_SP δ_αβ    (T1u).
    """
    function cubic_sp_model(; xQ = Vec3(0.0, 0.0, 0.0), perturb = false, decoupled = false, g_s = 0.0,
                              cartesian_phonon = false, epmat_outer_momentum = "el")
        model = sp_holstein_model(; ε_s = E_S, ε_p = E_P, t_ss = T_SS, V_sp = decoupled ? 0.0 : V_SP,
            V_ppσ = V_PPSIGMA, V_ppπ = V_PPPI, ω₀ = 0.01, g_s, alat = ALAT, epmat_outer_momentum,
            verbose = false)
        irvec = model.el_ham.irvec
        ham = reshape(copy(model.el_ham.op_r), model.nw, model.nw, :)
        for (ir, R) in enumerate(irvec)
            perturb && !iszero(R) && findfirst(!iszero, R) == 1 && (ham[1, 1, ir] *= 1.3)
            ham[:, :, ir] .*= cispi(dot(xQ, R))
        end
        model.el_ham = WannierObject(irvec, reshape(ham, model.nw^2, :))
        model.el_ham_R = wannier_object_multiply_R(model.el_ham, model.structure.lattice)
        model.el_vel = WannierObject(irvec, copy(model.el_ham_R.op_r))
        cartesian_phonon && cartesian_phonon_epmat!(model, irvec)
        model
    end

    const LAMBDA, C_SP = 0.03, 0.05

    # Three Cartesian displacement modes (Einstein phonon) and the vector epmat of `cubic_sp_model`
    function cartesian_phonon_epmat!(model, irvec)
        nw, nmodes, nr = model.nw, 3, length(irvec)
        mass, ω₀ = 1.0, 0.01
        g = zeros(ComplexF64, nw, nw, nmodes, nr, nr)       # (m, n, α, iRe, iRp)
        index = Dict(R => i for (i, R) in enumerate(irvec))
        i0 = index[Vec3(0, 0, 0)]
        for R in irvec
            iszero(R) && continue
            for α in 1:3
                g[1, 1, α, index[R], i0] = LAMBDA * R[α]
                g[1, 1, α, index[R], index[R]] = -LAMBDA * R[α]
            end
        end
        for α in 1:3
            g[1, 1 + α, α, i0, i0] = g[1 + α, 1, α, i0, i0] = C_SP
        end
        irvec_p = irvec
        model.epmat = if model.epmat_outer_momentum == "el"
            WannierObject(irvec, reshape(permutedims(g, (1, 2, 3, 5, 4)), nw^2 * nmodes * nr, nr); irvec_next = irvec_p)
        else
            WannierObject(irvec_p, reshape(g, nw^2 * nmodes * nr, nr); irvec_next = irvec)
        end
        model.nmodes = nmodes
        model.mass = fill(mass, nmodes)
        model.ph_dyn = WannierObject([Vec3(0, 0, 0)], reshape(Matrix(mass * ω₀^2 * I(nmodes)) .+ 0im, nmodes^2, 1))
        model.ph_dyn_R = wannier_object_multiply_R(model.ph_dyn, model.structure.lattice)
        model
    end
end
