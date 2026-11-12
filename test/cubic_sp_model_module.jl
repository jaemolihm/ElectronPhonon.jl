# A @testmodule, not a @testsnippet: evaluated once per process. A synthetic simple-cubic model with
# one atom and s + p_x, p_y, p_z Wannier orbitals on it (nearest-neighbour Slater-Koster hopping), whose
# on-site symmetry operators are known analytically, D_S = blockdiag(1, Scart). It is the fixture of
# `fit_onsite_symmetry_operators` and of `electron_eigenpairs(...; symmetry)`.
@testmodule CubicSP begin
    using ElectronPhonon
    using ElectronPhonon: Vec3, Mat3, Structure, Polar, WannierObject, symmetry_operations,
        wannier_object_multiply_R
    using LinearAlgebra

    export cubic_sp_model, analytic_D

    const ALAT = 5.0
    const E_S, E_P = 0.0, 0.5               # on-site energies (Ry)
    const T_SS, V_PPSIGMA, V_PPPI, V_SP = -0.10, 0.12, -0.04, 0.07

    # Analytic on-site operator of a spatial operation: s is invariant, p transforms as a vector.
    analytic_D(Scart) = [1.0 zeros(1, 3); zeros(3, 1) Matrix(Scart)]

    """
        cubic_sp_model(; xQ = Vec3(0, 0, 0), perturb = false) -> Model

    `H'(R) = exp(iπ xQ·R) H(R)`, i.e. the Hamiltonian `H(k + Q/2)`, with `el_sym = nothing`.
    `perturb` multiplies the s-s hopping along x by 1.3, which breaks the cubic symmetry.
    """
    function cubic_sp_model(; xQ = Vec3(0.0, 0.0, 0.0), perturb = false)
        nw, nmodes = 4, 1
        lattice = Mat3{Float64}(Diagonal(fill(ALAT, 3)))
        structure = Structure(ALAT, lattice, [1.0], [zeros(Vec3{Float64})], ["A"])

        irvec = [Vec3(0, 0, 0)]
        for i in 1:3, s in (-1, 1)
            push!(irvec, Vec3(ntuple(j -> j == i ? s : 0, 3)))
        end
        sort!(irvec, by = reverse)
        ham = zeros(ComplexF64, nw, nw, length(irvec))
        for (ir, R) in enumerate(irvec)
            if iszero(R)
                ham[1, 1, ir], ham[2, 2, ir], ham[3, 3, ir], ham[4, 4, ir] = E_S, E_P, E_P, E_P
                continue
            end
            i = findfirst(!iszero, R)
            ham[1, 1, ir] = T_SS * (perturb && i == 1 ? 1.3 : 1.0)
            for j in 1:3
                ham[1 + j, 1 + j, ir] = j == i ? V_PPSIGMA : V_PPPI
            end
            # <s,0|H|p_i,R> = R_i V_sp and <p_i,0|H|s,R> = -R_i V_sp
            ham[1, 1 + i, ir] = R[i] * V_SP
            ham[1 + i, 1, ir] = -R[i] * V_SP
        end
        for (ir, R) in enumerate(irvec)
            ham[:, :, ir] .*= cispi(dot(xQ, R))
        end

        el_ham = WannierObject(irvec, reshape(ham, nw^2, :))
        el_ham_R = wannier_object_multiply_R(el_ham, lattice)
        el_pos = WannierObject(irvec, zeros(ComplexF64, nw^2 * 3, length(irvec)))
        el_vel = WannierObject(irvec, copy(el_ham_R.op_r))
        irvec_ph = [Vec3(0, 0, 0)]
        ph_dyn = WannierObject(irvec_ph, fill(ComplexF64(0.01), nmodes^2, 1))
        ph_dyn_R = wannier_object_multiply_R(ph_dyn, lattice)
        ElectronPhonon.Model(; structure, symmetry = symmetry_operations(structure), nw, nmodes,
            wann_centers = [zeros(Vec3{Float64})], mass = [1.0],
            use_polar_dipole = false, polar_phonon = Polar(nothing), polar_eph = Polar(nothing),
            el_ham, el_ham_R, el_pos, el_vel, el_velocity_mode = :Direct,
            ph_dyn, ph_dyn_R, epmat = nothing, epmat_outer_momentum = "el", el_sym = nothing)
    end
end
