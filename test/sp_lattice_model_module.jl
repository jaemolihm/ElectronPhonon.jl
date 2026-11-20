# A @testmodule, not a @testsnippet: evaluated once per process. An s + p_x, p_y, p_z tight-binding model on a
# HEXAGONAL lattice (non-orthogonal, so the crystal-coordinate matrix S of an operation differs from its
# real-space matrix S_R = S^(-T)), with optional extra orbital-free atoms and a Cartesian e-ph matrix built from
# geometry only (no symmetry law), so that it is covariant under the point group by construction. It is the
# fixture of the `fit_onsite_symmetry_operators` / `unfold_eigenpairs` tests that `cubic_sp_model` cannot reach:
# on a cubic cell S == S_R and one atom has pi(kappa) = kappa, L_kappa = 0. Built next to `sp_holstein_model`
# instead of extending it: that model is simple cubic by definition and has no per-atom epmat.
@testmodule SPLattice begin
    using ElectronPhonon
    using ElectronPhonon: Vec3, Mat3, WannierObject, Structure, Model, Polar, symmetry_operations,
        wannier_object_multiply_R
    using LinearAlgebra

    export sp_lattice_model, analytic_D_sp

    const ALAT = 5.0
    const E_S, E_P = 0.0, 0.5               # on-site energies (Ry)
    const T_SS, V_SP, V_PPSIGMA, V_PPPI = -0.10, 0.07, 0.12, -0.04
    const C_EPMAT = (0.030, 0.021, 0.013, 0.017, 0.011, 0.019, 0.023, 0.015, 0.009)

    analytic_D_sp(Scart) = [1.0 zeros(1, 3); zeros(3, 1) Matrix(Scart)]

    """
        sp_lattice_model(; atoms = [Vec3(0, 0, 0)]) -> Model

    Hexagonal lattice `a1 = (1, 0, 0) a`, `a2 = (-1/2, √3/2, 0) a`, `a3 = (0, 0, 1.3) a`, `a = 5` bohr. The first
    atom of `atoms` (Cartesian, units of `a`) carries the s, p orbitals (Wannier centres at it, it must be at the
    origin); every other atom carries none. Hopping is Slater-Koster with the amplitude `exp(-2(|R|/a - 1))`
    for every `R` with `|R| <= 1.4 a`. `nmodes = 3 nat` Cartesian displacements (Einstein phonon) and
    `g(Re, Rp)[m, n, κα]` of the e-ph matrix a function of `e = R_e` (centre of the ket relative to the bra) and
    of `d = (e/2 - P)/a`, `P` the position of atom `κ` in cell `Rp`, with the tensor structure allowed
    for a vector perturbation (`D_S = blockdiag(1, Scart)`):
        ss:  c1 d_α          sp_β: c2 e_β d_α + c3 d_β d_α + c4 δ_βα          ps_β: same with c5, c6, c7
        p_β p_γ: c8 d_α δ_βγ + c9 (d_β δ_γα + d_γ δ_βα)
    all times `exp(-(|d|/0.4)^2)`. The `Rp` list holds every vector with `|R| <= 3.2 a`, so what the lists
    leave out is below 1e-8 of the largest entry.
    """
    function sp_lattice_model(; atoms = [Vec3(0.0, 0.0, 0.0)])
        nw, nat = 4, length(atoms)
        nmodes, ω₀, mass = 3 * nat, 0.01, 1.0
        lattice = Mat3{Float64}(ALAT .* [1.0 -0.5 0.0; 0.0 sqrt(3) / 2 0.0; 0.0 0.0 1.3])
        structure = Structure(ALAT, lattice, fill(mass, nat), atoms, [i == 1 ? "A" : "B" for i in 1:nat])
        symmetry = symmetry_operations(structure)

        function vectors_within(radius, ranges)
            Rs = [Vec3(n1, n2, n3) for n1 in ranges[1], n2 in ranges[2], n3 in ranges[3]
                  if norm(lattice * Vec3(n1, n2, n3)) <= radius * ALAT]
            sort!(vec(Rs); by = reverse)
        end
        irvec = vectors_within(1.4, (-2:2, -2:2, -2:2))

        # Slater-Koster H(R), D_S H(R) D_S' = H(S_R R) for D_S = blockdiag(1, Scart)
        ham = zeros(ComplexF64, nw, nw, length(irvec))
        for (ir, R) in enumerate(irvec)
            if iszero(R)
                ham[1, 1, ir] = E_S
                foreach(j -> ham[1 + j, 1 + j, ir] = E_P, 1:3)
                continue
            end
            r = lattice * R
            amplitude = exp(-2 * (norm(r) / ALAT - 1))
            n = r / norm(r)
            ham[1, 1, ir] = T_SS * amplitude
            for α in 1:3
                ham[1, 1 + α, ir] = V_SP * n[α] * amplitude
                ham[1 + α, 1, ir] = -V_SP * n[α] * amplitude
                for β in 1:3
                    ham[1 + α, 1 + β, ir] = amplitude * (V_PPSIGMA * n[α] * n[β] + V_PPPI * ((α == β) - n[α] * n[β]))
                end
            end
        end
        el_ham = WannierObject(irvec, reshape(ham, nw^2, :))
        el_ham_R = wannier_object_multiply_R(el_ham, lattice)
        el_pos = WannierObject(irvec, zeros(ComplexF64, nw^2 * 3, length(irvec)))
        el_vel = WannierObject(irvec, copy(el_ham_R.op_r))

        # e-ph matrix, (m, n, mode, Rp, Re) for epmat_outer_momentum = "el"
        irvec_p = vectors_within(3.2, (-4:4, -4:4, -3:3))
        c = C_EPMAT
        g = zeros(ComplexF64, nw, nw, nmodes, length(irvec_p), length(irvec))
        for (ie, Re) in enumerate(irvec), (ip, Rp) in enumerate(irvec_p), κ in 1:nat
            e = lattice * Re / ALAT
            d = (e / 2 - atoms[κ] - lattice * Rp / ALAT)
            h = exp(-(norm(d) / 0.4)^2)
            for α in 1:3
                ν = 3 * (κ - 1) + α
                g[1, 1, ν, ip, ie] = h * c[1] * d[α]
                for β in 1:3
                    g[1, 1 + β, ν, ip, ie] = h * (c[2] * e[β] * d[α] + c[3] * d[β] * d[α] + c[4] * (α == β))
                    g[1 + β, 1, ν, ip, ie] = h * (c[5] * e[β] * d[α] + c[6] * d[β] * d[α] + c[7] * (α == β))
                    for γ in 1:3
                        g[1 + β, 1 + γ, ν, ip, ie] = h * (c[8] * d[α] * (β == γ) +
                            c[9] * (d[β] * (γ == α) + d[γ] * (β == α)))
                    end
                end
            end
        end
        epmat = WannierObject(irvec, reshape(g, nw^2 * nmodes * length(irvec_p), length(irvec)); irvec_next = irvec_p)

        ph_dyn = WannierObject([Vec3(0, 0, 0)], reshape(Matrix(mass * ω₀^2 * I(nmodes)) .+ 0im, nmodes^2, 1))
        Model(; structure, symmetry, nw, nmodes,
            wann_centers = fill(zero(Vec3{Float64}), nw),
            mass = fill(mass, nmodes),
            use_polar_dipole = false, polar_phonon = Polar(nothing), polar_eph = Polar(nothing),
            el_ham, el_ham_R, el_pos, el_vel, el_velocity_mode = :Direct,
            ph_dyn, ph_dyn_R = wannier_object_multiply_R(ph_dyn, lattice),
            epmat, epmat_outer_momentum = "el", el_sym = nothing)
    end
end
