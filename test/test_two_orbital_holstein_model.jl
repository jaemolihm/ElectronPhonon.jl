@testsnippet TwoOrbitalHolsteinModel begin
    using LinearAlgebra
    using Random
    using ElectronPhonon: two_orbital_holstein_model

    τx = ComplexF64[0 1; 1 0]
    τz = ComplexF64[1 0; 0 -1]
    γ(xk) = 2 * (cos(2π * xk[1]) - cos(2π * xk[2]))

    # Closed forms of the docstring, k and q in crystal coordinates.
    hamiltonian_k(xk; t, t_x, M, ε₀) =
        (ε₀ - 2t * (cos(2π * xk[1]) + cos(2π * xk[2]))) * I(2) + M * τz + t_x * γ(xk) * τx
    vertex_kq(xk, xq; g₀, g_z, g_x) =
        g₀ * I(2) + g_z * τz + g_x * ComplexF64[0 γ(xk); γ(xk + xq) 0]
    ω_q(xq; ω₀, ω₁) = sqrt(ω₀^2 + ω₁^2 * (2 - cos(2π * xq[1]) - cos(2π * xq[2])))

    # Wannier-gauge ep(k, q) for the phonon mode, through the identity electron gauge.
    function wannier_gauge_ep(model, xk, xq)
        ph = compute_phonon_states(model, Kpoints([xq]), ["eigenvalue", "eigenvector"])[1]
        epmat = ElectronPhonon.get_interpolator(model.epmat)
        obj = ElectronPhonon.get_next_wannier_object(model.epmat)
        itp = ElectronPhonon.get_interpolator(obj)
        ep = zeros(ComplexF64, model.nw, model.nw, model.nmodes)
        uI = Matrix{ComplexF64}(I, model.nw, model.nw)
        if model.epmat_outer_momentum == "el"
            ElectronPhonon.compute_eph_RR_to_kR!(obj, epmat, xk, uI)
            ElectronPhonon.compute_eph_kR_to_kq!(ep, itp, xq, ph.u, uI)
        else
            ElectronPhonon.compute_eph_RR_to_Rq!(obj, epmat, xq, ph.u)
            ElectronPhonon.compute_eph_Rq_to_kq!(ep, itp, xk, uI, uI)
        end
        ep[:, :, 1], ph
    end
end

@testitem "Two-orbital Holstein model: H(k), phonon and ep_W" setup = [TwoOrbitalHolsteinModel] begin
    t, t_x, M, ε₀, ω₀, ω₁, g₀, g_z, g_x, alat, mass = 0.1, 0.03, 0.04, 0.02, 0.01, 0.004, 0.02, 0.006, 0.009, 5.0, 2.0
    ham_params = (; t, t_x, M, ε₀)
    coupling = (; g₀, g_z, g_x)

    Random.seed!(1234)
    xks = [Vec3(rand(2)..., 0.0) for _ in 1:6]
    xqs = [Vec3(rand(2)..., 0.0) for _ in 1:6]

    @testset "epmat_outer_momentum = $outer" for outer in ("el", "ph")
        model = two_orbital_holstein_model(; t, t_x, M, ε₀, ω₀, ω₁, g₀, g_z, g_x, alat, mass,
            epmat_outer_momentum = outer, verbose = false)
        @test model.nw == 2
        @test model.nmodes == 1
        @test model.structure.dimension == 2
        @test !model.use_polar_dipole

        # The interpolated eigenvectors diagonalize the analytic H(k): this fixes the
        # H(k) = ∑_R e^{ik·R} H(R) convention and not only the spectrum.
        el = compute_electron_states(model, Kpoints(xks), ["eigenvalue", "eigenvector"])
        for (ik, xk) in enumerate(xks)
            u = ElectronPhonon.no_offset_view(el[ik].u)
            @test u' * hamiltonian_k(xk; ham_params...) * u ≈ Diagonal(el[ik].e) atol=1e-12
        end

        # ep_W(k, q) = ĝ(k, q) √(2 ω₀) u_ph (Cartesian-displacement couplings, scale √(2 ω₀ m)).
        for xk in xks, xq in xqs
            ep, ph = wannier_gauge_ep(model, xk, xq)
            @test ph.e[1] ≈ ω_q(xq; ω₀, ω₁)
            @test abs(ph.u[1, 1]) ≈ 1 / sqrt(mass)
            @test ep ≈ sqrt(2 * ω₀) * (ph.u[1, 1] * sqrt(mass)) * vertex_kq(xk, xq; coupling...) atol=1e-13
        end
    end

    @testset "Hermiticity ĝ(k+q, -q)† = ĝ(k, q) of the closed form and the model" begin
        model = two_orbital_holstein_model(; t, t_x, M, ε₀, ω₀, g₀, g_z, g_x, verbose = false)
        for xk in xks, xq in xqs
            ep, _ = wannier_gauge_ep(model, xk, xq)
            ep_back, _ = wannier_gauge_ep(model, xk + xq, -xq)
            @test ep_back' ≈ ep atol=1e-13
        end
    end
end

@testitem "Two-orbital Holstein model: el_sym" setup = [TwoOrbitalHolsteinModel] begin
    t, t_x, M, ε₀, ω₀, g₀, g_x = 0.1, 0.03, 0.04, 0.02, 0.01, 0.02, 0.009
    model = two_orbital_holstein_model(; t, t_x, M, ε₀, ω₀, g₀, g_x, verbose = false)
    el_sym = model.el_sym
    @test el_sym.symmetry.nsym == 16
    @test !any(el_sym.symmetry.is_tr)
    @test all(op ∈ model.symmetry for op in el_sym.symmetry)

    Random.seed!(99)
    xks = [Vec3(rand(2)..., 0.0) for _ in 1:5]
    n_exchange = 0  # operations that exchange x and y
    for (isym, op) in enumerate(el_sym.symmetry)
        itp = ElectronPhonon.get_interpolator(el_sym.operators[isym])
        D = zeros(ComplexF64, 2, 2)
        global n_exchange += op.S[1, 2] != 0
        for xk in xks
            ElectronPhonon.get_symmetry_representation_wannier!(D, itp, xk, op.is_tr)
            @test D * D' ≈ I(2) atol=1e-14
            @test D * hamiltonian_k(xk; t, t_x, M, ε₀) * D' ≈
                hamiltonian_k(op.S * xk; t, t_x, M, ε₀) atol=1e-13
        end
    end
    # Half of D4h exchanges x and y: C4, C4³, 2 σ_d, 2 C2'' and their products with inversion.
    @test n_exchange == 8
end

@testitem "Two-orbital Holstein model: t_x = g_x = g_z = 0 reduction" setup = [TwoOrbitalHolsteinModel] begin
    t, M, ε₀, ω₀, g₀ = 0.1, 0.04, 0.02, 0.01, 0.02
    model = two_orbital_holstein_model(; t, t_x = 0.0, M, ε₀, ω₀, g₀, verbose = false)

    Random.seed!(7)
    xks = [Vec3(rand(2)..., 0.0) for _ in 1:5]
    xqs = [Vec3(rand(2)..., 0.0) for _ in 1:5]

    # Bands are the orbitals: ε(k) ± M with eigenvectors the unit vectors.
    el = compute_electron_states(model, Kpoints(xks), ["eigenvalue", "eigenvector"])
    for (ik, xk) in enumerate(xks)
        εk = ε₀ - 2t * (cos(2π * xk[1]) + cos(2π * xk[2]))
        @test el[ik].e ≈ [εk - M, εk + M] atol=1e-13
        @test abs.(ElectronPhonon.no_offset_view(el[ik].u)) ≈ [0 1; 1 0] atol=1e-13
    end

    # The vertex is g₀ √(2 ω₀) 𝟙 for every (k, q): diagonal and k, q independent.
    for xk in xks, xq in xqs
        ep, ph = wannier_gauge_ep(model, xk, xq)
        @test ep ≈ g₀ * sqrt(2 * ω₀) * (ph.u[1, 1] * sqrt(model.mass[1])) * I(2) atol=1e-13
    end
end
