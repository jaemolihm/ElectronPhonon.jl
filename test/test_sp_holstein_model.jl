@testitem "sp Holstein model" setup = [Common, CubicSP] begin
    using LinearAlgebra
    using Random
    using ElectronPhonon: sp_holstein_model

    ε_s, ε_p, t_ss, V_sp, V_ppσ, V_ppπ, ω₀, g_s, g_p, alat = 0.02, 0.5, -0.1, 0.07, 0.12, -0.04, 0.01, 0.03, -0.01, 5.0
    params = (; ε_s, ε_p, t_ss, V_sp, V_ppσ, V_ppπ, ω₀, g_s, g_p, alat, verbose = false)

    # H(k) of the docstring, with k in crystal coordinates.
    function hamiltonian_k(xk)
        c = cos.(2π .* xk)
        s = sin.(2π .* xk)
        H = zeros(ComplexF64, 4, 4)
        H[1, 1] = ε_s + 2t_ss * sum(c)
        for i in 1:3
            H[1 + i, 1 + i] = ε_p + 2V_ppπ * (sum(c) - c[i]) + 2V_ppσ * c[i]
            H[1, 1 + i] = 2im * V_sp * s[i]
            H[1 + i, 1] = -2im * V_sp * s[i]
        end
        H
    end
    vertex = Diagonal([g_s, g_p, g_p, g_p])

    Random.seed!(1234)
    xks = [Vec3(rand(3)...) for _ in 1:4]
    xqs = [Vec3(rand(3)...) for _ in 1:4]

    @testset "bands, phonon and g2, epmat_outer_momentum = $outer" for outer in ("el", "ph")
        model = sp_holstein_model(; params..., epmat_outer_momentum = outer)
        @test model.nw == 4
        @test model.nmodes == 1
        @test model.structure.dimension == 3
        @test model.el_sym === nothing

        # The interpolated eigenvectors diagonalize the analytic H(k), which fixes the sign
        # convention H(k) = ∑_R e^{ik·R} H(R) and not only the spectrum.
        el = compute_electron_states(model, Kpoints(xks), ["eigenvalue", "eigenvector"])
        for (ik, xk) in enumerate(xks)
            u = ElectronPhonon.no_offset_view(el[ik].u)
            @test u' * hamiltonian_k(xk) * u ≈ Diagonal(el[ik].e) atol=1e-12
        end

        ph = compute_phonon_states(model, Kpoints(xqs), ["eigenvalue", "eigenvector"])
        @test all(p.e[1] ≈ ω₀ for p in ph)

        # g2[m, n] = |⟨m, k+q| diag(g_s, g_p, g_p, g_p) |n, k⟩|², with the code's own eigenvectors.
        el_kq = compute_electron_states(model, Kpoints([xks[1] + xq for xq in xqs]), ["eigenvector"])
        epstate = EPState(model.nw, model.nmodes)
        epstate.el_k = el[1]
        epmat = ElectronPhonon.get_interpolator(model.epmat)
        obj = ElectronPhonon.get_next_wannier_object(model.epmat)
        itp = ElectronPhonon.get_interpolator(obj)
        u_k = ElectronPhonon.no_offset_view(epstate.el_k.u)
        if outer == "el"
            ElectronPhonon.compute_eph_RR_to_kR!(obj, epmat, xks[1], u_k)
        end
        for iq in eachindex(xqs)
            epstate.el_kq = el_kq[iq]
            epstate.ph = ph[iq]
            if outer == "el"
                ElectronPhonon.compute_eph_kR_to_kq!(epstate, itp, xqs[iq])
            else
                ElectronPhonon.compute_eph_RR_to_Rq!(obj, epmat, xqs[iq], epstate.ph.u)
                ElectronPhonon.compute_eph_Rq_to_kq!(epstate, itp, xks[1])
            end
            ElectronPhonon.epstate_set_g2!(epstate)
            u_kq = ElectronPhonon.no_offset_view(el_kq[iq].u)
            @test epstate.g2[:, :, 1] ≈ abs2.(u_kq' * vertex * u_k)
        end
    end

    @testset "symmetry: O_h × T, bands invariant" begin
        model = sp_holstein_model(; params...)
        @test model.symmetry.nsym == 96
        @test ElectronPhonon.check_group(model.symmetry)
        for xk in xks
            e0 = compute_electron_states(model, Kpoints(xk), ["eigenvalue"])[1].e
            for s in model.symmetry
                sk = Vec3(s.is_tr ? -s.S * xk : s.S * xk)
                @test compute_electron_states(model, Kpoints(sk), ["eigenvalue"])[1].e ≈ e0
            end
        end
    end

    @testset "fit_onsite_symmetry_operators == analytic D" begin
        model = sp_holstein_model(; params...)
        el_sym = fit_onsite_symmetry_operators(model)
        for (isym, s) in enumerate(model.symmetry)
            D = real(reshape(el_sym.operators[isym].op_r, 4, 4))
            @test min(norm(D - analytic_D(s.Scart)), norm(D + analytic_D(s.Scart))) < 1e-12
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError sp_holstein_model(; params..., ω₀ = 0.0)
        @test_throws ArgumentError sp_holstein_model(; params..., mass = -1.0)
        @test_throws ArgumentError sp_holstein_model(; params..., alat = 0.0)
        @test_throws ArgumentError sp_holstein_model(; params..., epmat_outer_momentum = "x")
    end
end
