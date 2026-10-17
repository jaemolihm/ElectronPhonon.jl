using Test
using LinearAlgebra
using Random
using ElectronPhonon
using ElectronPhonon: spin_orbit_holstein_model, holstein_model

@testset "spin-orbit Holstein model" begin
    t, M, λ_int, α_R, ω₀, g₀, g₁, alat, ε₀ = 0.1, 0.3, 0.05, 0.02, 0.01, 0.02, 0.005, 5.0, 0.04
    params = (; t, M, λ_int, α_R, ω₀, g₀, g₁, alat, ε₀, verbose = false)

    σ₀ = Matrix{ComplexF64}(I, 2, 2)
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]

    # H(k) of the docstring, with k in crystal coordinates.
    function hamiltonian_k(xk; t, M, λ_int, α_R, ε₀)
        skx, sky = sin(2π * xk[1]), sin(2π * xk[2])
        εk = ε₀ - 2t * (cos(2π * xk[1]) + cos(2π * xk[2]))
        (εk * kron(σ₀, σ₀) + M * kron(σz, σ₀) + λ_int * kron(σx, skx * σy - sky * σx)
         + α_R * kron(σ₀, sky * σx - skx * σy))
    end
    vertex = g₀ * kron(σ₀, σ₀) + g₁ * kron(σz, σ₀)

    Random.seed!(1234)
    xks = [Vec3(rand(2)..., 0.0) for _ in 1:5]
    xqs = [Vec3(rand(2)..., 0.0) for _ in 1:5]

    @testset "bands, phonon and g2, epmat_outer_momentum = $outer" for outer in ("el", "ph")
        model = spin_orbit_holstein_model(; params..., epmat_outer_momentum = outer)
        @test model.nw == 4
        @test model.nmodes == 1
        @test model.dimension == 2
        @test model.el_sym === nothing

        # The interpolated eigenvectors diagonalize the analytic H(k), which fixes the sign
        # convention H(k) = ∑_R e^{ik·R} H(R) and not only the spectrum.
        el = compute_electron_states(model, Kpoints(xks), ["eigenvalue", "eigenvector"])
        for (ik, xk) in enumerate(xks)
            u = ElectronPhonon.no_offset_view(el[ik].u)
            Hk = hamiltonian_k(xk; t, M, λ_int, α_R, ε₀)
            @test u' * Hk * u ≈ Diagonal(el[ik].e) atol=1e-12
        end

        ph = compute_phonon_states(model, Kpoints(xqs), ["eigenvalue", "eigenvector"])
        @test all(p.e[1] ≈ ω₀ for p in ph)

        # g2[m, n] = |⟨m, k+q| g₀ 𝟙 + g₁ τ_z |n, k⟩|², with the code's own eigenvectors.
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

    @testset "Kramers degeneracy iff α_R = 0" begin
        degenerate(e) = isapprox(e[1], e[2]; atol = 1e-12) && isapprox(e[3], e[4]; atol = 1e-12)
        for (α, expected) in ((0.0, true), (α_R, false))
            model = spin_orbit_holstein_model(; params..., α_R = α)
            el = compute_electron_states(model, Kpoints(xks), ["eigenvalue"])
            @test all(degenerate(el[ik].e) == expected for ik in eachindex(xks))
        end
    end

    @testset "symmetry: D4h × T for α_R = 0, C4v × T otherwise" begin
        for (α, nsym) in ((0.0, 32), (α_R, 16))
            model = spin_orbit_holstein_model(; params..., α_R = α)
            @test model.symmetry.nsym == nsym
            @test model.structure.symmetry.nsym == 32
            @test ElectronPhonon.check_group(model.symmetry)
            for xk in xks
                e0 = compute_electron_states(model, Kpoints(xk), ["eigenvalue"])[1].e
                for s in model.symmetry
                    sk = Vec3(s.is_tr ? -s.S * xk : s.S * xk)
                    @test compute_electron_states(model, Kpoints(sk), ["eigenvalue"])[1].e ≈ e0
                end
            end
        end

        # Energies cannot tell C4v × T from another 16-element group (T already maps k → -k),
        # so check the spin texture, an axial and T-odd vector:
        # ⟨σ⟩ₙ(Sk) = (-1)^{is_tr} det(R) R ⟨σ⟩ₙ(k). Bands are nondegenerate at generic k.
        model = spin_orbit_holstein_model(; params...)
        spin_ops = [kron(σ₀, σ) for σ in (σx, σy, σz)]
        function spin_texture(xk)
            el = compute_electron_states(model, Kpoints(xk), ["eigenvalue", "eigenvector"])[1]
            u = ElectronPhonon.no_offset_view(el.u)
            [Vec3(ntuple(i -> real(u[:, n]' * spin_ops[i] * u[:, n]), 3)) for n in 1:4]
        end
        for xk in xks
            s0 = spin_texture(xk)
            for s in model.symmetry
                sk = Vec3(s.is_tr ? -s.S * xk : s.S * xk)
                sign = (s.is_tr ? -1 : 1) * det(s.Scart)
                @test spin_texture(sk) ≈ [sign * s.Scart * v for v in s0] atol=1e-10
            end
        end
    end

    @testset "no SOC, no orbital splitting: four copies of the Holstein band" begin
        model = spin_orbit_holstein_model(; t, M = 0.0, λ_int = 0.0, α_R = 0.0, ω₀, g₀,
                                          alat, ε₀, verbose = false)
        holstein = holstein_model(; t, ω₀, g = g₀, alat, ε₀, dimension = 2, verbose = false)
        el = compute_electron_states(model, Kpoints(xks), ["eigenvalue"])
        el_ref = compute_electron_states(holstein, Kpoints(xks), ["eigenvalue"])
        @test all(el[ik].e ≈ fill(el_ref[ik].e[1], 4) for ik in eachindex(xks))
        @test model.epmat.op_r ≈ vec(holstein.epmat.op_r[1] * Matrix(I, 4, 4))
    end
end
