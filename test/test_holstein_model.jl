using Test
using LinearAlgebra
using Random
using ElectronPhonon
using ElectronPhonon: holstein_model, Structure

@testset "Holstein model" begin
    t, ω₀, g, alat, ε₀ = 0.1, 0.01, 0.02, 5.0, 0.05

    @testset "dim = $dim, epmat_outer_momentum = $outer" for dim in 1:3, outer in ("el", "ph")
        model = holstein_model(; t, ω₀, g, alat, ε₀, dimension = dim, epmat_outer_momentum = outer, verbose = false)

        @test model.nw == 1
        @test model.nmodes == 1
        @test model.dimension == dim
        @test length(model.mass) == model.nmodes
        @test model.volume ≈ 100^(3 - dim) * alat^3
        @test !model.use_polar_dipole
        @test model.el_sym === nothing

        Random.seed!(1234)
        xks = [Vec3(rand(3)) for _ in 1:5]
        xqs = [Vec3(rand(3)) for _ in 1:5]

        # ε_k = ε₀ - 2t ∑_{i=1}^{dim} cos(k · a_i), v_k = 2 t a sin(k · a_i) ê_i
        el = compute_electron_states(model, Kpoints(xks), ["eigenvalue", "eigenvector", "velocity"])
        for (ik, xk) in enumerate(xks)
            @test el[ik].e[1] ≈ ε₀ - 2t * sum(cos(2π * xk[i]) for i in 1:dim)
            v_ref = Vec3(ntuple(i -> i <= dim ? 2t * alat * sin(2π * xk[i]) : 0.0, 3))
            @test real.(el[ik].v[1, 1]) ≈ v_ref atol=1e-12
        end

        # Dispersionless phonon
        ph = compute_phonon_states(model, Kpoints(xqs), ["eigenvalue", "eigenvector"])
        @test all(p.e[1] ≈ ω₀ for p in ph)

        # g2 = |g|² for every (k, q)
        el_kq = compute_electron_states(model, Kpoints([xks[1] + xq for xq in xqs]), ["eigenvector"])
        epstate = EPState(model.nw, model.nmodes)
        epstate.el_k = el[1]
        epmat = ElectronPhonon.get_interpolator(model.epmat)
        obj = ElectronPhonon.get_next_wannier_object(model.epmat)
        itp = ElectronPhonon.get_interpolator(obj)
        if outer == "el"
            ElectronPhonon.get_eph_RR_to_kR!(obj, epmat, xks[1], ElectronPhonon.no_offset_view(epstate.el_k.u))
        end
        for iq in eachindex(xqs)
            epstate.el_kq = el_kq[iq]
            epstate.ph = ph[iq]
            if outer == "el"
                ElectronPhonon.get_eph_kR_to_kq!(epstate, itp, xqs[iq])
            else
                ElectronPhonon.get_eph_RR_to_Rq!(obj, epmat, xqs[iq], epstate.ph.u)
                ElectronPhonon.get_eph_Rq_to_kq!(epstate, itp, xks[1])
            end
            ElectronPhonon.epstate_set_g2!(epstate)
            @test epstate.g2[1, 1, 1] ≈ g^2
        end
    end

    @testset "symmetry never mixes hopping and inactive directions" begin
        # model.symmetry is spglib's group for the elongated cell, restricted to the
        # operations block-diagonal between 1:dimension and the rest. It is a subgroup of
        # what spglib returned, still a group, and free of block-mixing operations.
        for dim in 1:3
            model = holstein_model(; t, ω₀, g, alat, ε₀, dimension = dim, verbose = false)
            for symop in model.symmetry
                for i in 1:3, j in 1:3
                    if (i <= dim) != (j <= dim)
                        @test symop.S[i, j] == 0
                    end
                end
            end
            @test model.symmetry === model.structure.symmetry  # Structure applies it itself
            Ss = Set(model.symmetry.S)
            @test all(any(Sa * Sb == Sc for Sc in Ss) for Sa in Ss, Sb in Ss)  # closed
        end

        # On the elongated cell the restriction is a no-op — spglib already returns exactly
        # this group. Exercise it where it does bite, a cubic cell, so the check above cannot
        # pass just because the filter does nothing.
        cube = Mat3{Float64}(I(3))
        cubic = Structure(1.0, cube, [1.0], [zero(Vec3{Float64})], ["A"])
        @test cubic.symmetry.nsym == 96
        for (dim, nsym) in ((1, 32), (2, 32), (3, 96))
            @test restrict_symmetry_to_dimension(cubic.symmetry, dim, cube).nsym == nsym
            # Same thing through the Structure keyword.
            @test Structure(1.0, cube, [1.0], [zero(Vec3{Float64})], ["A"];
                            dimension = dim).symmetry.nsym == nsym
        end

        # The block-diagonal test is read off S in crystal coordinates, so it is only
        # meaningful if the lattice does not mix the blocks itself.
        skewed = Mat3{Float64}([1 0 0.3; 0 1 0; 0 0 10])
        @test_throws ArgumentError restrict_symmetry_to_dimension(cubic.symmetry, 2, skewed)
        @test_throws ArgumentError restrict_symmetry_to_dimension(cubic.symmetry, 0, cube)
    end

    @testset "symmetry is consistent with the dim-dimensional band" begin
        # A cubic cell would make spglib report operations mixing x and z, which the 1d and 2d
        # bands do not obey. Check every reported operation against the interpolated energies.
        for dim in 1:3
            model = holstein_model(; t, ω₀, g, alat, ε₀, dimension = dim, verbose = false)
            @test model.symmetry.nsym == (dim == 3 ? 96 : 32)
            Random.seed!(7)
            for _ in 1:5
                xk = Vec3(rand(3))
                e0 = compute_electron_states(model, Kpoints(xk), ["eigenvalue"])[1].e[1]
                for s in model.symmetry
                    sk = Vec3(s.is_tr ? -s.S * xk : s.S * xk)
                    e1 = compute_electron_states(model, Kpoints(sk), ["eigenvalue"])[1].e[1]
                    @test e1 ≈ e0
                end
            end
        end
    end

    @testset "velocity modes agree" begin
        # Position matrix elements vanish, so :Direct (the default) and :BerryConnection
        # must coincide.
        model = holstein_model(; t, ω₀, g, alat, ε₀, dimension = 3, verbose = false)
        @test model.el_velocity_mode === :Direct
        kpts = Kpoints([Vec3(0.1, 0.2, 0.3), Vec3(0.4, -0.15, 0.05)])
        v_direct = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"])
        model.el_velocity_mode = :BerryConnection
        v_berry = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"])
        @test all(v_direct[i].v[1, 1] == v_berry[i].v[1, 1] for i in 1:kpts.n)
    end

    @testset "through the e-ph driver, via BoltzmannCalculator" begin
        # Run the production calculator on the model and check two properties of the
        # scattering-out rate that are exact for Holstein and free of any prefactor
        # convention. Both follow from |g|² and ω being constant:
        #   Γ_k = 2π|g|² ∑_± (occupation) · D(ε_k ± ω₀)
        # so Γ depends on k only through ε_k, and scales as |g|².
        K = ElectronPhonon.unit_to_aru(:K)
        μ = 0.0  # fixed, so the occupation factors do not move when g changes

        function run_bte(g_value)
            model = holstein_model(; t, ω₀, g = g_value, alat, dimension = 2, verbose = false)
            occ = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 1.0, μlist = μ,
                model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac)
            calc = BoltzmannCalculator{Float64}(; occ,
                smearing_list = [SmearingType(:Gaussian, 2 * ω₀)], occupation_method = 5)
            res = ElectronPhonon.run_eph_over_k_and_kq(model, (6, 6, 1), (6, 6, 1);
                calculators = [calc], model.symmetry, window_k = (-Inf, Inf),
                window_kq = (-Inf, Inf), progress_print_step = 10^9, verbosity = 0)
            (calc, res)
        end

        calc, res = run_bte(g)

        # The phonon states come straight back from the driver — no calculator needed.
        @test all(p.e[1] ≈ ω₀ for p in res.ph_save)

        Sₒ = calc.Sₒ[1]
        e_i = calc.el_i.es
        @test length(Sₒ) == length(e_i) == calc.el_i.n
        @test all(>(0), Sₒ)

        # Γ is a function of ε alone. On the 6×6 grid the outer (IBZ) states include an
        # accidental degeneracy between the symmetry-inequivalent points (1/2, 0) and
        # (1/3, 1/6), both at ε = ε₀, so this is not implied by symmetry.
        ndegenerate = 0
        for i in eachindex(e_i), j in (i + 1):length(e_i)
            if isapprox(e_i[i], e_i[j]; atol = 1e-12)
                ndegenerate += 1
                @test Sₒ[i] ≈ Sₒ[j] rtol=1e-10
            end
        end
        @test ndegenerate >= 1       # a degenerate pair exists, so the check above ran
        @test !all(≈(Sₒ[1]), Sₒ)     # and Sₒ is not trivially constant across all states

        # Γ ∝ |g|², exactly: g2 = |g|² is the only g-dependent factor in the scatter.
        calc2, _ = run_bte(2 * g)
        @test calc2.el_i.es ≈ e_i
        @test calc2.Sₒ[1] ≈ 4 .* Sₒ rtol=1e-12
    end

    @testset "g and λ are alternative spellings of the same coupling" begin
        # λ = g² / (ω₀ · W/2) with half-bandwidth W/2 = 2·dimension·t.
        for dim in 1:3
            λ = 0.35
            from_λ = holstein_model(; t, ω₀, λ, alat, dimension = dim, verbose = false)
            g_expected = sqrt(2 * dim * λ * ω₀ * t)
            from_g = holstein_model(; t, ω₀, g = g_expected, alat, dimension = dim, verbose = false)
            @test from_λ.epmat.op_r ≈ from_g.epmat.op_r
            @test from_λ.epmat.op_r[1, findfirst(iszero, from_λ.epmat.irvec)] ≈
                g_expected * sqrt(2 * ω₀ * 1.0)

            # The half-bandwidth is 2·dimension·|t|, so the sign of t does not move λ.
            from_neg_t = holstein_model(; t = -t, ω₀, λ, alat, dimension = dim, verbose = false)
            @test from_neg_t.epmat.op_r ≈ from_λ.epmat.op_r
        end
    end

    @testset "mass cancels out" begin
        # The model is fixed by ω₀ and g alone; `mass` must not change any observable.
        ms = [holstein_model(; t, ω₀, g, alat, ε₀, dimension = 3, mass = m, verbose = false)
              for m in (1.0, 911.444)]
        xk, xq = Vec3(0.1, 0.2, 0.3), Vec3(0.3, -0.1, 0.25)
        for m in ms
            @test compute_phonon_states(m, Kpoints(xq), ["eigenvalue"])[1].e[1] ≈ ω₀
        end
        g2s = map(ms) do model
            el_k = compute_electron_states(model, Kpoints(xk), ["eigenvector"])[1]
            el_kq = compute_electron_states(model, Kpoints(xk + xq), ["eigenvector"])[1]
            ph = compute_phonon_states(model, Kpoints(xq), ["eigenvalue", "eigenvector"])[1]
            epstate = EPState(model.nw, model.nmodes)
            epstate.el_k, epstate.el_kq, epstate.ph = el_k, el_kq, ph
            obj = ElectronPhonon.get_next_wannier_object(model.epmat)
            ElectronPhonon.get_eph_RR_to_kR!(obj, ElectronPhonon.get_interpolator(model.epmat),
                xk, ElectronPhonon.no_offset_view(el_k.u))
            ElectronPhonon.get_eph_kR_to_kq!(epstate, ElectronPhonon.get_interpolator(obj), xq)
            ElectronPhonon.epstate_set_g2!(epstate)
            epstate.g2[1, 1, 1]
        end
        @test all(≈(g^2), g2s)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError holstein_model(; t, ω₀, g, dimension = 0, verbose = false)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, dimension = 4, verbose = false)
        @test_throws ArgumentError holstein_model(; t, ω₀ = -1.0, g, verbose = false)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, alat = 0.0, verbose = false)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, mass = 0.0, verbose = false)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, epmat_outer_momentum = "kq", verbose = false)
        # Exactly one of g and λ.
        @test_throws ArgumentError holstein_model(; t, ω₀, verbose = false)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, λ = 0.5, verbose = false)
        # Deriving g from λ needs a nonzero bandwidth.
        @test_throws ArgumentError holstein_model(; t = 0.0, ω₀, λ = 0.5, verbose = false)
    end
end
