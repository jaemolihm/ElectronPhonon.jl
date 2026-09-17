using Test
using LinearAlgebra
using Random
using ElectronPhonon
using ElectronPhonon: holstein_model, AbstractCalculator, OuterKLoop, EPData, OuterIteration

# A minimal calculator recording the range of g2 and ω seen by the driver. The driver calls
# `run_calculator!` from several threads, so the accumulation is guarded by a lock.
mutable struct _HolsteinG2Calc <: AbstractCalculator
    lock :: ReentrantLock
    g2 :: Vector{Float64}
    ω :: Vector{Float64}
    _HolsteinG2Calc() = new(ReentrantLock(), Float64[], Float64[])
end
ElectronPhonon.supports(::_HolsteinG2Calc, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_HolsteinG2Calc, ::Type{EPData}) = true
ElectronPhonon.setup_calculator!(c::_HolsteinG2Calc, backend, mode, kpts, qpts, el_states; kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_HolsteinG2Calc; kwargs...) = c
ElectronPhonon.calculator_begin!(::_HolsteinG2Calc, ::OuterIteration, ctx) = nothing
ElectronPhonon.calculator_end!(::_HolsteinG2Calc, ::OuterIteration, ctx) = nothing
function ElectronPhonon.run_calculator!(c::_HolsteinG2Calc, data::EPData, ctx)
    @lock c.lock begin
        append!(c.g2, vec(parent(data.epstate.g2)))
        append!(c.ω, data.epstate.ph.e)
    end
    c
end

@testset "Holstein model" begin
    t, ω₀, g, alat, ε₀ = 0.1, 0.01, 0.02, 5.0, 0.05

    @testset "dim = $dim, epmat_outer_momentum = $outer" for dim in 1:3, outer in ("el", "ph")
        model = holstein_model(; t, ω₀, g, alat, ε₀, dim, epmat_outer_momentum = outer)

        @test model.nw == 1
        @test model.nmodes == 1
        @test model.dim == dim
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

    @testset "symmetry is consistent with the dim-dimensional band" begin
        # A cubic cell would make spglib report operations mixing x and z, which the 1d and 2d
        # bands do not obey. Check every reported operation against the interpolated energies.
        for dim in 1:3
            model = holstein_model(; t, ω₀, g, alat, ε₀, dim)
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
        # Position matrix elements vanish, so :Direct and :BerryConnection must coincide.
        model = holstein_model(; t, ω₀, g, alat, ε₀, dim = 3)
        kpts = Kpoints([Vec3(0.1, 0.2, 0.3), Vec3(0.4, -0.15, 0.05)])
        v_berry = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"])
        model.el_velocity_mode = :Direct
        v_direct = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"])
        @test all(v_direct[i].v[1, 1] == v_berry[i].v[1, 1] for i in 1:kpts.n)
    end

    @testset "through the e-ph driver with symmetry" begin
        model = holstein_model(; t, ω₀, g, alat, dim = 2)
        calc = _HolsteinG2Calc()
        ElectronPhonon.run_eph_over_k_and_kq(model, (6, 6, 1), (6, 6, 1); calculators = [calc],
            model.symmetry, progress_print_step = 10^9,
            window_k = (-Inf, Inf), window_kq = (-Inf, Inf))
        @test !isempty(calc.g2)
        @test all(≈(g^2), calc.g2)
        @test all(≈(ω₀), calc.ω)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError holstein_model(; t, ω₀, g, dim = 0)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, dim = 4)
        @test_throws ArgumentError holstein_model(; t, ω₀ = -1.0, g)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, alat = 0.0)
        @test_throws ArgumentError holstein_model(; t, ω₀, g, epmat_outer_momentum = "kq")
    end
end
