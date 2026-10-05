using Test
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPBlock, supports,
    OuterKContext, OuterQContext, CPUBackend, calculator_begin_batch!, calculator_end_batch!, to_device

# Calculator-contract checks (CPU-only): the `supports` trait, the fail-early checks the drivers do
# at entry, `calculators` as a keyword, and the screening-disabled error.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

# A minimal well-formed outer-k calculator that just counts run_calculator! calls.
mutable struct _CountCalc <: AbstractCalculator
    n :: Threads.Atomic{Int}
    _CountCalc() = new(Threads.Atomic{Int}(0))
end
ElectronPhonon.supports(::_CountCalc, ::Type{OuterKLoop}) = true
ElectronPhonon.setup_calculator!(c::_CountCalc, backend, els_k, els_kq, phs; kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_CountCalc; kwargs...) = c
ElectronPhonon.run_calculator!(c::_CountCalc, ::EPBlock, ctx) = (Threads.atomic_add!(c.n, 1); c)
ElectronPhonon.calculator_begin_batch!(::_CountCalc, ctx) = nothing
ElectronPhonon.calculator_end_batch!(::_CountCalc, ctx) = nothing

# `_CountCalc` reading the band velocities, which `run_eph_over_k_and_q` cannot provide at k+q.
struct _VdiagCountCalc <: AbstractCalculator end
ElectronPhonon.supports(::_VdiagCountCalc, ::Type{OuterKLoop}) = true
ElectronPhonon.required_el_quantities(::_VdiagCountCalc) = [:vdiag]

# An outer-q calculator, for the outer-q entry checks.
struct _QCountCalc <: AbstractCalculator end
ElectronPhonon.supports(::_QCountCalc, ::Type{OuterQLoop}) = true

@testset "supports contract" begin
    c = _CountCalc()
    # Type arguments: declared true, undeclared default false.
    @test supports(c, OuterKLoop) == true
    @test supports(c, OuterQLoop) == false
    # Anything but a loop-tag type (an instance, a block type) must throw, not silently return false.
    @test_throws ErrorException supports(c, EPBlock)
    @test_throws ErrorException supports(c, OuterKLoop())
    @test_throws ErrorException supports(c, 5)
end

@testset "driver contract checks (CPU)" begin
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    grid = (4, 4, 4)

    # Model storage must match the loop order; reject mismatches before preparing any states.
    model_ph = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
    @test_throws "epmat_outer_momentum = \"el\"" ElectronPhonon.run_eph_over_k_and_kq(
        model_ph, grid, grid; calculators = [_CountCalc()], symmetry = nothing)
    @test_throws "epmat_outer_momentum = \"el\"" ElectronPhonon.run_eph_over_k_and_q(
        model_ph, grid, grid; calculators = [_CountCalc()], symmetry = nothing)
    @test_throws "epmat_outer_momentum = \"ph\"" ElectronPhonon.run_eph_over_q_and_k(
        model, grid, grid; calculators = [_QCountCalc()], symmetry = nothing)

    # Screening is disabled: any nontrivial screening_params errors at the driver entry.
    @test_throws ErrorException ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
        calculators = [_CountCalc()], symmetry = nothing, screening_params = 1,
        progress_print_step = 10^9)

    # `calculators` is a keyword argument: there is no positional form.
    @test_throws MethodError ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid, [_CountCalc()])

    # `run_eph_over_k_and_q` refuses a k eigenpair cache for an outer k list on no grid, and what
    # needs the k+q points on a grid: velocities at k+q and a k+q eigenpair cache.
    kq_run(; kw...) = ElectronPhonon.run_eph_over_k_and_q(model, grid, grid;
        calculators = [_CountCalc()], progress_print_step = 10^9, kw...)
    @test_throws ArgumentError ElectronPhonon.run_eph_over_k_and_q(model,
        Kpoints([Vec3(0.1, 0.2, 0.3)]), grid; calculators = [_CountCalc()],
        el_k_eigenpairs = ElectronPhonon.electron_eigenpairs(model, kpoints_grid(grid)))
    # The other drivers keep their grid requirement on the k points.
    @test_throws ArgumentError ElectronPhonon.run_eph_over_q_and_k(
        model_ph, Kpoints([Vec3(0.1, 0.2, 0.3)]), grid; calculators = [_QCountCalc()])
    # The q set of the outer-q loop is a point set, never a state selection.
    @test_throws "takes q points, not a state selection" ElectronPhonon.run_eph_over_q_and_k(
        model_ph, grid, filter_electron_states(grid, model.nw, model.el_ham, (-Inf, Inf));
        calculators = [_QCountCalc()])
    @test_throws ArgumentError kq_run(calculators = [_VdiagCountCalc()])
    @test_throws ArgumentError kq_run(el_kq_eigenpairs =
        ElectronPhonon.electron_eigenpairs(model, kpoints_grid(grid)))
end

@testset "fourier_mode on a CPU backend" begin
    model_el = ElectronPhonon.holstein_model(; t = 0.1, ω₀ = 0.01, g = 0.02, alat = 5.0, ε₀ = 0.05,
        dimension = 3, epmat_outer_momentum = "el", verbose = false)
    grid = (2, 2, 2)
    # `fourier_mode` selects the setup interpolation (`_check_run`, shared by both orders); a
    # batched mode is refused on the CPU.
    @test_throws ArgumentError ElectronPhonon.run_eph_over_k_and_kq(model_el, grid, grid;
        calculators = [_CountCalc()], fourier_mode = "batched")
end

# One bracket per outer batch, with no default: a calculator that defines none fails loudly.
mutable struct _NoBracketCalc <: AbstractCalculator end

@testset "loop context and the one bracket" begin
    ctx = OuterKContext(CPUBackend(), 3:5, 1)
    @test ctx isa OuterKContext{CPUBackend}
    @test (ctx.iks_batch, ctx.chunk) == (3:5, 1)
    ctx_q = OuterQContext(CPUBackend(), 2:2, 1)
    @test ctx_q isa OuterQContext{CPUBackend} && ctx_q.iqs_batch == 2:2
    @test_throws ErrorException calculator_begin_batch!(_NoBracketCalc(), ctx)
    @test_throws ErrorException calculator_end_batch!(_NoBracketCalc(), ctx)

    # The reference calculator's brackets are its own methods, never the erroring fallback.
    BC = ElectronPhonon.BoltzmannCalculator
    for f in (calculator_begin_batch!, calculator_end_batch!)
        m = which(f, (BC, OuterKContext{CPUBackend}))
        @test Base.unwrap_unionall(m.sig).parameters[2] !== AbstractCalculator
    end

    # Backend-routed `to_device`: the CPU backend is an identity (no CUDA needed on the host path).
    v = [1.0, 2.0, 3.0]
    @test to_device(CPUBackend(), v) === v
end

const CONTRACT_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
CONTRACT_GPU_AVAILABLE && CUDA.allowscalar(false)
include("calculator_contract_harness.jl")

# Every ElectronPhonon.jl calculator through the generic contract harness (MigdalEliashberg.jl's
# run the same harness from its own test_calculator_contract.jl).
@testset "calculator contract harness" begin
    models = (; el = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el"),
                ph = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph"))
    e_F = contract_fixtures().narrow.e_F
    K = unit_to_aru(:K); meV = unit_to_aru(:meV)
    entries = [
        (; name = "BoltzmannCalculator", orders = (OuterKLoop,),
           make = () -> BoltzmannCalculator{Float64}(;
               occ = ElectronOccupationParams(; Tlist = [300.0K, 600.0K], nlist = 4.0,
                   μlist = [e_F, e_F], volume = models.el.volume, nelec = 0,
                   spin_degeneracy = 2, occ_type = :FermiDirac),
               smearing_list = [SmearingType(:Gaussian, 50.0meV), SmearingType(:Gaussian, 100.0meV)]),
           outputs = function (c)
               ids_i, ids_f = contract_multiplet_ids(c.el_i), contract_multiplet_ids(c.el_f)
               Dict("Sₒ" => contract_group_sum(stack(c.Sₒ), 1, ids_i),
                    "Sᵢ" => contract_group_sum(contract_group_sum(stack(c.Sᵢ), 1, ids_i), 2, ids_f))
           end,
           # `bte_scattering_increments` summed over the modes of each pair, g2 = |ep|^2 / (2ω).
           reference = function (c, ref)
               Sₒ = zero.(c.Sₒ); Sᵢ = zero.(c.Sᵢ)
               contract_foreach_reference_pair(ref, c.el_i, c.el_f) do i, j, ep, ω, ek, ekq, wtkq
                   for (iT, (; μ, T)) in enumerate(c.occ), ν in eachindex(ω)
                       ω[ν] < c.omega_cutoff && continue
                       sₒ, sᵢ = ElectronPhonon.bte_scattering_increments(c.occupation_method,
                           ek, ekq, ω[ν], abs2(ep[ν]) / (2ω[ν]), wtkq, μ, T, c.smearing_list[iT])
                       Sₒ[iT][i] += sₒ; Sᵢ[iT][i, j] += sᵢ
                   end
               end
               (; c.el_i, c.el_f, Sₒ, Sᵢ)
           end),
    ]
    # The host eigensolve rotates levels split by less than `electron_degen_cutoff` with EPW's
    # degenerate gauge fix and the device one does not, so a multiplet sum weighted by a function of
    # each level's own energy moves by about split / smearing: 2.8e-7 on the ragged fixture (A100).
    check_calculator_contract(entries, models; gpu = CONTRACT_GPU_AVAILABLE, rtol_gpu = 1e-6)
end
