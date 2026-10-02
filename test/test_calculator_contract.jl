using Test
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPData, EPBlock, supports,
    LoopContext, CPUBackend, OuterIteration, calculator_begin!, calculator_end!, to_device

# Calculator-contract checks (CPU-only): the `supports` trait, the fail-early checks the drivers do
# at entry, the `calculators`-as-kwarg change, and the screening-disabled error.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

# An outer-k calculator without the per-point host payload: the per-point driver path must reject it
# up front.
mutable struct _BatchedOnlyKCalc <: AbstractCalculator end
ElectronPhonon.supports(::_BatchedOnlyKCalc, ::Type{OuterKLoop}) = true

# A minimal well-formed per-point outer-k calculator that just counts run_calculator! calls.
mutable struct _CountCalc <: AbstractCalculator
    n :: Int
    _CountCalc() = new(0)
end
ElectronPhonon.supports(::_CountCalc, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_CountCalc, ::Type{EPData}) = true
ElectronPhonon.setup_calculator!(c::_CountCalc, backend, mode, kpts, qpts, el_states; kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_CountCalc; kwargs...) = c
ElectronPhonon.run_calculator!(c::_CountCalc, ::EPData, ctx) = (c.n += 1; c)
ElectronPhonon.calculator_begin!(::_CountCalc, ::OuterIteration, ctx) = nothing
ElectronPhonon.calculator_end!(::_CountCalc, ::OuterIteration, ctx) = nothing

@testset "supports contract (DECISION-1)" begin
    c = _CountCalc()
    # Type arguments: declared true, undeclared default false.
    @test supports(c, OuterKLoop) == true
    @test supports(c, EPData) == true
    @test supports(c, OuterQLoop) == false
    @test supports(c, EPBlock) == false
    # Non-Type argument (a foot-gun) must throw, not silently return false.
    @test_throws ErrorException supports(c, OuterKLoop())
    @test_throws ErrorException supports(c, 5)
end

@testset "driver contract checks (CPU)" begin
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    grid = (4, 4, 4)

    # (a) A calculator without the per-point payload handed to the per-point path errors BEFORE the
    # loop starts.
    @test_throws ArgumentError ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
        calculators = [_BatchedOnlyKCalc()], symmetry = nothing, progress_print_step = 10^9,
        batched = false)

    # (d) Screening is disabled: any nontrivial screening_params errors at the driver entry.
    @test_throws ErrorException ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
        calculators = [_CountCalc()], symmetry = nothing, screening_params = 1,
        progress_print_step = 10^9)

    # (b) `calculators` is a keyword argument (hard change): the positional form is gone.
    @test_throws MethodError ElectronPhonon.run_eph_over_k_and_q(model, grid, grid, [_CountCalc()])

    # (b, cont.) The kwarg form runs end-to-end and the per-(k,q) host hook is actually called. The
    # per-point loops do not run in the current setup and bracket contract (ElectronPhonon.jl issue
    # #72, https://github.com/jaemolihm/ElectronPhonon.jl/issues/72).
    c = _CountCalc()
    @test_broken (ElectronPhonon.run_eph_over_k_and_q(model, grid, grid;
        calculators = [c], symmetry = nothing, progress_print_step = 10^9); c.n > 0)
end

@testset "driver rejects a batched fourier_mode on a CPU backend" begin
    # The message is the guard's own, so a match also shows it fired before any setup work.
    model_el = ElectronPhonon.holstein_model(; t = 0.1, ω₀ = 0.01, g = 0.02, alat = 5.0, ε₀ = 0.05,
        dimension = 3, epmat_outer_momentum = "el", verbose = false)
    model_ph = ElectronPhonon.holstein_model(; t = 0.1, ω₀ = 0.01, g = 0.02, alat = 5.0, ε₀ = 0.05,
        dimension = 3, epmat_outer_momentum = "ph", verbose = false)
    grid = (2, 2, 2)
    for fourier_mode in ("batched", "batched-gridopt")
        msg = "fourier_mode = \"$fourier_mode\" is not supported"
        @test_throws msg ElectronPhonon.run_eph_over_k_and_kq(model_el, grid, grid; fourier_mode)
        @test_throws msg ElectronPhonon.run_eph_over_k_and_kq(model_el, grid, grid; fourier_mode,
                                                                batched = true)
        @test_throws msg ElectronPhonon.run_eph_over_k_and_q(model_el, grid, grid; fourier_mode)
        @test_throws msg ElectronPhonon.run_eph_over_q_and_k(model_ph, grid, grid; fourier_mode)
        @test_throws msg ElectronPhonon.run_eph_over_q_and_k(model_ph, grid, grid; fourier_mode,
                                                               batched = true)
    end
end

# One bracket per outer batch, with no default: a calculator that defines none fails loudly.
mutable struct _NoBracketCalc <: AbstractCalculator end

@testset "loop context and the one bracket" begin
    ctx = LoopContext(CPUBackend(), OuterKLoop(), 3:5, 1)
    @test ctx isa LoopContext{CPUBackend, OuterKLoop}
    @test (ctx.batch, ctx.chunk) == (3:5, 1)
    @test LoopContext{CPUBackend, OuterKLoop} <: LoopContext{CPUBackend}
    @test_throws ErrorException calculator_begin!(_NoBracketCalc(), ctx)
    @test_throws ErrorException calculator_end!(_NoBracketCalc(), ctx)

    # The reference calculator's brackets are its own methods, never the erroring fallback.
    BC = ElectronPhonon.BoltzmannCalculator
    for f in (calculator_begin!, calculator_end!)
        m = which(f, (BC, LoopContext{CPUBackend, OuterKLoop}))
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
