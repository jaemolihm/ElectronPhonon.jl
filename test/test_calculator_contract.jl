using Test
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, ElPhDataPoint,
    ElPhDataOuterKBatched, supports

# Stage-2 calculator-contract checks (CPU-only): the `supports` trait, the fail-early payload checks
# the drivers do at entry, the `calculators`-as-kwarg change, and the screening-disabled error.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

# A batched-only calculator: declares the outer-k loop + device payload, but NOT the host
# `ElPhDataPoint`, so a CPU driver must reject it up front.
mutable struct _BatchedOnlyKCalc <: AbstractCalculator end
ElectronPhonon.supports(::_BatchedOnlyKCalc, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_BatchedOnlyKCalc, ::Type{ElPhDataOuterKBatched}) = true

# A minimal well-formed CPU outer-k calculator that just counts run_calculator! calls.
mutable struct _CountCalc <: AbstractCalculator
    n :: Int
    _CountCalc() = new(0)
end
ElectronPhonon.supports(::_CountCalc, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_CountCalc, ::Type{ElPhDataPoint}) = true
ElectronPhonon.setup_calculator!(c::_CountCalc, kpts, qpts, el_states; kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_CountCalc; kwargs...) = c
ElectronPhonon.run_calculator!(c::_CountCalc, ::ElPhDataPoint, ctx) = (c.n += 1; c)

@testset "supports contract (DECISION-1)" begin
    c = _CountCalc()
    # Type arguments: declared true, undeclared default false.
    @test supports(c, OuterKLoop) == true
    @test supports(c, ElPhDataPoint) == true
    @test supports(c, OuterQLoop) == false
    @test supports(c, ElPhDataOuterKBatched) == false
    # Non-Type argument (a foot-gun) must throw, not silently return false.
    @test_throws ErrorException supports(c, OuterKLoop())
    @test_throws ErrorException supports(c, 5)
end

@testset "driver contract checks (CPU)" begin
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    grid = (4, 4, 4)

    # (a) A batched-only calculator handed to a CPU driver errors BEFORE the loop starts.
    @test_throws ArgumentError ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
        calculators = [_BatchedOnlyKCalc()], symmetry = nothing, progress_print_step = 10^9)

    # (d) Screening is disabled: any nontrivial screening_params errors at the driver entry.
    @test_throws ErrorException ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
        calculators = [_CountCalc()], symmetry = nothing, screening_params = 1,
        progress_print_step = 10^9)

    # (b) `calculators` is a keyword argument (hard change): the positional form is gone.
    @test_throws MethodError ElectronPhonon.run_eph_over_k_and_q(model, grid, grid, [_CountCalc()])

    # (b, cont.) The kwarg form runs end-to-end and the per-(k,q) host hook is actually called.
    c = _CountCalc()
    ElectronPhonon.run_eph_over_k_and_q(model, grid, grid;
        calculators = [c], symmetry = nothing, progress_print_step = 10^9)
    @test c.n > 0
end
