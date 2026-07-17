using Test
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, ElPhDataPoint,
    ElPhDataOuterKBatched, supports, LoopContext, PointMode, BatchedMode, CPUBackend,
    GPUBackend, AbstractBackend, OuterIteration, OuterIterationBatch,
    calculator_begin!, calculator_end!, to_device

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

# Stage-3 (DECISION-6): brackets that differ by loop shape dispatch on the loop MODE, not the
# backend. A calculator with per-point-mode and batched-mode `OuterIteration` / `OuterIterationBatch`
# brackets; each records the (scope, mode) it fired under so we can assert selection is by mode.
mutable struct _ModeDispatchCalc <: AbstractCalculator
    fired :: Vector{Tuple{Symbol, Symbol}}
    _ModeDispatchCalc() = new(Tuple{Symbol, Symbol}[])
end
ElectronPhonon.calculator_begin!(c::_ModeDispatchCalc, ::OuterIteration, ::LoopContext{<:AbstractBackend, PointMode}) =
    (push!(c.fired, (:iter, :point)); c)
ElectronPhonon.calculator_begin!(c::_ModeDispatchCalc, ::OuterIterationBatch, ::LoopContext{<:AbstractBackend, BatchedMode}) =
    (push!(c.fired, (:batch, :batched)); c)

@testset "loop-mode bracket dispatch (DECISION-6)" begin
    # `LoopContext` carries the backend first, the mode second.
    ctx_pt = LoopContext(CPUBackend(), PointMode(), 1, 1:0, 4)
    ctx_bt = LoopContext(CPUBackend(), BatchedMode(), 0, 1:4, 4)
    @test ctx_pt isa LoopContext{CPUBackend, PointMode}
    @test ctx_bt isa LoopContext{CPUBackend, BatchedMode}
    # The backend-first order keeps the partial annotation `LoopContext{<:GPUBackend}` valid (any mode).
    @test LoopContext{CPUBackend, PointMode} <: LoopContext{CPUBackend}

    # A calculator's per-point `OuterIteration` bracket fires only in PointMode; the batched loop's
    # per-iteration `OuterIteration` bracket hits the default no-op (does NOT trigger the point path).
    c = _ModeDispatchCalc()
    calculator_begin!(c, OuterIteration(), ctx_pt)          # -> (:iter, :point)
    calculator_begin!(c, OuterIteration(), ctx_bt)          # -> default no-op (no method for BatchedMode)
    calculator_begin!(c, OuterIterationBatch(), ctx_bt)     # -> (:batch, :batched)
    calculator_begin!(c, OuterIterationBatch(), ctx_pt)     # -> default no-op (no method for PointMode)
    @test c.fired == [(:iter, :point), (:batch, :batched)]

    # The two reference calculators dispatch their real brackets by mode, not backend: the batched
    # loop fires per-k `OuterIteration` on a GPU backend, which must resolve to the default no-op
    # (not the CPU per-point reduction).
    BC = ElectronPhonon.BoltzmannCalculator
    @test which(calculator_begin!, (BC, OuterIteration, LoopContext{CPUBackend, PointMode})).module === ElectronPhonon
    # OuterIteration on any backend in BatchedMode -> AbstractCalculator no-op default.
    m_noop = which(calculator_begin!, (BC, OuterIteration, LoopContext{CPUBackend, BatchedMode}))
    @test m_noop.sig.parameters[2] === AbstractCalculator

    # Backend-routed `to_device`: the CPU backend is an identity (no CUDA needed on the host path).
    v = [1.0, 2.0, 3.0]
    @test to_device(CPUBackend(), v) === v
end
