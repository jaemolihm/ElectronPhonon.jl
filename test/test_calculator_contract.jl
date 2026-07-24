using Test
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPData,
    EPDataQBatched, supports, LoopContext, SingleMode, BatchedMode, CPUBackend,
    GPUBackend, AbstractBackend, OuterIteration, OuterIterationBatch,
    calculator_begin!, calculator_end!, to_device

# Stage-2 calculator-contract checks (CPU-only): the `supports` trait, the fail-early payload checks
# the drivers do at entry, the `calculators`-as-kwarg change, and the screening-disabled error.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

# A batched-only calculator: declares the outer-k loop + device payload, but NOT the host
# `EPData`, so a CPU driver must reject it up front.
mutable struct _BatchedOnlyKCalc <: AbstractCalculator end
ElectronPhonon.supports(::_BatchedOnlyKCalc, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_BatchedOnlyKCalc, ::Type{EPDataQBatched}) = true

# A minimal well-formed CPU outer-k calculator that just counts run_calculator! calls.
mutable struct _CountCalc <: AbstractCalculator
    n :: Int
    _CountCalc() = new(0)
end
ElectronPhonon.supports(::_CountCalc, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_CountCalc, ::Type{EPData}) = true
ElectronPhonon.setup_calculator!(c::_CountCalc, kpts, qpts, el_states; kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_CountCalc; kwargs...) = c
ElectronPhonon.run_calculator!(c::_CountCalc, ::EPData, ctx) = (c.n += 1; c)
# CPU-only (SingleMode): nothing per outer iteration, but there is no default bracket, so the no-op
# must be defined explicitly or the CPU loop's OuterIteration bracket would error.
ElectronPhonon.calculator_begin!(::_CountCalc, ::OuterIteration, ctx) = nothing
ElectronPhonon.calculator_end!(::_CountCalc, ::OuterIteration, ctx) = nothing

@testset "supports contract (DECISION-1)" begin
    c = _CountCalc()
    # Type arguments: declared true, undeclared default false.
    @test supports(c, OuterKLoop) == true
    @test supports(c, EPData) == true
    @test supports(c, OuterQLoop) == false
    @test supports(c, EPDataQBatched) == false
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
ElectronPhonon.calculator_begin!(c::_ModeDispatchCalc, ::OuterIteration, ::LoopContext{<:AbstractBackend, SingleMode}) =
    (push!(c.fired, (:iter, :point)); c)
ElectronPhonon.calculator_begin!(c::_ModeDispatchCalc, ::OuterIterationBatch, ::LoopContext{<:AbstractBackend, BatchedMode}) =
    (push!(c.fired, (:batch, :batched)); c)

@testset "loop-mode bracket dispatch (DECISION-6)" begin
    # `LoopContext` carries the backend first, the mode second.
    ctx_pt = LoopContext(CPUBackend(), SingleMode(), 1, 1:0, 4)
    ctx_bt = LoopContext(CPUBackend(), BatchedMode(), 0, 1:4, 4)
    @test ctx_pt isa LoopContext{CPUBackend, SingleMode}
    @test ctx_bt isa LoopContext{CPUBackend, BatchedMode}
    # The backend-first order keeps the partial annotation `LoopContext{<:GPUBackend}` valid (any mode).
    @test LoopContext{CPUBackend, SingleMode} <: LoopContext{CPUBackend}

    # `_ModeDispatchCalc` defines only OuterIteration/SingleMode and OuterIterationBatch/BatchedMode.
    # There is no default bracket, so the other two (scope, mode) combinations ERROR — a missing
    # bracket is loud, not a silent no-op. The two defined combinations still select by MODE, not
    # backend (the batched loop fires the per-k `OuterIteration` in BatchedMode).
    c = _ModeDispatchCalc()
    calculator_begin!(c, OuterIteration(), ctx_pt)                                  # -> (:iter, :point)
    @test_throws ErrorException calculator_begin!(c, OuterIteration(), ctx_bt)       # no BatchedMode method
    calculator_begin!(c, OuterIterationBatch(), ctx_bt)                             # -> (:batch, :batched)
    @test_throws ErrorException calculator_begin!(c, OuterIterationBatch(), ctx_pt)  # no SingleMode method
    @test c.fired == [(:iter, :point), (:batch, :batched)]

    # The reference calculators dispatch their real brackets by mode, not backend, and every combination
    # the loops fire resolves to a calculator-owned method — never the AbstractCalculator error fallback.
    # BoltzmannCalculator does real work in OuterIteration/SingleMode and defines an explicit no-op for
    # OuterIteration/BatchedMode (the batched outer-k loop fires the per-k bracket on a device backend).
    BC = ElectronPhonon.BoltzmannCalculator
    m_single = which(calculator_begin!, (BC, OuterIteration, LoopContext{CPUBackend, SingleMode}))
    @test Base.unwrap_unionall(m_single.sig).parameters[2] !== AbstractCalculator
    m_batched = which(calculator_begin!, (BC, OuterIteration, LoopContext{CPUBackend, BatchedMode}))
    @test Base.unwrap_unionall(m_batched.sig).parameters[2] !== AbstractCalculator

    # Backend-routed `to_device`: the CPU backend is an identity (no CUDA needed on the host path).
    v = [1.0, 2.0, 3.0]
    @test to_device(CPUBackend(), v) === v
end
