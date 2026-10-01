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
ElectronPhonon.setup_calculator!(c::_CountCalc, backend, mode, kpts, qpts, el_states; kwargs...) = c
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
    # bracket is loud, not a silent no-op. The two defined combinations select by MODE, not backend.
    c = _ModeDispatchCalc()
    calculator_begin!(c, OuterIteration(), ctx_pt)                                  # -> (:iter, :point)
    @test_throws ErrorException calculator_begin!(c, OuterIteration(), ctx_bt)       # no BatchedMode method
    calculator_begin!(c, OuterIterationBatch(), ctx_bt)                             # -> (:batch, :batched)
    @test_throws ErrorException calculator_begin!(c, OuterIterationBatch(), ctx_pt)  # no SingleMode method
    @test c.fired == [(:iter, :point), (:batch, :batched)]

    # The reference calculators dispatch their real brackets by mode, not backend, and every
    # combination the loops FIRE resolves to a calculator-owned method — never the AbstractCalculator
    # error fallback. For BoltzmannCalculator those are OuterIteration/SingleMode (the CPU outer-k
    # loop) and OuterIterationBatch/BatchedMode (the GPU outer-k loop). OuterIteration/BatchedMode is
    # NOT one of them: the GPU outer-k loop runs its q-tile loop outside the k loop, so a single k has
    # no begin/end point and no per-k bracket is fired; that combination correctly falls through to
    # the erroring fallback, which is what makes a calculator relying on it fail loudly.
    BC = ElectronPhonon.BoltzmannCalculator
    m_single = which(calculator_begin!, (BC, OuterIteration, LoopContext{CPUBackend, SingleMode}))
    @test Base.unwrap_unionall(m_single.sig).parameters[2] !== AbstractCalculator
    m_batch = which(calculator_begin!, (BC, OuterIterationBatch, LoopContext{CPUBackend, BatchedMode}))
    @test Base.unwrap_unionall(m_batch.sig).parameters[2] !== AbstractCalculator
    m_unfired = which(calculator_begin!, (BC, OuterIteration, LoopContext{CPUBackend, BatchedMode}))
    @test Base.unwrap_unionall(m_unfired.sig).parameters[2] === AbstractCalculator

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
