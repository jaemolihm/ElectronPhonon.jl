using Test
using ElectronPhonon
using ElectronPhonon: OuterKEngine, OuterQEngine, OuterKLoop, OuterQLoop, EPBlock, LoopContext,
    stage1!, stage2!, setup_calculator!, run_calculator!, calculator_begin!, calculator_end!,
    postprocess_calculator!, CPUBackend, gpu_backend, unit_to_aru

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")
isdefined(@__MODULE__, :eph_reference) || include("eph_reference_loop.jl")

const EPH_ENGINE_API_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# The calculator setup of the drivers, on a prepared engine.
_setup_on_engine!(calc, eng) = setup_calculator!(calc, eng.backend, eng.els_k, eng.els_kq, eng.phs;
    eng.sel_k, eng.sel_kq, nchunks_threads = length(eng.tiles),
    eng.n_outer_batch, eng.n_inner_tile, verbosity = 0)

@testset "direct e-ph engine and calculator API" begin
    keygrid = (10^6, 10^6, 10^6)
    kpts = Kpoints(Vec3(0.2513, 0.2487, 0.0129))
    qpts = Kpoints(Vec3(0.071, 0.023, 0.019))
    eV = unit_to_aru(:eV)
    window = (11.68eV - 0.5eV, 11.68eV + 3eV)
    model_el = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    model_ph = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
    ref = eph_reference_k_and_q(model_el, kpts, qpts, window, window, keygrid)
    @test length(ref.g2abs) == 1
    backends = Any[CPUBackend()]
    if EPH_ENGINE_API_GPU
        CUDA.allowscalar(false)
        push!(backends, gpu_backend())
    end
    for backend in backends, (Engine, model, Order) in
            ((OuterKEngine, model_el, OuterKLoop), (OuterQEngine, model_ph, OuterQLoop))
        calc = _PairRecorder(keygrid)
        eng = Engine(model, kpts, qpts; backend, calculators = [calc], window_k = window,
            window_kq = window, verbosity = 0)
        @test isempty(calc.g2abs) && calc.nw == 0  # construction runs no calculator hooks
        @test_throws ArgumentError stage2!(eng, 1, 1:1)
        @test_throws ArgumentError LoopContext(eng)
        _setup_on_engine!(calc, eng)
        @test calc.nw == model.nw && eng.sel_k !== nothing
        @test stage1!(eng, 1:1) === eng
        ctx = LoopContext(eng)
        @test ctx.order isa Order && ctx.batch == 1:1 && ctx.chunk == 1
        calculator_begin!(calc, ctx)
        block = stage2!(eng, 1, 1:1)
        @test block isa EPBlock{Order}
        @test block.els_k isa BatchedElectronState && block.els_kq isa BatchedElectronState
        @test block.phs isa BatchedPhononState
        @test block.els_k.nk == block.els_kq.nk == block.phs.nq == size(block.ep, 4) == 1
        @test size(block.ep) == (block.els_kq.nband_max, block.els_k.nband_max, model.nmodes, 1)
        @test block.dg === nothing
        run_calculator!(calc, block, ctx)
        calculator_end!(calc, ctx)
        postprocess_calculator!(calc; qpts = eng.qpts, symmetry = nothing)
        dev = compare_with_reference(ref, calc)
        @test dev.nmissing == dev.nextra == 0
        @test dev.g2_reldev < 1e-11 && dev.ω_dev < 1e-10
        @test_throws ArgumentError stage2!(eng, 2, 1:1)
        @test_throws BoundsError stage2!(eng, 1, 1:1; chunk = 2)
        @test_throws ArgumentError stage2!(eng, 1, 1:0)
        @test_throws BoundsError stage2!(eng, 1, 0:1)
        @test_throws BoundsError LoopContext(eng; chunk = 0)
        @test_throws ArgumentError stage1!(eng, 1:0)
        @test_throws BoundsError stage1!(eng, 0:1)
    end

    # Direct callers get the same filtering result as the production loop, not an unfinished matrix.
    eng = OuterKEngine(model_el, kpts, qpts; window_k = window, window_kq = window,
        energy_conservation_tol = 0.0, verbosity = 0)
    stage1!(eng, 1:1)
    @test stage2!(eng, 1, 1:1) === nothing
    @test_throws ArgumentError OuterKEngine(model_el, kpts, qpts; n_outer_batch = 0)
    @test_throws ArgumentError OuterKEngine(model_el, kpts, qpts; n_inner_tile = 0)
    @test_throws ArgumentError OuterKEngine(model_el, kpts, qpts; nchunks_threads = 0)
end

@testset "direct stage2 completes polar, derivative and phonon-basis options" begin
    kpts = Kpoints(Vec3(0.2513, 0.2487, 0.0129))
    qpts = Kpoints(Vec3(0.071, 0.023, 0.019))
    model_el = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "el")
    model_ph = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "ph")
    @test model_el.polar_eph.use
    backends = EPH_ENGINE_API_GPU ? Any[CPUBackend(), gpu_backend()] : Any[CPUBackend()]
    for backend in backends, basis in (:eigenmode, :cartesian)
        ek = OuterKEngine(model_el, kpts, qpts; backend, eph_phonon_basis = basis,
            covariant_derivative_of_g = true, verbosity = 0)
        eq = OuterQEngine(model_ph, kpts, qpts; backend, eph_phonon_basis = basis, verbosity = 0)
        stage1!(ek, 1:1); stage1!(eq, 1:1)
        bk, bq = stage2!(ek, 1, 1:1), stage2!(eq, 1, 1:1)
        @test sum(abs2, Array(bk.ep)) ≈ sum(abs2, Array(bq.ep)) rtol = 1e-11
        @test size(bk.dg) == (model_el.nw, model_el.nw, model_el.nmodes, 3, 1)
        @test all(isfinite, Array(bk.dg)) && sum(abs2, Array(bk.dg)) > 0
        @test bq.dg === nothing
    end
end

@testset "engine stage capacities, batch offsets and independent chunks" begin
    grid = (3, 3, 3)
    backends = EPH_ENGINE_API_GPU ? Any[CPUBackend(), gpu_backend()] : Any[CPUBackend()]
    for backend in backends, (Engine, momentum, Order) in
            ((OuterKEngine, "el", OuterKLoop), (OuterQEngine, "ph", OuterQLoop))
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = momentum)
        calc = _PairRecorder()
        eng = Engine(model, grid, grid; backend, calculators = [calc], verbosity = 0,
            n_outer_batch = 3, n_inner_tile = 2, nchunks_threads = 2,
            (Order === OuterKLoop ? (; inner_loop_kq = true) : (; precompute_el_kq = true))...)
        _setup_on_engine!(calc, eng)
        @test_throws ArgumentError stage1!(eng, 1:4)
        stage1!(eng, 3:4)  # nonzero offset, shorter than allocated stage-1 capacity
        ctx = LoopContext(eng)
        @test ctx.batch == 3:4
        @test_throws ArgumentError stage2!(eng, 2, 1:1)
        @test_throws ArgumentError stage2!(eng, 3, 1:3)
        # Different outer q points can safely read different stage-1 slices on different chunks.
        # On a GPU the single chunk remains on the caller's stream.
        if backend isa CPUBackend
            tasks = [Threads.@spawn stage2!(eng, i, 1:2; chunk) for (chunk, i) in enumerate(3:4)]
            blocks = fetch.(tasks)
            @test !Base.mightalias(blocks[1].ep, blocks[2].ep)
            for (chunk, block) in enumerate(blocks)
                run_calculator!(calc, block, LoopContext(eng; chunk))
            end
        else
            for i in 3:4
                block = stage2!(eng, i, 1:2)
                run_calculator!(calc, block, ctx)
            end
        end
        # Compare the directly produced blocks against the independent scalar reference.
        kpoints = Order === OuterKLoop ? Kpoints(2, eng.kpts.vectors[3:4], [0.5, 0.5], grid) :
            Kpoints(2, eng.kpts.vectors[1:2], [0.5, 0.5], grid)
        kqpoints = Order === OuterKLoop ? Kpoints(2, eng.kqpts.vectors[1:2], [0.5, 0.5], grid) : nothing
        reference_model = Order === OuterKLoop ? model : _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        ref = if Order === OuterKLoop
            eph_reference(reference_model, kpoints, kqpoints, (-Inf, Inf), (-Inf, Inf))
        else
            qs = Kpoints(2, eng.qpts.vectors[3:4], [0.5, 0.5], grid)
            eph_reference_k_and_q(reference_model, kpoints, qs, (-Inf, Inf), (-Inf, Inf), grid)
        end
        dev = compare_with_reference(ref, calc)
        @test dev.nmissing == dev.nextra == 0
        @test dev.g2_reldev < 1e-11 && dev.ω_dev < 1e-10
        # Stage-1 batch replacement is explicit and invalidates the previous outer indices.
        stage1!(eng, 5:5)
        @test LoopContext(eng).batch == 5:5
        @test_throws ArgumentError stage2!(eng, 3, 1:1)
    end
end
