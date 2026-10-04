using Test
using ElectronPhonon

# The "writing your own calculator" guide (docs/writing_a_calculator.md) contains a complete minimal
# example calculator between the <!-- doc-example:begin --> / <!-- doc-example:end --> sentinels.
# This test extracts that block VERBATIM, evaluates it, and runs it through the drivers on the Pb
# artifact model, so the documented example cannot rot.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

const CALCULATOR_GUIDE_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Extract the fenced Julia code between the doc-example sentinels.
function _extract_doc_example(md_path; tag = "doc-example")
    text = read(md_path, String)
    b = findfirst("<!-- $tag:begin -->", text)
    e = findfirst("<!-- $tag:end -->", text)
    (b === nothing || e === nothing) && error("doc-example sentinels not found in $md_path")
    block = text[last(b)+1 : first(e)-1]
    # Drop the ```julia … ``` fences, keep the code between them.
    lines = split(block, '\n')
    code = String[]
    infence = false
    for ln in lines
        s = strip(ln)
        if startswith(s, "```")
            infence = !infence
            continue
        end
        infence && push!(code, ln)
    end
    join(code, '\n')
end

@testset "writing_a_calculator.md example" begin
    guide = joinpath(@__DIR__, "..", "docs", "writing_a_calculator.md")
    @test isfile(guide)
    code = _extract_doc_example(guide)
    @test occursin("EphG2SumCalculator", code)

    # Evaluate the guide's example verbatim (defines the struct + interface methods).
    include_string(@__MODULE__, code)

    # Run it on the Pb artifact model (outer-k driver), with several outer batches and inner tiles,
    # so the per-chunk partials and their reduction over `ctx.batch` are exercised.
    # `invokelatest`: the calculator type + its interface methods were just defined by
    # `include_string`, so construct-and-run must execute at the latest world age to see them.
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    nk = 4
    calc = Base.invokelatest() do
        T = getfield(@__MODULE__, :EphG2SumCalculator)
        c = T()
        ElectronPhonon.run_eph_over_k_and_kq(model, (nk, nk, nk), (nk, nk, nk);
            calculators = [c], symmetry = nothing, progress_print_step = 10^9,
            n_outer_batch = 5, n_inner_tile = 7)
        c
    end

    @test length(calc.g2_per_k) == nk^3
    @test all(isfinite, calc.g2_per_k) && all(>(0), calc.g2_per_k)

    # The other drivers on the same full grids hand the calculator the same pairs, so they give the
    # same sums: `run_eph_over_k_and_q` (k + q solved per tile) and `run_eph_over_q_and_k` (outer q,
    # the outer-q `run_calculator!`), on the CPU and, when available, on a GPU (the broadcast
    # version of `pair_g2_sums!`).
    model_ph = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
    grid = (nk, nk, nk)
    common = (; symmetry = nothing, progress_print_step = 10^9, n_outer_batch = 5, n_inner_tile = 7,
              verbosity = 0)
    runs = Any[
        "outer k, inner q" => (; kw...) -> ElectronPhonon.run_eph_over_k_and_q(model, grid, grid; kw...),
        "outer q" => (; kw...) -> ElectronPhonon.run_eph_over_q_and_k(model_ph, grid, grid; kw...)]
    if CALCULATOR_GUIDE_GPU
        CUDA.allowscalar(false)
        gpu = ElectronPhonon.gpu_backend()
        push!(runs,
            "outer k, GPU" => (; kw...) -> ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid; backend = gpu, kw...),
            "outer q, GPU" => (; kw...) -> ElectronPhonon.run_eph_over_q_and_k(model_ph, grid, grid; backend = gpu, kw...))
    end
    for (name, run) in runs
        c = Base.invokelatest() do
            c = getfield(@__MODULE__, :EphG2SumCalculator)()
            run(; calculators = [c], common...)
            c
        end
        @test c.g2_per_k ≈ calc.g2_per_k rtol = 1e-10
    end

    # The direct-call example uses exactly the same calculator implementation and lifecycle.
    # The example reads the global `model` and defines `calc`, `kpts` and `qpts`; they are read back
    # at the latest world age, like the calculator type above.
    direct_code = _extract_doc_example(guide; tag = "doc-single-pair")
    Core.eval(@__MODULE__, :(model = $model))
    include_string(@__MODULE__, direct_code)
    Base.invokelatest() do
        c = getfield(@__MODULE__, :calc)
        @test length(c.g2_per_k) == 1
        @test all(isfinite, c.g2_per_k) && maximum(abs, c.g2_per_k) > 0
        driver = getfield(@__MODULE__, :EphG2SumCalculator)()
        ElectronPhonon.run_eph_over_k_and_q(model, getfield(@__MODULE__, :kpts), getfield(@__MODULE__, :qpts);
            calculators = [driver], verbosity = 0)
        @test driver.g2_per_k ≈ c.g2_per_k rtol = 1e-10

        # The broadcast version of `pair_g2_sums!` on the CPU gives the loop's sums.
        pair_g2_sums! = getfield(@__MODULE__, :pair_g2_sums!)
        eng = getfield(@__MODULE__, :eng)
        block = ElectronPhonon.stage2!(eng, 1, 1:1)
        by_loop = copy(pair_g2_sums!(c, block, 1, eng.backend))
        by_broadcast = invoke(pair_g2_sums!, Tuple{typeof(c), Any, Any, Any}, c, block, 1, eng.backend)
        @test by_broadcast ≈ by_loop rtol = 1e-12
    end
end
