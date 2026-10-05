using Test
using ElectronPhonon

# The "writing your own calculator" guide (docs/writing_a_calculator.md) contains the complete
# example calculator between the <!-- doc-example:begin --> / <!-- doc-example:end --> sentinels,
# a minimal CPU outer-k calculator and its run between the `doc-minimal` ones,
# and a driver run and a single-pair run of it between the `doc-driver` and `doc-single-pair` ones.
# This test extracts those blocks VERBATIM and evaluates them on the Pb artifact model, so the
# documented examples cannot rot.

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
    # so the per-chunk rows of `g2_per_k_buffer` and their reduction over `ctx.iks_batch` are exercised.
    # `invokelatest`: the calculator type + its interface methods were just defined by
    # `include_string`, so construct-and-run must execute at the latest world age to see them.
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    nk = 4
    calc, out = Base.invokelatest() do
        T = getfield(@__MODULE__, :EphG2SumCalculator)
        c = T()
        out = ElectronPhonon.run_eph_over_k_and_kq(model, (nk, nk, nk), (nk, nk, nk);
            calculators = [c], symmetry = nothing, progress_print_step = 10^9,
            n_outer_batch = 5, n_inner_tile = 7)
        c, out
    end

    @test length(calc.g2_per_k) == nk^3
    @test all(isfinite, calc.g2_per_k) && all(>(0), calc.g2_per_k)
    # `g2_avg`, reduced in the loop from the blocks' k weights, is the k-weighted sum of `g2_per_k`.
    @test calc.g2_avg ≈ sum(out.kpts.weights .* calc.g2_per_k) rtol = 1e-12

    # The other drivers on the same full grids hand the calculator the same pairs, so they give the
    # same sums: `run_eph_over_k_and_q` (k + q solved per tile) and `run_eph_over_q_and_k` (outer q),
    # on the CPU (the loop methods of `run_calculator!`) and, when available, on a GPU (the
    # broadcast methods), which compares all four methods.
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
        c, out_run = Base.invokelatest() do
            c = getfield(@__MODULE__, :EphG2SumCalculator)()
            out_run = run(; calculators = [c], common...)
            c, out_run
        end
        @test c.g2_per_k ≈ calc.g2_per_k rtol = 1e-10
        @test c.g2_avg ≈ sum(out_run.kpts.weights .* c.g2_per_k) rtol = 1e-12
        @test c.g2_avg ≈ calc.g2_avg rtol = 1e-10
    end

    # The minimal example, verbatim: it defines `MinimalG2Calculator` and runs it on the outer-k
    # driver, which the full calculator reproduces.
    Core.eval(@__MODULE__, :(epw_folder = $(_artifact_folder("pb"))))
    include_string(@__MODULE__, _extract_doc_example(guide; tag = "doc-minimal"))
    Base.invokelatest() do
        c_min, out_min = getfield(@__MODULE__, :calc_minimal), getfield(@__MODULE__, :out_minimal)
        @test length(c_min.g2_per_k) == out_min.kpts.n
        @test all(isfinite, c_min.g2_per_k) && all(>(0), c_min.g2_per_k)
        c_full = getfield(@__MODULE__, :EphG2SumCalculator)()
        ElectronPhonon.run_eph_over_k_and_kq(getfield(@__MODULE__, :model), (8, 8, 8), (8, 8, 8);
            calculators = [c_full], verbosity = 0)
        @test c_full.g2_per_k ≈ c_min.g2_per_k rtol = 1e-12
        @test c_min.g2_avg ≈ sum(out_min.kpts.weights .* c_min.g2_per_k) rtol = 1e-12
        @test c_full.g2_avg ≈ c_min.g2_avg rtol = 1e-12
    end

    # The driver example, verbatim. It reads the global `epw_folder` and defines `calc` (outer k,
    # symmetry-reduced), `calc_line` (outer k on a k line) and `calc_q` (outer q, full grid), with
    # the driver outputs `out`, `out_line` and `out_q`.
    include_string(@__MODULE__, _extract_doc_example(guide; tag = "doc-driver"))
    Base.invokelatest() do
        read_global(name) = getfield(@__MODULE__, name)
        c, c_line, c_q = read_global(:calc), read_global(:calc_line), read_global(:calc_q)
        out, out_line, out_q = read_global(:out), read_global(:out_line), read_global(:out_q)
        # Symmetry reduces the outer k of the outer-k run, and not the k of the outer-q run.
        @test out.kpts.n < out_q.kpts.n
        for (calc_run, out_run) in ((c, out), (c_line, out_line), (c_q, out_q))
            @test length(calc_run.g2_per_k) == out_run.kpts.n
            @test all(isfinite, calc_run.g2_per_k) && all(>(0), calc_run.g2_per_k)
            @test calc_run.g2_avg ≈ sum(out_run.kpts.weights .* calc_run.g2_per_k) rtol = 1e-12
        end
        # The irreducible-wedge sum equals the full-grid sum only as far as the interpolated g2 is
        # symmetric (6e-9 relative here).
        @test c.g2_avg ≈ c_q.g2_avg rtol = 1e-7
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
    end
end
