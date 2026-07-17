using Test
using ElectronPhonon

# The "writing your own calculator" guide (docs/writing_a_calculator.md) contains a complete minimal
# example calculator between the <!-- doc-example:begin --> / <!-- doc-example:end --> sentinels.
# This test extracts that block VERBATIM, evaluates it, and runs it through run_eph_over_k_and_kq on
# the Pb artifact model, so the documented example cannot rot.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

# Extract the fenced Julia code between the doc-example sentinels.
function _extract_doc_example(md_path)
    text = read(md_path, String)
    b = findfirst("<!-- doc-example:begin -->", text)
    e = findfirst("<!-- doc-example:end -->", text)
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

    # Run it on the Pb artifact model (outer-k driver). This is threaded (@threads over k+q chunks),
    # so it also exercises the per-chunk (id_chunk) thread-safety pattern the guide documents.
    # `invokelatest`: the calculator type + its interface methods were just defined by
    # `include_string`, so construct-and-run must execute at the latest world age to see them.
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    nk = 4
    calc = Base.invokelatest() do
        T = getfield(@__MODULE__, :EphG2SumCalculator)
        c = T()
        ElectronPhonon.run_eph_over_k_and_kq(model, (nk, nk, nk), (nk, nk, nk);
            calculators = [c], symmetry = nothing, progress_print_step = 10^9)
        c
    end

    @test length(calc.per_k) == nk^3
    @test all(isfinite, calc.per_k)
    # g2 = |ep|²/(2ω) can be negative where ω < 0 (Pb's soft acoustic modes on a coarse grid), so
    # only assert the result is finite and non-trivial (nonzero e-ph coupling recorded).
    @test maximum(abs, calc.per_k) > 0
end
