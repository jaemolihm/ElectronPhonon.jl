using Test
using ElectronPhonon
using ElectronPhonon: unit_to_aru, run_eph_over_k_and_kq, run_eph_over_q_and_k

# Features of the per-point loops that the batched loops do not provide yet (ElectronPhonon.jl
# issue #72, https://github.com/jaemolihm/ElectronPhonon.jl/issues/72), on the default loop. Each
# `@test_broken` asserts the real result, so it fails now and passes once the feature returns; Stage C
# turns it into `@test` then. Each assertion was checked to hold on the per-point loop of the
# previous release.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")
isdefined(@__MODULE__, :eph_reference) || include("eph_reference_loop.jl")

# Records Σ |dg[:, :, :, d]|² of every pair from the `dg` of the blocks (in-window bands, all modes).
mutable struct _DgRecorder <: ElectronPhonon.AbstractCalculator
    ngrid :: NTuple{3, Int}
    sums :: Dict{NTuple{6, Int}, Vector{Float64}}
    _DgRecorder() = new((0, 0, 0), Dict())
end
ElectronPhonon.supports(::_DgRecorder, ::Type{OuterKLoop}) = true
ElectronPhonon.calculator_begin!(::_DgRecorder, ctx) = nothing
ElectronPhonon.calculator_end!(::_DgRecorder, ctx) = nothing
ElectronPhonon.postprocess_calculator!(c::_DgRecorder; kwargs...) = c
ElectronPhonon.setup_calculator!(c::_DgRecorder, backend, els_k, els_kq, phs; kwargs...) =
    (c.ngrid = els_k.kpts.ngrid; c)
function ElectronPhonon.run_calculator!(c::_DgRecorder, block::EPBlock{OuterKLoop}, ctx)
    dg = Array(block.dg)   # (m, n, ν, d, j)
    nbk, nbkq = Array(block.els_k.nband)[1], Array(block.els_kq.nband)
    for (j, xq) in enumerate(block.xq)
        c.sums[_pair_key(block.xk, block.xk + xq, c.ngrid)] =
            [sum(abs2, dg[1:nbkq[j], 1:nbk, :, d, j]) for d in 1:3]
    end
end

@testset "features of the per-point loops on the default loop (#72)" begin
    # Outer-k polar: on a polar model (cubicBN), Σ |ep|² over every pair, band and mode, which does
    # not depend on the electron or phonon basis on a full window, agrees with the outer-q loop.
    # Measured 3.3e-16 relative on the per-point loops, 3³ grids.
    @testset "outer-k polar" begin
        grid = (3, 3, 3)
        total(rec) = sum(sum, values(rec.g2abs))
        rec_q = _PairRecorder()
        run_eph_over_q_and_k(_load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "ph"),
            grid, grid; calculators = [rec_q], use_symmetry = false, keep_all_qpts = true,
            progress_print_step = 10^9, verbosity = 0)
        model = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "el")
        @test model.polar_eph.use && length(rec_q.g2abs) == prod(grid)^2
        # Outer-k polar (#72).
        @test_broken (rec_k = _PairRecorder();
            run_eph_over_k_and_kq(model, grid, grid; calculators = [rec_k], symmetry = nothing,
                progress_print_step = 10^9, verbosity = 0);
            isapprox(total(rec_k), total(rec_q); rtol = 1e-12))
    end

    # Energy conservation: a `(:Fixed, 10σ)` cut drops only pairs whose Gaussian-smeared BTE
    # contribution is below exp(-100), so Sₒ and Sᵢ match the uncut run, and it does drop pairs: the
    # pair recorder running alongside sees fewer of them. Measured 0.0 and 1.1e-45 relative on the
    # per-point loop (Pb 6³, ±0.5 eV, σ = 20 meV).
    @testset "energy conservation" begin
        model = _load_model_from_artifacts("pb")
        eV, K, meV = unit_to_aru(:eV), unit_to_aru(:K), unit_to_aru(:meV)
        μ = 11.68eV; window = (μ - 0.5eV, μ + 0.5eV); σ = 20meV
        runbte(; kwargs...) = (c = BoltzmannCalculator{Float64}(;
                occ = ElectronOccupationParams(; Tlist = [300.0K], nlist = 4.0, μlist = μ,
                    volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
                smearing_list = [SmearingType(:Gaussian, σ)], occupation_method = 5);
            rec = _PairRecorder();
            run_eph_over_k_and_kq(model, (6, 6, 6), (6, 6, 6); calculators = [c, rec],
                symmetry = nothing, window_k = window, window_kq = window,
                progress_print_step = 10^9, verbosity = 0, kwargs...); (c, rec))
        c_all, rec_all = runbte()
        @test maximum(stack(c_all.Sₒ)) > 0 && length(rec_all.g2abs) > 0
        # Energy conservation (#72).
        @test_broken ((c_cut, rec_cut) = runbte(; energy_conservation = (:Fixed, 10σ));
            length(rec_cut.g2abs) < length(rec_all.g2abs) &&
            isapprox(stack(c_cut.Sₒ), stack(c_all.Sₒ); rtol = 1e-12) &&
            isapprox(stack(c_cut.Sᵢ), stack(c_all.Sᵢ); rtol = 1e-12))
    end

    # The covariant derivative of the e-ph matrix: Σ |dg[:, :, :, d]|² of every pair against
    # `eph_reference_dg`, which reuses the per-point loop's Wannier-center correction (not an
    # independent check of that term). Measured 6.3e-16 relative on the per-point loop at
    # `fourier_mode = "normal"` (Pb 3³, full window); at its default `"gridopt"` that loop is 0.82 off.
    @testset "covariant_derivative_of_g" begin
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        grid = (3, 3, 3)
        kpts = GridKpoints(kpoints_grid(grid))
        ref = eph_reference_dg(model, kpts, kpts)
        @test length(ref) == prod(grid)^2 && maximum(maximum, values(ref)) > 0
        # covariant_derivative_of_g (#72).
        @test_broken (rec = _DgRecorder();
            run_eph_over_k_and_kq(model, grid, grid; calculators = [rec], symmetry = nothing,
                covariant_derivative_of_g = true, progress_print_step = 10^9, verbosity = 0);
            keys(rec.sums) == keys(ref) && maximum(maximum(abs, rec.sums[key] - ref[key])
                for key in keys(ref)) < 1e-10 * maximum(maximum, values(ref)))
    end
end
