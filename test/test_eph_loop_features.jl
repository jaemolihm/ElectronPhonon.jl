using Test
using ElectronPhonon
using ElectronPhonon: unit_to_aru, run_eph_over_k_and_kq, run_eph_over_q_and_k, CPUBackend, gpu_backend

# Features the per-point loops of the previous release had, on the e-ph loop (ElectronPhonon.jl
# issue #72): the outer-k polar term, energy conservation and the covariant derivative. Each
# assertion was checked to hold on the per-point loop of the previous release.

const EPH_FEATURES_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")
isdefined(@__MODULE__, :eph_reference) || include("eph_reference_loop.jl")

# Records Σ |dg[:, :, :, d]|² of every pair from the `dg` of the blocks (in-window bands, all modes).
# The Dict is guarded by a lock: blocks of different thread chunks run concurrently.
mutable struct _DgRecorder <: ElectronPhonon.AbstractCalculator
    ngrid :: NTuple{3, Int}
    sums :: Dict{NTuple{6, Int}, Vector{Float64}}
    lock :: ReentrantLock
    _DgRecorder() = new((0, 0, 0), Dict(), ReentrantLock())
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
        s = [sum(abs2, dg[1:nbkq[j], 1:nbk, :, d, j]) for d in 1:3]
        lock(() -> c.sums[_pair_key(block.xk, block.xk + xq, c.ngrid)] = s, c.lock)
    end
end

@testset "outer-k polar, energy conservation, covariant derivative (#72)" begin
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
        backends = EPH_FEATURES_GPU_AVAILABLE ? Any[CPUBackend(), gpu_backend()] : Any[CPUBackend()]
        for backend in backends
            @test (rec_k = _PairRecorder();
                run_eph_over_k_and_kq(model, grid, grid; calculators = [rec_k], symmetry = nothing,
                    backend, progress_print_step = 10^9, verbosity = 0);
                isapprox(total(rec_k), total(rec_q); rtol = 1e-12))
        end
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
        @test ((c_cut, rec_cut) = runbte(; energy_conservation = (:Fixed, 10σ));
            length(rec_cut.g2abs) < length(rec_all.g2abs) &&
            isapprox(stack(c_cut.Sₒ), stack(c_all.Sₒ); rtol = 1e-12) &&
            isapprox(stack(c_cut.Sᵢ), stack(c_all.Sᵢ); rtol = 1e-12))
    end

    # Energy conservation on both orders, outer batch > 1, against the reference double loop with
    # the same `(:Fixed, tol)` filter applied to its pairs: the kept pairs and their |g|^2.
    @testset "energy conservation, both orders" begin
        model_el = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        model_ph = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
        eV = unit_to_aru(:eV)
        μ = 11.68eV; window = (μ - 0.5eV, μ + 3eV); tol = 0.01eV
        grid = (6, 6, 6)
        kpts = GridKpoints(kpoints_grid(grid))
        ref = eph_reference(model_el, kpts, kpts, window, window)
        conserves(key) = (elk = ref.el_k[key[1:3]]; elkq = ref.el_kq[key[4:6]];
            any(abs(elk.e[n] - elkq.e[m] - s * ω) <= tol for n in elk.rng, m in elkq.rng,
                ω in ref.ωq[key], s in (-1, 1)))
        ref_cut = (; g2abs = filter(kv -> conserves(kv[1]), ref.g2abs), ref.ωq, ref.el_k, ref.el_kq)
        @test 0 < length(ref_cut.g2abs) < length(ref.g2abs)
        common = (; window_k = window, window_kq = window, energy_conservation = (:Fixed, tol),
                  n_outer_batch = 7, n_inner_tile = 30, nchunks_threads = 4, verbosity = 0)
        for order in ("outer k", "outer q")
            rec = _PairRecorder()
            order == "outer k" ?
                run_eph_over_k_and_kq(model_el, grid, grid; calculators = [rec], symmetry = nothing, common...) :
                run_eph_over_q_and_k(model_ph, grid, grid; calculators = [rec], use_symmetry = false, common...)
            dev = compare_with_reference(ref_cut, rec)
            @test keys(rec.g2abs) == keys(ref_cut.g2abs)
            @test dev.g2_reldev < 1e-11
        end
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
        backends = EPH_FEATURES_GPU_AVAILABLE ? Any[CPUBackend(), gpu_backend()] : Any[CPUBackend()]
        for backend in backends
            @test (rec = _DgRecorder();
                run_eph_over_k_and_kq(model, grid, grid; calculators = [rec], symmetry = nothing,
                    backend, n_outer_batch = 5, covariant_derivative_of_g = true,
                    progress_print_step = 10^9, verbosity = 0);
                keys(rec.sums) == keys(ref) && maximum(maximum(abs, rec.sums[key] - ref[key])
                    for key in keys(ref)) < 1e-10 * maximum(maximum, values(ref)))
        end
    end
end
