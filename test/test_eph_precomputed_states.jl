using Test
using ElectronPhonon
using ElectronPhonon: electron_eigenpairs, gpu_backend, CPUBackend, AbstractBackend, xk_to_ik,
    xk_to_ik_unsafe, AbstractCalculator, OuterKLoop, OuterQLoop, EPData, EPDataQBatched,
    OuterIteration, OuterIterationBatch
using OffsetArrays: no_offset_view

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const PRECOMPUTED_STATES_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# The batched e-ph loop requires at least one calculator, and the three drivers hand out different
# payloads, so this probe declares every loop shape and payload they use. It records nothing: what
# this file asserts is the electron states a driver's setup produces, not the coupling its loop
# computes.
struct _PrecomputedStatesProbe <: AbstractCalculator end
ElectronPhonon.supports(::_PrecomputedStatesProbe, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_PrecomputedStatesProbe, ::Type{OuterQLoop}) = true
ElectronPhonon.supports(::_PrecomputedStatesProbe, ::Type{EPData}) = true
ElectronPhonon.supports(::_PrecomputedStatesProbe, ::Type{EPDataQBatched}) = true
ElectronPhonon.calculator_begin!(c::_PrecomputedStatesProbe, ::OuterIteration, ctx) = c
ElectronPhonon.calculator_end!(c::_PrecomputedStatesProbe, ::OuterIteration, ctx) = c
ElectronPhonon.calculator_begin!(c::_PrecomputedStatesProbe, ::OuterIterationBatch, ctx) = c
ElectronPhonon.calculator_end!(c::_PrecomputedStatesProbe, ::OuterIterationBatch, ctx) = c
ElectronPhonon.setup_calculator!(c::_PrecomputedStatesProbe, backend, mode, kpts, qpts, el_states;
                                 kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_PrecomputedStatesProbe; kwargs...) = c
ElectronPhonon.run_calculator!(c::_PrecomputedStatesProbe, ::EPData, ctx) = c
ElectronPhonon.run_calculator!(c::_PrecomputedStatesProbe, ::EPDataQBatched, ctx) = c

# Every field a run fills, compared with `==`: `v`/`rbar` are window views, so take them through
# `no_offset_view` (their axes are covered by `rng`). Same comparison as `_eigenpairs_state_equal`
# in test_electron_eigenpairs.jl, repeated here so this file runs on its own.
function _driver_state_equal(a::ElectronState, b::ElectronState)
    a.xk == b.xk && a.e_full == b.e_full && a.u_full == b.u_full && a.nband == b.nband &&
        a.rng == b.rng && a.vdiag == b.vdiag &&
        no_offset_view(a.v) == no_offset_view(b.v) &&
        no_offset_view(a.rbar) == no_offset_view(b.rbar)
end

# Whether two runs agree on `u_full` at every outer k point they both visit, and how many those
# are (pinned by the caller so the comparison cannot be vacuous).
function _shared_outer_u_agree(run_a, run_b)
    ok, nshared = true, 0
    for (ika, xk) in enumerate(run_a.kpts.vectors)
        # `xk` is a node of both runs' grid but need not be in `run_b`'s selection, so a miss is a
        # legitimate answer here and the unchecked lookup is the right one.
        ikb = xk_to_ik_unsafe(xk, run_b.kpts)
        ikb === nothing && continue
        nshared += 1
        ok &= run_a.el_k_save[ika].u_full == run_b.el_k_save[ikb].u_full
    end
    (ok, nshared)
end

# Largest `u_full` deviation between two runs over the outer k points they share.
function _shared_outer_u_deviation(run_a, run_b)
    d = 0.0
    for (ika, xk) in enumerate(run_a.kpts.vectors)
        ikb = xk_to_ik_unsafe(xk, run_b.kpts)
        ikb === nothing && continue
        d = max(d, maximum(abs, run_a.el_k_save[ika].u_full - run_b.el_k_save[ikb].u_full))
    end
    d
end

@testset "e-ph drivers with precomputed electron states" begin
    grid = (4, 4, 4)
    kgrid = GridKpoints(kpoints_grid(grid))
    # Two outer k selections that differ but overlap: 24 of the 64 k points are in both.
    subset(iks) = GridKpoints(Kpoints(kgrid.vectors[iks]; ngrid = kgrid.ngrid), kgrid.ngrid)
    sub_a, sub_b = subset(1:48), subset(25:64)

    @testset "run_eph_over_k_and_kq" begin
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        _run(kpts_in; kwargs...) = ElectronPhonon.run_eph_over_k_and_kq(model, kpts_in, grid;
            calculators = [_PrecomputedStatesProbe()], symmetry = nothing,
            progress_print_step = 10^9, verbosity = 0, kwargs...)

        arms = Tuple{AbstractBackend, String}[(CPUBackend(), "gridopt")]
        PRECOMPUTED_STATES_GPU_AVAILABLE && push!(arms, (gpu_backend(), "gridopt"))
        for (backend, fourier_mode) in arms
            cache = electron_eigenpairs(model, kgrid; backend, fourier_mode)
            run_a = _run(sub_a; backend, fourier_mode)
            run_a_cached = _run(sub_a; backend, fourier_mode,
                                el_k_eigenpairs = cache, el_kq_eigenpairs = cache)
            run_b_cached = _run(sub_b; backend, fourier_mode,
                                el_k_eigenpairs = cache, el_kq_eigenpairs = cache)

            # A cache is inert: it holds the same eigensolve output on the same H(k), so a
            # with-cache run reproduces the without-cache one bit for bit on both sides. Both arms
            # use the same `fourier_mode` -- H(k) is not bitwise equal between Fourier modes, and
            # inside a degenerate multiplet that difference is O(1) in `u`.
            @test all(_driver_state_equal.(run_a_cached.el_k_save, run_a.el_k_save))
            @test all(_driver_state_equal.(run_a_cached.el_kq_save, run_a.el_kq_save))

            # Every state a cached run produces carries the cache's own eigenpair, on the outer k
            # side and the inner k+q side alike. That is what makes the shared gauge structural:
            # two runs over different outer sets are anchored to the same `u` at every k point.
            e_cache, u_cache = Array(cache.e_full), Array(cache.u_full)
            from_cache(el) = (ik = xk_to_ik(el.xk, cache.kpts);
                              el.e_full == e_cache[:, ik] && el.u_full == u_cache[:, :, ik])
            @test all(from_cache, run_a_cached.el_k_save)
            @test all(from_cache, run_a_cached.el_kq_save)
            @test all(from_cache, run_b_cached.el_k_save)
            # Negative control: the claim must fail against a one-k offset in the cache, so a run
            # that mapped its k points wrongly could not leave it passing.
            @test !any(run_a_cached.el_k_save) do el
                ik = xk_to_ik(el.xk, cache.kpts)
                el.u_full == u_cache[:, :, mod1(ik + 1, cache.kpts.n)]
            end

            # The deliverable: the two runs agree on the eigenvector gauge at every shared k point.
            @test _shared_outer_u_agree(run_a_cached, run_b_cached) == (true, 24)

            # Recorded, not asserted: on Pb the two runs already agree without a cache, on both
            # backends -- H(k) here comes out bitwise independent of which other k points the run
            # visits, so the cache is a guarantee rather than a fix on this fixture. It is what
            # makes the agreement structural on a system with pervasive degeneracy.
            @info "shared-k u_full deviation without a cache" backend nk_shared = 24 deviation =
                _shared_outer_u_deviation(run_a, _run(sub_b; backend, fourier_mode))

            # A cache that does not cover every k point a run visits is an error, not a silent
            # recompute -- asserted once per side, so each kwarg's forwarding is covered on its own.
            partial = electron_eigenpairs(model, subset(1:60); backend, fourier_mode)
            @test_throws "is not one of its" _run(sub_b; backend, fourier_mode,
                                                 el_k_eigenpairs = partial)
            @test_throws "is not one of its" _run(sub_a; backend, fourier_mode,
                                                 el_kq_eigenpairs = partial)
        end

        # The eigenvectors really come from the cache: one built with the other Fourier mode
        # differs from what the run would have computed by O(1) inside a degenerate multiplet, so a
        # run that diagonalized H(k) itself could not reproduce it. CPU only -- the device path has
        # only the batched interpolator, so `fourier_mode` does not reach its eigensolve.
        cache_normal = electron_eigenpairs(model, kgrid; fourier_mode = "normal")
        run_gridopt = _run(sub_a; fourier_mode = "gridopt")
        run_normal_cache = _run(sub_a; fourier_mode = "gridopt",
                                el_k_eigenpairs = cache_normal, el_kq_eigenpairs = cache_normal)
        u_deviation(a, b) = maximum(eachindex(a)) do ik
            maximum(abs, a[ik].u_full - b[ik].u_full)
        end
        @test u_deviation(run_gridopt.el_k_save, run_normal_cache.el_k_save) > 1
        @test u_deviation(run_gridopt.el_kq_save, run_normal_cache.el_kq_save) > 1
    end

    # The k side is one edit in `_setup_electron_k`, shared by all three drivers, so the two
    # siblings need only the kwarg forward covered. Their k+q states are deliberately not wired.
    @testset "sibling drivers, k side" begin
        for (outer_momentum, driver) in (("el", ElectronPhonon.run_eph_over_k_and_q),
                                         ("ph", ElectronPhonon.run_eph_over_q_and_k))
            model = _load_model_from_artifacts("pb"; epmat_outer_momentum = outer_momentum)
            # `run_eph_over_k_and_q` takes `symmetry`, `run_eph_over_q_and_k` takes `use_symmetry`;
            # both default to the model's symmetry, which the k+q grid path here must not reduce.
            sym_kwargs = outer_momentum == "el" ? (; symmetry = nothing) :
                (; use_symmetry = false, keep_all_qpts = true)
            _run(; kwargs...) = driver(model, sub_a, grid;
                calculators = [_PrecomputedStatesProbe()], fourier_mode = "gridopt",
                progress_print_step = 10^9, verbosity = 0, sym_kwargs..., kwargs...)

            run_plain = _run()
            cache = electron_eigenpairs(model, kgrid; fourier_mode = "gridopt")
            @test all(_driver_state_equal.(_run(el_k_eigenpairs = cache).el_k_save,
                                           run_plain.el_k_save))
            # Same teeth as above: a cache built with the other Fourier mode must change `u`.
            run_normal_cache = _run(el_k_eigenpairs =
                electron_eigenpairs(model, kgrid; fourier_mode = "normal"))
            @test maximum(eachindex(run_plain.el_k_save)) do ik
                maximum(abs, run_plain.el_k_save[ik].u_full -
                             run_normal_cache.el_k_save[ik].u_full)
            end > 1
        end
    end
end
