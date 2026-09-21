using Test
using ElectronPhonon
using ElectronPhonon: electron_eigenpairs, gpu_backend, CPUBackend, AbstractBackend, xk_to_ik,
    xk_to_ik_unsafe, AbstractCalculator, OuterKLoop, OuterQLoop, EPData, EPDataQBatched,
    OuterIteration, OuterIterationBatch

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
ElectronPhonon.calculator_begin!(::_PrecomputedStatesProbe, ::OuterIteration, ctx) = nothing
ElectronPhonon.calculator_end!(::_PrecomputedStatesProbe, ::OuterIteration, ctx) = nothing
ElectronPhonon.calculator_begin!(::_PrecomputedStatesProbe, ::OuterIterationBatch, ctx) = nothing
ElectronPhonon.calculator_end!(::_PrecomputedStatesProbe, ::OuterIterationBatch, ctx) = nothing
ElectronPhonon.setup_calculator!(c::_PrecomputedStatesProbe, backend, mode, kpts, qpts, el_states;
                                 kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_PrecomputedStatesProbe; kwargs...) = c
ElectronPhonon.run_calculator!(c::_PrecomputedStatesProbe, ::EPData, ctx) = c
ElectronPhonon.run_calculator!(c::_PrecomputedStatesProbe, ::EPDataQBatched, ctx) = c

# Largest deviation between two runs' states, position by position (the two runs must share a
# k-point list). `_electron_state_equal` lives in common_models_from_artifacts.jl.
_u_deviation(a, b) = maximum(ik -> maximum(abs, a[ik].u_full - b[ik].u_full), eachindex(a))
_e_deviation(a, b) = maximum(ik -> maximum(abs, a[ik].e_full - b[ik].e_full), eachindex(a))

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

@testset "e-ph drivers with precomputed electron and phonon states" begin
    grid = (4, 4, 4)
    kgrid = GridKpoints(kpoints_grid(grid))
    # Two outer k selections that differ but overlap: 24 of the 64 k points are in both.
    subset(iks) = GridKpoints(Kpoints(kgrid.vectors[iks]; ngrid = kgrid.ngrid), kgrid.ngrid)
    sub_a, sub_b = subset(1:48), subset(25:64)

    @testset "run_eph_over_k_and_kq" begin
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        _run(kpts_in, kqpts_in = grid; kwargs...) = ElectronPhonon.run_eph_over_k_and_kq(
            model, kpts_in, kqpts_in; calculators = [_PrecomputedStatesProbe()],
            symmetry = nothing, progress_print_step = 10^9, verbosity = 0, kwargs...)

        arms = Tuple{AbstractBackend, String}[(CPUBackend(), "gridopt")]
        PRECOMPUTED_STATES_GPU_AVAILABLE && push!(arms, (gpu_backend(), "gridopt"))
        for (backend, fourier_mode) in arms
            cache = electron_eigenpairs(model, kgrid; backend, fourier_mode)
            # The same eigenpairs rotated by one k point: a cache that is well-formed, on the right
            # backend and covers every k point, but holds the wrong eigenvector at each of them.
            # This is what gives the assertions below teeth on *either* backend -- a run that
            # ignored the cache and diagonalized H(k) itself would be unaffected by it. (A cache
            # built at the other `fourier_mode` only works as a tooth on CPU: both
            # `electron_eigenpairs`' device branch and the device eigensolve hardcode
            # `fourier_mode = "batched"`.)
            perm = [mod1(ik + 1, kgrid.n) for ik in 1:kgrid.n]
            bad = Eigenpairs(model.nw, kgrid, cache.e_full[:, perm],
                                     cache.u_full[:, :, perm])

            run_a = _run(sub_a; backend, fourier_mode)
            run_a_cached = _run(sub_a; backend, fourier_mode,
                                el_k_eigenpairs = cache, el_kq_eigenpairs = cache)
            run_b_cached = _run(sub_b; backend, fourier_mode,
                                el_k_eigenpairs = cache, el_kq_eigenpairs = cache)
            run_a_bad = _run(sub_a; backend, fourier_mode,
                             el_k_eigenpairs = bad, el_kq_eigenpairs = bad)

            # A cache is inert: it holds the same eigensolve output on the same H(k), so a
            # with-cache run reproduces the without-cache one bit for bit on both sides. Both arms
            # use the same `fourier_mode` -- H(k) is not bitwise equal between Fourier modes, and
            # inside a degenerate multiplet that difference is O(1) in `u`.
            @test all(_electron_state_equal.(run_a_cached.el_k_save, run_a.el_k_save))
            @test all(_electron_state_equal.(run_a_cached.el_kq_save, run_a.el_kq_save))

            # Teeth for that claim, and for every claim below it: the wrong cache must change what
            # the run produces, on the outer k side and on the inner k+q side.
            @test _u_deviation(run_a.el_k_save, run_a_bad.el_k_save) > 1
            @test _u_deviation(run_a.el_kq_save, run_a_bad.el_kq_save) > 1
            @test _e_deviation(run_a.el_k_save, run_a_bad.el_k_save) > 0.1
            @test _e_deviation(run_a.el_kq_save, run_a_bad.el_kq_save) > 0.1

            # Every state a cached run produces carries the cache's own eigenpair, on the outer k
            # side and the inner k+q side alike. That is what makes the shared gauge structural:
            # two runs over different outer sets are anchored to the same `u` at every k point.
            e_cache, u_cache = Array(cache.e_full), Array(cache.u_full)
            from_cache(el) = (ik = xk_to_ik(el.xk, cache.kpts);
                              el.e_full == e_cache[:, ik] && el.u_full == u_cache[:, :, ik])
            @test all(from_cache, run_a_cached.el_k_save)
            @test all(from_cache, run_a_cached.el_kq_save)
            @test all(from_cache, run_b_cached.el_k_save)
            @test !any(from_cache, run_a_bad.el_k_save)

            # The deliverable: the two runs agree on the eigenvector gauge at every shared k point.
            # The pinned 24 is a vacuity guard on the overlap, not a physical claim. Note the
            # agreement itself already holds on Pb *without* a cache -- H(k) here comes out bitwise
            # independent of which other k points a run visits, on both backends -- so what the
            # cache buys on this fixture is the guarantee, measured by the wrong-cache teeth above.
            # On a system with pervasive degeneracy it is what makes the agreement structural.
            @test _shared_outer_u_agree(run_a_cached, run_b_cached) == (true, 24)
            @test _shared_outer_u_agree(run_a_bad, run_b_cached) == (false, 24)

            # A cache that does not cover every k point a run visits is an error, not a silent
            # recompute -- asserted once per side, so each kwarg's forwarding is covered on its
            # own. The `ArgumentError` reaches the caller wrapped by the threaded state loop, so
            # match the message rather than the exception type. It must name the k point on both
            # backends: unguarded, a device cache would instead fail inside the indexing kernel,
            # as a bare `KernelException`.
            partial = electron_eigenpairs(model, subset(1:60); backend, fourier_mode)
            @test_throws "does not cover" _run(sub_b; backend, fourier_mode,
                                               el_k_eigenpairs = partial)
            @test_throws "does not cover" _run(sub_a; backend, fourier_mode,
                                               el_kq_eigenpairs = partial)

            # A prebuilt k+q `FilteredBandStates` takes its own early return in
            # `_setup_electron_kq`, which the grid runs above never enter -- and it is the branch
            # the cross driver uses, since it passes prebuilt selections for both positionals.
            sel_kq = ElectronPhonon.filter_electron_states(grid, model.nw, model.el_ham,
                                                           (-Inf, Inf); fourier_mode)
            prebuilt = _run(sub_a, sel_kq; backend, fourier_mode)
            prebuilt_cached = _run(sub_a, sel_kq; backend, fourier_mode,
                                   el_kq_eigenpairs = cache)
            prebuilt_bad = _run(sub_a, sel_kq; backend, fourier_mode, el_kq_eigenpairs = bad)
            @test all(_electron_state_equal.(prebuilt_cached.el_kq_save, prebuilt.el_kq_save))
            @test _u_deviation(prebuilt.el_kq_save, prebuilt_bad.el_kq_save) > 1
        end

        # `el_kq_eigenpairs` also composes with `el_kq_from_unfolding = true` (the cache is looked
        # up at the irreducible points and the unfolding rotation carries its gauge into the star),
        # and the driver's docstring says so, but that combination is NOT exercised anywhere in
        # this repo: `unfold_ElectronStates` reads `model.el_sym.operators`, and `el_sym` is
        # `nothing` for every test artifact model, so the path throws with or without a cache.
        # test/test_unfold.jl is commented out of runtests.jl for the same reason.
        @test _load_model_from_artifacts("pb"; load_epmat = false).el_sym === nothing

        # A cache built with the other Fourier mode is a second, independent tooth on CPU: it
        # differs from what the run would have computed by O(1) inside a degenerate multiplet.
        cache_normal = electron_eigenpairs(model, kgrid; fourier_mode = "normal")
        run_gridopt = _run(sub_a; fourier_mode = "gridopt")
        run_normal_cache = _run(sub_a; fourier_mode = "gridopt",
                                el_k_eigenpairs = cache_normal, el_kq_eigenpairs = cache_normal)
        @test _u_deviation(run_gridopt.el_k_save, run_normal_cache.el_k_save) > 1
        @test _u_deviation(run_gridopt.el_kq_save, run_normal_cache.el_kq_save) > 1
    end

    # The phonon cache is `run_eph_over_k_and_kq`'s alone: it is the driver whose q-point set is
    # derived (`combine_kpoint_grids`) rather than given, so a caller can only reach those phonons
    # through this kwarg. What it buys is a second run over the same q points inheriting the first
    # run's eigenmode basis, which `g` depends on inside a degenerate multiplet.
    @testset "run_eph_over_k_and_kq, phonon side" begin
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        _run(kpts_in, kqpts_in = grid; kwargs...) = ElectronPhonon.run_eph_over_k_and_kq(
            model, kpts_in, kqpts_in; calculators = [_PrecomputedStatesProbe()],
            symmetry = nothing, progress_print_step = 10^9, verbosity = 0,
            fourier_mode = "gridopt", kwargs...)

        backends = AbstractBackend[CPUBackend()]
        PRECOMPUTED_STATES_GPU_AVAILABLE && push!(backends, gpu_backend())
        for backend in backends
            plain = _run(sub_a; backend)
            qpts = plain.qpts
            cache = _phonon_eigenpairs(plain.ph_save, qpts, backend)
            rot = circshift(1:qpts.n, 1)
            rotated = Eigenpairs(model.nmodes, qpts, cache.e_full[:, rot], cache.u_full[:, :, rot])

            cached = _run(sub_a; backend, ph_eigenpairs = cache)
            bad = _run(sub_a; backend, ph_eigenpairs = rotated)
            @test qpts.n > 1                      # so the rotation is a real permutation
            @test all(_phonon_state_equal.(cached.ph_save, plain.ph_save))
            @test !any(_phonon_state_equal.(bad.ph_save, plain.ph_save))

            # A cache over the wrong q-point set is an error, not a silent recompute. The driver
            # derives its q points from the two k grids, so a caller cannot check the coverage
            # itself -- this is the only thing standing between it and a wrong basis.
            sub_q = GridKpoints(Kpoints(qpts.vectors[1:qpts.n-1]; ngrid = qpts.ngrid), qpts.ngrid)
            partial = Eigenpairs(model.nmodes, sub_q, cache.e_full[:, 1:qpts.n-1],
                                 cache.u_full[:, :, 1:qpts.n-1])
            @test_throws "does not cover" _run(sub_a; backend, ph_eigenpairs = partial)

            # An electron cache carries `nbasis = nw`, so the wrong species is caught at the entry.
            @test_throws "eigenpairs holds nbasis" _run(sub_a; backend,
                ph_eigenpairs = electron_eigenpairs(model, kgrid; backend,
                                                    fourier_mode = "gridopt"))
        end

        # On incommensurate k / k+q grids there is no q-point set at all -- the phonons are solved
        # per (k, q) inside the loop -- so a cache cannot be honoured and is refused rather than
        # ignored.
        @test_throws "requires commensurate" _run((2, 2, 2), (3, 3, 3);
            ph_eigenpairs = Eigenpairs(model.nmodes, GridKpoints(kpoints_grid((2, 2, 2))),
                                       zeros(model.nmodes, 8),
                                       zeros(ComplexF64, model.nmodes, model.nmodes, 8)))
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
            @test all(_electron_state_equal.(_run(el_k_eigenpairs = cache).el_k_save,
                                             run_plain.el_k_save))
            # Same wrong-cache tooth as above, so the inertness claim cannot pass vacuously.
            perm = [mod1(ik + 1, kgrid.n) for ik in 1:kgrid.n]
            bad = Eigenpairs(model.nw, kgrid, cache.e_full[:, perm],
                                     cache.u_full[:, :, perm])
            @test _u_deviation(run_plain.el_k_save,
                               _run(el_k_eigenpairs = bad).el_k_save) > 1
        end
    end
end
