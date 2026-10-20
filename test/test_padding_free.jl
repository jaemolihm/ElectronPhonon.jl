@testsnippet PaddingFree begin
    using ElectronPhonon: OuterKEngine, OuterQEngine, OuterKLoop, OuterQLoop, EPBlock, OuterKContext, OuterQContext,
        BatchedElectronState, BatchedPhononState, stage1!, stage2!, setup_calculator!, run_calculator!,
        calculator_begin_batch!, calculator_end_batch!, postprocess_calculator!, run_eph_over_k_and_kq,
        run_eph_over_k_and_q, run_eph_over_q_and_k, CPUBackend, gpu_backend, unit_to_aru

    # The rule of the padding-free e-ph engine: no arithmetic on the box padding of a
    # `BatchedElectronState` (local band `n > nband[k]`). Every block must be exact (`nband[j] ==
    # nband_max` on both electron sides), and no calculation may use the padding. Two checks per route:
    # * structural: a recorder asserts the extents of every block it is handed;
    # * poison: the recorder overwrites the padding of the resident states with NaN, and the engine
    #   API run also fills every stage-1 buffer with NaN before `stage1!` and every tile buffer before
    #   each `stage2!`. Every output must then be bitwise the unpoisoned run's, and every block's
    #   `ep` / `dg` finite.
    # A padded result is dropped by an index guard, so the output comparison alone passes on code that
    # computes on the padding; the structural check is what tests exactness. For the k+q states solved
    # per tile, the block's energies are also compared with an independent solve. `BROKEN` lists the
    # assertions the engine does not meet yet.

    const PADDING_FREE_GPU = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end

    # Assertions the engine does not meet yet, as (route, flag); none.
    const BROKEN = Set{Tuple{Symbol, Symbol}}()

    # Overwrite every entry of `els` with `n > nband[k]` (any band index of the field) by NaN.
    function _poison_padding!(els::BatchedElectronState)
        nband = Array(els.nband)
        bandaxes = (e = (1,), u = (2,), vdiag = (2,), v = (2, 3), rbar = (2, 3))
        for name in keys(bandaxes)
            x = getfield(els, name)
            x === nothing && continue
            a = Array(x)
            nan = eltype(a) <: Complex ? complex(NaN, NaN) : NaN
            for idx in CartesianIndices(a)
                k = idx[end]
                any(ax -> idx[ax] > nband[k], bandaxes[name]) && (a[idx] = nan)
            end
            copyto!(x, a)
        end
        els
    end

    # Fill the float arrays of a state container (a tile's gathered states) with NaN.
    function _poison_all!(x)
        nan!(a::AbstractArray{<:Complex}) = fill!(a, complex(NaN, NaN))
        nan!(a::AbstractArray{<:AbstractFloat}) = fill!(a, NaN)
        nan!(a) = nothing
        for name in fieldnames(typeof(x))
            nan!(getfield(x, name))
        end
        x
    end

    # Fill the stage-1 and tile scratch of an engine with NaN: everything stage 2 must write before it
    # reads. The cartesian identity basis `u_ph_id` is data, not scratch.
    function _poison_scratch!(eng; stage1 = true)
        for name in (stage1 ? (:ep_kR, :dg_kR, :g_fourier, :P_mk, :mxks, :g_q, :g_rot, :ep_Rq) : ())
            hasfield(typeof(eng), name) || continue
            x = getfield(eng, name)
            x === nothing || _poison_all!((; x))
        end
        for tile in eng.tiles, name in fieldnames(typeof(tile))
            name === :u_ph_id && continue
            x = getfield(tile, name)
            if x isa BatchedElectronState || x isa BatchedPhononState
                _poison_all!(x)
            else
                _poison_all!((; x))
            end
        end
        eng
    end

    # Records the structure of every block, and optionally poisons the resident padding at setup.
    mutable struct _PaddingRule <: ElectronPhonon.AbstractCalculator
        poison :: Bool
        nblocks :: Int
        k_box :: Bool       # every block's k side has nband[j] == nband_max
        kq_box :: Bool
        ep_shape :: Bool    # size(ep)[1:2] == (els_kq.nband_max, els_k.nband_max)
        finite :: Bool      # every `ep` / `dg` entry is finite
        kq_energies :: Vector{Tuple{ElectronPhonon.Vec3{Float64}, Vector{Float64}}}  # solved k+q: (x_{k+q}, e)
        lock :: ReentrantLock
        _PaddingRule(poison) = new(poison, 0, true, true, true, true, [], ReentrantLock())
    end
    ElectronPhonon.supports(::_PaddingRule, ::Type{<:ElectronPhonon.LoopTag}) = true
    ElectronPhonon.calculator_begin_batch!(::_PaddingRule, ctx) = nothing
    ElectronPhonon.calculator_end_batch!(::_PaddingRule, ctx) = nothing
    ElectronPhonon.postprocess_calculator!(c::_PaddingRule; kwargs...) = c
    function ElectronPhonon.setup_calculator!(c::_PaddingRule, backend, els_k, els_kq, phs; kwargs...)
        if c.poison
            _poison_padding!(els_k)
            els_kq === nothing || _poison_padding!(els_kq)
        end
        c
    end
    function ElectronPhonon.run_calculator!(c::_PaddingRule, tile::EPBlock, ctx)
        nbk, nbkq = Array(tile.els_k.nband), Array(tile.els_kq.nband)
        kq_energies = tile isa EPBlock{OuterQLoop} && tile.ikq === nothing ? Array(tile.els_kq.e) : nothing
        k_box = kq_box = true
        kq_e = Tuple{ElectronPhonon.Vec3{Float64}, Vector{Float64}}[]
        n_items = 0
        for s in ElectronPhonon.eachsegment(tile; ep = Array(tile.ep)), j in s.pairs
            jj = j - first(s.pairs) + 1
            # every pair of a class has exactly the class's bands on both sides
            jk, jkq = s.k_pt + (s.k_shared ? 1 : jj), s.kq_pt + jj
            k_box &= nbk[jk] == s.b_k
            kq_box &= nbkq[jkq] == s.b_kq
            if kq_energies !== nothing
                slot0 = tile.seg_host[ElectronPhonon.SEG_KQ_SLOT, s.c]
                first_slot = slot0 + (jj - 1) * s.b_kq
                push!(kq_e, (tile.xk[j] + (tile.xq isa ElectronPhonon.Vec3 ? tile.xq : tile.xq[j]), kq_energies[first_slot + 1:first_slot + nbkq[jkq]]))
            end
        end
        for s in ElectronPhonon.eachsegment(tile; ep = Array(tile.ep))
            n_items += length(s.items)
        end
        ep_shape = n_items == length(tile.ep) && size(tile.seg_host, 2) >= 1
        finite = all(isfinite, Array(tile.ep)) && (tile.dg === nothing || all(isfinite, Array(tile.dg)))
        lock(c.lock) do
            c.nblocks += 1
            c.k_box &= k_box; c.kq_box &= kq_box; c.ep_shape &= ep_shape; c.finite &= finite
            append!(c.kq_energies, kq_e)
        end
    end

    # A calculator that asks for the pairs' phonon eigenvectors, which an outer-q block does not carry.
    struct _WantsU <: ElectronPhonon.AbstractCalculator end
    ElectronPhonon.supports(::_WantsU, ::Type{OuterQLoop}) = true
    ElectronPhonon.required_ph_quantities(::_WantsU) = [:u]

    # The outputs to compare: the pairs' |g|^2 (and the covariant derivative sums).
    function _check_rule(rule, route; poison)
        @test rule.nblocks > 0
        @test rule.ep_shape
        for (flag, value) in ((:k_box, rule.k_box), (:kq_box, rule.kq_box), (:finite, rule.finite))
            flag === :finite && !poison && continue
            if (route, flag) in BROKEN
                @test_broken value
            else
                @test value
            end
        end
    end

    # The recorded energies at the solved k+q points against `compute_electron_states` at the same points.
    function _kq_energies_match(model, rule, window)
        xs = first.(rule.kq_energies)
        isempty(xs) && return false
        ref = compute_electron_states(model, Kpoints(length(xs), xs, ones(length(xs)), (1, 1, 1)),
            ["eigenvalue"], window; fourier_mode = "normal")
        all(zip(ref, last.(rule.kq_energies))) do (el, e)
            length(e) == el.nband && maximum(abs, e .- el.e_full[el.rng]; init = 0.0) < 1e-9
        end
    end

    function _run_route(route, models, grid, window; backend, poison, dg = false, kwargs...)
        pair, rule = _PairRecorder(), _PaddingRule(poison)
        calcs = Any[pair, rule]
        dgrec = dg ? _DgRecorder() : nothing
        dg && push!(calcs, dgrec)
        # Outer q: tiles of several q (3 per batch, 400 pairs), so a tile holds pairs of more than one q.
        sizes = route in (:k_kq, :k_q) ? (; n_inner_tile = 16) : (; n_inner_tile = 400, n_outer_batch = 3)
        common = (; calculators = calcs, backend, window_k = window, window_kq = window, sizes...,
                  progress_print_step = 10^9, verbosity = 0, symmetry = nothing, kwargs...)
        if route === :k_kq
            run_eph_over_k_and_kq(models.el, grid, grid; common..., covariant_derivative_of_g = dg)
        elseif route === :k_q
            run_eph_over_k_and_q(models.el, grid, grid; common..., covariant_derivative_of_g = dg)
        else
            run_eph_over_q_and_k(models.ph, grid, grid; common..., precompute_el_kq = route === :q_k_pre)
        end
        (; pair, rule, dgrec)
    end

    # The same route through the engine API: stage 1 and stage 2 over every outer point and tile.
    function _run_engine(route, models, grid, window; backend, poison, dg = false, kwargs...)
        pair, rule = _PairRecorder(), _PaddingRule(poison)
        calcs = Any[pair, rule]
        dgrec = dg ? _DgRecorder() : nothing
        dg && push!(calcs, dgrec)
        outer_k = route in (:k_kq, :k_q)
        Engine = outer_k ? OuterKEngine : OuterQEngine
        extra = route === :k_kq ? (; inner_loop_kq = true, covariant_derivative_of_g = dg) :
                route === :k_q ? (; covariant_derivative_of_g = dg) : (; precompute_el_kq = route === :q_k_pre)
        sizes = outer_k ? (; n_outer_batch = 2, n_inner_tile = 16) : (; n_outer_batch = 3, n_inner_tile = 400)
        eng = Engine(outer_k ? models.el : models.ph, grid, grid; backend, calculators = calcs, window_k = window,
            window_kq = window, sizes..., verbosity = 0, extra..., kwargs...)
        for c in calcs
            setup_calculator!(c, eng.backend, eng.els_k, eng.els_kq, eng.phs;
                order = outer_k ? OuterKLoop() : OuterQLoop(), eng.sel_k, eng.sel_kq, nchunks_threads = 1,
                eng.n_outer_batch, eng.n_inner_tile, verbosity = 0)
        end
        poison && _poison_scratch!(eng)
        outer_pts, inner_pts = outer_k ? (eng.kpts, outer_k && route === :k_kq ? eng.kqpts : eng.qpts) :
                                         (eng.qpts, eng.kpts)
        for batch in Iterators.partition(1:outer_pts.n, eng.n_outer_batch)
            stage1!(eng, batch)
            ctx = outer_k ? OuterKContext(eng) : OuterQContext(eng)
            for c in calcs
                calculator_begin_batch!(c, ctx)
            end
            # Outer k: one k per tile. Outer q: every q of the batch in one tile, over k ranges that fit.
            nk_tile = outer_k ? eng.n_inner_tile : eng.n_inner_tile ÷ length(batch)
            for iout in (outer_k ? batch : (batch,)), tile in Iterators.partition(1:inner_pts.n, nk_tile)
                poison && _poison_scratch!(eng; stage1 = false)
                tl = stage2!(eng, iout, tile)
                tl === nothing && continue
                for c in calcs
                    run_calculator!(c, tl, ctx)
                end
            end
            for c in calcs
                calculator_end_batch!(c, ctx)
            end
        end
        (; pair, rule, dgrec)
    end

    function _same_outputs(a, b)
        a.pair.g2abs == b.pair.g2abs && a.pair.ωq == b.pair.ωq &&
            (a.dgrec === nothing || a.dgrec.sums == b.dgrec.sums)
    end

    # The same pairs, equal to `rtol` of the largest |g|^2 and of the largest frequency: two tilings or
    # thread partitions differ by GEMM and eigensolve round-off only. `|g|^2` per entry is gauge dependent inside
    # a degenerate band or phonon multiplet (a different eigensolve round-off picks another basis), so it is
    # compared on multiplet sums, as `compare_with_reference` does. Returns the first failing part, or `nothing`.
    function _close_failure(a, b; rtol = 1e-12, tol_degen = 1e-6)
        keys(a.pair.g2abs) == keys(b.pair.g2abs) || return :pairs
        scale = maximum(maximum, values(a.pair.g2abs))
        ωscale = maximum(maximum, values(a.pair.ωq))
        for k in keys(a.pair.g2abs)
            maximum(abs, a.pair.ωq[k] .- b.pair.ωq[k]) <= rtol * ωscale || return :ω
            # the band multiplets of the window bands, from the first run's energies
            nn, mm = findall(!isnan, a.pair.e_k[k]), findall(!isnan, a.pair.e_kq[k])
            for gm in contract_multiplets(a.pair.e_kq[k][mm], tol_degen), gn in contract_multiplets(a.pair.e_k[k][nn], tol_degen),
                    gν in contract_multiplets(a.pair.ωq[k], tol_degen)
                sa = sum(a.pair.g2abs[k][mm[gm], nn[gn], gν]); sb = sum(b.pair.g2abs[k][mm[gm], nn[gn], gν])
                abs(sa - sb) <= rtol * scale || return :g2
            end
        end
        nothing
    end
end

@testitem "padding-free e-ph engine" setup = [Common, PaddingFree, EphReferenceLoop] begin
    eV = unit_to_aru(:eV)
    pb = (el = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el"),
          ph = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph"))
    bn = (el = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "el"),
          ph = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "ph"))
    @test bn.el.polar_eph.use
    # Ragged windows: the band counts of the k and the k+q points differ.
    fixtures = [(:pb, pb, (6, 6, 6), (11.68eV - 0.5eV, 11.68eV + 3eV)),
                (:cubicBN, bn, (3, 3, 3), (2.0eV, 5.0eV))]
    backends = PADDING_FREE_GPU ? Any[CPUBackend(), gpu_backend()] : Any[CPUBackend()]
    for backend in backends, (name, models, grid, window) in fixtures,
            route in (:k_kq, :k_q, :q_k_pre, :q_k_solved)
        @testset "$(nameof(typeof(backend))) $name $route" begin
            ref = _run_route(route, models, grid, window; backend, poison = false)
            _check_rule(ref.rule, route; poison = false)
            run = _run_route(route, models, grid, window; backend, poison = true)
            _check_rule(run.rule, route; poison = true)
            @test _same_outputs(ref, run)
            # Outer q: the multi-q tiles against one q per tile (the shape of a tile without the q grouping).
            if route in (:q_k_pre, :q_k_solved)
                one_q = _run_route(route, models, grid, window; backend, poison = false, n_inner_tile = 16, n_outer_batch = 1)
                @test _close_failure(ref, one_q) === nothing
            end
            # Through the engine API, with the tile and stage-1 scratch poisoned as well.
            eref = _run_engine(route, models, grid, window; backend, poison = false)
            erun = _run_engine(route, models, grid, window; backend, poison = true)
            _check_rule(erun.rule, route; poison = true)
            if route === :q_k_solved
                @test _kq_energies_match(models.el, run.rule, window)
                @test _kq_energies_match(models.el, erun.rule, window)
            end
            @test _same_outputs(eref, erun)
            @test _close_failure(ref, eref) === nothing
        end
    end
    # Polar and covariant derivative, outer k.
    for backend in backends, route in (:k_kq, :k_q)
        @testset "$(nameof(typeof(backend))) cubicBN $route dg" begin
            ref = _run_route(route, bn, (3, 3, 3), (2.0eV, 5.0eV); backend, poison = false, dg = true)
            run = _run_route(route, bn, (3, 3, 3), (2.0eV, 5.0eV); backend, poison = true, dg = true)
            _check_rule(run.rule, route; poison = true)
            @test !isempty(run.dgrec.sums) && _same_outputs(ref, run)
            erun = _run_engine(route, bn, (3, 3, 3), (2.0eV, 5.0eV); backend, poison = true, dg = true)
            @test _close_failure(ref, erun) === nothing
        end
    end
    # A finite energy-conservation tolerance on the inner k+q route (selected on the device on a GPU).
    for backend in backends
        @testset "$(nameof(typeof(backend))) pb k_kq finite tolerance" begin
            kw = (; energy_conservation_tol = 0.1eV)
            ref = _run_route(:k_kq, pb, (6, 6, 6), fixtures[1][4]; backend, poison = false, kw...)
            run = _run_route(:k_kq, pb, (6, 6, 6), fixtures[1][4]; backend, poison = true, kw...)
            _check_rule(run.rule, :k_kq; poison = true)
            @test _same_outputs(ref, run)
        end
    end
    # Outer q refuses a calculator that lists :u in required_ph_quantities, at the start of the run.
    @test_throws "eigenvectors" run_eph_over_q_and_k(pb.ph, (2, 2, 2), (2, 2, 2); calculators = [_WantsU()],
        symmetry = nothing, verbosity = 0)
end
