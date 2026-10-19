@testsnippet EngineBytes begin
    using ElectronPhonon: OuterKLoop, OuterQLoop, OuterKEngine, OuterQEngine, engine_bytes, stage1!,
        _run_options, _setup_states, unit_to_aru, _select_qs_per_outer_batch, _phonon_state_bytes

    const ENGINE_BYTES_GPU = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end

    # `CUDA.@allocated f()`, expanded at run time: the macro needs `CUDA` when the file is lowered, which
    # fails where CUDA is not installed (CI) even though the GPU branch never runs there.
    _device_allocated(f) = @eval CUDA.@allocated $f()

    # Bytes of the device memory reachable from `x` (fields, tuples, vectors). A view of a CuArray is a
    # CuArray at the same pointer, so a buffer is counted once, at the largest size it is reached with.
    function _collect_device_buffers!(buffers, seen, x)
        (isbits(x) || x isa Module || x isa DataType || haskey(seen, x)) && return buffers
        seen[x] = nothing
        if x isa CuArray
            buffers[UInt(pointer(x))] = max(get(buffers, UInt(pointer(x)), 0), sizeof(x))
        elseif x isa Array
            eltype(x) <: Number || foreach(y -> _collect_device_buffers!(buffers, seen, y), x)
        elseif x isa AbstractArray
            _collect_device_buffers!(buffers, seen, parent(x))
        else
            for i in 1:nfields(x)
                isdefined(x, i) && _collect_device_buffers!(buffers, seen, getfield(x, i))
            end
        end
        buffers
    end
    _device_bytes(x) = sum(values(_collect_device_buffers!(Dict{UInt, Int}(), IdDict{Any, Nothing}(), x)); init = 0)
end

# `engine_bytes` is what `plan_batch` sizes the inner tile from, before the engine exists: it must
# cover the device arrays the engine constructor allocates (summed `sizeof`) and what one `stage1!`
# allocates (`CUDA.@allocated`), without overstating them by much.

@testitem "engine_bytes against the engines' device allocations" setup = [Common, EngineBytes] begin
    if !ENGINE_BYTES_GPU
        @info "CUDA not available/functional — skipping engine_bytes test"
    else
        CUDA.allowscalar(false)
        backend = ElectronPhonon.gpu_backend()
        eV = unit_to_aru(:eV); e_F = 11.68eV
        window = (e_F - 0.5eV, e_F + 3eV)
        grid = (6, 6, 6)
        el_qty, ph_qty = [:u, :e], [:u, :e]
        nb, ntile = 7, 40
        # (measured, counted) of the device-selection cases, by whether the phonons are per batch.
        selection_cases = Dict{Bool, Tuple{Int, Int}}()
        # The last two cases select the pairs on the device (a finite energy_conservation_tol), the
        # last with the phonons built per outer batch.
        for (order, mom, dg, tol, per_batch) in ((OuterKLoop(), "el", false, Inf, false),
                (OuterKLoop(), "el", true, Inf, false), (OuterQLoop(), "ph", false, Inf, false),
                (OuterKLoop(), "el", false, 1e-3, false), (OuterKLoop(), "el", false, 1e-3, true))
            model = _load_model_from_artifacts("pb"; epmat_outer_momentum = mom)
            # The device-selection cases run on a finer k+q grid, whose table of all modes outgrows
            # the loop's buffers.
            grid_kq = isfinite(tol) ? (24, 24, 24) : grid
            run_options(max_device_bytes) = _run_options(model; inner_loop_kq = order isa OuterKLoop,
                backend, window_k = window, window_kq = window, symmetry = nothing,
                energy_conservation_tol = tol, n_outer_batch = nb, max_device_bytes, verbosity = 0)
            st = _setup_states(order, model, grid, grid_kq, run_options(nothing))
            if per_batch
                # A cap just under twice the table of all modes sends the run through the ω pass.
                cap = 2 * st.qpts.n * _phonon_state_bytes(Float64, model.nmodes, run_options(nothing).ph_qty) - 1
                st = _setup_states(order, model, grid, grid_kq, run_options(cap))
                @test st.phonon_batches !== nothing
            end
            # The q of each outer batch's kept pairs: the phonons per outer batch, and the batch with
            # the most of them, whose stage 1 the count holds in either case.
            iqs_per_batch = isfinite(tol) ? _select_qs_per_outer_batch(backend, st.kpts, st.kqpts,
                st.qpts, st.els_k, st.els_kq, per_batch ? st.phonon_batches.ω_kept : st.phs.e, nb, tol) : nothing
            phonon_batches = per_batch ? (; ω_kept = st.phonon_batches.ω_kept, iqs_per_batch) : nothing
            nbk = st.els_k.nband_max
            nbkq = st.els_kq === nothing ? model.nw : st.els_kq.nband_max
            common = (; n_outer_batch = nb, n_inner_tile = ntile, nchunks = 1,
                      eph_phonon_basis = :eigenmode)
            if order isa OuterKLoop
                bytes = engine_bytes(OuterKEngine, model; nband_max_k = nbk, nband_max_kq = nbkq,
                    nk = st.kpts.n, nkq = st.kqpts.n, el_qty, ph_qty,
                    inner_loop_kq = true, device_pair_selection = isfinite(tol),
                    nq_grid = prod(st.qpts.ngrid), nmodes_kept = st.phs.nmodes, phonon_batches,
                    backend, covariant_derivative_of_g = dg, eph_phonon_basis = :eigenmode)
                eng = OuterKEngine(model, backend, st.els_k, st.els_kq, st.phs, el_qty, ph_qty;
                    st.kpts, st.kqpts, st.qpts, covariant_derivative_of_g = dg,
                    energy_conservation_tol = tol, phonon_batches, common...)
                ib = iqs_per_batch === nothing ? 1 : argmax(length.(iqs_per_batch))
                run1 = () -> stage1!(eng, (ib - 1) * nb + 1:min(ib * nb, st.kpts.n))
            else
                bytes = engine_bytes(OuterQEngine, model; nband_max_k = nbk, nband_max_kq = nbkq,
                    nk = st.kpts.n, el_qty, ph_qty,
                    precompute_el_kq = false, eph_phonon_basis = :eigenmode)
                eng = OuterQEngine(model, backend, st.els_k, st.els_kq, st.phs, el_qty, ph_qty;
                    st.kpts, st.qpts, common...)
                run1 = () -> stage1!(eng, 1:nb)
            end
            run1()
            CUDA.synchronize()
            transient = _device_allocated(run1)
            # The planner runs after resident states (and the frequencies of the per-batch pair
            # selection) are built: their bytes are already unavailable in free_bytes, so the engine
            # budget counts newly allocated scratch, not those aliases.
            held = _device_bytes(eng) - _device_bytes((st.els_k, st.els_kq, st.phs,
                                                   st.phonon_batches === nothing ? nothing : st.phonon_batches.ω_kept))
            counted = bytes.persistent + bytes.per_outer * nb + bytes.per_pair * ntile
            @info "engine_bytes" order mom dg tol per_batch held transient counted ratio = (held + transient) / counted
            # Everything the engine holds and its stage 1 allocates is counted (0.925-1.000 measured,
            # Pb, A100)...
            @test held + transient <= 1.02 * counted
            # ...and the count is not a loose upper bound. The outer-k stage-1 rotation transients are
            # counted at the box width, while each band class of the batch allocates at its own.
            @test held + transient >= (order isa OuterKLoop ? 0.9 : 0.95) * counted
            isfinite(tol) && (selection_cases[per_batch] = (held + transient, counted))
        end
        # What the per-batch phonons add (their storage, index buffers and per-batch solve) is counted
        # on its own, not hidden by the rest of the engine.
        Δmeasured, Δcounted = selection_cases[true] .- selection_cases[false]
        @info "engine_bytes, per-batch phonons" Δmeasured Δcounted ratio = Δmeasured / Δcounted
        @test 0.95 * Δcounted <= Δmeasured <= 1.02 * Δcounted
    end
end

# `stage2!` runs once per block, on the tile's preallocated scratch: with resident k+q states (no
# per-tile eigensolve) it allocates no array of the block's size, on the CPU or on a device. On a
# device the outer-q k list is staged per call (24 bytes per point).
@testitem "stage2! allocates no scratch" setup = [Common, EngineBytes] begin
    using ElectronPhonon: stage2!
    eV = unit_to_aru(:eV); e_F = 11.68eV
    window = (e_F - 0.5eV, e_F + 3eV)
    grid = (6, 6, 6)
    el_qty, ph_qty = [:u, :e], [:u, :e]
    nb, ntile = 3, 40
    backends = Any[ElectronPhonon.CPUBackend()]
    ENGINE_BYTES_GPU && push!(backends, ElectronPhonon.gpu_backend())
    for backend in backends, order in (OuterKLoop(), OuterQLoop())
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = order isa OuterKLoop ? "el" : "ph")
        options = _run_options(model; inner_loop_kq = order isa OuterKLoop, backend,
            window_k = window, window_kq = window, symmetry = nothing, 
            precompute_el_kq = order isa OuterQLoop, verbosity = 0)
        st = _setup_states(order, model, grid, grid, options)
        common = (; n_outer_batch = nb, n_inner_tile = ntile, nchunks = 1,
                  eph_phonon_basis = :eigenmode)
        if order isa OuterKLoop
            eng = OuterKEngine(model, backend, st.els_k, st.els_kq, st.phs, el_qty, ph_qty;
                st.kpts, st.kqpts, st.qpts, covariant_derivative_of_g = false, common...)
        else
            eng = OuterQEngine(model, backend, st.els_k, st.els_kq, st.phs, el_qty, ph_qty;
                st.kpts, st.kqpts, st.qpts, common...)
        end
        stage1!(eng, 1:nb)
        # The worker behind `stage2!`, on the concrete fields, so the dispatch is not measured.
        tile_workspace = ElectronPhonon._workspace_fields(eng.tiles[1])
        eng_fields = ElectronPhonon._workspace_fields(eng)
        run2 = () -> ElectronPhonon._stage2!(order, eng_fields, tile_workspace, 1, 1:ntile)
        run2()
        nbytes = if backend isa ElectronPhonon.CPUBackend
            @allocated run2()
        else
            CUDA.synchronize(); _device_allocated(run2)
        end
        @info "stage2! allocations" order backend = nameof(typeof(backend)) nbytes
        # Only small view/dispatch wrappers and the staged k list may be allocated; the rotation
        # scratch alone would be O(1e5) bytes for this fixture.
        @test nbytes <= 24 * ntile + 2048
    end
end
