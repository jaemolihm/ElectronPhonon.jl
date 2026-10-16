using Test
using ElectronPhonon
using ElectronPhonon: OuterKLoop, OuterQLoop, OuterKEngine, OuterQEngine, engine_bytes, stage1!,
    _run_options, _setup_states, unit_to_aru, _select_qs_per_outer_batch

# `engine_bytes` is what `plan_batch` sizes the inner tile from, before the engine exists: it must
# cover the device arrays the engine constructor allocates (summed `sizeof`) and what one `stage1!`
# allocates (`CUDA.@allocated`), without overstating them by much.

const ENGINE_BYTES_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# `CUDA.@allocated f()`, expanded at run time: the macro needs `CUDA` when the file is lowered, which
# fails where CUDA is not installed (CI) even though the GPU branch never runs there.
_device_allocated(f) = @eval CUDA.@allocated $f()

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

# Bytes of every device array reachable from `x` (fields, tuples, vectors), each array once.
function _device_bytes(x, seen = IdDict{Any, Nothing}())
    (isbits(x) || x isa Module || x isa DataType || haskey(seen, x)) && return 0
    seen[x] = nothing
    x isa CuArray && return sizeof(x)
    x isa Array && return eltype(x) <: Number ? 0 : sum(y -> _device_bytes(y, seen), x; init = 0)
    x isa AbstractArray && return _device_bytes(parent(x), seen)
    sum(i -> isdefined(x, i) ? _device_bytes(getfield(x, i), seen) : 0, 1:nfields(x); init = 0)
end

@testset "engine_bytes against the engines' device allocations" begin
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
        # (measured, counted) of the device-selection cases, by phonon storage.
        selection_cases = Dict{Symbol, Tuple{Int, Int}}()
        # The last two cases select the pairs on the device (a finite energy_conservation_tol), the
        # last with the phonons built per outer batch.
        for (order, mom, dg, tol, phonon_storage) in ((OuterKLoop(), "el", false, Inf, :auto),
                (OuterKLoop(), "el", true, Inf, :auto), (OuterQLoop(), "ph", false, Inf, :auto),
                (OuterKLoop(), "el", false, 1e-3, :auto), (OuterKLoop(), "el", false, 1e-3, :per_batch))
            model = _load_model_from_artifacts("pb"; epmat_outer_momentum = mom)
            options = _run_options(model; inner_loop_kq = order isa OuterKLoop, backend,
                window_k = window, window_kq = window, symmetry = nothing,
                energy_conservation_tol = tol, phonon_storage, verbosity = 0)
            st = _setup_states(order, model, grid, grid, options)
            # Phonons per outer batch: the largest batch's q, and the engine's phonon solve.
            iqs_per_batch = st.ω_all === nothing ? nothing : _select_qs_per_outer_batch(backend,
                st.kpts, st.kqpts, st.qpts, st.els_k, st.els_kq, st.ω_all, nb, tol)
            nq_per_batch = iqs_per_batch === nothing ? nothing : maximum(length, iqs_per_batch)
            nbk = st.els_k.nband_max
            nbkq = st.els_kq === nothing ? model.nw : st.els_kq.nband_max
            common = (; n_outer_batch = nb, n_inner_tile = ntile, nchunks = 1,
                      eph_phonon_basis = :eigenmode)
            if order isa OuterKLoop
                bytes = engine_bytes(OuterKEngine, model; nband_max_k = nbk, nband_max_kq = nbkq,
                    nk = st.kpts.n, nkq = st.kqpts.n, el_qty, ph_qty,
                    inner_loop_kq = true, device_pair_selection = isfinite(tol),
                    nq_grid = prod(st.qpts.ngrid), nmodes_kept = st.phs.nmodes, nq_per_batch,
                    nq = st.qpts.n, backend, covariant_derivative_of_g = dg, eph_phonon_basis = :eigenmode)
                # The per-batch count needs the q set's size and the backend of the solve.
                nq_per_batch === nothing || @test_throws ArgumentError engine_bytes(OuterKEngine, model;
                    nband_max_k = nbk, nband_max_kq = nbkq, nk = st.kpts.n, nkq = st.kqpts.n, el_qty,
                    ph_qty, inner_loop_kq = true, device_pair_selection = true, nq_per_batch,
                    covariant_derivative_of_g = dg, eph_phonon_basis = :eigenmode)
                eng = OuterKEngine(model, backend, st.els_k, st.els_kq, st.phs, el_qty, ph_qty;
                    st.kpts, st.kqpts, st.qpts, covariant_derivative_of_g = dg,
                    energy_conservation_tol = tol, st.ω_all, iqs_per_batch, common...)
                run1 = () -> stage1!(eng, 1:nb)
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
            held = _device_bytes(eng) - _device_bytes((st.els_k, st.els_kq, st.phs, st.ω_all))
            counted = bytes.persistent + bytes.per_outer * nb + bytes.per_pair * ntile
            @info "engine_bytes" order mom dg tol phonon_storage held transient counted ratio = (held + transient) / counted
            # Everything the engine holds and its stage 1 allocates is counted (0.925-1.000 measured,
            # Pb, A100)...
            @test held + transient <= 1.02 * counted
            # ...and the count is not a loose upper bound. The outer-k stage-1 rotation transients are
            # counted at the box width, while each band class of the batch allocates at its own.
            @test held + transient >= (order isa OuterKLoop ? 0.9 : 0.95) * counted
            isfinite(tol) && (selection_cases[phonon_storage] = (held + transient, counted))
        end
        # What the per-batch phonons add (their storage, index buffers and per-batch solve) is counted
        # on its own, not hidden by the rest of the engine. The first batch's solve stands in for the
        # largest batch's, which the count holds.
        Δmeasured, Δcounted = selection_cases[:per_batch] .- selection_cases[:auto]
        @info "engine_bytes, per-batch phonons" Δmeasured Δcounted ratio = Δmeasured / Δcounted
        @test 0.95 * Δcounted <= Δmeasured <= 1.02 * Δcounted
    end
end

# `stage2!` runs once per block, on the tile's preallocated scratch: with resident k+q states (no
# per-tile eigensolve) it allocates no array of the block's size, on the CPU or on a device. On a
# device the outer-q k list is staged per call (24 bytes per point).
@testset "stage2! allocates no scratch" begin
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
