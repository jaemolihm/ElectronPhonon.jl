using Test
using ElectronPhonon
using ElectronPhonon: OuterKLoop, OuterQLoop, OuterKEngine, OuterQEngine, engine_bytes, stage1!,
    _setup_states, unit_to_aru, EPMAT_CHUNK_BYTES

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
        diskdir = mktempdir()
        chunk_bytes = EPMAT_CHUNK_BYTES[]
        # The disk arms stream the epmat in chunks of 3 columns, of which the engine holds one.
        try
            for (order, mom, dg, disk) in ((OuterKLoop(), "el", false, false), (OuterKLoop(), "el", true, false),
                                     (OuterKLoop(), "ph", false, false), (OuterQLoop(), "ph", false, false),
                                     (OuterQLoop(), "el", false, false), (OuterKLoop(), "el", false, true),
                                     (OuterKLoop(), "ph", false, true), (OuterQLoop(), "ph", false, true),
                                     (OuterQLoop(), "el", false, true))
                model = _load_model_from_artifacts("pb"; epmat_outer_momentum = mom)
                if disk
                    EPMAT_CHUNK_BYTES[] = 3 * sizeof(ComplexF64) * size(model.epmat.op_r, 1)
                    model = _disk_epmat_model(model, mkpath(joinpath(diskdir, mom)))
                end
                st = _setup_states(order, model, grid, grid, el_qty, ph_qty; backend, window_k = window,
                    window_kq = window, symmetry = nothing, precompute_el_kq = false, keep_all_qpts = true,
                    eph_phonon_basis = :eigenmode, fourier_mode = "gridopt", mpi_comm_k = nothing,
                    el_k_eigenpairs = nothing, el_kq_eigenpairs = nothing, ph_eigenpairs = nothing,
                    verbosity = 0)
                nbk = st.el_k.nband_max
                nbkq = st.el_kq === nothing ? model.nw : st.el_kq.nband_max
                common = (; n_outer_batch = nb, n_inner_tile = ntile, nchunks = 1, drop_pairs = false,
                          eph_phonon_basis = :eigenmode)
                if order isa OuterKLoop
                    bytes = engine_bytes(OuterKEngine, model; nband_max_k = nbk, nband_max_kq = nbkq,
                        nk = st.kpts.n, nkq = st.kqpts.n, el_qty, ph_qty, drop_pairs = false,
                        covariant_derivative_of_g = dg, eph_phonon_basis = :eigenmode)
                    eng = OuterKEngine(model, backend, st.el_k, st.el_kq, st.ph, el_qty, ph_qty;
                        st.kpts, st.kqpts, st.qpts, covariant_derivative_of_g = dg, common...)
                    run1 = () -> stage1!(eng, st.el_k, st.kpts, 1:nb)
                else
                    bytes = engine_bytes(OuterQEngine, model; nband_max_k = nbk, nband_max_kq = nbkq,
                        nk = st.kpts.n, n_outer_batch = nb, el_qty, ph_qty, drop_pairs = false,
                        precompute_el_kq = false, eph_phonon_basis = :eigenmode)
                    eng = OuterQEngine(model, backend, st.el_k, st.el_kq, st.ph, el_qty, ph_qty;
                        st.kpts, st.qpts, common...)
                    run1 = () -> stage1!(eng, st.ph, st.qpts, 1:nb, :eigenmode)
                end
                run1()
                CUDA.synchronize()
                transient = _device_allocated(run1)
                held = _device_bytes(eng)
                counted = bytes.persistent + bytes.per_outer * nb + bytes.per_pair * ntile
                @info "engine_bytes" order mom dg disk held transient counted ratio = (held + transient) / counted
                # Everything the engine holds and its stage 1 allocates is counted (0.982-1.000 measured,
                # Pb, all nine arms)...
                @test held + transient <= 1.02 * counted
                # ...and the count is not a loose upper bound.
                @test held + transient >= 0.95 * counted
            end
        finally
            EPMAT_CHUNK_BYTES[] = chunk_bytes
        end
    end
end

# `stage2!` runs once per block, on the tile's preallocated scratch: it allocates no array of the
# block's size, on the CPU or on a device (the k list of an outer-q block is staged per call, 24
# bytes per point).
@testset "stage2! allocates no scratch" begin
    using ElectronPhonon: stage2!, reshape_view_batched_electron_states
    eV = unit_to_aru(:eV); e_F = 11.68eV
    window = (e_F - 0.5eV, e_F + 3eV)
    grid = (6, 6, 6)
    el_qty, ph_qty = [:u, :e], [:u, :e]
    nb, ntile = 3, 40
    backends = Any[ElectronPhonon.CPUBackend()]
    ENGINE_BYTES_GPU && push!(backends, ElectronPhonon.gpu_backend())
    for backend in backends, order in (OuterKLoop(), OuterQLoop())
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = order isa OuterKLoop ? "el" : "ph")
        st = _setup_states(order, model, grid, grid, el_qty, ph_qty; backend, window_k = window,
            window_kq = window, symmetry = nothing, precompute_el_kq = false, keep_all_qpts = true,
            eph_phonon_basis = :eigenmode, fourier_mode = "gridopt", mpi_comm_k = nothing,
            el_k_eigenpairs = nothing, el_kq_eigenpairs = nothing, ph_eigenpairs = nothing, verbosity = 0)
        common = (; n_outer_batch = nb, n_inner_tile = ntile, nchunks = 1, drop_pairs = false,
                  eph_phonon_basis = :eigenmode)
        if order isa OuterKLoop
            eng = OuterKEngine(model, backend, st.els_k, st.els_kq, st.phs, el_qty, ph_qty;
                st.kpts, st.kqpts, st.qpts, covariant_derivative_of_g = false, common...)
            stage1!(eng, st.els_k, st.kpts, 1:nb)
            t = eng.tiles[1]
            pairs = (; n = ntile, iouter = 1, phase = view(t.P_kq, :, 1:ntile),
                     phs = view(t.phs, 1:ntile),
                     els_kq = view(st.els_kq, 1:ntile))
        else
            eng = OuterQEngine(model, backend, st.els_k, st.els_kq, st.phs, el_qty, ph_qty;
                st.kpts, st.qpts, common...)
            stage1!(eng, st.phs, st.qpts, 1:nb, :eigenmode)
            t = eng.tiles[1]
            pairs = (; n = ntile, els_k = view(t.els_k, 1:ntile),
                     els_kq = reshape_view_batched_electron_states(t.els_kq, model.nw, ntile),
                     phs = view(st.phs, 1:1), xk = view(st.kpts.vectors, 1:ntile))
        end
        stage2!(eng, t, pairs)
        nbytes = if backend isa ElectronPhonon.CPUBackend
            @allocated stage2!(eng, t, pairs)
        else
            CUDA.synchronize(); _device_allocated(() -> stage2!(eng, t, pairs))
        end
        @info "stage2! allocations" order backend = nameof(typeof(backend)) nbytes
        # Measured (Pb): 80 and 112 bytes on the CPU with Julia 1.11, 0 and 32 with 1.13 (ccqlin059);
        # 32 and 992 bytes (the k list) on the GPU (A100). The outer-q stage-2 scratch (g, tmp,
        # uk_rep) alone is of order 1e5 bytes here.
        @test nbytes <= 24 * ntile + 2048
    end
end
