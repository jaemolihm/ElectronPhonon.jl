using Test
using ElectronPhonon
using ElectronPhonon: OuterKLoop, OuterQLoop, OuterKEngine, OuterQEngine, engine_bytes, stage1!,
    _setup_states, unit_to_aru

# `engine_bytes` is what `plan_batch` sizes the inner tile from, before the engine exists: it must
# cover the device arrays the engine constructor allocates (summed `sizeof`) and what one `stage1!`
# allocates (`CUDA.@allocated`), without overstating them by much.

const ENGINE_BYTES_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

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
        for (order, mom, dg) in ((OuterKLoop(), "el", false), (OuterKLoop(), "el", true),
                                 (OuterKLoop(), "ph", false), (OuterQLoop(), "ph", false),
                                 (OuterQLoop(), "el", false))
            model = _load_model_from_artifacts("pb"; epmat_outer_momentum = mom)
            st = _setup_states(order, model, grid, grid, el_qty, ph_qty; backend, window_k = window,
                window_kq = window, symmetry = nothing, precompute_el_kq = false, keep_all_qpts = true,
                eph_phonon_basis = :eigenmode, mpi_comm_k = nothing, el_k_eigenpairs = nothing,
                el_kq_eigenpairs = nothing, ph_eigenpairs = nothing, fill_padding_nan = false,
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
            transient = CUDA.@allocated run1()
            held = _device_bytes(eng)
            counted = bytes.persistent + bytes.per_outer * nb + bytes.per_pair * ntile
            @info "engine_bytes" order mom dg held transient counted ratio = (held + transient) / counted
            # Everything the engine holds and its stage 1 allocates is counted (0.992-1.000 measured,
            # Pb, all five arms)...
            @test held + transient <= 1.02 * counted
            # ...and the count is not a loose upper bound.
            @test held + transient >= 0.95 * counted
        end
    end
end
