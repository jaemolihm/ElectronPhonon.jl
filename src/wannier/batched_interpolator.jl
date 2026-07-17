using LinearAlgebra

export register_kpoints!
export clear_registered_kpoints!
export get_fourier_batched!

"""
    BatchedWannierInterpolator{T, WT}

Stateful Wannier interpolator that batches Fourier transformations for sequential k-point queries.
Composes a stateless [`BatchedFourierCore`](@ref) (the whole-batch Fourier engine) with a
[`SequentialQueryCache`](@ref) (the order-enforced per-k query bookkeeping).

# Usage
1. Create interpolator: `itp = BatchedWannierInterpolator(wannier_obj; batch_size=32)`
2. Register k-points: `register_kpoints!(itp, kpoint_list)`
3. Query sequentially: `get_fourier!(op_k, itp, xk)` for each xk in order

# Performance
Uses BLAS3 matrix-matrix multiplication instead of BLAS2 matrix-vector multiplication,
providing ~3-5x speedup for sequential k-point queries.

# Backends
The buffer arrays follow the backend of `parent.op_r`, so a GPU parent (a `WannierObject`
whose `op_r` is a `CuMatrix`) keeps the whole cached batch on the device. For the GPU /
whole-batch use case, prefer [`get_fourier_batched!`](@ref) over per-k `get_fourier!`,
which avoids a device→host copy per k-point.

# Data freshness
The interpolator interpolates fresh from `parent.op_r` on every call and never consults
`parent._id` (a GridOpt-family concern): it is stateless with respect to the parent data, so a
caller may mutate `parent.op_r` in place between calls without any cache invalidation.

# Notes
- K-points must be registered before querying
- K-points must be queried in the exact order they were registered
- Out-of-order or unregistered queries will throw an error
"""
mutable struct BatchedWannierInterpolator{T, WT <: AbstractWannierObject, MC, MR, VC} <: AbstractWannierInterpolator{T}
    # Stateless whole-batch Fourier engine (owns the parent + GEMM scratch)
    const core::BatchedFourierCore{T, WT, MC, MR}

    # Order-enforced per-k query bookkeeping
    const cache::SequentialQueryCache{T}

    # Cached results from the current batch (ndata × batch_size), on parent's backend
    cached_results::MC

    # Output buffer (per-k query API)
    const out::VC

    # Buffers for intermediate calculations
    const buffer::VC
    const buffer2::VC

    # Buffer for diagonalization (eigensolve scratch, used by the wannier_to_bloch eigen drivers
    # when this interpolator drives the per-k Hamiltonian/dynamical-matrix eigensolve)
    const ws::HermitianEigenWsSYEV{Complex{T},T}
end

function BatchedWannierInterpolator(parent::WT; batch_size::Int=32, xk_tol=sqrt(eps(T))/100) where {WT <: AbstractWannierObject{T}} where {T}
    core = BatchedFourierCore(parent; batch_cap=batch_size)
    cache = SequentialQueryCache{T}(; xk_tol)
    ws = HermitianEigenWsSYEV{Complex{T},T}()

    cached_results = _alloc_array(parent, Complex{T}, parent.ndata, batch_size)
    out    = _alloc_array(parent, Complex{T}, parent.ndata)
    buffer = _alloc_array(parent, Complex{T}, 0)
    buffer2 = _alloc_array(parent, Complex{T}, 0)

    BatchedWannierInterpolator{T, WT, typeof(cached_results), typeof(core.irvec_mat), typeof(out)}(
        core, cache, cached_results, out, buffer, buffer2, ws)
end

# `parent`, `nr`, and `batch_size` live in the composed core; forward them so callers and the
# generic AbstractWannierInterpolator helpers keep working unchanged.
@inline function Base.getproperty(obj::BatchedWannierInterpolator, name::Symbol)
    if name === :parent
        getfield(obj, :core).parent
    elseif name === :nr
        getfield(getfield(obj, :core).parent, :nr)
    elseif name === :batch_size
        getfield(obj, :core).batch_cap
    else
        getfield(obj, name)
    end
end


"""
    clear_registered_kpoints!(obj::BatchedWannierInterpolator)

Clear registered k-points and cached results. Resets the interpolator to initial state.
"""
clear_registered_kpoints!(obj::BatchedWannierInterpolator) = clear_registered_kpoints!(obj.cache)


"""
    register_kpoints!(obj::BatchedWannierInterpolator, xk_list)

Register a sequence of k-points that will be queried via `get_fourier!`.
The k-points must be queried in the exact order they are registered.

# Arguments
- `obj`: BatchedWannierInterpolator
- `xk_list`: Vector of k-points to be registered

# Notes
- Clears any previously registered k-points and cached results
- K-points MUST be queried in the same order via `get_fourier!`
- Querying out-of-order or unregistered k-points will throw an error
"""
register_kpoints!(obj::BatchedWannierInterpolator, xk_list) = register_kpoints!(obj.cache, xk_list)


"""
    get_fourier!(op_k, obj::BatchedWannierInterpolator{T}, xk)

Compute Fourier transform at k-point xk.

# Behavior
- xk must match the next registered k-point in sequence
- If this k-point starts a new batch, triggers batched computation
- Otherwise returns cached result from the current batch
- Throws error if xk doesn't match the expected next k-point

# Arguments
- `op_k`: Output vector of size ndata
- `obj`: BatchedWannierInterpolator
- `xk`: k-point (Vec3)

# Errors
- If no k-points are registered
- If xk doesn't match the next expected k-point
- If all registered k-points have been exhausted
"""
@timing "get_fourier" function get_fourier!(op_k, obj::BatchedWannierInterpolator{T, WT}, xk) where {T, WT}
    (; cache) = obj
    ndata = obj.core.parent.ndata
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == ndata
    op_k_1d = _reshape(op_k, (length(op_k),))

    current_index = _next_query_index(cache, xk)

    # Check if we need to compute a new batch
    if current_index < cache.cached_batch_start || current_index > cache.cached_batch_end
        _compute_batch!(obj, current_index)
    end

    # Return cached result
    cache_offset = current_index - cache.cached_batch_start + 1
    @views op_k_1d .= obj.cached_results[1:ndata, cache_offset]

    # Advance to next k-point
    cache.current_index += 1

    return op_k
end


"""
    _compute_batch!(obj::BatchedWannierInterpolator, start_idx::Int)

Internal function: Compute a batch of k-points starting from the given index, storing the
result in `obj.cached_results` and recording the batch range in `obj.cache`.
"""
function _compute_batch!(obj::BatchedWannierInterpolator{T, WT}, start_idx::Int) where {T, WT}
    (; core, cache, cached_results) = obj
    ndata = core.parent.ndata

    # Determine batch range
    batch_start = start_idx
    batch_end = min(start_idx + core.batch_cap - 1, length(cache.registered_kpoints))
    batch_len = batch_end - batch_start + 1

    xks = @view cache.registered_kpoints[batch_start:batch_end]
    @views fourier_batched!(cached_results[1:ndata, 1:batch_len], core, xks)

    # Update cache metadata
    cache.cached_batch_start = batch_start
    cache.cached_batch_end = batch_end

    nothing
end


"""
    get_fourier_batched!(out, obj::BatchedWannierInterpolator, xk_list)

Fourier-transform `obj` at all k-points in `xk_list` at once, writing into `out`
(`(ndata, nk)` on the backend of `obj.parent.op_r`). `length(xk_list)` may be larger than
`obj.batch_size`: the k-points are processed internally in chunks of `batch_size` on the
stateless [`BatchedFourierCore`](@ref), and the whole result is kept on the backend (no per-k
device→host copy). This is the entry point for GPU / whole-batch use.

Does not touch the [`SequentialQueryCache`](@ref) queue state, so it is independent of any
in-progress per-k `get_fourier!` sequence.
"""
function get_fourier_batched!(out, obj::BatchedWannierInterpolator{T}, xk_list) where {T}
    core = obj.core
    ndata = core.parent.ndata
    nk = length(xk_list)
    @assert size(out) == (ndata, nk)
    cap = core.batch_cap
    start = 1
    while start <= nk
        stop = min(start + cap - 1, nk)
        xks = @view xk_list[start:stop]
        @views fourier_batched!(out[:, start:stop], core, xks)
        start = stop + 1
    end
    out
end


"""
    skip_registered_kpoint!(obj::BatchedWannierInterpolator)

Skip one registered k-point in the sequence and advance the current index.
"""
function skip_registered_kpoint!(obj::BatchedWannierInterpolator)
    obj.cache.current_index += 1
    nothing
end
