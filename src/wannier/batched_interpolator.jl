using LinearAlgebra

export register_kpoints!
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
Every buffer is allocated on `backend`, which must be the backend `parent.op_r` already lives on
(checked at construction), so a GPU parent keeps the whole cached batch on the device. For the
GPU / whole-batch use case, prefer [`get_fourier_batched!`](@ref) over per-k `get_fourier!`,
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
mutable struct BatchedWannierInterpolator{T, WT <: WannierObject, BT, MC, MR, VC} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated. The same object the core holds: duplicated here as a
    # `const` reference so the generic `AbstractWannierInterpolator` `getproperty` covers all four
    # interpolator types uniformly. Both fields are `const`, so they cannot desync.
    const parent::WT

    # Block width of the composed engine. Duplicated here for the same reason as `parent`: both
    # are `const` and set from one argument, so they cannot desync, and `itp.batch_size` then means
    # the same thing on every interpolator type that has a batch.
    const batch_size::Int

    # Stateless whole-batch Fourier engine (owns the GEMM scratch)
    const core::BatchedFourierCore{T, WT, BT, MC, MR}

    # Order-enforced per-k query bookkeeping
    const cache::SequentialQueryCache{T}

    # Cached results from the current batch (ndata × batch_size), on `core.backend`. Written only
    # by `_compute_batch!` and read only by `get_fourier!`, both unreachable until a k-point is
    # registered — so it is allocated on the first `register_kpoints!` and stays `nothing` for the
    # whole-batch callers, which are the majority and never query per k.
    cached_results::Union{Nothing, MC}

    # Output buffer (per-k query API)
    const out::VC

    # Buffer for intermediate calculations. Read via `_reshape_buffer` by the per-k
    # `wannier_to_bloch.jl` drivers, which this interpolator's `get_fourier!` API feeds.
    const buffer::VC

    # Buffer for diagonalization, used by the per-k eigensolve drivers (`get_el_eigen!`,
    # `get_ph_eigen!`) when `compute_eigenvalues_el` / `compute_electron_states` is run with
    # `fourier_mode = "batched"`.
    const ws::HermitianEigenWsSYEV{Complex{T},T}
end

function BatchedWannierInterpolator(parent::WT; backend::AbstractBackend = CPUBackend(),
        batch_size::Int = 32, xk_tol = sqrt(eps(T))/100) where {WT <: WannierObject{T}} where {T}
    bs = batch_size
    core = BatchedFourierCore(parent; backend, batch_size)
    cache = SequentialQueryCache{T}(; xk_tol)

    ws = HermitianEigenWsSYEV{Complex{T},T}()

    out    = alloc(backend, Complex{T}, parent.ndata)
    buffer = alloc(backend, Complex{T}, 0)

    BatchedWannierInterpolator{T, WT, typeof(backend), typeof(core.phase),
                               typeof(core.irvec_mat), typeof(out)}(
        parent, bs, core, cache, nothing, out, buffer, ws)
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
- Allocates the per-k cached-result buffer on the first call
"""
function register_kpoints!(obj::BatchedWannierInterpolator{T}, xk_list) where {T}
    (; parent, core) = obj
    register_kpoints!(obj.cache, xk_list)
    if obj.cached_results === nothing
        obj.cached_results = alloc(core.backend, Complex{T}, parent.ndata, core.batch_size)
    end
    nothing
end


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
@timing "get_fourier" function get_fourier!(op_k, obj::BatchedWannierInterpolator{T, WT, BT, MC},
        xk) where {T, WT, BT, MC}
    (; cache, parent) = obj
    ndata = parent.ndata
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
    # Non-`nothing` here: `_next_query_index` above throws unless `register_kpoints!` has run.
    @views op_k_1d .= (obj.cached_results::MC)[1:ndata, cache_offset]

    # Advance to next k-point
    cache.current_index += 1

    return op_k
end


"""
    _compute_batch!(obj::BatchedWannierInterpolator, start_idx::Int)

Internal function: Compute a batch of k-points starting from the given index, storing the
result in `obj.cached_results` and recording the batch range in `obj.cache`.
"""
function _compute_batch!(obj::BatchedWannierInterpolator{T, WT, BT, MC},
        start_idx::Int) where {T, WT, BT, MC}
    (; core, cache, parent) = obj
    cached_results = obj.cached_results::MC
    ndata = parent.ndata

    # Determine batch range
    batch_start = start_idx
    batch_end = min(start_idx + core.batch_size - 1, length(cache.registered_kpoints))
    batch_len = batch_end - batch_start + 1

    # Stage this block's k-points only. On `CPUBackend` that is a zero-copy `reinterpret` view of
    # the queue and allocates nothing; staging the whole queue once instead would be an O(n_registered)
    # copy per `register_kpoints!`, which the per-q threaded loops call once per q.
    xks = @view cache.registered_kpoints[batch_start:batch_end]
    @views _fourier_batched!(cached_results[1:ndata, 1:batch_len], core,
                             _kpoints_to_device_matrix(core.backend, xks))

    # Update cache metadata
    cache.cached_batch_start = batch_start
    cache.cached_batch_end = batch_end

    nothing
end


"""
    get_fourier_batched!(out, obj::BatchedWannierInterpolator, xk_list::AbstractVector)
    get_fourier_batched!(out, obj::BatchedWannierInterpolator, xkmat::AbstractMatrix)

Fourier-transform `obj` at all the given k-points at once, writing into `out` (`(ndata, nk)` on the
interpolator's backend). Total: any `nk ≥ 0` is accepted, and an `nk` larger than the interpolator's
`batch_size` is processed in blocks of that width on the partial [`_fourier_batched!`](@ref). The
whole result is kept on the backend (no per-k device→host copy). This is the entry point for GPU /
whole-batch use.

A host k-list is staged onto the backend once here, not once per block. A caller that already holds
the staged `(3 × nk)` matrix — several interpolators over one k-list, as in
`_compute_electron_states_device!` — passes that instead and pays the transfer once for all of them.

Does not touch the [`SequentialQueryCache`](@ref) queue state, so it is independent of any
in-progress per-k `get_fourier!` sequence.
"""
get_fourier_batched!(out, obj::BatchedWannierInterpolator, xk_list::AbstractVector) =
    get_fourier_batched!(out, obj, _kpoints_to_device_matrix(obj.core.backend, xk_list))

function get_fourier_batched!(out, obj::BatchedWannierInterpolator, xkmat::AbstractMatrix)
    core = obj.core
    ndata = obj.parent.ndata
    nk = size(xkmat, 2)
    @assert size(xkmat, 1) == 3
    @assert size(out) == (ndata, nk)
    # Once per call, off the block loop: an array on the wrong side is named here instead of
    # surfacing as a mixed host/device broadcast (or a scalar-indexing error) inside the phase build.
    check_on_backend(core.backend, xkmat, "xkmat")
    cap = core.batch_size
    start = 1
    while start <= nk
        stop = min(start + cap - 1, nk)
        @views _fourier_batched!(out[:, start:stop], core, xkmat[:, start:stop])
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
