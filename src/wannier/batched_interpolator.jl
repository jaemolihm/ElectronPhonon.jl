using LinearAlgebra

export register_kpoints!
export clear_registered_kpoints!

"""
    BatchedWannierInterpolator{T, WT}

Stateful Wannier interpolator that batches Fourier transformations for sequential k-point queries.

# Usage
1. Create interpolator: `itp = BatchedWannierInterpolator(wannier_obj; batch_size=32)`
2. Register k-points: `register_kpoints!(itp, kpoint_list)`
3. Query sequentially: `get_fourier!(op_k, itp, xk)` for each xk in order

# Performance
Uses BLAS3 matrix-matrix multiplication instead of BLAS2 matrix-vector multiplication,
providing ~3-5x speedup for sequential k-point queries.

# Notes
- K-points must be registered before querying
- K-points must be queried in the exact order they were registered
- Out-of-order or unregistered queries will throw an error
"""
mutable struct BatchedWannierInterpolator{T, WT <: AbstractWannierObject} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated
    const parent::WT

    # Batch size (number of k-points to process together)
    const batch_size::Int

    # Registered k-points to be queried (queue)
    registered_kpoints::Vector{Vec3{T}}

    # Current position in the registered queue
    current_index::Int

    # Cached results from the current batch
    # cached_results[i] corresponds to registered_kpoints[batch_start + i - 1]
    cached_results::Matrix{Complex{T}}      # (ndata × batch_size)

    # Index range of currently cached batch
    cached_batch_start::Int
    cached_batch_end::Int

    # Buffers for batched Fourier transform
    phase_batch::Matrix{Complex{T}}         # (nr × batch_size)

    # Buffer for intermediate calculations
    buffer::Vector{Complex{T}}

    # Buffer for diagonalization
    ws::HermitianEigenWsSYEV{Complex{T},T}

    # Tolerance for k-point comparison
    const xk_tol::T

    function BatchedWannierInterpolator(parent::WT; batch_size::Int=32, xk_tol=sqrt(eps(T))/100) where {WT <: AbstractWannierObject{T}} where {T}
        nr = length(parent.irvec)
        ws = HermitianEigenWsSYEV{Complex{T},T}()
        new{T, WT}(
            parent,
            batch_size,
            Vec3{T}[],              # registered_kpoints
            1,                       # current_index
            zeros(Complex{T}, parent.ndata, batch_size),  # cached_results
            0,                       # cached_batch_start
            0,                       # cached_batch_end
            zeros(Complex{T}, nr, batch_size),  # phase_batch
            Complex{T}[],            # buffer
            ws,
            T(xk_tol)
        )
    end
end


"""
    clear_registered_kpoints!(obj::BatchedWannierInterpolator)

Clear registered k-points and cached results. Resets the interpolator to initial state.
"""
function clear_registered_kpoints!(obj::BatchedWannierInterpolator)
    empty!(obj.registered_kpoints)
    obj.current_index = 1
    obj.cached_batch_start = 0
    obj.cached_batch_end = 0
    nothing
end


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
function register_kpoints!(obj::BatchedWannierInterpolator{T}, xk_list) where {T}
    # Clear previous state
    clear_registered_kpoints!(obj)

    # Register new k-points
    append!(obj.registered_kpoints, xk_list)
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
@timing "get_fourier" function get_fourier!(op_k, obj::BatchedWannierInterpolator{T, WT}, xk) where {T, WT}
    (; parent, current_index, registered_kpoints, xk_tol) = obj
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == parent.ndata
    op_k_1d = _reshape(op_k, (length(op_k),))

    # Check if we have registered k-points
    if isempty(registered_kpoints)
        error("No k-points registered. Call register_kpoints! before using get_fourier!")
    end

    # Check if we've exhausted all registered k-points
    if current_index > length(registered_kpoints)
        error("All registered k-points have been exhausted. Current index: $current_index, total registered: $(length(registered_kpoints))")
    end

    # Check if xk matches the expected next k-point
    expected_xk = registered_kpoints[current_index]
    if !isapprox(xk, expected_xk; atol=xk_tol)
        error("K-point mismatch! Expected $(expected_xk) (index $current_index), got $(xk)")
    end

    # Check if we need to compute a new batch
    if current_index < obj.cached_batch_start || current_index > obj.cached_batch_end
        # Need to compute new batch
        _compute_batch!(obj, current_index)
    end

    # Return cached result
    cache_offset = current_index - obj.cached_batch_start + 1
    @views op_k_1d .= obj.cached_results[1:parent.ndata, cache_offset]

    # Advance to next k-point
    obj.current_index += 1

    return op_k
end


"""
    _compute_batch!(obj::BatchedWannierInterpolator, start_idx::Int)

Internal function: Compute a batch of k-points starting from the given index.
Uses BLAS3 matrix-matrix multiplication for efficiency.
"""
function _compute_batch!(obj::BatchedWannierInterpolator{T, WT}, start_idx::Int) where {T, WT}
    (; parent, batch_size, registered_kpoints, phase_batch, cached_results) = obj

    # Determine batch range
    batch_start = start_idx
    batch_end = min(start_idx + batch_size - 1, length(registered_kpoints))
    batch_len = batch_end - batch_start + 1

    # Compute phases for all k-points in this batch
    for (ik_local, ik_global) in enumerate(batch_start:batch_end)
        xk = registered_kpoints[ik_global]
        @views phase_batch[:, ik_local] .= cispi.(2 .* dot.(parent.irvec, Ref(xk)))
    end

    # Batched matrix-matrix multiplication
    if WT <: DiskWannierObject
        # For disk objects, still need to loop over R
        @views cached_results[:, 1:batch_len] .= 0
        for ir in 1:parent.nr
            op_r_ir = read_op_r(parent, ir)
            for ik_local in 1:batch_len
                @views cached_results[1:parent.ndata, ik_local] .+=
                    phase_batch[ir, ik_local] .* op_r_ir
            end
        end
    else
        # BLAS3 gemm: much faster than multiple BLAS2 gemv calls
        @views mul!(cached_results[1:parent.ndata, 1:batch_len],
                   parent.op_r[1:parent.ndata, :],
                   phase_batch[:, 1:batch_len])
    end

    # Update cache metadata
    obj.cached_batch_start = batch_start
    obj.cached_batch_end = batch_end

    nothing
end
