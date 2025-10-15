using LinearAlgebra

"""
    BatchedGridoptWannierInterpolator{T, WT}

Stateful Wannier interpolator combining GridOpt optimization with batched k3 computation.

# Usage
1. Create interpolator: `itp = BatchedGridoptWannierInterpolator(wannier_obj; batch_size=32)`
2. Register k-points: `register_kpoints!(itp, kpoint_list)`
3. Query sequentially: `get_fourier!(op_k, itp, xk)` for each xk in order

# Performance
Combines GridOpt's staged transformation (for k1, k2) with BLAS3 batching (for k3),
providing optimal performance for grid k-point queries.

# Notes
- K-points must be registered before querying
- K-points should be ordered with k1 changing slowest, k3 changing fastest
- Out-of-order or unregistered queries will throw an error
"""
mutable struct BatchedGridoptWannierInterpolator{T, WT <: AbstractWannierObject} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated
    const parent::WT

    # Batch size for k3 dimension
    const batch_size::Int

    # GridOpt for staged transformation (handles k1, k2)
    const gridopt::GridOpt{T}

    # Registered k-points to be queried (queue)
    registered_kpoints::Vector{Vec3{T}}

    # Current position in the registered queue
    current_index::Int

    # Cached results from the current k3 batch
    # cached_results[i] corresponds to registered_kpoints[batch_start + i - 1]
    cached_results::Matrix{Complex{T}}      # (ndata × batch_size)

    # Index range of currently cached batch
    cached_batch_start::Int
    cached_batch_end::Int

    # Batched phases for k3 dimension
    phase_3_batch::Matrix{Complex{T}}       # (nr_3 × batch_size)

    # Output buffer
    out::Vector{Complex{T}}

    # Buffer for intermediate calculations
    buffer::Vector{Complex{T}}

    # Buffer for diagonalization
    ws::HermitianEigenWsSYEV{Complex{T},T}

    # Check if `gridopt` is up-to-date with `parent`
    _id::Int

    # Tolerance for k-point comparison
    const xk_tol::T

    function BatchedGridoptWannierInterpolator(parent::WT; batch_size::Int=32, threads=false, xk_tol=sqrt(eps(T))/100) where {WT <: AbstractWannierObject{T}} where {T}
        gridopt = GridOpt(T, parent.irvec, parent.ndata, threads)
        ws = HermitianEigenWsSYEV{Complex{T},T}()
        nr_3 = gridopt.nr_3

        new{T, WT}(
            parent,
            batch_size,
            gridopt,
            Vec3{T}[],              # registered_kpoints
            1,                       # current_index
            zeros(Complex{T}, parent.ndata, batch_size),  # cached_results
            0,                       # cached_batch_start
            0,                       # cached_batch_end
            zeros(Complex{T}, nr_3, batch_size),  # phase_3_batch
            zeros(Complex{T}, parent.ndata),  # out
            Complex{T}[],            # buffer
            ws,
            parent._id,
            T(xk_tol)
        )
    end
end


"""
    clear_registered_kpoints!(obj::BatchedGridoptWannierInterpolator)

Clear registered k-points and cached results. Resets the interpolator to initial state.
"""
function clear_registered_kpoints!(obj::BatchedGridoptWannierInterpolator)
    empty!(obj.registered_kpoints)
    obj.current_index = 1
    obj.cached_batch_start = 0
    obj.cached_batch_end = 0
    reset_gridopt!(obj.gridopt)
    nothing
end


"""
    register_kpoints!(obj::BatchedGridoptWannierInterpolator, xk_list)

Register a sequence of k-points that will be queried via `get_fourier!`.
The k-points must be queried in the exact order they are registered.

# Arguments
- `obj`: BatchedGridoptWannierInterpolator
- `xk_list`: Vector of k-points to be registered

# Notes
- Clears any previously registered k-points and cached results
- K-points MUST be queried in the same order via `get_fourier!`
- For optimal performance, k-points should be ordered with k1 slowest, k3 fastest
- Querying out-of-order or unregistered k-points will throw an error
"""
function register_kpoints!(obj::BatchedGridoptWannierInterpolator{T}, xk_list) where {T}
    # Clear previous state
    clear_registered_kpoints!(obj)

    # Register new k-points
    append!(obj.registered_kpoints, xk_list)
    nothing
end


"""
    get_fourier!(op_k, obj::BatchedGridoptWannierInterpolator{T}, xk)

Compute Fourier transform at k-point xk.

# Behavior
- xk must match the next registered k-point in sequence
- Updates GridOpt cache if (k1, k2) changed
- If this k-point starts a new k3 batch, triggers batched computation
- Otherwise returns cached result from the current batch
- Throws error if xk doesn't match the expected next k-point

# Arguments
- `op_k`: Output vector of size ndata
- `obj`: BatchedGridoptWannierInterpolator
- `xk`: k-point (Vec3)

# Errors
- If no k-points are registered
- If xk doesn't match the next expected k-point
- If all registered k-points have been exhausted
"""
@timing "get_fourier" function get_fourier!(op_k, obj::BatchedGridoptWannierInterpolator{T, WT}, xk) where {T, WT}
    (; parent, gridopt, current_index, registered_kpoints, xk_tol) = obj
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == parent.ndata
    ndata = parent.ndata
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

    # Check if parent data changed
    if obj._id != parent._id
        reset_gridopt!(gridopt)
        obj._id = parent._id
    end

    # Update GridOpt cache if (k1, k2) changed
    if !isapprox(xk[1], gridopt.k1; atol=xk_tol)
        gridopt_set23!(gridopt, parent, xk[1], ndata)
        # k1 changed, so we need new k3 batch
        obj.cached_batch_start = 0
        obj.cached_batch_end = 0
    end
    if !isapprox(xk[2], gridopt.k2; atol=xk_tol)
        gridopt_set3!(gridopt, xk[2], ndata)
        # k2 changed, so we need new k3 batch
        obj.cached_batch_start = 0
        obj.cached_batch_end = 0
    end

    # Check if we need to compute a new k3 batch
    if current_index < obj.cached_batch_start || current_index > obj.cached_batch_end
        # Need to compute new batch
        _compute_k3_batch!(obj, current_index)
    end

    # Return cached result
    cache_offset = current_index - obj.cached_batch_start + 1
    @views op_k_1d .= obj.cached_results[1:ndata, cache_offset]

    # Advance to next k-point
    obj.current_index += 1

    return op_k
end


"""
    _compute_k3_batch!(obj::BatchedGridoptWannierInterpolator, start_idx::Int)

Internal function: Compute a batch of k-points with same (k1, k2) but different k3.
Uses BLAS3 matrix-matrix multiplication for efficiency.
"""
function _compute_k3_batch!(obj::BatchedGridoptWannierInterpolator{T}, start_idx::Int) where {T}
    (; parent, gridopt, batch_size, registered_kpoints, phase_3_batch, cached_results, xk_tol) = obj
    ndata = parent.ndata
    nr_3 = gridopt.nr_3

    # Find batch range: all k-points with same (k1, k2)
    batch_start = start_idx
    batch_end = batch_start
    current_k1 = gridopt.k1
    current_k2 = gridopt.k2

    # Extend batch_end while (k1, k2) match and within batch_size
    while batch_end < length(registered_kpoints) &&
          batch_end - batch_start + 1 < batch_size

        next_k = registered_kpoints[batch_end + 1]
        if isapprox(next_k[1], current_k1; atol=xk_tol) &&
           isapprox(next_k[2], current_k2; atol=xk_tol)
            batch_end += 1
        else
            break
        end
    end

    batch_len = batch_end - batch_start + 1

    # Compute phases for k3 values in this batch
    @views for (ik_local, ik_global) in enumerate(batch_start:batch_end)
        k3 = registered_kpoints[ik_global][3]
        @. phase_3_batch[:, ik_local] = cispi(2 * k3 * gridopt.irvec_3)
    end

    # BLAS3: Batch multiplication
    # cached_results = op_r_3 * phase_3_batch
    # (ndata × batch_len) = (ndata × nr_3) * (nr_3 × batch_len)
    @views mul!(cached_results[1:ndata, 1:batch_len],
                gridopt.op_r_3[1:ndata, :],
                phase_3_batch[:, 1:batch_len])

    # Update cache metadata
    obj.cached_batch_start = batch_start
    obj.cached_batch_end = batch_end

    nothing
end
