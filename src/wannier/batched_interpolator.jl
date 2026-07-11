using LinearAlgebra

export register_kpoints!
export clear_registered_kpoints!
export get_fourier_batched!

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

# Backends
The buffer arrays follow the backend of `parent.op_r`, so a GPU parent (a `WannierObject`
whose `op_r` is a `CuMatrix`) keeps the whole cached batch on the device. For the GPU /
whole-batch use case, prefer [`get_fourier_batched!`](@ref) over per-k `get_fourier!`,
which avoids a device→host copy per k-point.

# Notes
- K-points must be registered before querying
- K-points must be queried in the exact order they were registered
- Out-of-order or unregistered queries will throw an error
"""
# Type parameters of the buffer arrays follow the backend of `parent.op_r`:
#   MC — complex matrix : CPU `Matrix{Complex{T}}`, GPU `CuMatrix{Complex{T}}`
#   MR — real matrix    : CPU `Matrix{T}`,          GPU `CuMatrix{T}`
#   VC — complex vector : CPU `Vector{Complex{T}}`, GPU `CuVector{Complex{T}}`
# (For a DiskWannierObject parent the buffers fall back to host arrays.)
mutable struct BatchedWannierInterpolator{T, WT <: AbstractWannierObject, MC, MR, VC} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated
    const parent::WT

    # Batch size (number of k-points to process together)
    const batch_size::Int

    # Registered k-points to be queried (queue)
    registered_kpoints::Vector{Vec3{T}}

    # Current position in the registered queue
    current_index::Int

    # Cached results from the current batch (ndata × batch_size), on parent's backend
    cached_results::MC

    # Index range of currently cached batch
    cached_batch_start::Int
    cached_batch_end::Int

    # Buffers for batched Fourier transform, on parent's backend
    phase_batch::MC                         # (nr × batch_size) complex
    rdotk::MR                               # (nr × batch_size) real, scratch for irvec·k
    const irvec_mat::MR                     # (nr × 3) real R-vectors

    # Output buffer (per-k query API)
    out::VC

    # Buffer for intermediate calculations
    buffer::VC
    buffer2::VC

    # Buffer for diagonalization
    ws::HermitianEigenWsSYEV{Complex{T},T}

    # Tolerance for k-point comparison
    const xk_tol::T
end

# Allocate an array on the same backend as `parent.op_r`. For DiskWannierObject (which has
# no in-memory op_r) fall back to host memory.
_alloc_array(parent::AbstractWannierObject, ::Type{S}, dims...) where {S} = zeros(S, dims...)
_alloc_array(parent::WannierObject, ::Type{S}, dims...) where {S} = similar(parent.op_r, S, dims...)

function BatchedWannierInterpolator(parent::WT; batch_size::Int=32, xk_tol=sqrt(eps(T))/100) where {WT <: AbstractWannierObject{T}} where {T}
    nr = length(parent.irvec)
    ws = HermitianEigenWsSYEV{Complex{T},T}()

    cached_results = _alloc_array(parent, Complex{T}, parent.ndata, batch_size)
    phase_batch    = _alloc_array(parent, Complex{T}, nr, batch_size)
    rdotk          = _alloc_array(parent, T, nr, batch_size)

    # R-vectors as an (nr × 3) real matrix on the backend, for the GEMM phase computation.
    irvec_host = Matrix{T}(undef, nr, 3)
    for ir in 1:nr, d in 1:3
        irvec_host[ir, d] = parent.irvec[ir][d]
    end
    irvec_mat = _alloc_array(parent, T, nr, 3)
    copyto!(irvec_mat, irvec_host)

    out    = _alloc_array(parent, Complex{T}, parent.ndata)
    buffer = _alloc_array(parent, Complex{T}, 0)
    buffer2 = _alloc_array(parent, Complex{T}, 0)

    BatchedWannierInterpolator{T, WT, typeof(cached_results), typeof(irvec_mat), typeof(out)}(
        parent,
        batch_size,
        Vec3{T}[],               # registered_kpoints
        1,                       # current_index
        cached_results,
        0,                       # cached_batch_start
        0,                       # cached_batch_end
        phase_batch,
        rdotk,
        irvec_mat,
        out,
        buffer,
        buffer2,
        ws,
        T(xk_tol),
    )
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
function register_kpoints!(obj::BatchedWannierInterpolator, xk_list)
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
    _compute_phase_batch!(obj, batch_start, batch_end)

Fill `obj.phase_batch[:, 1:batch_len]` with `cispi(2 * irvec . xk)` for the k-points in the
batch, using a single GEMM (`irvec_mat * xkmat`) followed by a broadcast. Works on any
backend (CPU or GPU) since it uses no scalar indexing of the device buffers.
"""
function _compute_phase_batch!(obj::BatchedWannierInterpolator{T}, batch_start::Int, batch_end::Int) where {T}
    (; registered_kpoints, irvec_mat, rdotk, phase_batch) = obj
    batch_len = batch_end - batch_start + 1

    # k-point matrix (3 × batch_len), built on the host then copied to the backend. For a CPU
    # parent this copy is a redundant host→host move, but it is what lets the same code feed a
    # GPU parent (host→device); CPU batching is rarely used, so this is not specialized away.
    # TODO: `xkmat_host` and `xkmat` allocate a 3×batch_len host+device buffer on every call (once
    # per k-tile / q-chunk in the hot GPU loop); hoist them into the interpolator's persistent
    # buffers so this driver is allocation-free like the rest of `KRtoKQWorkspace`.
    xkmat_host = Matrix{T}(undef, 3, batch_len)
    for (ik_local, ik_global) in enumerate(batch_start:batch_end)
        xkmat_host[:, ik_local] .= registered_kpoints[ik_global]
    end
    xkmat = _alloc_array(obj.parent, T, 3, batch_len)
    copyto!(xkmat, xkmat_host)

    @views mul!(rdotk[:, 1:batch_len], irvec_mat, xkmat)          # (nr × batch_len) real
    @views @. phase_batch[:, 1:batch_len] = cispi(2 * rdotk[:, 1:batch_len])
    nothing
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

    # Compute phases for all k-points in this batch (one GEMM + broadcast)
    _compute_phase_batch!(obj, batch_start, batch_end)

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


"""
    get_fourier_batched!(out, obj::BatchedWannierInterpolator, xk_list)

Fourier-transform `obj` at all k-points in `xk_list` at once, writing into `out`
(`(ndata, nk)` on the backend of `obj.parent.op_r`). `length(xk_list)` may be larger than
`obj.batch_size`: the k-points are processed internally in chunks of `batch_size` reusing the
batched machinery, and the whole result is kept on the backend (no per-k device→host copy).
This is the entry point for GPU / whole-batch use.
"""
function get_fourier_batched!(out, obj::BatchedWannierInterpolator{T}, xk_list) where {T}
    ndata = obj.parent.ndata
    nk = length(xk_list)
    @assert size(out) == (ndata, nk)
    register_kpoints!(obj, xk_list)
    start = 1
    while start <= nk
        _compute_batch!(obj, start)
        len = obj.cached_batch_end - obj.cached_batch_start + 1
        @views out[:, start:start+len-1] .= obj.cached_results[1:ndata, 1:len]
        start += len
    end
    out
end


"""
    skip_registered_kpoint!(obj::BatchedWannierInterpolator)

Skip one registered k-point in the sequence and advance the current index.
"""
function skip_registered_kpoint!(obj::BatchedWannierInterpolator)
    obj.current_index += 1
    nothing
end
