using LinearAlgebra

# Allocate an array on the same backend as `parent.op_r`. For DiskWannierObject (which has
# no in-memory op_r) fall back to host memory.
_alloc_array(parent::AbstractWannierObject, ::Type{S}, dims...) where {S} = zeros(S, dims...)
_alloc_array(parent::WannierObject, ::Type{S}, dims...) where {S} = similar(parent.op_r, S, dims...)


"""
    BatchedFourierCore{T, WT, MC, MR}

Stateless whole-batch Wannier → Bloch Fourier engine. Holds only the persistent GEMM scratch
needed to interpolate up to `batch_cap` k-points in a single call; it has no notion of
registration order or cache position (that is [`SequentialQueryCache`](@ref)'s job).

Interpolates fresh from `parent.op_r` on every call and never consults `parent._id`, so it is
stateless with respect to the parent data (`_id` is a GridOpt-family concern).

# Type parameters
The buffer arrays follow the backend of `parent.op_r`:
- `MC` — complex matrix : CPU `Matrix{Complex{T}}`, GPU `CuMatrix{Complex{T}}`
- `MR` — real matrix    : CPU `Matrix{T}`,          GPU `CuMatrix{T}`

(For a `DiskWannierObject` parent the buffers fall back to host arrays.)
"""
struct BatchedFourierCore{T, WT <: AbstractWannierObject, MC, MR}
    # Parent WannierObject to be interpolated
    parent::WT

    # Maximum number of k-points a single `fourier_batched!` call handles
    batch_cap::Int

    # R-vectors as an (nr × 3) real matrix on the backend, for the GEMM phase computation
    irvec_mat::MR

    # Scratch for the batched phase computation, on parent's backend
    rdotk::MR                # (nr × batch_cap) real, scratch for irvec·k
    phase::MC                # (nr × batch_cap) complex

    # Persistent k-point matrix (3 × batch_cap): built on the host, copied to the backend.
    xkmat_host::Matrix{T}    # (3 × batch_cap) host
    xkmat::MR                # (3 × batch_cap) on backend
end

function BatchedFourierCore(parent::WT; batch_cap::Int=32) where {WT <: AbstractWannierObject{T}} where {T}
    nr = length(parent.irvec)

    # R-vectors as an (nr × 3) real matrix on the backend, for the GEMM phase computation.
    irvec_host = Matrix{T}(undef, nr, 3)
    for ir in 1:nr, d in 1:3
        irvec_host[ir, d] = parent.irvec[ir][d]
    end
    irvec_mat = _alloc_array(parent, T, nr, 3)
    copyto!(irvec_mat, irvec_host)

    rdotk = _alloc_array(parent, T, nr, batch_cap)
    phase = _alloc_array(parent, Complex{T}, nr, batch_cap)

    xkmat_host = Matrix{T}(undef, 3, batch_cap)
    xkmat = _alloc_array(parent, T, 3, batch_cap)

    BatchedFourierCore{T, WT, typeof(phase), typeof(irvec_mat)}(
        parent, batch_cap, irvec_mat, rdotk, phase, xkmat_host, xkmat)
end


"""
    fourier_batched!(out, core::BatchedFourierCore, xks)

Fourier-transform `core.parent` at all k-points in `xks` at once, writing into `out`
(`(ndata, length(xks))` on the backend of `core.parent.op_r`). `length(xks)` must be
`≤ core.batch_cap`. Uses one GEMM for the phases (`irvec_mat * xkmat`) and one GEMM for the
transform (`op_r * phase`), so it works on any backend (no scalar indexing of device buffers).
"""
function fourier_batched!(out, core::BatchedFourierCore{T, WT}, xks) where {T, WT}
    (; parent, irvec_mat, rdotk, phase, xkmat_host, xkmat) = core
    ndata = parent.ndata
    nk = length(xks)
    @assert nk <= core.batch_cap
    @assert size(out) == (ndata, nk)

    # k-point matrix (3 × nk), built in the persistent host buffer then copied to the backend.
    # For a CPU parent this copy is a redundant host→host move, but it is what lets the same code
    # feed a GPU parent (host→device); CPU batching is rarely used, so this is not specialized away.
    @views for (ik_local, xk) in enumerate(xks)
        xkmat_host[:, ik_local] .= xk
    end
    # Columns 1:nk are contiguous (column-major), so copy the first 3·nk linear elements. The
    # linear-range `copyto!` handles the host→device transfer without scalar indexing (a
    # host-SubArray → device-view `copyto!` would fall back to scalar indexing on the GPU).
    copyto!(xkmat, 1, xkmat_host, 1, 3 * nk)

    @views mul!(rdotk[:, 1:nk], irvec_mat, xkmat[:, 1:nk])       # (nr × nk) real
    @views @. phase[:, 1:nk] = cispi(2 * rdotk[:, 1:nk])

    if WT <: DiskWannierObject
        # For disk objects, still need to loop over R
        out .= 0
        for ir in 1:parent.nr
            op_r_ir = read_op_r(parent, ir)
            for ik_local in 1:nk
                @views out[:, ik_local] .+= phase[ir, ik_local] .* op_r_ir
            end
        end
    else
        # BLAS3 gemm: much faster than multiple BLAS2 gemv calls
        @views mul!(out, parent.op_r[1:ndata, :], phase[:, 1:nk])
    end
    out
end


"""
    SequentialQueryCache{T}

Shared adapter that turns a stateless batched Fourier engine into the order-enforced per-k
`get_fourier!` query API. Records the registered k-point queue, the current position, and the
index range of the batch currently cached by the owning interpolator.

The interpolator (e.g. [`BatchedWannierInterpolator`](@ref)) owns the cached-result buffer and
the batch-computation routine; this cache only tracks the queue bookkeeping that used to be
duplicated across the batched interpolator types.
"""
mutable struct SequentialQueryCache{T}
    # Registered k-points to be queried (queue)
    registered_kpoints::Vector{Vec3{T}}

    # Current position in the registered queue
    current_index::Int

    # Index range of the batch currently cached by the owning interpolator
    cached_batch_start::Int
    cached_batch_end::Int

    # Tolerance for k-point comparison
    const xk_tol::T
end

SequentialQueryCache{T}(; xk_tol=sqrt(eps(T))/100) where {T} =
    SequentialQueryCache{T}(Vec3{T}[], 1, 0, 0, T(xk_tol))


function clear_registered_kpoints!(cache::SequentialQueryCache)
    empty!(cache.registered_kpoints)
    cache.current_index = 1
    cache.cached_batch_start = 0
    cache.cached_batch_end = 0
    nothing
end

function register_kpoints!(cache::SequentialQueryCache, xk_list)
    clear_registered_kpoints!(cache)
    append!(cache.registered_kpoints, xk_list)
    nothing
end

"""
    _next_query_index(cache::SequentialQueryCache, xk) -> Int

Validate `xk` against the registered queue (order-enforced), returning the current index.
Throws if no k-points are registered, all are exhausted, or `xk` does not match the next
expected k-point.
"""
function _next_query_index(cache::SequentialQueryCache, xk)
    (; current_index, registered_kpoints, xk_tol) = cache

    if isempty(registered_kpoints)
        error("No k-points registered. Call register_kpoints! before using get_fourier!")
    end

    if current_index > length(registered_kpoints)
        error("All registered k-points have been exhausted. Current index: $current_index, total registered: $(length(registered_kpoints))")
    end

    expected_xk = registered_kpoints[current_index]
    if !isapprox(xk, expected_xk; atol=xk_tol)
        error("K-point mismatch! Expected $(expected_xk) (index $current_index), got $(xk)")
    end

    current_index
end
