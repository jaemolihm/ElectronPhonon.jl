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
