"""
Custom reshape function that always returns ReshapedArray. If the input is ReshapedArray,
use its parent directly as a parent of the output array.
Regarding the use of ReshapedArray, see
https://discourse.julialang.org/t/passing-views-to-function-without-allocation/51992/12
https://github.com/ITensor/NDTensors.jl/issues/32
TODO: Do we need to check size? (prod(size(x)) >= prod(dims) and prod(x.dims) >= prod(dims))?)
"""
_reshape(x::AbstractArray, dims) = Base.ReshapedArray(x, dims, ())
_reshape(x::Base.ReshapedArray, dims) = Base.ReshapedArray(x.parent, dims, ())

"""
    _reshape_buffer(buffer::Vector{T}, dims::NTuple{N, Int}) where {T, N}
Get preallocated buffer as a ReshapedArray.
Resize buffer if the allocated size is smaller than the requested size.
"""
function _reshape_buffer(buffer::AbstractVector{T}, dims::NTuple{N, Int}) where {T, N}
    n = prod(dims)
    if length(buffer) < n
        resize!(buffer, n)
    end
    Base.ReshapedArray(view(buffer, 1:n), dims, ())
end

"""
    reshape_buffer_view(buffer, dims...; offset = 0) -> AbstractArray

View the prod(dims) elements of a preallocated buffer after its first `offset` with shape dims and
dense strides. A smaller box uses consecutive storage rather than the original strides; no elements
are moved, and views at disjoint offsets are disjoint segments of one buffer. On a device this
preserves the device array type required by batched GEMMs. The buffer must already have enough
capacity. Unlike _reshape_buffer, this does not resize storage or force a Base.ReshapedArray
wrapper.
"""
reshape_buffer_view(buffer, dims...; offset::Int = 0) =
    reshape(view(vec(buffer), offset .+ (1:prod(dims))), dims)
