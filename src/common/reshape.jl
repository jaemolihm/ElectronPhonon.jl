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