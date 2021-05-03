# Adapted from DFTK.jl/src/common/types.jl

using StaticArrays

# Frequently-used static types
const Mat3{T} = SMatrix{3, 3, T, 9} where T
const Vec3{T} = SVector{3, T} where T
const Vec2{T} = SVector{2, T} where T
