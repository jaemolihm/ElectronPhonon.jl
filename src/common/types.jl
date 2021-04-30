# Adapted from DFTK.jl/src/common/types.jl

using StaticArrays

# Frequently-used static types
const Mat3{T} = SMatrix{3, 3, T, 9} where T
const Vec3{T} = SVector{3, T} where T
const Vec2{T} = SVector{2, T} where T

# Symmetry operations (S, tau)
# S: rotation matrix in reciprocal crystal coordinates
# tau: fractional translation in real-space crystal coordinates
const SymOp = Tuple{Mat3{Int}, Vec3{Float64}}
identity_symop() = (Mat3{Int}(I), Vec3(zeros(3)))