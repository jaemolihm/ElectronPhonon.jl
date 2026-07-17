using LinearAlgebra

export update_op_r!

abstract type AbstractWannierObject{T<:Real} end
Base.eltype(::Type{<:AbstractWannierObject{T}}) where {T} = Complex{T}

"Check validity of input for WannierObject constructor"
function check_wannierobject(irvec::Vector{Vec3{Int}}, op_r)
    if size(op_r, 2) != length(irvec)
        println("size(op_r, 2)=$(size(op_r, 2)) must equal length(irvec)=$(length(irvec))")
        return false
    end
    if ! issorted(irvec, by=x->reverse(x))
        println("irvec is not sorted. irvec must be sorted by reverse(r)")
        return false
    end
    return true
end

"Data in coarse real-space grid for a single operator"
Base.@kwdef mutable struct WannierObject{T, AT <: AbstractMatrix{Complex{T}}} <: AbstractWannierObject{T}
    const nr::Int
    const irvec::Vector{Vec3{Int}}
    # `op_r` is an `AbstractMatrix` so that it may live on a device (e.g. a `CuMatrix` from
    # the CUDA extension) as well as in host memory. `irvec` always stays on the host.
    const op_r::AT
    ndata::Int # Size of the Fourier-transformed data matrix
               # By default, ndata = size(op_r, 1). If one only wants a part of op_r to be
               # transformed, one may use ndata to be smaller.

    # For a higher-order WannierObject, the irvec to be used to Fourier transform op_k.
    const irvec_next::Union{Nothing,Vector{Vec3{Int}}}

    # When op_r is updated, increment _id.
    # This is used to check if interpolators are up-to-date.
    _id::Int
end

# In-memory (host) `WannierObject`: `op_r` is a plain `Matrix`. `Model` stores only host
# objects (device, e.g. `CuMatrix`-backed, objects are created transiently by `to_device` and
# never stored back), so its fields use this concrete type to stay type-stable — the widened
# `WannierObject{T, AT}` would otherwise leave them as an abstract `UnionAll`.
const HostWannierObject{T} = WannierObject{T, Matrix{Complex{T}}}

function WannierObject(irvec::Vector{Vec3{Int}}, op_r; irvec_next=nothing, sort=false, ndata=nothing)
    nr = length(irvec)

    # Sort R vectors
    if sort
        inds = sortperm(irvec, by=x->reverse(x))
        irvec_ = irvec[inds]
        op_r_ = op_r[:, inds]
    else
        irvec_ = irvec
        op_r_ = op_r
    end

    if ! check_wannierobject(irvec_, op_r_)
        error("WannierObject constructor check failed")
    end
    # `ndata` defaults to the full first dimension; pass a smaller value to make a
    # partial-transform object (only its first `ndata` rows of `op_r` are Fourier-transformed).
    ndata_ = ndata === nothing ? size(op_r_, 1) : ndata
    # FIXME: better type parameter for WannierObject
    T = eltype(op_r_).parameters[1]  # ComplexF64 -> Float64
    WannierObject{T, typeof(op_r_)}(nr=nr, irvec=irvec_, op_r=op_r_, ndata=ndata_,
        irvec_next=irvec_next, _id=0,
    )
end

function Base.show(io::IO, obj::AbstractWannierObject)
    print(io, typeof(obj), "(nr=$(obj.nr), ndata=$(obj.ndata))")
end

# WannierObject(nr, irvec::Array{Int,2}, op_r) = WannierObject(nr, reinterpret(Vec3{Int}, vec(irvec))[:], op_r)

function wannier_object_multiply_R(obj::AbstractWannierObject{T}, lattice) where {T}
    # Put R to the slowest index
    opR_r = zeros(Complex{T}, (obj.ndata, 3, obj.nr))
    for ir = 1:obj.nr
        @views for i = 1:3
            opR_r[:, i, ir] .= im .* obj.op_r[:, ir] .* dot(lattice[i, :], obj.irvec[ir])
        end
    end
    WannierObject(obj.irvec, reshape(opR_r, (obj.ndata*3, obj.nr)); obj.irvec_next)
end

"""
    update_op_r!(obj, op_r_new; rows = axes(obj.op_r, 1))

Set `obj.op_r[rows, :] .= op_r_new` in place and bump `obj._id` to invalidate any
`_id`-tracking (GridOpt-family) interpolator caches. `op_r_new` may have any shape as long as
its length matches the destination. Pass `rows` to write only a leading row range of a
partial-transform object (`ndata < size(op_r, 1)`); the default writes the whole `op_r`.

This is the single entry point that writes `obj._id`.
"""
function update_op_r!(obj, op_r_new; rows = axes(obj.op_r, 1))
    dst = @view obj.op_r[rows, :]
    @assert length(dst) == length(op_r_new)
    @assert eltype(obj.op_r) == eltype(op_r_new)
    # Reshape and set dst .= op_r_new without allocation (skip the reshape when shapes match,
    # keeping the device-view assignment identical to a plain broadcast copy).
    if size(dst) == size(op_r_new)
        dst .= op_r_new
    else
        dst .= _reshape(op_r_new, size(dst))
    end
    obj._id += 1
    obj
end

"""
    get_next_wannier_object(parent :: WannierObject{T}; ndata = nothing) where {T}
For a higher-order WannierObject (e.g. `g(Rₑ, Rₚ) → g(k, Rₚ) → g(k, q)`), return a
child WannierObject whose `child.irvec` is `parent.irvec_next`.

`op_r` is always allocated at the full child width. Pass `ndata` (≤ the full width) to make
the child *born* partial-transform — only its first `ndata` rows are Fourier-transformed —
instead of shrinking `ndata` on the object after construction.
"""
function get_next_wannier_object(parent :: WannierObject{T}; ndata = nothing) where {T}
    if parent.irvec_next === nothing
        throw(ArgumentError("irvec_next must be set"))
    end
    nr_child = length(parent.irvec_next)
    mod(parent.ndata, nr_child) == 0 || throw(ArgumentError("ndata must be divisible by length(irvec_next)"))
    ndata_child = div(parent.ndata, nr_child)
    if ndata !== nothing && !(1 <= ndata <= ndata_child)
        throw(ArgumentError("ndata=$ndata must be in 1:$ndata_child"))
    end
    WannierObject(parent.irvec_next, zeros(Complex{T}, (ndata_child, nr_child)); ndata)
end
