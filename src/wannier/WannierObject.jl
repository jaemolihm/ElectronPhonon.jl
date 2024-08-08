using LinearAlgebra

export update_op_r!

abstract type AbstractWannierObject{T<:Real} end
Base.eltype(::Type{<:AbstractWannierObject{T}}) where {T} = Complex{T}

"Check validity of input for WannierObject constructor"
function check_wannierobject(irvec::Vector{Vec3{Int}}, op_r)
    if size(op_r, 2) != length(irvec)
        @error ("size(op_r, 2)=$(size(op_r, 2)) must equal length(irvec)=$(length(irvec))")
        return false
    end
    if ! issorted(irvec, by=x->reverse(x))
        @error "irvec is not sorted. irvec must be sorted by reverse(r)"
        return false
    end
    return true
end

"Data in coarse real-space grid for a single operator"
Base.@kwdef mutable struct WannierObject{T} <: AbstractWannierObject{T}
    const nr::Int
    const irvec::Vector{Vec3{Int}}
    const op_r::Array{Complex{T},2}
    ndata::Int # Size of the Fourier-transformed data matrix
               # By default, ndata = size(op_r, 1). If one only wants a part of op_r to be
               # transformed, one may use ndata to be smaller.

    # For a higher-order WannierObject, the irvec to be used to Fourier transform op_k.
    const irvec_next::Union{Nothing,Vector{Vec3{Int}}}

    # When op_r is updated, increment _id.
    # This is used to check if interpolators are up-to-date.
    _id::Int
end

function WannierObject(irvec::Vector{Vec3{Int}}, op_r; irvec_next=nothing, sort=false)
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
    T = eltype(op_r_).parameters[1]
    WannierObject{T}(nr=nr, irvec=irvec_, op_r=op_r_, ndata=size(op_r_, 1),
        irvec_next=irvec_next, _id=0,
    )
end

function Base.show(io::IO, obj::AbstractWannierObject)
    print(io, typeof(obj), "(nr=$(obj.nr), ndata=$(obj.ndata))")
end

# WannierObject(nr, irvec::Array{Int,2}, op_r) = WannierObject(nr, reinterpret(Vec3{Int}, vec(irvec))[:], op_r)

function wannier_object_multiply_R(obj::AbstractWannierObject{T}, lattice) where {T}
    opR_r = zeros(Complex{T}, (obj.ndata, 3, obj.nr))
    for ir = 1:obj.nr
        @views for i = 1:3
            opR_r[:, i, ir] .= im .* obj.op_r[:, ir] .* dot(lattice[i, :], obj.irvec[ir])
        end
    end
    WannierObject(obj.irvec, reshape(opR_r, (obj.ndata*3, obj.nr)))
end

function update_op_r!(obj, op_r_new)
    @assert length(obj.op_r) == length(op_r_new)
    @assert eltype(obj.op_r) == eltype(op_r_new)
    # Reshape and set obj.op_r .= op_r_new without allocation
    obj.op_r .= _reshape(op_r_new, size(obj.op_r))
    obj._id += 1
end
