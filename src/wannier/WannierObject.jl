using LinearAlgebra

export update_op_r!

abstract type AbstractWannierObject{T<:Real} end
Base.eltype(::Type{<:AbstractWannierObject{T}}) where {T} = Complex{T}

"""
    SymmWSLayout

Index data of an e-ph `WannierObject` stored in the y layout of EPW `use_ws = 'symmetric'`:
`op_r[(a, iy), iRe]` holds the element `a = (i, j, mode)` at R_p = y[iy] + fld(R_e, 2), so the
columns (R_e) form groups of equal shift s = fld(R_e, 2) and a group's `nr_y` y vectors land on R_p
`irvec_next[ip_of_y[:, g]]`. The object's `irvec` lists R_e group by group (not sorted), column
range `group_start[g]:group_start[g+1]-1` being group `g`. The output of the R_e transform lives on
the `nr_p` vectors of `irvec_next` (`ndata = nw^2 nmodes nr_p`), as for the expanded storage.
Built by [`read_epmat_symmws_y`](@ref); consumed by the batched Fourier engine
(`_fourier_batched!` of [`BatchedFourierCore`](@ref)).
"""
struct SymmWSLayout
    nr_p::Int
    nr_y::Int
    group_start::Vector{Int}
    ip_of_y::Matrix{Int}
end

"""
    SymmWSLayout(irvec_e, y_file, irvec_p)

Layout of the columns `irvec_e` (R_e, already grouped by contiguous equal `fld.(R_e, 2)`) with the
y vectors `y_file` (row blocks, in the order of the stored rows) on the R_p list `irvec_p`. Called by
`read_epmat_symmws_y`; it throws if the groups are not contiguous or a y + s is missing in `irvec_p`.
"""
function SymmWSLayout(irvec_e, y_file, irvec_p)
    index_p = Dict(R => i for (i, R) in enumerate(irvec_p))
    shifts = [fld.(R, 2) for R in irvec_e]
    group_start = [1; [i for i in 2:length(shifts) if shifts[i] != shifts[i-1]]; length(shifts) + 1]
    unique(shifts[group_start[1:end-1]]) == shifts[group_start[1:end-1]] ||
        throw(ArgumentError("irvec_e is not grouped by fld.(R_e, 2)"))
    ip_of_y = [index_p[y + shifts[group_start[g]]] for y in y_file, g in 1:length(group_start)-1]
    SymmWSLayout(length(irvec_p), length(y_file), group_start, ip_of_y)
end

"Check validity of input for WannierObject constructor"
function check_wannierobject(irvec::Vector{Vec3{Int}}, op_r; sorted = true)
    if size(op_r, 2) != length(irvec)
        println("size(op_r, 2)=$(size(op_r, 2)) must equal length(irvec)=$(length(irvec))")
        return false
    end
    if sorted && ! issorted(irvec, by=x->reverse(x))
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

    # Set for the y layout of EPW `use_ws = 'symmetric'` (see [`SymmWSLayout`](@ref)); `nothing` otherwise.
    # Only the batched Fourier engine transforms such an object.
    const symmws::Union{Nothing, SymmWSLayout} = nothing
end

# In-memory (host) `WannierObject`: `op_r` is a plain `Matrix`. `Model` stores only host
# objects (device, e.g. `CuMatrix`-backed, objects are created transiently by `to_device` and
# never stored back), so its fields use this concrete type to stay type-stable — the widened
# `WannierObject{T, AT}` would otherwise leave them as an abstract `UnionAll`.
const HostWannierObject{T} = WannierObject{T, Matrix{Complex{T}}}

function WannierObject(irvec::Vector{Vec3{Int}}, op_r; irvec_next=nothing, sort=false, ndata=nothing, symmws=nothing)
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

    # The y layout lists R_e group by group (see `SymmWSLayout`), not sorted
    if ! check_wannierobject(irvec_, op_r_; sorted = symmws === nothing)
        error("WannierObject constructor check failed")
    end
    # `ndata` defaults to the full first dimension; pass a smaller value to make a
    # partial-transform object (only its first `ndata` rows of `op_r` are Fourier-transformed).
    ndata_ = ndata === nothing ? size(op_r_, 1) : ndata
    # FIXME: better type parameter for WannierObject
    T = eltype(op_r_).parameters[1]  # ComplexF64 -> Float64
    WannierObject{T, typeof(op_r_)}(nr=nr, irvec=irvec_, op_r=op_r_, ndata=ndata_,
        irvec_next=irvec_next, _id=0, symmws=symmws,
    )
end

function Base.show(io::IO, obj::AbstractWannierObject)
    print(io, typeof(obj), "(nr=$(obj.nr), ndata=$(obj.ndata))")
end

# WannierObject(nr, irvec::Array{Int,2}, op_r) = WannierObject(nr, reinterpret(Vec3{Int}, vec(irvec))[:], op_r)

function wannier_object_multiply_R(obj::AbstractWannierObject{T}, lattice) where {T}
    obj.symmws === nothing || throw(ArgumentError("a WannierObject in the y layout (epmat_symmws_storage = \"y\") has no R-weighted copy"))
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
    get_next_wannier_object(parent :: WannierObject{T}; ndata_child = nothing) where {T}
For a higher-order WannierObject (e.g. `g(Rₑ, Rₚ) → g(k, Rₚ) → g(k, q)`), return a
child WannierObject whose `child.irvec` is `parent.irvec_next`.

`op_r` is always allocated at the full child width. Pass `ndata_child` (≤ the full width) to make
the child *born* partial-transform — only its first `ndata_child` rows are Fourier-transformed —
instead of shrinking `ndata` on the object after construction.
"""
function get_next_wannier_object(parent :: WannierObject{T}; ndata_child = nothing) where {T}
    if parent.irvec_next === nothing
        throw(ArgumentError("irvec_next must be set"))
    end
    nr_child = length(parent.irvec_next)
    mod(parent.ndata, nr_child) == 0 || throw(ArgumentError("ndata must be divisible by length(irvec_next)"))
    ndata_child_full = div(parent.ndata, nr_child)
    if ndata_child !== nothing && !(1 <= ndata_child <= ndata_child_full)
        throw(ArgumentError("ndata_child=$ndata_child must be in 1:$ndata_child_full"))
    end
    WannierObject(parent.irvec_next, zeros(Complex{T}, (ndata_child_full, nr_child)); ndata = ndata_child)
end


"""
    _irvec_to_device_matrix(backend, irvec, ::Type{T}) -> (nr × 3) real matrix

R-vectors of a `WannierObject` as an `(nr × 3)` real matrix on `backend`, the layout
[`build_fourier_phase!`](@ref) expects. Built on the host and copied over once.
"""
function _irvec_to_device_matrix(backend, irvec, ::Type{T}) where {T}
    nr = length(irvec)
    to_device_copy(backend, T[irvec[ir][d] for ir in 1:nr, d in 1:3])
end
