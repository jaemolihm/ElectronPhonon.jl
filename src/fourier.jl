# __precompile__(true)

# module FourierModule

using Parameters
using SharedArrays
using LinearAlgebra
using Base.Threads

export AbstractWannierObject
export WannierObject
export get_fourier!
export update_op_r!

abstract type AbstractWannierObject{T<:Real} end

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
Base.@kwdef struct WannierObject{T} <: AbstractWannierObject{T}
    nr::Int
    irvec::Vector{Vec3{Int}}
    op_r::Array{Complex{T},2}
    ndata::Int # First dimension of op_r

    # For gridopt Fourier transform
    gridopts::Vector{GridOpt{T}}

    # Allocated buffer for normal Fourier transform
    rdotks::Vector{Vector{T}}
    phases::Vector{Vector{Complex{T}}}

    # For a higher-order WannierObject, the irvec to be used to Fourier transform op_k.
    irvec_next::Union{Nothing,Vector{Vec3{Int}}}
end

function WannierObject(irvec::Vector{Vec3{Int}}, op_r; irvec_next=nothing)
    nr = length(irvec)
    if ! check_wannierobject(irvec, op_r)
        error("WannierObject constructor check failed")
    end
    T = eltype(op_r).parameters[1]
    WannierObject{T}(nr=nr, irvec=irvec, op_r=op_r, ndata=size(op_r, 1),
        gridopts=[GridOpt{T}() for i=1:Threads.nthreads()],
        rdotks=[zeros(T, nr) for i=1:Threads.nthreads()],
        phases=[zeros(Complex{T}, nr) for i=1:Threads.nthreads()],
        irvec_next=irvec_next,
        )
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

function get_phase_expikr!(obj, xk, tid)
    rdotk = obj.rdotks[tid]
    phase = obj.phases[tid]
    for (ir, r) in enumerate(obj.irvec)
        rdotk[ir] = dot(r, 2pi*xk)
    end
    phase .= cis.(rdotk)
    phase
end

function update_op_r!(obj, op_r_new)
    # Regarding the use of ReshapedArray, see
    # https://discourse.julialang.org/t/passing-views-to-function-without-allocation/51992/12
    # https://github.com/ITensor/NDTensors.jl/issues/32
    @assert length(obj.op_r) == length(op_r_new)
    @assert eltype(obj.op_r) == eltype(op_r_new)
    # Reshape and set obj.op_r .= op_r_new without allocation
    obj.op_r .= Base.ReshapedArray(op_r_new, size(obj.op_r), ())
    for gridopt in obj.gridopts
        gridopt.k1 = NaN
        gridopt.k2 = NaN
    end
end

"Fourier transform real-space operator to momentum-space operator"
@timing "get_fourier" function get_fourier!(op_k, obj::AbstractWannierObject{T}, xk; mode="normal") where {T}
    # Regarding the use of ReshapedArray, see
    # https://discourse.julialang.org/t/passing-views-to-function-without-allocation/51992/12
    # https://github.com/ITensor/NDTensors.jl/issues/32

    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == obj.ndata

    op_k_1d = Base.ReshapedArray(op_k, (length(op_k),), ())

    if mode == "normal"
        phase = get_phase_expikr!(obj, xk, threadid())
        _get_fourier_normal!(op_k_1d, obj, xk, phase)
    elseif mode == "gridopt"
        _get_fourier_gridopt!(op_k_1d, obj, xk)
    else
        error("mode must be normal or gridopt")
    end
    return
end

"Fourier transform real-space operator to momentum-space operator using a
pre-computed phase factor"
@timing "get_fourier" function get_fourier!(op_k, obj::AbstractWannierObject{T}, xk, phase; mode="normal") where {T}
    # Regarding the use of ReshapedArray, see
    # https://discourse.julialang.org/t/passing-views-to-function-without-allocation/51992/12
    # https://github.com/ITensor/NDTensors.jl/issues/32

    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == obj.ndata
    @assert eltype(phase) == Complex{T}
    @assert length(phase) == obj.nr

    op_k_1d = Base.ReshapedArray(op_k, (length(op_k),), ())

    _get_fourier_normal!(op_k_1d, obj, xk, phase)
end

"Fourier transform real-space operator to momentum-space operator with a
pre-computed phase factor"
function _get_fourier_normal!(op_k_1d, obj::AbstractWannierObject{T}, xk, phase) where {T}
    mul!(op_k_1d, obj.op_r, phase)
    return
end

"Fourier transform real-space operator to momentum-space operator with grid optimization"
function _get_fourier_gridopt!(op_k_1d, obj::AbstractWannierObject{T}, xk) where {T}
    tid = Threads.threadid()
    gridopt = obj.gridopts[tid]

    if ! gridopt.is_initialized
        gridopt_initialize!(gridopt, obj.irvec, obj.op_r)
    end

    if ! isapprox(xk[1], gridopt.k1, atol=1.e-9)
        gridopt_set23!(gridopt, obj.irvec, obj.op_r, xk[1])
    end
    if ! isapprox(xk[2], gridopt.k2, atol=1.e-9)
        gridopt_set3!(gridopt, xk[2])
    end

    gridopt_get3!(op_k_1d, gridopt, xk[3])
end

# end # FourierModule
