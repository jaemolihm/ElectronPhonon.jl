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

"Check validity of input"
function check_wannierobject(nr, irvec::Array{Int,2}, op_r)
    if size(irvec) != (3, nr)
        @error ("size(irvec) must be (3, nr=$nr), not $(size(irvec))")
        return false
    end
    if ! issorted(irvec, by=x->reverse(x))
        @error "irvec is not sorted. irvec must be sorted by reverse(r)"
        return false
    end
    return true
end

"Check validity of input for WannierObject constructor"
function check_wannierobject(nr, irvec::Vector{Vec3{Int}}, op_r)
    if length(irvec) != nr
        @error ("length(irvec) must be nr=$nr, not $(length(irvec))")
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
end

function WannierObject(nr, irvec::Vector{Vec3{Int}}, op_r)
    if ! check_wannierobject(nr, irvec, op_r)
        error("WannierObject constructor check failed")
    end
    T = eltype(op_r).parameters[1]
    WannierObject{T}(nr=nr, irvec=irvec, op_r=op_r, ndata=size(op_r, 1),
        gridopts=[GridOpt{T}() for i=1:Threads.nthreads()],
        rdotks=[zeros(Float64, nr) for i=1:Threads.nthreads()],
        phases=[zeros(Complex{Float64}, nr) for i=1:Threads.nthreads()]
        )
end

WannierObject(T, nr, irvec::Array{Int,2}, op_r) = WannierObject(nr, reinterpret(Vec3{Int}, vec(irvec)), op_r)

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
    @assert size(obj.op_r) == size(op_r_new)
    obj.op_r .= op_r_new
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

    @assert eltype(op_k) == Complex{Float64}
    @assert length(op_k) == obj.ndata
    @assert eltype(phase) == Complex{T}
    @assert length(phase) == obj.nr

    op_k_1d = Base.ReshapedArray(op_k, (length(op_k),), ())

    _get_fourier_normal!(op_k_1d, obj, xk, phase)
end

"Fourier transform real-space operator to momentum-space operator with a
pre-computed phase factor"
@timing "normal" function _get_fourier_normal!(op_k_1d, obj::AbstractWannierObject{T}, xk, phase) where {T}
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

    if isnan(gridopt.k1) || abs(gridopt.k1 - xk[1]) > 1.e-9
        gridopt_set23!(gridopt, obj.irvec, obj.op_r, xk[1])
    end
    if isnan(gridopt.k2) || abs(gridopt.k2 - xk[2]) > 1.e-9
        gridopt_set3!(gridopt, xk[2])
    end

    gridopt_get3!(op_k_1d, gridopt, xk[3])
end

# end # FourierModule
