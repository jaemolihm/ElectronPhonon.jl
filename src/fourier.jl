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
struct WannierObject{T} <: AbstractWannierObject{T}
    nr::Int
    irvec::Vector{Vec3{Int}}
    op_r::Array{Complex{T},2}

    # For gridopt Fourier transform
    gridopts::Vector{GridOpt{T}}

    # Allocated buffer for normal Fourier transform
    rdotks::Vector{Vector{T}}
    phases::Vector{Vector{Complex{T}}}
end

function WannierObject(nr, irvec::Vector{Vec3{Int}}, op_r)
    check = check_wannierobject(nr, irvec, op_r)
    if ! check
        error("WannierObject constructor check failed")
    end
    T = eltype(op_r).parameters[1]
    WannierObject{T}(nr, reinterpret(Vec3{Int}, vec(irvec)), op_r,
        [GridOpt{T}() for i=1:Threads.nthreads()],
        [zeros(Float64, nr) for i=1:Threads.nthreads()],
        [zeros(Complex{Float64}, nr) for i=1:Threads.nthreads()]
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
function get_fourier!(op_k, obj::AbstractWannierObject{T}, xk, phase_input=nothing; mode="normal") where {T}
    # Regarding the use of ReshapedArray, see
    # https://discourse.julialang.org/t/passing-views-to-function-without-allocation/51992/12
    # https://github.com/ITensor/NDTensors.jl/issues/32

    @assert eltype(op_k) == Complex{Float64}
    @assert length(op_k) == size(obj.op_r, 1)

    op_k_1d = Base.ReshapedArray(op_k, (length(op_k),), ())

    if mode == "normal"
        if phase_input === nothing
            phase = get_phase_expikr!(obj, xk, threadid())
        else
            # If phase is given, we assume that the given phase is equal to the above.
            # TODO: Add debugging check of above condition
            phase = phase_input
        end

        mul!(op_k_1d, obj.op_r, phase)
        return
    elseif mode == "gridopt"
        tid = Threads.threadid()
        gridopt = obj.gridopts[tid]

        if ! gridopt.is_initialized
            # println("Initializing obj.gridopts[$tid]")
            gridopt_initialize!(gridopt, obj.irvec, obj.op_r)
        end

        if isnan(gridopt.k1) || abs(gridopt.k1 - xk[1]) > 1.e-9
            gridopt_set23!(gridopt, obj.irvec, obj.op_r, xk[1])
        end
        if isnan(gridopt.k2) || abs(gridopt.k2 - xk[2]) > 1.e-9
            gridopt_set3!(gridopt, xk[2])
        end

        gridopt_get3!(op_k_1d, gridopt, xk[3])
    else
        error("mode must be normal or gridopt")
    end
end


# end # FourierModule
