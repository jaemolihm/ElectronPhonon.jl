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

# TODO: Make GridOpt{T}, make AbstractGridOpt (?)

"Real-space data with reduced dimensions for mode=gridopt in get_fourier"
@with_kw mutable struct GridOpt
    is_initialized::Bool = false
    k1::Float64 = NaN
    nr_23::Int = 0
    irvec_23::Vector{Vec2{Int}} = Vector{Vec2{Int}}()
    irmap_rng_23::Array{UnitRange{Int64},1} = Array{UnitRange{Int64},1}()
    op_r_23::Array{Complex{Float64},2} = zeros(Complex{Float64}, 1, 1)
    k2::Float64 = NaN
    nr_3::Int = 0
    irvec_3::Vector{Int} = zeros(Int, 1)
    irmap_rng_3::Array{UnitRange{Int64},1} = Array{UnitRange{Int64},1}()
    op_r_3::Array{Complex{Float64},2} = zeros(Complex{Float64}, 1, 1)

    # Cache for Fourier transformation
    rdotk::Vector{Float64} = zeros(Float64, 1)
    phase::Vector{Complex{Float64}} = zeros(Complex{Float64}, 1)
end

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
"Check validity of input"
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
    gridopts::Vector{GridOpt}

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
        [GridOpt() for i=1:Threads.nthreads()],
        [zeros(Float64, nr) for i=1:Threads.nthreads()],
        [zeros(Complex{Float64}, nr) for i=1:Threads.nthreads()]
        )
end

WannierObject(T, nr, irvec::Array{Int,2}, op_r) = WannierObject(nr, reinterpret(Vec3{Int}, vec(irvec)), op_r)

"Fourier transform real-space operator to momentum-space operator"
function get_fourier!(op_k, obj, xk, phase_input=nothing; mode="normal")
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

        rdotk = gridopt.rdotk
        phase = gridopt.phase
        rdotk .= 2pi .* xk[3] .* gridopt.irvec_3
        phase .= cis.(rdotk)

        mul!(op_k_1d, gridopt.op_r_3, phase)
        return
    else
        error("mode must be normal or gridopt")
    end
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
    @assert size(obj.op_r) == size(op_r_new)
    obj.op_r .= op_r_new
    for gridopt in obj.gridopts
        gridopt.k1 = NaN
        gridopt.k2 = NaN
    end
end

function gridopt_initialize!(gridopt, irvec, op_r)
    # Here, we assume that irvec is sorted according to (r[3], r[2], r[1]).

    # Initialize 23
    gridopt.k1 = NaN
    gridopt.irvec_23 = unique([Vec2(r[2:3]) for r in irvec])
    gridopt.nr_23 = length(gridopt.irvec_23)
    gridopt.op_r_23 = zeros(ComplexF64, size(op_r, 1), gridopt.nr_23)
    gridopt.irmap_rng_23 = Array{UnitRange{Int64},1}()
    for r_23 in gridopt.irvec_23
        ir_first = findfirst(map(x -> x[2:3] == r_23, irvec))
        ir_last = findlast(map(x -> x[2:3] == r_23, irvec))
        push!(gridopt.irmap_rng_23, ir_first:ir_last)
    end

    # Initialize 3
    gridopt.k2 = NaN
    gridopt.irvec_3 = unique([r[2] for r in gridopt.irvec_23])
    gridopt.nr_3 = length(gridopt.irvec_3)
    gridopt.op_r_3 = zeros(ComplexF64, size(op_r, 1), gridopt.nr_3)
    gridopt.irmap_rng_3 = Array{UnitRange{Int64},1}()
    for r_3 in gridopt.irvec_3
        ir_first = findfirst(map(x -> x[2] == r_3, gridopt.irvec_23))
        ir_last = findlast(map(x -> x[2] == r_3, gridopt.irvec_23))
        push!(gridopt.irmap_rng_3, ir_first:ir_last)
    end

    # Initialize cache data
    gridopt.rdotk = zeros(Float64, gridopt.nr_3)
    gridopt.phase = zeros(Complex{Float64}, gridopt.nr_3)

    @info "Initializing gridopt"
    @info "nr=$(length(irvec)), nr_23=$(gridopt.nr_23), nr_3=$(gridopt.nr_3)"

    gridopt.is_initialized = true
end

function gridopt_set23!(gridopt, irvec, op_r, k)
    gridopt.k1 = k
    gridopt.k2 = NaN
    gridopt.op_r_23 .= 0.0
    phase = [cis(2pi * k * r[1]) for r in irvec]
    @views @inbounds for (ir_23, ir_rng) in enumerate(gridopt.irmap_rng_23)
        mul!(gridopt.op_r_23[:, ir_23], op_r[:, ir_rng], phase[ir_rng])
    end
end

function gridopt_set3!(gridopt, k)
    gridopt.k2 = k
    gridopt.op_r_3 .= 0.0
    phase = [cis(2pi * k * r[1]) for r in gridopt.irvec_23]
    @views @inbounds for (ir_3, ir_rng) in enumerate(gridopt.irmap_rng_3)
        mul!(gridopt.op_r_3[:, ir_3], gridopt.op_r_23[:, ir_rng], phase[ir_rng])
    end
end

# end # FourierModule
