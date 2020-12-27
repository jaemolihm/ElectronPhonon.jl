
using Parameters
using SharedArrays
using LinearAlgebra
using Base.Threads

"Real-space data with reduced dimensions for mode=gridopt in get_fourier"
@with_kw mutable struct GridOpt{T<:Real}
    is_initialized::Bool = false
    # Data for (k1, R2, R3)
    k1::Float64 = NaN
    nr_23::Int = 0
    irvec_23::Vector{Vec2{Int}} = Vector{Vec2{Int}}()
    irmap_rng_23::Array{UnitRange{Int64},1} = Array{UnitRange{Int64},1}()
    op_r_23::Array{Complex{T},2} = zeros(Complex{T}, 1, 1)

    # Data for (k1, k2, R3)
    k2::Float64 = NaN
    nr_3::Int = 0
    irvec_3::Vector{Int} = zeros(Int, 1)
    irmap_rng_3::Array{UnitRange{Int64},1} = Array{UnitRange{Int64},1}()
    op_r_3::Array{Complex{T},2} = zeros(Complex{T}, 1, 1)

    # Cache for Fourier transformation
    rdotk::Vector{T} = zeros(T, 1)
    phase::Vector{Complex{T}} = zeros(Complex{T}, 1)
end

function gridopt_initialize_irvec!(gridopt, irvec)
    # Here, we assume that irvec is sorted according to (r[3], r[2], r[1]).

    # Initialize 23
    gridopt.k1 = NaN
    gridopt.irvec_23 = unique([Vec2(r[2:3]) for r in irvec])
    gridopt.nr_23 = length(gridopt.irvec_23)
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
    gridopt.irmap_rng_3 = Array{UnitRange{Int64},1}()
    for r_3 in gridopt.irvec_3
        ir_first = findfirst(map(x -> x[2] == r_3, gridopt.irvec_23))
        ir_last = findlast(map(x -> x[2] == r_3, gridopt.irvec_23))
        push!(gridopt.irmap_rng_3, ir_first:ir_last)
    end

end

function gridopt_initialize!(gridopt::GridOpt{T}, irvec, op_r) where {T}
    # Here, we assume that irvec is sorted according to (r[3], r[2], r[1]).

    gridopt_initialize_irvec!(gridopt, irvec)

    # Initialize 23
    gridopt.op_r_23 = zeros(Complex{T}, size(op_r, 1), gridopt.nr_23)

    # Initialize 3
    gridopt.op_r_3 = zeros(Complex{T}, size(op_r, 1), gridopt.nr_3)

    # Initialize cache data
    gridopt.rdotk = zeros(T, gridopt.nr_3)
    gridopt.phase = zeros(Complex{T}, gridopt.nr_3)

    @info "Initializing gridopt"
    @info "nr=$(length(irvec)), nr_23=$(gridopt.nr_23), nr_3=$(gridopt.nr_3)"

    gridopt.is_initialized = true
end

# TODO: Rename to gridopt_compute_krr?
@timing "s23" function gridopt_set23!(gridopt::GridOpt{T}, irvec, op_r, k) where {T}
    gridopt.k1 = k
    gridopt.k2 = NaN
    gridopt.op_r_23 .= 0.0
    phase = [cis(2pi * k * r[1]) for r in irvec]
    @views @inbounds for (ir_23, ir_rng) in enumerate(gridopt.irmap_rng_23)
        mul!(gridopt.op_r_23[:, ir_23], op_r[:, ir_rng], phase[ir_rng])
    end
end

# TODO: Rename to gridopt_compute_kkr?
@timing "s3" function gridopt_set3!(gridopt::GridOpt{T}, k) where {T}
    gridopt.k2 = k
    gridopt.op_r_3 .= 0.0
    phase = [cis(2pi * k * r[1]) for r in gridopt.irvec_23]
    @views @inbounds for (ir_3, ir_rng) in enumerate(gridopt.irmap_rng_3)
        mul!(gridopt.op_r_3[:, ir_3], gridopt.op_r_23[:, ir_rng], phase[ir_rng])
    end
end

@timing "g3" function gridopt_get3!(op_k_1d, gridopt::GridOpt{T}, k) where {T}
    rdotk = gridopt.rdotk
    phase = gridopt.phase
    rdotk .= 2pi .* k .* gridopt.irvec_3
    phase .= cis.(rdotk)

    mul!(op_k_1d, gridopt.op_r_3, phase)
    return
end
