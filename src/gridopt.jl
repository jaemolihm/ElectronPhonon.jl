
using Parameters
using SharedArrays
using LinearAlgebra
using Base.Threads

"Real-space data with reduced dimensions for mode=gridopt in get_fourier"
@with_kw mutable struct GridOpt{T<:Real}
    is_initialized::Bool = false
    ndata::Int = 0 # Size of the Fourier-transformed data matrix

    # Data for (k1, R2, R3)
    k1::Float64 = NaN
    nr_23::Int = 0
    irvec_23::Vector{Vec2{Int}} = Vector{Vec2{Int}}(undef, 0)
    irmap_rng_23::Array{UnitRange{Int64},1} = Array{UnitRange{Int64},1}(undef, 0)
    op_r_23::Array{Complex{T},2} = Matrix{Complex{T}}(undef, 0, 0)

    # Data for (k1, k2, R3)
    k2::Float64 = NaN
    nr_3::Int = 0
    irvec_3::Vector{Int} = Vector{Int}(undef, 0)
    irmap_rng_3::Array{UnitRange{Int64},1} = Array{UnitRange{Int64},1}(undef, 0)
    op_r_3::Array{Complex{T},2} = Matrix{Complex{T}}(undef, 0, 0)

    # Cache for Fourier transformation
    phase::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0)
    phase_23::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0)
    phase_3::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0)
    rdotk_3::Vector{T} = Vector{T}(undef, 0)
end

function gridopt_initialize_irvec!(gridopt, irvec)
    # Here, we assume that irvec is sorted according to (r[3], r[2], r[1]).

    # Initialize 23
    gridopt.k1 = NaN
    @views gridopt.irvec_23 = unique([Vec2(r[2:3]) for r in irvec])
    gridopt.nr_23 = length(gridopt.irvec_23)
    gridopt.irmap_rng_23 = Array{UnitRange{Int64},1}()
    @views for r_23 in gridopt.irvec_23
        ir_first = findfirst(x -> x[2:3] == r_23, irvec)
        ir_last = findlast(x -> x[2:3] == r_23, irvec)
        push!(gridopt.irmap_rng_23, ir_first:ir_last)
    end

    # Initialize 3
    gridopt.k2 = NaN
    gridopt.irvec_3 = unique([r[2] for r in gridopt.irvec_23])
    gridopt.nr_3 = length(gridopt.irvec_3)
    gridopt.irmap_rng_3 = Array{UnitRange{Int64},1}()
    @views for r_3 in gridopt.irvec_3
        ir_first = findfirst(x -> x[2] == r_3, gridopt.irvec_23)
        ir_last = findlast(x -> x[2] == r_3, gridopt.irvec_23)
        push!(gridopt.irmap_rng_3, ir_first:ir_last)
    end

end

function gridopt_initialize!(gridopt::GridOpt{T}, obj) where {T}
    # Here, we assume that irvec is sorted according to (r[3], r[2], r[1]).

    gridopt_initialize_irvec!(gridopt, obj.irvec)
    gridopt.ndata = obj.ndata

    # Initialize 23
    gridopt.op_r_23 = zeros(Complex{T}, size(obj.op_r, 1), gridopt.nr_23)

    # Initialize 3
    gridopt.op_r_3 = zeros(Complex{T}, size(obj.op_r, 1), gridopt.nr_3)

    # Initialize cache data
    gridopt.phase = zeros(Complex{T}, length(obj.irvec))
    gridopt.phase_23 = zeros(Complex{T}, gridopt.nr_23)
    gridopt.phase_3 = zeros(Complex{T}, gridopt.nr_3)
    gridopt.rdotk_3 = zeros(T, gridopt.nr_3)

    @info "Initializing gridopt"
    @info "nr=$(length(obj.irvec)), nr_23=$(gridopt.nr_23), nr_3=$(gridopt.nr_3)"

    gridopt.is_initialized = true
end

# TODO: Rename to gridopt_compute_krr?
@timing "s23" function gridopt_set23!(gridopt::GridOpt{T}, irvec, op_r, k) where {T}
    gridopt.k1 = k
    gridopt.k2 = NaN
    gridopt.op_r_23 .= 0
    phase = gridopt.phase
    for (ir, r) in enumerate(irvec)
        phase[ir] = cispi(2 * k * r[1])
    end
    rng_data = 1:gridopt.ndata
    @views @inbounds for (ir_23, ir_rng) in enumerate(gridopt.irmap_rng_23)
        mul!(gridopt.op_r_23[rng_data, ir_23], op_r[rng_data, ir_rng], phase[ir_rng])
    end
end

# TODO: Rename to gridopt_compute_kkr?
@timing "s3" function gridopt_set3!(gridopt::GridOpt{T}, k) where {T}
    gridopt.k2 = k
    gridopt.op_r_3 .= 0
    phase = gridopt.phase_23
    for (ir, r) in enumerate(gridopt.irvec_23)
        phase[ir] = cispi(2 * k * r[1])
    end
    rng_data = 1:gridopt.ndata
    @views @inbounds for (ir_3, ir_rng) in enumerate(gridopt.irmap_rng_3)
        mul!(gridopt.op_r_3[rng_data, ir_3], gridopt.op_r_23[rng_data, ir_rng], phase[ir_rng])
    end
end

@timing "g3" function gridopt_get3!(op_k_1d, gridopt::GridOpt{T}, k) where {T}
    rdotk = gridopt.rdotk_3
    phase = gridopt.phase_3
    rdotk .= k .* gridopt.irvec_3
    phase .= cispi.(2 .* rdotk)

    @views mul!(op_k_1d, gridopt.op_r_3[1:gridopt.ndata, :], phase)
    return
end
