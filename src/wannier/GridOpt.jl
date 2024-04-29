using Base.Threads
using LinearAlgebra

# TODO: Add threads option to GridOpt or WannierInterpolator (to enable only for epmat)

mutable struct GridOpt{T<:Real}
    # TODO: Remove ndata
    const ndata::Int # Size of the Fourier-transformed data matrix

    # Data for (k1, R2, R3)
    k1::T
    const nr_23::Int
    const irvec_23::Vector{Vec2{Int}}
    const irmap_rng_23::Vector{UnitRange{Int}}
    const op_r_23::Array{Complex{T},2}

    # Data for (k1, k2, R3)
    k2::T
    const nr_3::Int
    const irvec_3::Vector{Int}
    const irmap_rng_3::Vector{UnitRange{Int}}
    const op_r_3::Array{Complex{T},2}

    # Cache for Fourier transformation
    const phase::Vector{Complex{T}}
    const phase_23::Vector{Complex{T}}
    const phase_3::Vector{Complex{T}}

    function GridOpt(::Type{T}, irvec::Vector{Vec3{Int}}, ndata::Int) where {T}
        # Here, we assume that irvec is sorted according to (r[3], r[2], r[1]).

        # Initialize 23
        k1 = NaN
        irvec_23 = unique!([Vec2(r[2:3]) for r in irvec])
        nr_23 = length(irvec_23)
        irmap_rng_23 = UnitRange{Int}[]
        @views for r_23 in irvec_23
            ir_first = findfirst(x -> x[2:3] == r_23, irvec)
            ir_last = findlast(x -> x[2:3] == r_23, irvec)
            push!(irmap_rng_23, ir_first:ir_last)
        end

        # Initialize 3
        k2 = NaN
        irvec_3 = unique!([r[2] for r in irvec_23])
        nr_3 = length(irvec_3)
        irmap_rng_3 = UnitRange{Int}[]
        for r_3 in irvec_3
            ir_first = findfirst(x -> x[2] == r_3, irvec_23)
            ir_last = findlast(x -> x[2] == r_3, irvec_23)
            push!(irmap_rng_3, ir_first:ir_last)
        end

        # Initialize 23
        op_r_23 = zeros(Complex{T}, ndata, nr_23)

        # Initialize 3
        op_r_3 = zeros(Complex{T}, ndata, nr_3)

        # Initialize cache data
        phase = zeros(Complex{T}, length(irvec))
        phase_23 = zeros(Complex{T}, nr_23)
        phase_3 = zeros(Complex{T}, nr_3)

        new{T}(ndata, k1, nr_23, irvec_23, irmap_rng_23, op_r_23,
            k2, nr_3, irvec_3, irmap_rng_3, op_r_3,
            phase, phase_23, phase_3)
    end
end

# TODO: Rename to gridopt_compute_krr?
@timing "s23" function gridopt_set23!(gridopt::GridOpt{T}, parent::WT, k, ndata) where {T, WT}
    gridopt.k1 = k
    gridopt.k2 = NaN
    gridopt.op_r_23 .= 0
    phase = gridopt.phase
    for (ir, r) in enumerate(parent.irvec)
        phase[ir] = cispi(2 * k * r[1])
    end
    rng_data = 1:ndata
    # @views for (ir_23, ir_rng) in enumerate(gridopt.irmap_rng_23)
    #     ir_rng = gridopt.irmap_rng_23[ir_23]
    #     if WT <: DiskWannierObject
    #         gridopt.op_r_23[rng_data, ir_23] .= 0
    #         for ir in ir_rng
    #             gridopt.op_r_23[rng_data, ir_23] .+= phase[ir] .* read_op_r(parent, ir)[rng_data]
    #         end
    #     else
    #         mul!(gridopt.op_r_23[rng_data, ir_23], parent.op_r[rng_data, ir_rng], phase[ir_rng])
    #     end
    # end
    if ndata > 100_000
        @views @threads for ir_23 in eachindex(gridopt.irmap_rng_23)
            ir_rng = gridopt.irmap_rng_23[ir_23]
            if WT <: DiskWannierObject
                gridopt.op_r_23[rng_data, ir_23] .= 0
                for ir in ir_rng
                    gridopt.op_r_23[rng_data, ir_23] .+= phase[ir] .* read_op_r(parent, ir)[rng_data]
                end
            else
                mul!(gridopt.op_r_23[rng_data, ir_23], parent.op_r[rng_data, ir_rng], phase[ir_rng])
            end
        end
    else
        @views for (ir_23, ir_rng) in enumerate(gridopt.irmap_rng_23)
            ir_rng = gridopt.irmap_rng_23[ir_23]
            if WT <: DiskWannierObject
                gridopt.op_r_23[rng_data, ir_23] .= 0
                for ir in ir_rng
                    gridopt.op_r_23[rng_data, ir_23] .+= phase[ir] .* read_op_r(parent, ir)[rng_data]
                end
            else
                mul!(gridopt.op_r_23[rng_data, ir_23], parent.op_r[rng_data, ir_rng], phase[ir_rng])
            end
        end
    end
end

# TODO: Rename to gridopt_compute_kkr?
@timing "s3" function gridopt_set3!(gridopt::GridOpt{T}, k, ndata) where {T}
    gridopt.k2 = k
    gridopt.op_r_3 .= 0
    phase = gridopt.phase_23
    for (ir, r) in enumerate(gridopt.irvec_23)
        phase[ir] = cispi(2 * k * r[1])
    end
    rng_data = 1:ndata
    # @views for (ir_3, ir_rng) in enumerate(gridopt.irmap_rng_3)
    #     mul!(gridopt.op_r_3[rng_data, ir_3], gridopt.op_r_23[rng_data, ir_rng], phase[ir_rng])
    # end
    if ndata > 100_000
        @views @threads for ir_3 in eachindex(gridopt.irmap_rng_3)
            ir_rng = gridopt.irmap_rng_3[ir_3]
            mul!(gridopt.op_r_3[rng_data, ir_3], gridopt.op_r_23[rng_data, ir_rng], phase[ir_rng])
        end
    else
        @views for (ir_3, ir_rng) in enumerate(gridopt.irmap_rng_3)
            mul!(gridopt.op_r_3[rng_data, ir_3], gridopt.op_r_23[rng_data, ir_rng], phase[ir_rng])
        end
    end
end

@timing "g3" function gridopt_get3!(op_k_1d, gridopt::GridOpt{T}, k, ndata) where {T}
    phase = gridopt.phase_3
    @. phase = cispi(2 * k * gridopt.irvec_3)

    @views mul!(op_k_1d, gridopt.op_r_3[1:ndata, :], phase)
    return
end

"""
    reset_gridopt!(gridopt::GridOpt)
Reset the `GridOpt`. Must be called when the parent data is modified.
"""
function reset_gridopt!(gridopt::GridOpt)
    gridopt.k1 = NaN
    gridopt.k2 = NaN
end
