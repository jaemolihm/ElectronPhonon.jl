using Base.Threads: @threads, nthreads

# TODO: This is not used in the current code.
# Currently, fourier_mode = "gridopt" + DiskWannierObject uses op_r from disk, but
# stores op_r_23, op_r_3 in memory. DiskGridOpt can be implemented and used as a new
# fourier_mode = "gridopt_disk" to store op_r_23, op_r_3 in disk.

"Real-space data with reduced dimensions for fourier_mode=gridopt in get_fourier
with reading writing op_r to disk."
Base.@kwdef mutable struct DiskGridOpt{T<:Real}
    is_initialized::Bool = false
    # Data for (k1, R2, R3)
    k1::Float64 = NaN
    nr_23::Int = 0
    irvec_23::Vector{Vec2{Int}} = Vector{Vec2{Int}}(undef, 0)
    irmap_rng_23::Array{UnitRange{Int},1} = Vector{UnitRange{Int}}(undef, 0)
    filename_23::String = ""

    # Data for (k1, k2, R3)
    k2::Float64 = NaN
    nr_3::Int = 0
    irvec_3::Vector{Int} = Vector{Int}(undef, 0)
    irmap_rng_3::Array{UnitRange{Int},1} = Vector{UnitRange{Int}}(undef, 0)
    filename_3::String = ""

    # Cache for Fourier transformation
    phase::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0)
    phase_23::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0)
    phase_3::Vector{Complex{T}} = Vector{Complex{T}}(undef, 0)
    rdotk_3::Vector{T} = Vector{T}(undef, 0)

    # # Buffer for writing op_r_23, op_r_3 (used only in gridopt)
    # buffers_w::Vector{Vector{Complex{T}}}
    # buffers_w=[zeros(Complex{T}, length(rng)) for rng in ranges],
end







"Fourier transform real-space operator to momentum-space operator with grid optimization"
function _get_fourier_gridopt!(op_k_1d, obj::DiskWannierObject{T}, xk) where {T}
    tid = Threads.threadid()
    gridopt = obj.gridopts[tid]

    if ! gridopt.is_initialized
        # println("Initializing obj.gridopts[$tid]")
        gridopt_initialize!(gridopt, obj.irvec, obj.tag)
    end

    if isnan(gridopt.k1) || abs(gridopt.k1 - xk[1]) > 1.e-9
        gridopt_set23!(gridopt, xk[1], obj)
    end
    if isnan(gridopt.k2) || abs(gridopt.k2 - xk[2]) > 1.e-9
        gridopt_set3!(gridopt, xk[2], obj)
    end

    gridopt_get3!(op_k_1d, gridopt, xk[3], obj)
end

function gridopt_initialize!(gridopt::DiskGridOpt{T}, irvec, tag) where {T}
    # Here, we assume that irvec is sorted according to (r[3], r[2], r[1]).

    gridopt_initialize_irvec!(gridopt, irvec)

    # TODO: use MPI id
    gridopt.filename_23 = "tmp_$(tag)_mpi$(mpi_myrank())_23.bin"
    gridopt.filename_3 = "tmp_$(tag)_mpi$(mpi_myrank())_3.bin"

    # Initialize cache data
    gridopt.phase = zeros(Complex{T}, length(irvec))
    gridopt.phase_23 = zeros(Complex{T}, gridopt.nr_23)
    gridopt.phase_3 = zeros(Complex{T}, gridopt.nr_3)
    gridopt.rdotk_3 = zeros(T, gridopt.nr_3)

    @info "Initializing gridopt"
    @info "nr=$(length(irvec)), nr_23=$(gridopt.nr_23), nr_3=$(gridopt.nr_3)"

    gridopt.is_initialized = true
end

# TODO: Rename to gridopt_compute_krr?
@timing "disk_s23" function gridopt_set23!(gridopt::DiskGridOpt{T}, k, obj) where {T}
    gridopt.k1 = k
    gridopt.k2 = NaN
    phase = gridopt.phase
    for (ir, r) in enumerate(obj.irvec)
        phase[ir] = cispi(2 * k * r[1])
    end

    # Remove filename_23 if exists
    rm(joinpath(obj.dir, gridopt.filename_23), force=true)
    fws = [open(joinpath(obj.dir, gridopt.filename_23), "w") for i=1:nthreads()]

    @threads for irng in axes(obj.ranges, 1)
        rng = obj.ranges[irng]
        buffer_r = obj.buffers_r[irng]
        buffer_w = obj.buffers_w[irng]

        # TODO: filename_23 -> op_r_23_file?
        fr = open(joinpath(obj.dir, obj.filename), "r")
        fw = fws[threadid()]

        for (ir_23, ir_rng) in enumerate(gridopt.irmap_rng_23)
            buffer_w .= 0
            for ir in ir_rng
                # Read buffer_r = op_r[rng, ir]
                seek(fr, sizeof(Complex{T}) * (obj.ndata*(ir-1) + rng[1]-1))
                read!(fr, buffer_r)

                buffer_w .+= buffer_r .* phase[ir]
            end
            # Write op_r_23[rng, ir_23] = buffer_w
            seek(fw, sizeof(Complex{T}) * (obj.ndata*(ir_23-1) + rng[1]-1))
            write(fw, buffer_w)
        end
        close(fr)
    end
    map(close, fws)
end


# TODO: Rename to gridopt_compute_kkr?
@timing "disk_s3" function gridopt_set3!(gridopt::DiskGridOpt{T}, k, obj) where {T}
    gridopt.k2 = k
    phase = gridopt.phase_23
    for (ir, r) in enumerate(gridopt.irvec_23)
        phase[ir] = cispi(2 * k * r[1])
    end

    # Remove filename_3 if exists
    rm(joinpath(obj.dir, gridopt.filename_3), force=true)
    fws = [open(joinpath(obj.dir, gridopt.filename_3), "w") for i=1:nthreads()]

    @threads for irng in axes(obj.ranges, 1)
        rng = obj.ranges[irng]
        buffer_r = obj.buffers_r[irng]
        buffer_w = obj.buffers_w[irng]

        # TODO: filename_23 -> op_r_23_file?
        fr = open(joinpath(obj.dir, gridopt.filename_23), "r")
        fw = fws[threadid()]

        for (ir_3, ir_rng) in enumerate(gridopt.irmap_rng_3)
            buffer_w .= 0
            for ir in ir_rng
                # Read buffer_r = op_r_23[rng, ir]
                seek(fr, sizeof(Complex{T}) * (obj.ndata*(ir-1) + rng[1]-1))
                read!(fr, buffer_r)

                buffer_w .+= buffer_r .* phase[ir]
            end
            # Write op_r_3[rng, ir_3] = buffer_w
            seek(fw, sizeof(Complex{T}) * (obj.ndata*(ir_3-1) + rng[1]-1))
            write(fw, buffer_w)
        end
        close(fr)
    end
    map(close, fws)
end


@timing "disk_g3" function gridopt_get3!(op_k_1d, gridopt::DiskGridOpt{T}, k, obj) where {T}
    rdotk = gridopt.rdotk_3
    phase = gridopt.phase_3
    rdotk .= k .* gridopt.irvec_3
    phase .= cispi.(2 .* rdotk)

    op_k_1d .= 0
    @threads for irng in axes(obj.ranges, 1)
        rng = obj.ranges[irng]
        buffer_r = obj.buffers_r[irng]

        f = open(joinpath(obj.dir, gridopt.filename_3), "r")
        @views @inbounds for i in 1:gridopt.nr_3
            # Seek and read op_r[rng, i] from file
            seek(f, sizeof(Complex{T}) * (obj.ndata*(i-1) + rng[1]-1))
            read!(f, buffer_r)
            op_k_1d[rng] .+= buffer_r .* phase[i]
        end
        close(f)
    end
    return
end
