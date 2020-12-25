
# Struct and functions for disk-buffered Wannier objects

export DiskWannierObject1

"Data in coarse real-space grid for a single operator"
Base.@kwdef struct DiskWannierObject1{T} <: AbstractWannierObject{T}
    nr::Int
    irvec::Vector{Vec3{Int}}
    ndata::Int # First dimension of op_r

    # # For gridopt Fourier transform
    # gridopts::Vector{DiskGridOpt{T}}

    # Allocated buffer for normal Fourier transform
    rdotks::Vector{Vector{T}}
    phases::Vector{Vector{Complex{T}}}

    # For reading data from file
    dir::String
    filename::String
    op_r_buffers::Vector{Vector{Complex{T}}}

    # For multithreading IO
    ranges::Vector{UnitRange{Int}}
end

function DiskWannierObject1(T, nr, irvec::Vector{Vec3{Int}}, ndata, dir, filename)
    ranges = split_evenly(1:ndata, nthreads())
    DiskWannierObject1{T}(nr=nr, irvec=irvec, ndata=ndata,
        dir=dir, filename=filename,
        rdotks=[zeros(T, nr) for i=1:nthreads()],
        phases=[zeros(Complex{T}, nr) for i=1:nthreads()],
        ranges=ranges,
        op_r_buffers=[zeros(Complex{T}, length(rng)) for rng in ranges],
        )
end

function DiskWannierObject1(T, nr, irvec::Array{Int,2}, ndata, dir, filename)
    irvec_svector = reinterpret(Vec3{Int}, vec(irvec))
    DiskWannierObject1(nr, irvec_svector, ndata, dir, filename)
end

"Fourier transform real-space operator to momentum-space operator"
function get_fourier!(op_k, obj::DiskWannierObject1{T}, xk; mode="normal") where {T}
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == obj.ndata

    op_k_1d = Base.ReshapedArray(op_k, (length(op_k),), ())

    if mode == "normal"
        phase = get_phase_expikr!(obj, xk, threadid())
        get_fourier!(op_k, obj, xk, phase, mode="normal")
    elseif mode == "gridopt"
        error("gridopt not implemented for DiskWannierObject")
    else
        error("mode must be normal or gridopt")
    end
end

"Fourier transform real-space operator to momentum-space operator with a
pre-computed phase factor"
function get_fourier!(op_k, obj::DiskWannierObject1{T}, xk, phase; mode="normal") where {T}
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == obj.ndata
    @assert eltype(phase) == Complex{T}
    @assert length(phase) == obj.nr

    if mode != "normal"
        error("get_fourier! with pre-computed phase works only for mode=normal.")
    end

    op_k_1d = Base.ReshapedArray(op_k, (length(op_k),), ())

    # Read op_r from file and sum over R
    op_k_1d .= 0
    @threads for (rng, buffer) in collect(zip(obj.ranges, obj.op_r_buffers))
        f = open(joinpath(obj.dir, obj.filename), "r")
        @views @inbounds for i in 1:obj.nr
            seek(f, sizeof(Complex{T}) * (obj.ndata*(i-1) + rng[1]-1))
            read!(f, buffer)
            op_k_1d[rng] .+= buffer .* phase[i]
        end
        close(f)
    end
    return
end
