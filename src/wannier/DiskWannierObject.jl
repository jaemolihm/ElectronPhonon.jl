# Struct and functions for disk-buffered Wannier objects

"Data in coarse real-space grid for a single operator"
Base.@kwdef struct DiskWannierObject{T} <: AbstractWannierObject{T}
    nr::Int
    irvec::Vector{Vec3{Int}}
    ndata::Int # First dimension of op_r

    # For a higher-order WannierObject, the irvec to be used to Fourier transform op_k.
    irvec_next::Union{Nothing,Vector{Vec3{Int}}}

    # For reading data from file
    tag::String
    dir::String
    filename::String

    # Buffer for reading op_r
    op_r_buffer::Vector{Complex{T}}

    # Tag used to check if interpolators are up-to-date.
    _id::Int
end


function DiskWannierObject(T, tag, nr, irvec::Vector{Vec3{Int}}, ndata, dir, filename;
        irvec_next=nothing)
    DiskWannierObject{T}(; tag, nr, irvec, ndata, dir, filename, irvec_next,
        op_r_buffer = zeros(Complex{T}, ndata), _id=0)
end

function DiskWannierObject(T, tag, nr, irvec::Array{Int,2}, ndata, dir, filename; irvec_next=nothing)
    irvec_svector = reinterpret(Vec3{Int}, vec(irvec))
    DiskWannierObject(T, tag, nr, irvec_svector, ndata, dir, filename; irvec_next)
end


function read_op_r(obj::DiskWannierObject, ir)
    # Seek and read op_r[rng, i] from file
    T = eltype(obj)
    f = open(joinpath(obj.dir, obj.filename), "r")
    seek(f, sizeof(T) * obj.ndata * (ir - 1))
    read!(f, obj.op_r_buffer)
    close(f)
    obj.op_r_buffer
end
