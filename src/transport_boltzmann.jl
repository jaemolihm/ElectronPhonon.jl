"""
Boltzmann transport of electrons.
"""

using StaticArrays
using HDF5

Base.@kwdef struct BTEdata{T <: Real}
    """
    Describes scatterings between a single k point and other k' points.
    """
    # Information about the k point
    xk::SVector{3,T} # Crystal coordinates of k. (3,)
    nbandk::Int64 # Number of bands at k.
    ek::Vector{T} # Energy of bands at k. (nbandk,)

    # Information about k' points
    nkp::Int64 # Number of k' points
    xkp::Vector{SVector{3,T}} # Crystal coordinates of k'. (3,)
    nbandkp_max::Int64 # Maximum number of bands at k'
    nbandkp::Vector{Int64} # Number of bands at each k'. (nkp,)
    ekp::Matrix{T} # Energy of bands at k'. (nbandkp_max, nkp)

    # Scattering due to electron-phonon coupling. q = k' - k
    nmodes::Int64 # Number of phonon modes
    ωq::Matrix{T} # Energy of phonon modes at q. (nmodes, nkp)
    g2::Array{T, 4} # Squared el-ph matrix elements. (nbandkp_max, nbandk, nmodes, nkp)
end

# function BTEdata(T, nbandk, nkp, nbandkp_max, nmodes)
#     BTEdata{T}(xk=zeros(SVector{3,T}), nbandk=nbandk, ek=zeros(T, nbandk),
#         nkp=nkp, xkp=zeros(SVector{3,T}, nkp), nbandkp_max=nbandkp_max,
#         nbandkp=zeros(Int64, nkp), ekp=zeros(T, nbandkp_max, nkp),
#         nmodes=nmodes, ωq=zeros(T, nmodes, nkp),
#         g2=zeros(T, nbandkp_max, nbandk, nmodes, nkp))
# end

function create_dataset_wrapper(parent, path, type, space::Tuple)
    create_dataset(parent, path, datatype(type), dataspace(space))
end

function create_dataset_wrapper(parent, path, type, len::Integer)
    create_dataset(parent, path, datatype(type), dataspace((len,)))
end

function open_BTEdata_file_for_write(filename, nk, nbandk_max, nkp_max, nbandkp_max, nmodes)
    T = Float64
    fid = h5open(filename, "w")
    for ik in 1:nk
        g = create_group(fid, "electron/ik$ik")
        create_dataset_wrapper(g, "xk", T, 3)
        create_dataset_wrapper(g, "nbandk", Int64, 1)
        create_dataset_wrapper(g, "ek", T, nbandk_max)
        create_dataset_wrapper(g, "nkp", Int64, 1)
        create_dataset_wrapper(g, "xkp", T, (3, nkp_max))
        create_dataset_wrapper(g, "nbandkp_max", Int64, 1)
        create_dataset_wrapper(g, "nbandkp", Int64, nkp_max)
        create_dataset_wrapper(g, "ekp", T, (nbandkp_max, nkp_max))
        create_dataset_wrapper(g, "nmodes", Int64, 1)
        create_dataset_wrapper(g, "omegaq", T, (nmodes, nkp_max))
        create_dataset_wrapper(g, "g2", T, (nbandkp_max, nbandk_max, nmodes, nkp_max))
        close(g)
    end
    write_attribute(fid["electron"], "nk", nk)
    write_attribute(fid["electron"], "nbandk_max", nbandk_max)
    fid
end

@timing "dump_BTEdata" function dump_BTEdata(g, btedata::BTEdata{T}) where {T}
    nkp = btedata.nkp
    g["xk"][:] = btedata.xk
    g["nbandk"][:] = btedata.nbandk
    g["ek"][1:btedata.nbandk] = btedata.ek
    g["nkp"][:] = btedata.nkp
    g["xkp"][:,1:nkp] = reshape(reinterpret(T, btedata.xkp), 3, :)
    g["nbandkp_max"][:] = btedata.nbandkp_max
    g["nbandkp"][:] = btedata.nbandkp
    g["ekp"][:,1:nkp] = btedata.ekp
    g["nmodes"][:] = btedata.nmodes
    g["omegaq"][:,1:nkp] = btedata.ωq
    g["g2"][:,:,:,1:nkp] = btedata.g2
end

# @timing "load_BTEdata"
function load_BTEdata(T, g)
    BTEdata{T}(
        xk = SVector{3,T}(read(g["xk"])),
        nbandk = read(g["nbandk"])[1],
        ek = read(g["ek"]),
        nkp = read(g["nkp"])[1],
        xkp = collect(vec(reinterpret(SVector{3,Float64}, read(g["xkp"])))),
        nbandkp_max = read(g["nbandkp_max"])[1],
        nbandkp = read(g["nbandkp"]),
        ekp = read(g["ekp"]),
        nmodes = read(g["nmodes"])[1],
        ωq = read(g["omegaq"]),
        g2 = read(g["g2"]),
    )
end