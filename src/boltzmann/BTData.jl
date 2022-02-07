
# General routines for BTData (Boltzmann Transport Data) objects
#
# Basic data types (Int, Float, Complex, Array, StaticArray, ...) are converted when writing
# into a HDF5 friendly type via `_data_julia_to_hdf5` and then written to file. When reading,
# data are first read from the file and converted to the desired type via `_data_hdf5_to_julia`.
# Function `_data_hdf5_to_julia` ensures type stability via type assertion.
#
# Composite data types (structs) are treated field by field, in dump_BTData and load_BTData.
#
# TODO: Why not JLD2?
# TODO: Cleanup. Merge _data_julia_to_hdf5 and dump_BTData.

export AbstractBTData
export load_BTData
export dump_BTData

abstract type AbstractBTData{T <: Real} end

"""
    _data_julia_to_hdf5(x)
Transform `x` into a type that can be written to a HDF5 file. Default case.
"""
_data_julia_to_hdf5(x) = x

"""
    _data_hdf5_to_julia(x, ::Type{T}) where T
Transform `x` which was read from HDF5 file into type `T`.
"""
_data_hdf5_to_julia(x, ::Type{T}) where T = T(x)::T

# Tuple
_data_julia_to_hdf5(x::Tuple) = collect(x)

# BitArray (Needed because HDF5 fails because strides(::BitArray) is not defined.)
_data_julia_to_hdf5(x::BitArray) = Array(x)

# StaticArray: transform to an Array
function _data_julia_to_hdf5(x::T) where {T <: SArray}
    collect(reshape(reinterpret(eltype(T), x), size(T)...))
end
function _data_hdf5_to_julia(x, ::Type{T}) where {T <: SArray}
    sarray_size = size(T)
    if size(x) != sarray_size
        error("Size of x $(size(x)) not consistent with Type $T")
    end
    T(x)::T
end

# AbstractArray of StaticArrays: convert to a higher dimensional Array
function _data_julia_to_hdf5(x::AbstractArray{T}) where {T <: SArray}
    collect(reshape(reinterpret(eltype(T), x), size(T)..., size(x)...))
end
function _data_hdf5_to_julia(x, ::Type{T}) where {T <: AbstractArray{ET}} where {ET <: SArray}
    sarray_size = size(eltype(T))
    if size(x)[1:length(sarray_size)] != sarray_size
        error("Size of x $(size(x)) not consistent with Type $T")
    end
    array_size = size(x)[length(sarray_size)+1:end]
    return collect(reshape(reinterpret(eltype(T), vec(x)), array_size))::T
end

# UnitRange
_data_julia_to_hdf5(x::UnitRange) = [x.start, x.stop]
_data_hdf5_to_julia(x, ::Type{T}) where {T <: UnitRange} = (x[1]:x[2])::T

# AbstractArray of UnitRange
function _data_julia_to_hdf5(x::AbstractArray{T}) where {T <: UnitRange{FT}} where {FT}
    collect(reshape(reinterpret(FT, x), 2, size(x)...))
end
function _data_hdf5_to_julia(x, ::Type{T}) where {T <: AbstractArray{ET}} where {ET <: UnitRange}
    ET.(selectdim(x, 1, 1), selectdim(x, 1, 2))::T
end


"""
    dump_BTData(f, obj::T) where {T}
Dump BTData object `obj` to an HDF5 file or group `f`.
"""
@timing "dump_BTData" function dump_BTData(f, obj::T) where {T}
    for name in fieldnames(T)
        f[String(name)] = _data_julia_to_hdf5(getfield(obj, name))
    end
end

"""
    load_BTData(f, ::Type{T}) where {T}
Load BTData object of type `BTType` from an HDF5 file or group `f`.
"""
@timing "load_BTData" function load_BTData(f, ::Type{T}) where {T}
    data = []
    for (i, name) in enumerate(fieldnames(T))
        FieldType = T.types[i]
        push!(data, _data_hdf5_to_julia(read(f, String(name)), FieldType))
    end
    T(data...)::T
end