
"General routines for BTData (Boltzmann Transport Data) objects"

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
    _data_julia_to_hdf5(x::T) where {T <: SArray}
Transform Transform a staticarray to an array."""
function _data_julia_to_hdf5(x::T) where {T <: SArray}
    collect(reshape(reinterpret(eltype(T), x), size(T)...))
end

"""
    _data_julia_to_hdf5(x::AbstractArray{T}) where {T <: SArray}
Transform an array of staticarrays to a high-dimensional array.
"""
function _data_julia_to_hdf5(x::AbstractArray{T}) where {T <: SArray}
    collect(reshape(reinterpret(eltype(T), x), size(T)..., size(x)...))
end

"""
    _data_hdf5_to_julia(x, OutType::Type)
Transform `x` into type `OutType`.
The only nontrivial part is transforming an array to a StaticArray or an array of StaticArrays.
"""
function _data_hdf5_to_julia(x, OutType::Type)
    if x isa OutType
        return x
    end
    # Transform an array to a staticarray
    if OutType <: SArray
        sarray_size = size(OutType)
        if size(x) != sarray_size
            error("Size of x $(size(x)) not consistent with OutType $OutType")
        end
        return OutType(x)
    end
    # Transform a high-dimensional array to an array of staticarrays
    if OutType <: AbstractArray && eltype(OutType) <: SArray
        sarray_size = size(eltype(OutType))
        if size(x)[1:length(sarray_size)] != sarray_size
            error("Size of x $(size(x)) not consistent with OutType $OutType")
        end
        array_size = size(x)[length(sarray_size)+1:end]
        return collect(reshape(reinterpret(eltype(OutType), vec(x)), array_size))
    end

    # If the function has not returned yet, the conversion is not implemented.
    error("Conversion of $(typeof(x)) into $OutType not implemented")
end


"""
    dump_BTData(f, obj::AbstractBTData{T}) where {T}
Dump BTData object `obj` to an HDF5 file or group `f`.
"""
@timing "dump_BTData" function dump_BTData(f, obj::AbstractBTData{T}) where {T}
    for n in fieldnames(typeof(obj))
        f[String(n)] = _data_julia_to_hdf5(getfield(obj, n))
    end
end

"""
    load_BTData(f, BTType::DataType)
Load BTData object of type `BTType` from an HDF5 file or group `f`.
"""
@timing "load_BTData" function load_BTData(f, BTType::DataType)
    if ! (BTType <: AbstractBTData)
        error("BTType must be a subtype of AbstractBTData")
    end
    data = []
    for (i, n) in enumerate(fieldnames(BTType))
        T = BTType.types[i]
        push!(data, _data_hdf5_to_julia(read(f, String(n)), T))
    end
    BTType(data...)
end