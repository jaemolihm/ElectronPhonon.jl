using Dictionaries

# Matrix element and energy conservation for a scattering process
const ElPhScatteringElement{T} = NamedTuple{(:mel, :econv_p, :econv_m), Tuple{Complex{T}, Bool, Bool}}

# Map (ik, ib, ikq, jb, imode) to a ElPhScatteringElement
const QMEElPhScatteringData{T} = Dictionary{CartesianIndex{5}, ElPhScatteringElement{T}}

@inline function load_scatteringdata(f)
    mel = _data_hdf5_to_julia(read(f, "mel"), Vector{Complex{Float64}})
    econv_p = _data_hdf5_to_julia(read(f, "econv_p"), BitVector)
    econv_m = _data_hdf5_to_julia(read(f, "econv_m"), BitVector)
    ib = _data_hdf5_to_julia(read(f, "ib"), Vector{Int16})
    jb = _data_hdf5_to_julia(read(f, "jb"), Vector{Int16})
    imode = _data_hdf5_to_julia(read(f, "imode"), Vector{Int16})
    ik = _data_hdf5_to_julia(read(f, "ik"), Vector{Int})
    ikq = _data_hdf5_to_julia(read(f, "ikq"), Vector{Int})

    g(mel, econv_p, econv_m) = (;mel, econv_p, econv_m)
    scat = Dictionary(CartesianIndex.(ik, ib, ikq, jb, imode), g.(mel, econv_p, econv_m))
    scat
end
