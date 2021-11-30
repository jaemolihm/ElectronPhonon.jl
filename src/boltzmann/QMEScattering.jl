using Dictionaries

# Matrix element and energy conservation for a scattering process
const ElPhScatteringElement{T} = NamedTuple{(:mel, :econv_p, :econv_m), Tuple{Complex{T}, Bool, Bool}}

# Map (ikq, ib, jb, imode) to a ElPhScatteringElement. Works only if ik is constant over the dataset.
const QMEElPhScatteringData{T} = Dictionary{CartesianIndex{3}, ElPhScatteringElement{T}}

"""
For a dataset at a given constant ik, create a map scat[ikq][CI(ib, jb, imode)] = (;mel, econv_p, econv_m).
"""
function load_BTData(f, ::Type{QMEElPhScatteringData{FT}}) where FT
    mel = _data_hdf5_to_julia(read(f, "mel"), Vector{Complex{FT}})
    econv_p = _data_hdf5_to_julia(read(f, "econv_p"), BitVector)
    econv_m = _data_hdf5_to_julia(read(f, "econv_m"), BitVector)
    ib = _data_hdf5_to_julia(read(f, "ib"), Vector{Int16})
    jb = _data_hdf5_to_julia(read(f, "jb"), Vector{Int16})
    imode = _data_hdf5_to_julia(read(f, "imode"), Vector{Int16})
    ik = _data_hdf5_to_julia(read(f, "ik"), Vector{Int})
    ikq = _data_hdf5_to_julia(read(f, "ikq"), Vector{Int})

    @assert all(ik .== ik[1])

    g(mel, econv_p, econv_m) = (;mel, econv_p, econv_m)
    scat = Dictionary{Int, QMEElPhScatteringData3{FT}}()

    cis = CartesianIndex.(ib, jb, imode)
    gs = g.(mel, econv_p, econv_m)
    inds = falses(length(ikq))
    @views for i in unique(ikq)
        inds .= ikq .== i
        insert!(scat, i, Dictionary(cis[inds], gs[inds]))
    end
    scat
end