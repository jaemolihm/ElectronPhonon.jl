using SparseArrays

struct ElPhVertexElement{FT}
    mel::Complex{FT}
    econv_p::Bool
    econv_m::Bool
end
Base.zero(::Type{<:ElPhVertexElement}) = nothing

Base.:+(x::ElPhVertexElement, y::Number) = ElPhVertexElement(x.mel + y, x.econv_p, x.econv_m)

"""
Electron-phonon vertex dataset for given ik point.
Access via dset[ikq, iband, jband, imode]`, where `iband` and `jband` are band indices at
`k` and `k+q`, respectively.
"""
struct ElPhVertexDataset{FT}
    data::SparseMatrixCSC{ElPhVertexElement{FT}, Int}
    nband_i::Int
    nband_j::Int
    nband_ignore_i::Int
    nband_ignore_j::Int
    nmodes::Int
    nkq::Int
end

@inline function _index(d::ElPhVertexDataset, ikq, ib, jb, imode)
    (; nband_i, nband_j, nband_ignore_i, nband_ignore_j, nmodes, nkq) = d
    @boundscheck 1 <= imode <= nmodes || throw(BoundsError())
    nband_ignore_i + 1 <= ib <= nband_ignore_i + nband_i || return nothing
    nband_ignore_j + 1 <= jb <= nband_ignore_j + nband_j || return nothing
    1 <= ikq <= nkq || return nothing
    i = ib - nband_ignore_i + (jb - nband_ignore_j - 1) * nband_i
    j = imode + (ikq - 1) * nmodes
    i, j
end

@inline function Base.getindex(d::ElPhVertexDataset{FT}, ikq, ib, jb, imode) where {FT}
    inds = _index(d, ikq, ib, jb, imode)
    inds === nothing ? zero(ElPhVertexElement{FT}) : d.data[inds...]
end

@inline function Base.setindex!(d::ElPhVertexDataset, x, ikq, ib, jb, imode)
    inds = _index(d, ikq, ib, jb, imode)
    inds === nothing && throw(BoundsError())
    d.data[inds...] = x
end

function load_BTData(f, ::Type{ElPhVertexDataset{FT}}) where FT
    mel = read(f, "mel")::Vector{Complex{FT}}
    econv_p = read(f, "econv_p")::Vector{Bool}
    econv_m = read(f, "econv_m")::Vector{Bool}
    ib = Int.(read(f, "ib")::Vector{Int16})
    jb = Int.(read(f, "jb")::Vector{Int16})
    imode = Int.(read(f, "imode")::Vector{Int16})
    ik = read(f, "ik")::Vector{Int}
    ikq = read(f, "ikq")::Vector{Int}

    @assert all(ik .== ik[1])

    nband_ignore_i = minimum(ib) - 1
    nband_ignore_j = minimum(jb) - 1
    nband_i = maximum(ib) - nband_ignore_i
    nband_j = maximum(jb) - nband_ignore_j
    nmodes = maximum(imode)
    nkq = maximum(ikq)

    Is = @. ib - nband_ignore_i + (jb - nband_ignore_j - 1) * nband_i
    Js = @. imode + (ikq - 1) * nmodes
    Vs = ElPhVertexElement.(mel, econv_p, econv_m)
    data = sparse(Is, Js, Vs, nband_i * nband_j, nkq * nmodes)

    ElPhVertexDataset(data, nband_i, nband_j, nband_ignore_i, nband_ignore_j, nmodes, nkq)
end

"""
MatrixElementDataset{T}
Matrix element dataset for given ik point.
Access via dset[ikq, iband, jband]`, where `iband` and `jband` are band indices at
`k` and `k+q`, respectively.
"""
struct MatrixElementDataset{T}
    data::SparseMatrixCSC{T, Int}
    nband_i::Int
    nband_j::Int
    nband_ignore_i::Int
    nband_ignore_j::Int
    nkq::Int
end

@inline function Base.getindex(d::MatrixElementDataset{T}, ikq, ib, jb) where {T}
    (; nband_i, nband_j, nband_ignore_i, nband_ignore_j, nkq) = d
    nband_ignore_i + 1 <= ib <= nband_ignore_i + nband_i || return zero(T)
    nband_ignore_j + 1 <= jb <= nband_ignore_j + nband_j || return zero(T)
    1 <= ikq <= nkq || return zero(T)
    i = ib - nband_ignore_i + (jb - nband_ignore_j - 1) * nband_i
    j = ikq
    d.data[i, j]
end

function load_BTData(f, ::Type{MatrixElementDataset{T}}) where {T}
    mel = read(f, "mel")::Vector{T}
    ib = Int.(read(f, "ib")::Vector{Int16})
    jb = Int.(read(f, "jb")::Vector{Int16})
    ik = read(f, "ik")::Vector{Int}
    ikq = read(f, "ikq")::Vector{Int}

    @assert all(ik .== ik[1])

    nband_ignore_i = minimum(ib) - 1
    nband_ignore_j = minimum(jb) - 1
    nband_i = maximum(ib) - nband_ignore_i
    nband_j = maximum(jb) - nband_ignore_j
    nkq = maximum(ikq)

    Is = @. ib - nband_ignore_i + (jb - nband_ignore_j - 1) * nband_i
    Js = ikq
    data = sparse(Is, Js, mel, nband_i * nband_j, nkq)

    MatrixElementDataset(data, nband_i, nband_j, nband_ignore_i, nband_ignore_j, nkq)
end