using SparseArrays

struct ElPhVertexElement{FT}
    mel::Complex{FT}
    econv_p::Bool
    econv_m::Bool
end
Base.zero(::Type{<:ElPhVertexElement}) = nothing

struct ElPhVertexDataset{FT}
    # FIXME: different nband for ib and jb
    data::SparseMatrixCSC{ElPhVertexElement{FT}, Int}
    nband::Int
    nband_ignore::Int
    nmodes::Int
end

function Base.getindex(d::ElPhVertexDataset, ikq, ib, jb, imode)
    (; nband_ignore, nband, nmodes) = d
    @boundscheck begin
        1 <= imode <= nmodes || throw(BoundsError())
        nband_ignore + 1 <= ib <= nband_ignore + nband || throw(BoundsError())
        nband_ignore + 1 <= jb <= nband_ignore + nband || throw(BoundsError())
    end
    # i = ib - nband_ignore + (jb - nband_ignore - 1) * nband + (imode - 1) * nband^2
    # j = ikq
    i = ib - nband_ignore + (jb - nband_ignore - 1) * nband
    j = imode + (ikq - 1) * nmodes
    d.data[i, j]
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

    nband_ignore = min(minimum(ib), minimum(jb)) - 1
    nband = max(maximum(ib), maximum(jb)) - nband_ignore
    nmodes = maximum(imode)

    Is = @. ib - nband_ignore + (jb - nband_ignore - 1) * nband
    Js = @. imode + (ikq - 1) * nmodes
    Vs = ElPhVertexElement.(mel, econv_p, econv_m)
    data = sparse(Is, Js, Vs)

    ElPhVertexDataset(data, nband, nband_ignore, nmodes)
end