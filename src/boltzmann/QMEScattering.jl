using Dictionaries

# Matrix element and energy conservation for a scattering process
const ElPhScatteringElement{T} = NamedTuple{(:mel, :econv_p, :econv_m), Tuple{Complex{T}, Bool, Bool}}

# Map (ik, ib, ikq, jb, imode) to a ElPhScatteringElement
const QMEElPhScatteringData{T} = Dictionary{CartesianIndex{5}, ElPhScatteringElement{T}}

@inline function load_scatteringdata(g)
    mel = g["mel"]::Vector{Complex{Float64}}
    econv_p = g["econv_p"]::BitVector
    econv_m = g["econv_m"]::BitVector
    ib = g["ib"]::Vector{Int16}
    jb = g["jb"]::Vector{Int16}
    imode = g["imode"]::Vector{Int16}
    ik = g["ik"]::Vector{Int}
    ikq = g["ikq"]::Vector{Int}

    nscat = length(mel)
    scat = QMEElPhScatteringData{Float64}(sizehint=nscat)
    for iscat in 1:nscat
        key = CI(ik[iscat], ib[iscat], ikq[iscat], jb[iscat], imode[iscat])
        insert!(scat, key, (mel=mel[iscat], econv_p=econv_p[iscat], econv_m=econv_m[iscat]))
    end
    scat
end

# fid = jldopen(filename, "r")
# @code_warntype load_scatteringdata(fid["scattering/ik$ik"])
# close(fid)