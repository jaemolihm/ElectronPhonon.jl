
"Scatterings in Boltzmann transport calculation"

export load_ElPhBTModel

mutable struct ElPhScatteringData{T <: Real} <: AbstractBTData{T}
    # Each index (i = 1, ..., n) represents a single scattering process.
    # State indices refers to a state defined by a BTStates object.
    n::Int # Number of scattering processes
    const ind_el_i::Vector{Int} # Index of electron initial states
    const ind_el_f::Vector{Int} # Index of electron final states
    const ind_ph::Vector{Int} # Index of phonon states
    const sign_ph::Vector{Int} # Sign of phonon energy. +1 for emission, -1 for absorption.
    const mel::Vector{T} # Scattering matrix elements. (Squared e-ph matrix elements.)
end
function ElPhScatteringData{T}(n=0) where {T}
    ElPhScatteringData{T}(0, zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(T, n))
end

# Indexing
function Base.getindex(scat::ElPhScatteringData{T}, i::Int) where {T}
    1 <= i <= scat.n || throw(BoundsError(scat, i))
    (ind_el_i=scat.ind_el_i[i], ind_el_f=scat.ind_el_f[i], ind_ph=scat.ind_ph[i], sign_ph=scat.sign_ph[i], mel=scat.mel[i])
end
Base.firstindex(scat::ElPhScatteringData) = 1
Base.lastindex(scat::ElPhScatteringData) = scat.n

# Iterating
Base.iterate(scat::ElPhScatteringData{T}, state=1) where {T} = state > scat.n ? nothing : (scat[state], state+1)
Base.length(scat::ElPhScatteringData) = scat.n
Base.eltype(::Type{ElPhScatteringData{T}}) where {T} = NamedTuple{(:ind_el_i, :ind_el_f, :ind_ph, :sign_ph, :mel), Tuple{Int64, Int64, Int64, Int64, T}}

# TODO: Enable ElPhScatteringData to have AbstractArray so that filtered data becomes a view, not a copy.
function Base.filter(f, scat::ElPhScatteringData{T}) where {T}
    inds = map(s -> f(s), scat)
    ElPhScatteringData{T}(sum(inds), scat.ind_el_i[inds], scat.ind_el_f[inds], scat.ind_ph[inds], scat.sign_ph[inds], scat.mel[inds])
end

Base.empty!(scat::ElPhScatteringData) = (scat.n = 0; scat)
function Base.push!(scat::ElPhScatteringData, data::Tuple{Int, Int, Int, Int, T}) where {T}
    scat.n += 1
    scat.n > length(scat.ind_el_i) && error("number of scatterings exceeded the size of ElPhScatteringData")
    scat.ind_el_i[scat.n] = data[1]
    scat.ind_el_f[scat.n] = data[2]
    scat.ind_ph[scat.n] = data[3]
    scat.sign_ph[scat.n] = data[4]
    scat.mel[scat.n] = data[5]
end

@views function Base.vcat(scatterings::ElPhScatteringData{T}...) where {T}
    n = sum(scat.n for scat in scatterings)
    ind_el_i = reduce(vcat, scat.ind_el_i[1:scat.n] for scat in scatterings)
    ind_el_f = reduce(vcat, scat.ind_el_f[1:scat.n] for scat in scatterings)
    ind_ph = reduce(vcat, scat.ind_ph[1:scat.n] for scat in scatterings)
    sign_ph = reduce(vcat, scat.sign_ph[1:scat.n] for scat in scatterings)
    mel = reduce(vcat, scat.mel[1:scat.n] for scat in scatterings)
    ElPhScatteringData{T}(n, ind_el_i, ind_el_f, ind_ph, sign_ph, mel)
end

"""
    dump_BTData(f, obj::ElPhScatteringData{T}, num) where {T}
Dump BTData object `obj` to an HDF5 file or group `f`.
- `num`: Number of scattering processes to save.
"""
@timing "dump_BTData" function dump_BTData(f, obj::ElPhScatteringData{T}, n=obj.n) where {T}
    if n > obj.n
        error("Number of scattering processes $n cannot be greater than obj.n $(obj.n)")
    end
    f["n"] = n
    @views f["ind_el_i"] = obj.ind_el_i[1:n]
    @views f["ind_el_f"] = obj.ind_el_f[1:n]
    @views f["ind_ph"] = obj.ind_ph[1:n]
    @views f["sign_ph"] = obj.sign_ph[1:n]
    @views f["mel"] = obj.mel[1:n]
end

"""
    dump_BTData(f, objs::Vector{ElPhScatteringData{T}}, ns)
Dump BTData objects in `objs` to an HDF5 file or group `f`.
- `ns`: Number of scattering processes to save for each object in `objs`.
TODO: clean up. Maybe use TypedTables...
"""
@timing "dump_BTData" function dump_BTData(f, objs::Vector{ElPhScatteringData{T}},
                                           ns=[obj.n for obj in objs]) where {T}
    for (obj, n) in zip(objs, ns)
        if n > obj.n
            error("Number of scattering processes $n cannot be greater than obj.n $(obj.n)")
        end
    end
    ntot = sum(ns)
    f["n"] = _data_julia_to_hdf5(ntot)
    dset1 = create_dataset(f, "ind_el_i", datatype(Int), (ntot,))
    dset2 = create_dataset(f, "ind_el_f", datatype(Int), (ntot,))
    dset3 = create_dataset(f, "ind_ph", datatype(Int), (ntot,))
    dset4 = create_dataset(f, "sign_ph", datatype(Int), (ntot,))
    dset5 = create_dataset(f, "mel", datatype(T), (ntot,))
    i = 0
    @views for (obj, n) in zip(objs, ns)
        n == 0 && continue
        dset1[i+1:i+n] = obj.ind_el_i[1:n]
        dset2[i+1:i+n] = obj.ind_el_f[1:n]
        dset3[i+1:i+n] = obj.ind_ph[1:n]
        dset4[i+1:i+n] = obj.sign_ph[1:n]
        dset5[i+1:i+n] = obj.mel[1:n]
        i += n
    end
end

function _dump_array_to_hdf5()
    dset = create_dataset(f, String(name), datatype(eltype(fieldtype(ElPhScatteringData{T}, name))), (num_tot,))
    i = 0
    @views for (obj, num) in zip(objs, nums)
        dset[i+1:i+num] = getfield(obj, name)[1:num]
        i += num
    end
end


"""
Defines a Boltzmann transport problem with electron-phonon scattering
"""
struct ElPhBTModel{T <: Real}
    el_i::BTStates{T} # Initial electron state
    el_f::BTStates{T} # Final electron state
    ph::BTStates{T} # Phonon state
    scattering::ElPhScatteringData{T} # Scattering processes
end

function load_ElPhBTModel(filename, ::Type{FT}=Float64) where FT
    # TODO: Optimize
    fid = h5open(filename, "r")
    el_i = load_BTData(open_group(fid, "initialstate_electron"), BTStates{FT})
    el_f = load_BTData(open_group(fid, "finalstate_electron"), BTStates{FT})
    ph = load_BTData(open_group(fid, "phonon"), BTStates{FT})

    # Read scattering
    group_scattering = open_group(fid, "scattering")
    nscat_tot = mapreduce(g -> read(g, "n"), +, group_scattering)
    scattering = ElPhScatteringData{FT}(nscat_tot)
    for key in keys(group_scattering)
        g = open_group(group_scattering, key)
        scat_i = load_BTData(g, typeof(scattering))
        rng = (scattering.n + 1) : (scattering.n + scat_i.n)
        scattering.ind_el_i[rng] .= scat_i.ind_el_i
        scattering.ind_el_f[rng] .= scat_i.ind_el_f
        scattering.ind_ph[rng] .= scat_i.ind_ph
        scattering.sign_ph[rng] .= scat_i.sign_ph
        scattering.mel[rng] .= scat_i.mel
        scattering.n += scat_i.n
    end
    close(fid)
    ElPhBTModel(el_i, el_f, ph, scattering)
end
