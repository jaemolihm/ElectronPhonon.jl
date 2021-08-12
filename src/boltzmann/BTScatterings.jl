
"Scatterings in Boltzmann transport calculation"

export load_ElPhBTModel

struct ElPhScatteringData{T <: Real} <: AbstractBTData{T}
    # Each index (i = 1, ..., n) represents a single scattering process.
    # State indices refers to a state defined by a BTStates object.
    n::Int # Number of scattering processes
    ind_el_i::Vector{Int} # Index of electron initial states
    ind_el_f::Vector{Int} # Index of electron final states
    ind_ph::Vector{Int} # Index of phonon states
    sign_ph::Vector{Int} # Sign of phonon energy. +1 for emission, -1 for absorption.
    mel::Vector{T} # Scattering matrix elements. (Squared e-ph matrix elements.)
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

function ElPhScatteringData{T}(n) where {T}
    ElPhScatteringData{T}(n, zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(T, n))
end
ElPhScatteringData{T}() where {T} = ElPhScatteringData{T}(0)

function concatenate_scattering(scatterings::ElPhScatteringData{T}...) where {T}
    n = sum([scat.n for scat in scatterings])
    ind_el_i = vcat([scat.ind_el_i for scat in scatterings]...)
    ind_el_f = vcat([scat.ind_el_f for scat in scatterings]...)
    ind_ph = vcat([scat.ind_ph for scat in scatterings]...)
    sign_ph = vcat([scat.sign_ph for scat in scatterings]...)
    mel = vcat([scat.mel for scat in scatterings]...)
    ElPhScatteringData{T}(n, ind_el_i, ind_el_f, ind_ph, sign_ph, mel)
end

"""
    dump_BTData(f, obj::ElPhScatteringData{T}, num) where {T}
Dump BTData object `obj` to an HDF5 file or group `f`.
- `num`: Number of scattering processes to save.
"""
@timing "dump_BTData" function dump_BTData(f, obj::ElPhScatteringData{T}, num) where {T}
    if num > obj.n
        error("Number of scattering processes $num cannot be greater than obj.n $(obj.n)")
    end
    # @views for name in fieldnames(typeof(obj))
    #     if name == :n
    #         f[String(name)] = _data_julia_to_hdf5(num)
    #     else
    #         f[String(name)] = _data_julia_to_hdf5(getfield(obj, name)[1:num])
    #     end
    # end
    f["n"] = num
    @views f["ind_el_i"] = obj.ind_el_i[1:num]
    @views f["ind_el_f"] = obj.ind_el_f[1:num]
    @views f["ind_ph"] = obj.ind_ph[1:num]
    @views f["sign_ph"] = obj.sign_ph[1:num]
    @views f["mel"] = obj.mel[1:num]
end

"""
    dump_BTData(f, objs::Vector{ElPhScatteringData{T}}, nums)
Dump BTData objects in `objs` to an HDF5 file or group `f`.
- `nums`: Number of scattering processes to save for each object in `objs`.
TODO: clean up. Maybe use TypedTables...
"""
@timing "dump_BTData" function dump_BTData(f, objs::Vector{ElPhScatteringData{T}}, nums) where {T}
    for (obj, num) in zip(objs, nums)
        if num > obj.n
            error("Number of scattering processes $num cannot be greater than obj.n $(obj.n)")
        end
    end
    num_tot = sum(nums)
    # for name in fieldnames(eltype(objs))
    #     if name == :n
    #         f[String(name)] = _data_julia_to_hdf5(num_tot)
    #     else
    #         dset = create_dataset(f, String(name), datatype(eltype(fieldtype(ElPhScatteringData{T}, name))), (num_tot,))
    #         i = 0
    #         @views for (obj, num) in zip(objs, nums)
    #             dset[i+1:i+num] = getfield(obj, name)[1:num]
    #             i += num
    #         end
    #     end
    # end
    f["n"] = _data_julia_to_hdf5(num_tot)
    dset1 = create_dataset(f, "ind_el_i", datatype(Int), (num_tot,))
    dset2 = create_dataset(f, "ind_el_f", datatype(Int), (num_tot,))
    dset3 = create_dataset(f, "ind_ph", datatype(Int), (num_tot,))
    dset4 = create_dataset(f, "sign_ph", datatype(Int), (num_tot,))
    dset5 = create_dataset(f, "mel", datatype(T), (num_tot,))
    i = 0
    @views for (obj, num) in zip(objs, nums)
        dset1[i+1:i+num] = obj.ind_el_i[1:num]
        dset2[i+1:i+num] = obj.ind_el_f[1:num]
        dset3[i+1:i+num] = obj.ind_ph[1:num]
        dset4[i+1:i+num] = obj.sign_ph[1:num]
        dset5[i+1:i+num] = obj.mel[1:num]
        i += num
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

function load_ElPhBTModel(filename, T=Float64)
    # TODO: Optimize
    fid = h5open(filename, "r")
    el_i = load_BTData(fid["initialstate_electron"], EPW.BTStates{T})
    el_f = load_BTData(fid["finalstate_electron"], EPW.BTStates{T})
    ph = load_BTData(fid["phonon"], EPW.BTStates{T})

    # Read scattering
    f = fid["scattering"]
    nscat_tot = mapreduce(g -> read(g, "n"), +, f)
    scattering = EPW.ElPhScatteringData{T}(nscat_tot)
    nscat_cumulative = 0
    for g in f
        scat_i = load_BTData(g, typeof(scattering))
        nscat_i = scat_i.n
        rng = nscat_cumulative+1:nscat_cumulative+nscat_i
        scattering.ind_el_i[rng] .= scat_i.ind_el_i
        scattering.ind_el_f[rng] .= scat_i.ind_el_f
        scattering.ind_ph[rng] .= scat_i.ind_ph
        scattering.sign_ph[rng] .= scat_i.sign_ph
        scattering.mel[rng] .= scat_i.mel
        nscat_cumulative += nscat_i
    end
    close(fid)
    ElPhBTModel(el_i, el_f, ph, scattering)
end
