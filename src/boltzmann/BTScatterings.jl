
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
    @views for name in fieldnames(typeof(obj))
        if name == :n
            f[String(name)] = _data_julia_to_hdf5(num)
        else
            f[String(name)] = _data_julia_to_hdf5(getfield(obj, name)[1:num])
        end
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
    nscat_tot = sum([read(f["ik$ik"], "n") for ik in 1:el_i.nk])
    scattering = EPW.ElPhScatteringData{T}(nscat_tot)
    nscat_cumulative = 0
    for ik = 1:el_i.nk
        g = f["ik$ik"]
        scat_ik = load_BTData(g, typeof(scattering))
        nscat_ik = scat_ik.n
        rng = nscat_cumulative+1:nscat_cumulative+nscat_ik
        scattering.ind_el_i[rng] .= scat_ik.ind_el_i
        scattering.ind_el_f[rng] .= scat_ik.ind_el_f
        scattering.ind_ph[rng] .= scat_ik.ind_ph
        scattering.sign_ph[rng] .= scat_ik.sign_ph
        scattering.mel[rng] .= scat_ik.mel
        nscat_cumulative += nscat_ik
    end
    close(fid)
    ElPhBTModel(el_i, el_f, ph, scattering)
end
