# Struct for quantum kinetic equation including interband coherence

using Parameters: @with_kw
using Dictionaries

"""
    QMEStates{T <: Real} <: AbstractBTData{T}
States for quantum master equation. Includes off-diagonal coherence between eigenstates.
"""
@with_kw struct QMEStates{T <: Real} <: AbstractBTData{T}
    # Number of coherences and states
    n::Int
    # Number of bands
    nband::Int
    # Energy
    e1::Vector{T} = T[]
    # Energy
    e2::Vector{T} = T[]
    # Velocity in Cartesian coordinates
    v::Vector{Vec3{Complex{T}}} = Vec3{Complex{T}}[]
    # Band index of the states
    ib1::Vector{Int} = Int[]
    # Band index of the states
    ib2::Vector{Int} = Int[]
    # Index of k points
    ik::Vector{Int} = Int[]
    # Range of bands at each k point
    ib_rng::Vector{UnitRange{Int}} = UnitRange{Int}[]
    # Number of occupied states per unit cell that are not explicitly included, e.g. because
    # they are below the window. Used only for electrons.
    nstates_base::T = 0
    # k points
    kpts::GridKpoints{T}
    # index map
    indmap::Dictionaries.Dictionary{CartesianIndex{3}, Int} = states_index_map(n, ib1, ib2, ik)
end

# Indexing
function Base.getindex(s::QMEStates, i::Int)
    1 <= i <= s.n || throw(BoundsError(s, i))
    (ib1=s.ib1[i], ib2=s.ib2[i], e1=s.e1[i], e2=s.e2[i], ik=s.ik[i], v=s.v[i],
    xks=s.kpts.vectors[s.ik[i]], k_weight=s.kpts.weights[s.ik[i]])
end
Base.firstindex(s::QMEStates) = 1
Base.lastindex(s::QMEStates) = s.n


# HDF IO
# TODO: Move kpoints IO to kpoints.jl. Rename load_BTData -> load_from_hdf(?)
# TODO: Merge _data_hdf5_to_julia and dump_BTData
@timing "dump_BTData" function dump_BTData(f, obj::QMEStates{T}) where {T}
    for name in fieldnames(typeof(obj))
        if name === :kpts
            g = create_group(f, String(name))
            dump_BTData(g, getfield(obj, name))
        elseif name === :indmap
            continue
        else
            f[String(name)] = _data_julia_to_hdf5(getfield(obj, name))
        end
    end
end

function dump_BTData(f, obj::AbstractKpoints{T}) where T
    for name in fieldnames(typeof(obj))
        if name !== :_xk_hash_to_ik
            f[String(name)] = _data_julia_to_hdf5(getfield(obj, name))
        end
    end
end

@timing "load_BTData" function load_BTData(f, T::Type{QMEStates{FT}}) where FT
    data = []
    for (i, name) in enumerate(fieldnames(T))
        if name === :kpts
            push!(data, (name => load_BTData(f[String(name)], T.types[i])))
        elseif name === :indmap
            continue
        else
            push!(data, (name => _data_hdf5_to_julia(read(f, String(name)), T.types[i])))
        end
    end
    QMEStates(; data...)
end

function load_BTData(f, ::Type{GridKpoints{FT}}) where FT
    kpts = load_BTData(f, Kpoints{FT})
    GridKpoints(kpts)
end


"""
- `offdiag_cutoff`: Maximum interband energy difference to include the off-diagonal part.
"""
@timing "el_to_BTC" function electron_states_to_QMEStates(el_states::Vector{ElectronState{T}},
        kpts::EPW.AbstractKpoints{T}, offdiag_cutoff, nstates_base=zero(T)) where {T <: Real}
    @assert kpts.n == length(el_states)
    nk = length(el_states)

    e1 = T[]
    e2 = T[]
    v = Vec3{Complex{T}}[]
    ib1_list = Int[]
    ib2_list = Int[]
    ik_list = Int[]
    ib_rng = UnitRange{Int}[]
    n = 0
    for ik in 1:nk
        el = el_states[ik]
        el.nband == 0 && continue
        for ib2 in el.rng, ib1 in el.rng
            if abs(el.e[ib1] - el.e[ib2]) <= offdiag_cutoff
                n += 1
                push!(e1, el.e[ib1])
                push!(e2, el.e[ib2])
                push!(ib1_list, ib1 + el.nband_ignore)
                push!(ib2_list, ib2 + el.nband_ignore)
                push!(ik_list, ik)
                push!(v, el.v[ib1, ib2])
            end
        end
        push!(ib_rng, el.rng .+ el.nband_ignore)
    end
    iband_min = minimum(el.rng_full.start for el in el_states if length(el.nband) > 0)
    iband_max = maximum(el.rng_full.stop for el in el_states if length(el.nband) > 0)
    nband = iband_max - iband_min + 1
    QMEStates(; n, nband, e1, e2, v, ib1=ib1_list, ib2=ib2_list, ik=ik_list, ib_rng, nstates_base, kpts=GridKpoints(kpts))
end

function BTStates(s::QMEStates)
    # Take only diagonal elements from QMEStates and create BTStates
    inds = s.ib1 .== s.ib2
    EPW.BTStates(sum(inds), s.kpts.n, s.nband, s.e1[inds], real.(s.v[inds]), s.kpts.weights[s.ik[inds]],
                 s.kpts.vectors[s.ik[inds]], s.ib1[inds], s.kpts.ngrid, s.nstates_base)
end

"""
    states_index_map(states)
Create a map (ib1, ib2, ik) => i
"""
function states_index_map(states::QMEStates)
    @warn "DEPRECATED"
    states_index_map(states.n, states.ib1, states.ib2, states.ik)
end

function states_index_map(n, ib1, ib2, ik)
    index_map = Dictionary{CI{3}, Int}()
    for i in 1:n
        insert!(index_map, CI(ib1[i], ib2[i], ik[i]), i)
    end
    index_map
end

"""
    get_1d_index(states, ib1, ib2, ik)
Return the 1d index of the state `(ib1, ib2, ik)` in `states`. Return 0 if such state does not
exist in `states`.
"""
@inline function get_1d_index(states::QMEStates, ib1, ib2, ik)
    get(states.indmap, CI(ib1, ib2, ik), 0)
end

@inline function hasstate(states::QMEStates, ib1, ib2, ik)
    haskey(states.indmap, CI(ib1, ib2, ik))
end