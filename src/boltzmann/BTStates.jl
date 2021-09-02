"States in Boltzmann transport calculation"

using Parameters: @with_kw

"""
    @with_kw struct BTStates{T <: Real} <: AbstractBTData{T}
Each index (i = 1, ..., n) represents a single state in the Brillouin zone.
"""
@with_kw struct BTStates{T <: Real} <: AbstractBTData{T}
    # Number of states
    n::Int
    # Number of k points
    nk::Int
    # Number of bands
    nband::Int
    # Energy
    e::Vector{T} = Vector{T}()
    # Velocity in Cartesian coordinates
    vdiag::Vector{Vec3{T}} = Vector{Vec3{T}}()
    # Brillouin zone weights
    k_weight::Vector{T} = Vector{T}()
    # crystal momentum of the states
    xks::Vector{Vec3{T}} = Vector{Vec3{T}}()
    # Band index of the states
    iband::Vector{Int} = Vector{Int}()
    # Grid size
    ngrid::NTuple{3, Int}
end
# TODO: Add kshift

# Indexing
function Base.getindex(s::BTStates{T}, i::Int) where {T}
    1 <= i <= s.n || throw(BoundsError(s, i))
    (xks=s.xks[i], iband=s.iband[i], e=s.e[i], vdiag=s.vdiag[i], k_weight=s.k_weight[i])
end
Base.firstindex(s::BTStates) = 1
Base.lastindex(s::BTStates) = s.n

"""
    electron_states_to_BTStates(el_states::Vector{ElectronState{T}},
    kpts::EPW.Kpoints{T}) where {T <: Real}
Transform Vector of ElectronState to a BTState.
# Output
- A BTState object
- `imap`: imap[ib, ik] is the index of the state in BTStates (for ib in `el_states[ik].rng`)
"""
@timing "el_to_BT" function electron_states_to_BTStates(el_states::Vector{ElectronState{T}},
        kpts::EPW.AbstractKpoints{T}) where {T <: Real}
    nk = length(el_states)
    n = sum([el.nband for el in el_states])
    ngrid = kpts.ngrid
    max_nband_bound = maximum([el.nband_bound for el in el_states])
    imap = zeros(Int, max_nband_bound, nk)

    e = zeros(T, n)
    vdiag = zeros(Vec3{T}, n)
    k_weight = zeros(T, n)
    xks = zeros(Vec3{T}, n)
    iband = zeros(Int, n)
    istate = 0
    iband_min = el_states[1].nw + 1
    iband_max = -1
    for ik in 1:nk
        el = el_states[ik]
        if el.nband == 0
            continue
        end
        iband_min = min(iband_min, el.rng_full[1])
        iband_max = max(iband_max, el.rng_full[end])
        for ib in el.rng
            istate += 1
            e[istate] = el.e[ib]
            vdiag[istate] = el.vdiag[ib]
            k_weight[istate] = kpts.weights[ik]
            xks[istate] = kpts.vectors[ik]
            iband[istate] = ib + el.nband_ignore
            imap[ib, ik] = istate
        end
    end
    nband = iband_max - iband_min + 1
    BTStates{T}(n, nk, nband, e, vdiag, k_weight, xks, iband, ngrid), imap
end


"""
    phonon_states_to_BTStates(ph_states::Vector{PhononState{T}},
    kpts::EPW.Kpoints{T}) where {T <: Real}
Transform Vector of PhononState to a BTState.
# Output
- A BTState object
- `imap`: imap[ib, ik] is the index of the state in BTStates (for ib in `ph_states[ik].rng`)
"""
@timing "ph_to_BT" function phonon_states_to_BTStates(ph_states::Vector{PhononState{T}},
        kpts::EPW.AbstractKpoints{T}) where {T <: Real}
    nk = length(ph_states)
    nmodes = ph_states[1].nmodes
    ngrid = kpts.ngrid
    imap = zeros(Int, nmodes, nk)

    nband = nmodes
    n = nmodes * nk
    e = zeros(T, n)
    vdiag = zeros(Vec3{T}, n)
    k_weight = zeros(T, n)
    xks = zeros(Vec3{T}, n)
    iband = zeros(Int, n)
    istate = 0
    for ik in 1:nk
        ph = ph_states[ik]
        for imode in 1:nmodes
            istate += 1
            e[istate] = ph.e[imode]
            vdiag[istate] = ph.vdiag[imode]
            k_weight[istate] = kpts.weights[ik]
            xks[istate] = kpts.vectors[ik]
            iband[istate] = imode
            imap[imode, ik] = istate
        end
    end
    BTStates{T}(n, nk, nband, e, vdiag, k_weight, xks, iband, ngrid), imap
end

# TODO: Make ElectronState and PhononState similar so only one function is needed.

"""
    states_index_map(states, symmetry=nothing)
Create a map (xk_int, iband) => i
"""
function states_index_map(states, symmetry=nothing; xk_shift=Vec3(0, 0, 0))
    index_map = Dict{NTuple{4, Int}, Int}()
    for i in 1:states.n
        xk_int = mod.(round.(Int, (states.xks[i] - xk_shift) .* states.ngrid), states.ngrid)
        index_map[(xk_int.data..., states.iband[i])] = i
        if symmetry !== nothing
            for (S, is_tr) in zip(symmetry.S, symmetry.is_tr)
                Sk = is_tr * S * states.xks[i]
                Sk_int = mod.(round.(Int, Sk .* states.ngrid), states.ngrid)
                key = (Sk_int.data..., states.iband[i])
                if ! haskey(index_map, key)
                    index_map[key] = i
                end
            end
        end
    end
    index_map
end

function mpi_gather(s::BTStates{FT}, comm::MPI.Comm) where {FT}
    # FIXME: nk is incorrect
    n = mpi_sum(s.n, comm)
    iband = mpi_gather(s.iband, comm)
    e = mpi_gather(s.e, comm)
    vdiag = mpi_gather(s.vdiag, comm)
    k_weight = mpi_gather(s.k_weight, comm)
    xks = mpi_gather(s.xks, comm)

    # If ngrid is not same among processers, set ngrid to (0,0,0).
    ngrid_root = mpi_bcast(s.ngrid, comm)
    all_ngrids_same = EPW.mpi_reduce(s.ngrid == ngrid_root, &, comm)
    ngrid = all_ngrids_same ? ngrid_root : (0, 0, 0)

    if mpi_isroot(comm)
        @assert n == length(iband) == length(e) == length(vdiag) == length(k_weight) == length(xks)
        nband = length(Set(iband))
        nk = -1
        BTStates{FT}(n=n, nk=nk, nband=nband, e=e, vdiag=vdiag, k_weight=k_weight, xks=xks, iband=iband, ngrid=ngrid)
    else
        BTStates{FT}(n=0, nk=0, nband=0, ngrid=ngrid)
    end
end