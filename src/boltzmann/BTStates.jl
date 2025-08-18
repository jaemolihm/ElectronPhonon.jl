"States in Boltzmann transport calculation"

using Parameters: @with_kw
using Dictionaries
using OffsetArrays

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
    # Number of occupied states per unit cell that are not explicitly included, e.g. because
    # they are below the window. Used only for electrons.
    nstates_base::T = 0
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
    kpts::Kpoints{T}) where {T <: Real}
Transform Vector of ElectronState to a BTState.
# Output
- A BTState object
- `imap`: imap[ib, ik] is the index of the state in BTStates (for ib in `el_states[ik].rng`)
"""
@timing "el_to_BT" function electron_states_to_BTStates(el_states::Vector{ElectronState{T}},
        kpts::AbstractKpoints{T}, nstates_base=0) where {T <: Real}
    nk = length(el_states)
    n = sum([el.nband for el in el_states])
    ngrid = kpts.ngrid
    iband_min = minimum(el.rng.start for el in el_states if el.nband > 0)
    iband_max = maximum(el.rng.stop  for el in el_states if el.nband > 0)
    nband = iband_max - iband_min + 1
    imap = OffsetArray(zeros(Int, nband, nk), iband_min:iband_max, :)

    e = zeros(T, n)
    vdiag = zeros(Vec3{T}, n)
    k_weight = zeros(T, n)
    xks = zeros(Vec3{T}, n)
    iband = zeros(Int, n)
    istate = 0
    for ik in 1:nk
        el = el_states[ik]
        if el.nband == 0
            continue
        end
        for ib in el.rng
            istate += 1
            e[istate] = el.e[ib]
            vdiag[istate] = el.vdiag[ib]
            k_weight[istate] = kpts.weights[ik]
            xks[istate] = kpts.vectors[ik]
            iband[istate] = ib
            imap[ib, ik] = istate
        end
    end
    BTStates{T}(n, nk, nband, e, vdiag, k_weight, xks, iband, ngrid, nstates_base), imap
end


"""
    phonon_states_to_BTStates(ph_states::Vector{PhononState{T}},
    kpts::Kpoints{T}) where {T <: Real}
Transform Vector of PhononState to a BTState.
# Output
- A BTState object
- `imap`: imap[ib, ik] is the index of the state in BTStates (for ib in `ph_states[ik].rng`)
"""
@timing "ph_to_BT" function phonon_states_to_BTStates(ph_states::Vector{PhononState{T}},
        kpts::AbstractKpoints{T}) where {T <: Real}
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
    BTStates{T}(;n, nk, nband, e, vdiag, k_weight, xks, iband, ngrid), imap
end

# TODO: Make ElectronState and PhononState similar so only one function is needed.

"""
    states_index_map(states, symmetry=nothing)
Create a map such that map[CartesianIndex(xk_int)][iband] = i.
"""
function states_index_map(states, symmetry=nothing; xk_shift=Vec3(0, 0, 0))
    index_map = Dictionary{CI{3}, Vector{Int}}()
    nband = states.nband
    for i in 1:states.n
        iband = states.iband[i]
        xk_int = mod.(round.(Int, (states.xks[i] - xk_shift) .* states.ngrid), states.ngrid)
        key = CI(xk_int...)
        if ! haskey(index_map, key)
            insert!(index_map, key, zeros(Int, nband))
        end
        index_map[key][iband] = i
        if symmetry !== nothing
            for (S, is_tr) in zip(symmetry.S, symmetry.is_tr)
                Sk = is_tr * S * states.xks[i]
                Sk_int = mod.(round.(Int, Sk .* states.ngrid), states.ngrid)
                key = CI(Sk_int...)
                if ! haskey(index_map, key)
                    insert!(index_map, key, zeros(Int, nband))
                end
                index_map[key][iband] = i
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
    # FIXME: Should this be done? (mpi_sum for nelec_below_window is called in filter.jl)
    # nstates_base = mpi_sum(s.nstates_base, comm)

    # If ngrid is not same among processers, set ngrid to (0,0,0).
    ngrid_root = mpi_bcast(s.ngrid, comm)
    all_ngrids_same = mpi_reduce(s.ngrid == ngrid_root, &, comm)
    ngrid = all_ngrids_same ? ngrid_root : (0, 0, 0)

    if mpi_isroot(comm)
        @assert n == length(iband) == length(e) == length(vdiag) == length(k_weight) == length(xks)
        nband = length(Set(iband))
        nk = -1
        BTStates{FT}(; n, nk, nband, e, vdiag, k_weight, xks, iband, ngrid, nstates_base)
    else
        BTStates{FT}(n=0, nk=0, nband=0, ngrid=ngrid)
    end
end


"""
    find_unfolding_indices(el_i, el_f, symmetry)
    => ind_and_isym_map :: Vector{NTuple{2, Int}}

Build a map that unfolds a velocity (polar, time-reversal odd) on `el_i` to `el_f` by `symmetry`.
``S(ind_and_isym_map[i][2]) * el_i[ind_and_isym_map[i][1]] -> el_f[i]``

If there are multiple symmetries that map f to i, return only the first one.
The `i`-th element of `ind_and_isym_map` always corresponds to the `i`-th element of `el_f`.
"""
function find_unfolding_indices(el_i :: BTStates, el_f :: BTStates, symmetry)
    ind_and_isym_map = fill((0, 0), el_f.n)

    for i in 1:el_f.n
        xk_f = el_f[i].xks
        ib = el_f[i].iband

        found = false

        for (isym, S) in enumerate(symmetry)
            for j in 1:el_i.n
                if el_i[j].iband == ib
                    k_i = el_i[j].xks
                    Sk_i = apply_symop(S, k_i, :momentum)
                    dk = Sk_i - xk_f
                    if all(abs.(dk .- round.(dk)) .< 1e-6)
                        ind_and_isym_map[i] = (j, isym)
                        found = true
                        break
                    end
                end
            end
            found && break
        end

        found || error("Cannot find the corresponding state in el_i for i = $i, e_f = $(el_f[i])")
    end

    ind_and_isym_map
end


"""
    find_unfolding_indices_with_duplicates(el_i, el_f, symmetry)
    => ind_and_isym_map :: Vector{NTuple{4, Int}}

Build a map that unfolds a velocity (polar, time-reversal odd) on `el_i` to `el_f` by `symmetry`.
``S(ind_and_isym_map[i][2]) * el_i[ind_and_isym_map[i][1]] -> el_f[ind_and_isym_map[3]]``
There are `ind_and_isym_map[4]` symmetries with this mapping.

If there are multiple symmetries that map f to i, return everything.
Now `i`-th element of `ind_and_isym_map` does not necessarily correspond to the `i`-th
element of `el_f`.
"""
function find_unfolding_indices_with_duplicates(el_i :: BTStates, el_f :: BTStates, symmetry)
    ind_and_isym_map = NTuple{4, Int}[]
    list_tmp = NTuple{3, Int}[]

    for i in 1:el_f.n
        xk_f = el_f[i].xks
        ib = el_f[i].iband

        # To count the degeneracy, store the indices on a temporary list
        empty!(list_tmp)

        for (isym, S) in enumerate(symmetry)
            for j in 1:el_i.n
                if el_i[j].iband == ib
                    k_i = el_i[j].xks
                    Sk_i = apply_symop(S, k_i, :momentum)
                    dk = Sk_i - xk_f
                    if all(abs.(dk .- round.(dk)) .< 1e-6)
                        push!(list_tmp, (j, isym, i))
                    end
                end
            end
        end

        ndegen = length(list_tmp)
        for ii in list_tmp
            push!(ind_and_isym_map, (ii..., ndegen))
        end

        ndegen == 0 && error("Cannot find the corresponding state in el_i for i = $i, e_f = $(el_f[i])")
    end

    ind_and_isym_map
end

"""
    _BTE_unfold_δf(δf_i, el_f, indmap, symmetry) => δf_f
Unfold `δf_i` (polar, time-reversal odd) on irreducible BZ to the full BZ with states `el_f`.
`indmap` must be precomputed using `find_unfolding_indices_with_duplicates(el_i, el_f, symmetry)`.
"""
function _BTE_unfold_δf(δf_i, el_f, indmap, symmetry)
    δf_f = zeros(eltype(δf_i), el_f.n)
    for (indi, isym, indf, ndegen) in indmap
        δf_f[indf] += apply_symop(symmetry[isym], δf_i[indi], :momentum_cartesian) / ndegen
    end
    δf_f
end
