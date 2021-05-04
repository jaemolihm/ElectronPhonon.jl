
"States in Boltzmann transport calculation"

struct BTStates{T <: Real} <: AbstractBTData{T}
    # Each index (i = 1, ..., n) represents a single state in the Brillouin zone.
    n::Int # Number of states
    nk::Int # Number of k points
    nband::Int # Number of bands
    e::Vector{T} # Energy
    vdiag::Vector{Vec3{T}} # Velocity in Cartesian coordinates
    k_weight::Vector{T} # Brillouin zone weights
    xks::Vector{Vec3{T}} # crystal momentum of the states
    iband::Vector{Int} # Band index of the states
    ngrid::NTuple{3, Int} # Grid size
end


"""
    electron_states_to_BTStates(el_states::Vector{ElectronState{T}},
    kpts::EPW.Kpoints{T}) where {T <: Real}
Transform Vector of ElectronState to a BTState.
# Output
- A BTState object
- `imap`: imap[ib, ik] is the index of the state in BTStates (for ib in `el_states[ik].rng`)
"""
@timing "el_to_BT" function electron_states_to_BTStates(el_states::Vector{ElectronState{T}},
        kpts::EPW.Kpoints{T}) where {T <: Real}
    nk = length(el_states)
    n = sum([el.nband for el in el_states])
    ngrid = kpts.ngrid
    max_nband = maximum([el.nband for el in el_states])
    imap = zeros(Int, max_nband, nk)

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
            iband[istate] = el.rng_full[ib]
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
        kpts::EPW.Kpoints{T}) where {T <: Real}
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