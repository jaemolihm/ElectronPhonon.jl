
# Electron eigenvalue and eigenvector at a single k point

import Base.@kwdef

using EPW.WanToBloch: get_el_eigen!, get_el_velocity_diag!

export ElectronState
export copyto!
export set_window!
export set_eigen!
export set_velocity_diag!

@kwdef mutable struct ElectronState{T <: Real}
    nw::Int # Number of Wannier functions
    e_full::Vector{T} # Eigenvalues at all bands
    u_full::Matrix{Complex{T}} # Electron eigenvectors

    # Variables related to the energy window.
    # The bands inside the window must be included in the range
    # nband_ignore+1:nband_ignore+nband_bound.
    # Indexing a full array: arr_full[rng_full]
    # Indexing a filtered array: arr[rng]
    # rng_full = rng .+ nband_ignore
    # nband = length(rng)
    nband_bound::Int # Upper bound of possible nband
    nband_ignore::Int # Number of low-lying ignored bands
    nband::Int # Number of bands inside the energy window
    rng_full::UnitRange{Int} # Index of bands inside the energy window for the full index
    rng::UnitRange{Int} # Index of bands inside the energy window for the offset index
    e::Vector{T} # Eigenvalues at bands inside the energy window
    vdiag::Matrix{T} # Diagonal components of band velocity inside the energy window
end

function ElectronState(T, nw, nband_bound=nw, nband_ignore=0)
    @assert nband_bound > 0
    @assert nband_ignore >= 0
    @assert nband_bound + nband_ignore <= nw

    ElectronState{T}(
        nw=nw,
        e_full=zeros(T, nw),
        u_full=zeros(Complex{T}, nw, nw),
        nband_bound=nband_bound,
        nband_ignore=nband_ignore,
        nband=0,
        rng_full=1:0,
        rng=1:0,
        e=zeros(T, nband_bound),
        vdiag=zeros(T, 3, nband_bound),
    )
end

""" get_u(el)
Return eigenvector for bands inside the window."""
get_u(el) = view(el.u_full, :, el.rng_full)


function copyto!(dest::ElectronState, src::ElectronState)
    if dest.nband_bound < src.nband_bound
        throw(ArgumentError("src.nband_bound ($(src.nband_bound)) cannot be greater " *
            "than dest.nband_bound ($(dest.nband_bound))"))
    end
    if dest.nband_ignore != src.nband_ignore
        throw(ArgumentError("src.nband_ignore ($(src.nband_ignore)) must be " *
            "equal to dest.nband_ignore ($(dest.nband_ignore))"))
    end
    if dest.nw != src.nw
        throw(ArgumentError("src.nw ($(src.nw)) must be " *
            "equal to dest.nw ($(dest.nw))"))
    end
    dest.nband_ignore = src.nband_ignore
    dest.nband = src.nband
    dest.e_full .= src.e_full
    dest.u_full .= src.u_full
    dest.rng_full = src.rng_full
    dest.rng = src.rng
    @views dest.e[src.rng] .= src.e[src.rng]
    @views dest.vdiag[:, src.rng] .= src.vdiag[:, src.rng]
    dest
end

"""
    set_window!(el::ElectronState, window=(-Inf,Inf))
Find out the bands inside the window and set el.nband, el.rng and el.rng_full.
Return true if no bands are selected.
FIXME: return false if no bands are selected..
"""
function set_window!(el::ElectronState, window=(-Inf,Inf))
    ibands = EPW.inside_window(el.e_full, window...)
    # If no bands are selected, return true.
    if isempty(ibands)
        return true
    end
    if ibands[1] <= el.nband_ignore
        throw(BoundsError("Selected bands ($(ibands[1]):$(ibands[end])) must not include " *
            "bands 1 to nband_ignore ($(el.nband_ignore))."))
    end
    if ibands[end] - ibands[1] + 1 > el.nband_bound
        throw(ArgumentError("Number of selected bands ($(ibands[end] - ibands[1] + 1)) " *
            "cannot exceed nband_bound ($(el.nband_bound))."))
    end

    el.rng_full = ibands[1]:ibands[end]
    el.rng = el.rng_full .- el.nband_ignore
    el.nband = length(el.rng)

    @views el.e[el.rng] .= el.e_full[el.rng_full]
    false
end

# Define wrappers of WanToBloch functions

"""
    set_eigen!(el::ElectronState, el_ham, xk, fourier_mode="normal")
Compute electron eigenenergy and eigenvector and save them in el.
"""
function set_eigen!(el::ElectronState, el_ham, xk, fourier_mode="normal")
    get_el_eigen!(el.e_full, el.u_full, el.nw, el_ham, xk, fourier_mode)
end

"""
    set_velocity_diag!(el::ElectronState, el_ham_R, xk, fourier_mode="normal")
Compute electron band velocity, only the band-diagonal part.
"""
function set_velocity_diag!(el::ElectronState, el_ham_R, xk, fourier_mode="normal")
    uk = get_u(el)
    velocity_diag = view(el.vdiag, :, el.rng)
    get_el_velocity_diag!(velocity_diag, el.nw, el_ham_R, xk, uk, fourier_mode)
end