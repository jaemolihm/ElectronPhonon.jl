
# Electron eigenvalue and eigenvector at a single k point

import Base.@kwdef

export ElectronState
export set_window!
export set_eigen!
export set_velocity_diag!

@kwdef mutable struct ElectronState{T <: Real}
    nw::Int # Number of Wannier functions
    e_full::Vector{T} # Eigenvalues at all bands
    u_full::Matrix{Complex{T}} # Electron eigenvectors

    # Variables related to the energy window.
    # Applying to full array: arr_full[rng]
    # Applying to filtered array: arr[1:nband]
    nband_bound::Int # Upper bound of possible nband
    nband::Int # Number of bands inside the energy window
    rng::UnitRange{Int} # Index of bands inside the energy window
    e::Vector{T} # Eigenvalues at bands inside the energy window
    vdiag::Matrix{T} # Diagonal components of band velocity inside the energy window
end

function ElectronState(T, nw; nband_bound=nothing)
    if nband_bound === nothing
        nband_bound = nw
    end
    @assert nband_bound > 0
    @assert nband_bound <= nw

    ElectronState{T}(
        nw=nw,
        e_full=zeros(T, nw),
        u_full=zeros(Complex{T}, nw, nw),
        nband_bound=nband_bound,
        nband=0,
        rng=1:0,
        e=zeros(T, nband_bound),
        vdiag=zeros(T, 3, nband_bound),
    )
end

""" get_u(el)
Return eigenvector for bands inside the window."""
get_u(el) = view(el.u_full, :, el.rng)

"""
    set_window!(el::ElectronState, window=(-Inf,Inf))
Find out the bands inside the window and set el.nband and el.rng.
Return true if no bands are selected.
FIXME: return false if no bands are selected..
"""
function set_window!(el::ElectronState, window=(-Inf,Inf))
    ibands = EPW.inside_window(el.e_full, window...)
    # If no bands are selected, return true.
    if isempty(ibands)
        return true
    end

    el.rng = ibands[1]:ibands[end]
    el.nband = length(el.rng)
    if el.nband > el.nband_bound
        throw(ArgumentError("Number of selected bands ($(el.nband)) cannot exceed " *
        "nband_bound ($(el.nband_bound))."))
    end

    @views el.e[1:el.nband] .= el.e_full[el.rng]
    false
end

# Define wrappers of WanToBloch functions

"""
    get_eigen!(el::ElectronState, el_ham, xk, fourier_mode="normal")
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
    velocity_diag = view(el.vdiag, :, 1:el.nband)
    get_el_velocity_diag!(velocity_diag, el.nw, el_ham_R, xk, uk, fourier_mode)
end