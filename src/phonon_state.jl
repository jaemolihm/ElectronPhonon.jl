
# Phonon eigenvalue and eigenvector at a single q point

import Base.@kwdef

using EPW.WanToBloch: get_ph_eigen!

export PhononState
export copyto!
# export set_eigen!
# export set_velocity_diag!

@kwdef mutable struct PhononState{T <: Real}
    nmodes::Int # Number of modes
    e::Vector{T} # Phonon eigenavlues
    u::Matrix{Complex{T}} # Phonon eigenmodes
    # buffers
    occupation::Vector{T} # Phonon occupation number
end

function PhononState(T, nmodes)
    PhononState{T}(
        nmodes=nmodes,
        e=zeros(T, nmodes),
        u=zeros(Complex{T}, nmodes, nmodes),
        occupation=zeros(T, nmodes),
    )
end

function copyto!(dest::PhononState, src::PhononState)
    if dest.nmodes < src.nmodes
        throw(ArgumentError("src.nmodes ($(src.nmodes)) cannot be greater " *
            "than dest.nmodes ($(dest.nmodes))"))
    end
    dest.e .= src.e
    dest.u .= src.u
    dest.occupation .= src.occupation
    dest
end

function set_occupation!(ph::PhononState, T)
    for i = 1:ph.nmodes
        ph.occupation[i] = occ_boson(ph.e[i] / T)
    end
end

# Define wrappers of WanToBloch functions

"""
    set_eigen!(ph::PhononState, model, xk, polar=nothing; fourier_mode="normal")
Compute electron eigenenergy and eigenvector and save them in el.
"""
function set_eigen!(ph::PhononState, model, xk, fourier_mode="normal")
    get_ph_eigen!(ph.e, ph.u, model.ph_dyn, model.mass, xk,
        model.polar_phonon, fourier_mode=fourier_mode)
end

# """
#     set_velocity_diag!(el::ElectronState, el_ham_R, xk, fourier_mode="normal")
# Compute electron band velocity, only the band-diagonal part.
# """
# function set_velocity_diag!(el::ElectronState, el_ham_R, xk, fourier_mode="normal")
#     uk = get_u(el)
#     velocity_diag = view(el.vdiag, :, el.rng)
#     get_el_velocity_diag!(velocity_diag, el.nw, el_ham_R, xk, uk, fourier_mode)
# end