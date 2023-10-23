
# Phonon eigenvalue and eigenvector at a single q point

using EPW.WanToBloch: get_ph_eigen!

export PhononState
export copyto!
export set_occupation!
export set_eigen!
export set_eigen_valueonly!
export set_velocity_diag!

Base.@kwdef mutable struct PhononState{T <: Real}
    nmodes::Int # Number of modes
    e::Vector{T} # Phonon eigenavlues
    u::Matrix{Complex{T}} # Phonon eigenmodes
    vdiag::Vector{Vec3{T}} # Diagonal components of band velocity in Cartesian coordinates.
    occupation::Vector{T} # Phonon occupation number
    eph_dipole_coeff::Vector{Complex{T}} # dipole potential coefficients for e-ph coupling
end

function PhononState{T}(nmodes) where {T}
    PhononState{T}(
        nmodes=nmodes,
        e=zeros(T, nmodes),
        u=zeros(Complex{T}, nmodes, nmodes),
        vdiag=zeros(Vec3{T}, nmodes),
        occupation=zeros(T, nmodes),
        eph_dipole_coeff=zeros(Complex{T}, nmodes)
    )
end

PhononState(nmodes, ::Type{FT}=Float64) where FT = PhononState{FT}(nmodes)

function Base.copyto!(dest::PhononState, src::PhononState)
    if dest.nmodes < src.nmodes
        throw(ArgumentError("src.nmodes ($(src.nmodes)) cannot be greater " *
            "than dest.nmodes ($(dest.nmodes))"))
    end
    dest.e .= src.e
    dest.u .= src.u
    dest.occupation .= src.occupation
    dest.eph_dipole_coeff .= src.eph_dipole_coeff
    for i in 1:src.nmodes
        dest.vdiag[i] = src.vdiag[i]
    end
    dest
end

function set_occupation!(ph::PhononState, T)
    for i = 1:ph.nmodes
        ph.occupation[i] = occ_boson(ph.e[i], T)
    end
end
get_occupation(ph::PhononState, T) = occ_boson.(ph.e, T)

# Define wrappers of WanToBloch functions

"""
    set_eigen!(ph::PhononState, model, xk, polar=nothing; fourier_mode="normal")
Compute electron eigenenergy and eigenvector and save them in el.
"""
function set_eigen!(ph::PhononState, model, xk; fourier_mode="normal")
    get_ph_eigen!(ph.e, ph.u, model.ph_dyn, model.mass, xk, model.polar_phonon; fourier_mode)
end

"""
    set_eigen_valueonly!(ph::PhononState, model, xk; fourier_mode="normal")
Compute electron eigenenergy and save them in el.
"""
function set_eigen_valueonly!(ph::PhononState, model, xk; fourier_mode="normal")
    get_ph_eigen_valueonly!(ph.e, model.ph_dyn, model.mass, xk, model.polar_phonon; fourier_mode)
end

"""
    set_velocity_diag!(ph::PhononState, ph_dyn_R, xk; fourier_mode="normal")
Compute electron band velocity, only the band-diagonal part.
"""
function set_velocity_diag!(ph::PhononState{T}, model, xk; fourier_mode="normal") where {T}
    @views vdiag = reshape(reinterpret(T, ph.vdiag), 3, ph.nmodes)
    get_ph_velocity_diag!(vdiag, model.ph_dyn_R, xk, ph.u; fourier_mode)
    # get_ph_velocity_diag! calculates the derivative of the dynamical matrix, but the
    # phonon frequency is sqrt of the eigenvalue of the dynamical matrix.
    # We use dw/dk = (d(w^2)/dk) / (2 * w).
    @views for imode in 1:ph.nmodes
        if ph.e[imode] < omega_acoustic
            continue
        end
        vdiag[:, imode] ./= 2 .* ph.e[imode]
    end
end

"""
    set_eph_dipole_coeff!(ph::PhononState{T}, model, xk)
Compute the coefficients for the dipole electron-phonon coupling. The phonon eigenstates must
be already set.
"""
function set_eph_dipole_coeff!(ph::PhononState{T}, model, xk) where {T}
    get_eph_dipole_coeffs!(ph.eph_dipole_coeff, xk, model.polar_eph, ph.u)
end
