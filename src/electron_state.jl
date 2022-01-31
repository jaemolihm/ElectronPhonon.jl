
# Electron eigenvalue and eigenvector at a single k point

using EPW.AllocatedLAPACK: epw_syev!
using EPW.WanToBloch: get_el_eigen!, get_el_velocity_diag_berry_connection!,
                      get_el_velocity_berry_connection!, get_el_velocity_direct!

export ElectronState
export copyto!
export set_window!
export set_occupation!
export set_eigen!
export set_eigen_valueonly!
export set_velocity_diag!
export set_position!

# TODO: Remove nband_bound. nband >= length(rng) can hold. Create a function `trim(el::ElectronState)`
#       (or even trim!) that reduces nband to length(rng).

Base.@kwdef mutable struct ElectronState{T <: Real}
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

    # These arrays are defined only for bands inside the window
    e::Vector{T} # Eigenvalues at bands inside the energy window
    vdiag::Vector{Vec3{T}} # Diagonal components of band velocity in Cartesian coordinates.
    v::Matrix{Vec3{Complex{T}}} # Velocity matrix in Cartesian coordinates.
    rbar::Matrix{Vec3{Complex{T}}} # Position matrix in Cartesian coordinates (without the Hamiltonian derivative term).
    occupation::Vector{T} # Electron occupation number
end

function ElectronState{T}(nw, nband_bound=nw, nband_ignore=0) where {T}
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
        vdiag=fill(zeros(Vec3{T}), (nband_bound,)),
        v=fill(zeros(Vec3{Complex{T}}), (nband_bound, nband_bound)),
        rbar=fill(zeros(Vec3{Complex{T}}), (nband_bound, nband_bound)),
        occupation=zeros(T, nband_bound),
    )
end

ElectronState(nw, ::Type{FT}=Float64; nband_bound=nw, nband_ignore=0) where FT = ElectronState{FT}(nw, nband_bound, nband_ignore)

""" get_u(el)
Return eigenvector for bands inside the window."""
get_u(el) = view(el.u_full, :, el.rng_full)

# This defines el.u, but it is type unstable
# Base.getproperty(el::ElectronState, v::Symbol) = v === :u ? view(el.u_full, :, el.rng_full) : getfield(el, v)

function Base.copyto!(dest::ElectronState, src::ElectronState)
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
    for ib in src.rng
        dest.e[ib] = src.e[ib]
        dest.vdiag[ib] = src.vdiag[ib]
        for jb in src.rng
            dest.v[jb, ib] = src.v[jb, ib]
        end
    end
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
        el.nband = 0
        el.rng = 1:0
        el.rng_full = 1:0
        return true
    end
    if ibands[1] <= el.nband_ignore
        throw(ArgumentError("Selected bands ($(ibands[1]):$(ibands[end])) must not include " *
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

function set_occupation!(el::ElectronState, μ, T)
    for i in el.rng
        el.occupation[i] = occ_fermion(el.e[i] - μ, T)
    end
end

# Define wrappers of WanToBloch functions

"""
    set_eigen!(el::ElectronState, model, xk, fourier_mode="normal")
Compute electron eigenenergy and eigenvector and save them in el.
"""
function set_eigen!(el::ElectronState, model, xk, fourier_mode="normal")
    get_el_eigen!(el.e_full, el.u_full, el.nw, model.el_ham, xk, fourier_mode)

    # Set window to the default value: [nband_ignore+1, nband_ignore+nband_bound].
    el.rng_full = el.nband_ignore+1:el.nband_ignore+el.nband_bound
    el.rng = el.rng_full .- el.nband_ignore
    el.nband = el.nband_bound
    @views el.e[el.rng] .= el.e_full[el.rng_full]
end

"""
    set_eigen_valueonly!(el::ElectronState, model, xk, fourier_mode="normal")
Compute electron eigenenergy and save them in el.
"""
function set_eigen_valueonly!(el::ElectronState, model, xk, fourier_mode="normal")
    get_el_eigen_valueonly!(el.e_full, el.nw, model.el_ham, xk, fourier_mode)

    # Set window to the default value: [nband_ignore+1, nband_ignore+nband_bound].
    el.rng_full = el.nband_ignore+1:el.nband_ignore+el.nband_bound
    el.rng = el.rng_full .- el.nband_ignore
    el.nband = el.nband_bound
    @views el.e[el.rng] .= el.e_full[el.rng_full]
end

"""
    set_velocity_diag!(el::ElectronState, model, xk, fourier_mode="normal")
Compute electron band velocity, only the band-diagonal part.
"""
function set_velocity_diag!(el::ElectronState{FT}, model, xk, fourier_mode="normal") where {FT}
    uk = get_u(el)
    if model.el_velocity_mode === :Direct
        # For direct Wannier interpolation, there is no faster way to calculate only the diagonal part.
        # So we just calculate the full velocity matrix and set take the diagonal part.
        @views velocity = reshape(reinterpret(Complex{FT}, el.v[el.rng, el.rng]), 3, el.nband, el.nband)
        get_el_velocity_direct!(velocity, el.nw, model.el_vel, xk, uk, fourier_mode)
        for i in el.rng
            el.vdiag[i] = real.(el.v[i, i])
        end
    elseif model.el_velocity_mode === :BerryConnection
        # For Berry connection method, we ignore the Berry connection contribution which is
        # zero for the diagonal part.
        @views velocity_diag = reshape(reinterpret(FT, el.vdiag[el.rng]), 3, el.nband)
        get_el_velocity_diag_berry_connection!(velocity_diag, el.nw, model.el_ham_R, xk, uk, fourier_mode)
    else
        throw(ArgumentError("model.el_velocity_mode must be :Direct or :BerryConnection, not $(model.el_velocity_mode)."))
    end
end

"""
    set_velocity(el::ElectronState, model, xk, fourier_mode="normal"; skip_rbar=false)
Compute electron band velocity.
- `skip_rbar`: If true, assume `el.rbar` is already calculated and skip `set_position!`.
"""
function set_velocity!(el::ElectronState{FT}, model, xk, fourier_mode="normal"; skip_rbar=false) where {FT}
    uk = get_u(el)
    @views velocity = reshape(reinterpret(Complex{FT}, el.v[el.rng, el.rng]), 3, el.nband, el.nband)
    if model.el_velocity_mode === :Direct
        get_el_velocity_direct!(velocity, el.nw, model.el_vel, xk, uk, fourier_mode)
    elseif model.el_velocity_mode === :BerryConnection
        # Need to set el.rbar first.
        skip_rbar || set_position!(el, model, xk, fourier_mode)
        @views rbar = el.rbar[el.rng, el.rng]
        get_el_velocity_berry_connection!(velocity, el.nw, model.el_ham_R, el.e, xk, uk, rbar, fourier_mode)
    else
        throw(ArgumentError("model.el_velocity_mode must be :Direct or :BerryConnection, not $(model.el_velocity_mode)."))
    end
end

"""
    set_position!(el::ElectronState, model, xk, fourier_mode="normal")
Compute electron position matrix elements.
"""
function set_position!(el::ElectronState{FT}, model, xk, fourier_mode="normal") where {FT}
    uk = get_u(el)
    @views rbar = reshape(reinterpret(Complex{FT}, el.rbar[el.rng, el.rng]), 3, el.nband, el.nband)
    get_el_velocity_direct!(rbar, el.nw, model.el_pos, xk, uk, fourier_mode)
end