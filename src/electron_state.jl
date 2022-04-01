
# Electron eigenvalue and eigenvector at a single k point

using OffsetArrays
using OffsetArrays: no_offset_view

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

"""
TODO: Implement the following.
For `e_full` and `u_full`, values for all bands are stored.
When accessing fields without the `_full` postfix (`e`, `u`, `v`, `vdiag`, `rbar`, `occupation`),
an OffsetArray is returned. These OffsetArrays are indiced by the physical band indices, which
are listed in `rng_full`.
To get an array with 1-based indexing, use `OffsetArrays.no_offset_view`.
"""
Base.@kwdef mutable struct ElectronState{T <: Real}
    nw::Int # Number of Wannier functions
    e_full::Vector{T} # Eigenvalues at all bands
    u_full::Matrix{Complex{T}} # Electron eigenvectors

    # Variables related to the energy window.
    # Indexing a full array: arr_full[rng_full]
    # Indexing a filtered array: arr[rng]
    # rng_full = rng .+ nband_ignore
    # rng = 1:nband
    nband_bound::Int # Upper bound of possible nband
    nband_ignore::Int # Number of low-lying ignored bands
    nband::Int # Number of bands inside the energy window
    rng_full::UnitRange{Int} # Index of bands inside the energy window for the full (physical) index
    rng::UnitRange{Int} # Index of bands inside the energy window for the offset index

    # These arrays are defined only for bands inside the window
    vdiag::Vector{Vec3{T}} # Diagonal components of band velocity in Cartesian coordinates.
    v::Matrix{Vec3{Complex{T}}} # Velocity matrix in Cartesian coordinates.
    rbar::Matrix{Vec3{Complex{T}}} # Position matrix in Cartesian coordinates (without the Hamiltonian derivative term).
    occupation::Vector{T} # Electron occupation number
end

function ElectronState{T}(nw, nband_bound=nw) where {T}
    ElectronState{T}(
        nw=nw,
        e_full=zeros(T, nw),
        u_full=zeros(Complex{T}, nw, nw),
        nband_bound=nband_bound,
        nband_ignore=0,
        nband=0,
        rng_full=1:0,
        rng=1:0,
        vdiag=fill(zeros(Vec3{T}), (nband_bound,)),
        v=fill(zeros(Vec3{Complex{T}}), (nband_bound, nband_bound)),
        rbar=fill(zeros(Vec3{Complex{T}}), (nband_bound, nband_bound)),
        occupation=zeros(T, nband_bound),
    )
end

ElectronState(nw, ::Type{FT}=Float64; nband_bound=nw) where FT = ElectronState{FT}(nw, nband_bound)

function Base.getproperty(el::ElectronState, name::Symbol)
    if name === :u
        # Return eigenvectors for bands inside the window.
        view(getfield(el, :u_full), :, getfield(el, :rng_full))
    elseif name === :e
        # Return eigenvalues for bands inside the window.
        view(getfield(el, :e_full), getfield(el, :rng_full))
    elseif name === :vdiag
        view(getfield(el, :vdiag), getfield(el, :rng))
    elseif name === :v
        view(getfield(el, name), getfield(el, :rng), getfield(el, :rng))
    elseif name === :rbar
        OffsetArray(view(getfield(el, name), getfield(el, :rng), getfield(el, :rng)), getfield(el, :rng_full), getfield(el, :rng_full))
    else
        getfield(el, name)
    end
end

function Base.copyto!(dest::ElectronState, src::ElectronState)
    if dest.nband_bound < src.nband_bound
        throw(ArgumentError("src.nband_bound ($(src.nband_bound)) cannot be greater " *
            "than dest.nband_bound ($(dest.nband_bound))"))
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
    dest.vdiag .= src.vdiag
    dest.v .= src.v
    dest.rbar .= src.rbar
    dest
end

"""
    Base.resize!(el::ElectronState{FT}, nband_bound=el.nband) where FT
Resize `el` so that the arrays have size `nband_bound`. Data except `e_full` and `u_full` are deleted.
"""
function Base.resize!(el::ElectronState{FT}, nband_bound=el.nband) where FT
    el.nband_bound = nband_bound
    el.vdiag = zeros(Vec3{FT}, nband_bound)
    el.v = zeros(Vec3{Complex{FT}}, nband_bound, nband_bound)
    el.rbar = zeros(Vec3{Complex{FT}}, nband_bound, nband_bound)
    el.occupation = zeros(FT, nband_bound)
end

"""
    set_window!(el::ElectronState, window=(-Inf, Inf))
Find out the bands inside the window and set el.nband, el.rng and el.rng_full.
"""
function set_window!(el::ElectronState, window=(-Inf, Inf))
    ibands = EPW.inside_window(el.e_full, window...)
    # If no bands are selected, return true.
    if isempty(ibands)
        el.nband_ignore = 0
        el.nband = 0
        el.rng = 1:0
        el.rng_full = 1:0
    else
        el.rng_full = ibands[1]:ibands[end]
        el.nband_ignore = ibands[1] - 1
        el.nband = length(el.rng_full)
        el.rng = 1:el.nband
        if el.nband > el.nband_bound
            # If el.nband is greater than nband_bound, resize the arrays.
            resize!(el)
        end
    end
    return el
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

    # Reset window to a dummy value
    el.nband = 0
    el.rng_full = 1:0
    el.rng = 1:0
end

"""
    set_eigen_valueonly!(el::ElectronState, model, xk, fourier_mode="normal")
Compute electron eigenenergy and save them in el.
"""
function set_eigen_valueonly!(el::ElectronState, model, xk, fourier_mode="normal")
    get_el_eigen_valueonly!(el.e_full, el.nw, model.el_ham, xk, fourier_mode)

    # Reset window to a dummy value
    el.nband = 0
    el.rng_full = 1:0
    el.rng = 1:0
end

"""
    set_velocity_diag!(el::ElectronState, model, xk, fourier_mode="normal")
Compute electron band velocity, only the band-diagonal part.
"""
function set_velocity_diag!(el::ElectronState{FT}, model, xk, fourier_mode="normal") where {FT}
    if model.el_velocity_mode === :Direct
        # For direct Wannier interpolation, there is no faster way to calculate only the diagonal part.
        # So we just calculate the full velocity matrix and set take the diagonal part.
        velocity = reshape(reinterpret(Complex{FT}, el.v), 3, el.nband, el.nband)
        get_el_velocity_direct!(velocity, el.nw, model.el_vel, xk, el.u, fourier_mode)
        for i = 1:el.nband
            el.vdiag[i] = real.(el.v[i, i])
        end
    elseif model.el_velocity_mode === :BerryConnection
        # For Berry connection method, we ignore the Berry connection contribution which is
        # zero for the diagonal part.
        velocity_diag = reshape(reinterpret(FT, el.vdiag), 3, el.nband)
        get_el_velocity_diag_berry_connection!(velocity_diag, el.nw, model.el_ham_R, xk, el.u, fourier_mode)
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
    velocity = reshape(reinterpret(Complex{FT}, el.v), 3, el.nband, el.nband)
    if model.el_velocity_mode === :Direct
        get_el_velocity_direct!(velocity, el.nw, model.el_vel, xk, el.u, fourier_mode)
    elseif model.el_velocity_mode === :BerryConnection
        # Need to set el.rbar first.
        skip_rbar || set_position!(el, model, xk, fourier_mode)
        get_el_velocity_berry_connection!(velocity, el.nw, model.el_ham_R, el.e, xk, el.u,
            no_offset_view(el.rbar), fourier_mode)
    else
        throw(ArgumentError("model.el_velocity_mode must be :Direct or :BerryConnection, not $(model.el_velocity_mode)."))
    end
end

"""
    set_position!(el::ElectronState, model, xk, fourier_mode="normal")
Compute electron position matrix elements.
"""
function set_position!(el::ElectronState{FT}, model, xk, fourier_mode="normal") where {FT}
    rbar = reshape(reinterpret(Complex{FT}, no_offset_view(el.rbar)), 3, el.nband, el.nband)
    get_el_velocity_direct!(rbar, el.nw, model.el_pos, xk, el.u, fourier_mode)
end