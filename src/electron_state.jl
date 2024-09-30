
# Electron eigenvalue and eigenvector at a single k point

using OffsetArrays
using OffsetArrays: no_offset_view

export ElectronState
export copyto!
export set_window!
export set_occupation!
export set_eigen!
export set_eigen_valueonly!
export set_velocity_diag!
export set_position!
export compute_symmetry_representation!
export get_occupation

# TODO: Remove nband_bound. nband >= length(rng) can hold. Create a function `trim(el::ElectronState)`
#       (or even trim!) that reduces nband to length(rng).

"""
For `e_full` and `u_full`, values for all bands are stored.
When accessing fields without the `_full` postfix (`e`, `u`, `v`, `vdiag`, `rbar`, `occupation`),
an OffsetArray is returned. These OffsetArrays are indiced by the physical band indices, which
are listed in `rng`.
To get an array with 1-based indexing, use `OffsetArrays.no_offset_view`.
"""
Base.@kwdef mutable struct ElectronState{T <: Real}
    nw::Int # Number of Wannier functions
    e_full::Vector{T} # Eigenvalues at all bands
    u_full::Matrix{Complex{T}} # Electron eigenvectors

    # Variables related to the energy window.
    nband_bound::Int # Upper bound of possible nband
    nband::Int # Number of bands inside the energy window
    rng::UnitRange{Int} # Physical band indices of bands inside the energy window

    # These arrays are defined only for bands inside the window
    vdiag::Vector{Vec3{T}} # Diagonal components of band velocity in Cartesian coordinates.
    v::Matrix{Vec3{Complex{T}}} # Velocity matrix in Cartesian coordinates.
    rbar::Matrix{Vec3{Complex{T}}} # Position matrix in Cartesian coordinates (without the Hamiltonian derivative term).
    occupation::Vector{T} # Electron occupation number
end

function ElectronState{T}(nw, nband_bound=0) where {T}
    ElectronState{T}(
        nw=nw,
        e_full=zeros(T, nw),
        u_full=zeros(Complex{T}, nw, nw),
        nband_bound=nband_bound,
        nband=0,
        rng=1:0,
        vdiag=fill(zeros(Vec3{T}), (nband_bound,)),
        v=fill(zeros(Vec3{Complex{T}}), (nband_bound, nband_bound)),
        rbar=fill(zeros(Vec3{Complex{T}}), (nband_bound, nband_bound)),
        occupation=zeros(T, nband_bound),
    )
end

ElectronState(nw, nband_bound=0) = ElectronState{Float64}(nw, nband_bound)

function Base.getproperty(el::ElectronState, name::Symbol)
    if name === :u
        OffsetArray(view(getfield(el, :u_full), :, getfield(el, :rng)), :, getfield(el, :rng))
    elseif name === :e
        OffsetArray(view(getfield(el, :e_full), getfield(el, :rng)), getfield(el, :rng))
    elseif name === :vdiag || name === :occupation
        OffsetArray(view(getfield(el, name), 1:getfield(el, :nband)), getfield(el, :rng))
    elseif name === :v || name === :rbar
        OffsetArray(view(getfield(el, name), 1:getfield(el, :nband), 1:getfield(el, :nband)),
                    getfield(el, :rng), getfield(el, :rng))
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
        throw(ArgumentError("src.nw ($(src.nw)) must be equal to dest.nw ($(dest.nw))"))
    end
    dest.nband = src.nband
    dest.e_full .= src.e_full
    dest.u_full .= src.u_full
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
Find out the bands inside the window and set el.nband and el.rng.
"""
function set_window!(el::ElectronState, window=(-Inf, Inf))
    ibands = inside_window(el.e_full, window...)
    if isempty(ibands)
        el.nband = 0
        el.rng = 1:0
    else
        el.rng = ibands[1]:ibands[end]
        el.nband = length(el.rng)
        if el.nband > el.nband_bound
            # If el.nband is greater than nband_bound, resize the arrays.
            resize!(el)
        end
    end
    return el
end

function set_occupation!(el::ElectronState, μ, T; occ_type = :FermiDirac)
    for i in el.rng
        el.occupation[i] = occ_fermion(el.e[i] - μ, T; occ_type)
    end
    el.occupation
end

function get_occupation(el::ElectronState, μ, T; occ_type = :FermiDirac)
    occ_fermion.(el.e .- μ, T; occ_type)
end


# Define wrappers of wannier_to_bloch functions

"""
    set_eigen!(el::ElectronState, ham, xk)
Compute electron eigenenergy and eigenvector and save them in el.
"""
function set_eigen!(el::ElectronState, ham, xk)
    get_el_eigen!(el.e_full, el.u_full, el.nw, ham, xk)

    # Reset window to a dummy value
    el.nband = 0
    el.rng = 1:0
end

"""
    set_eigen_valueonly!(el::ElectronState, ham, xk)
Compute electron eigenenergy and save them in el.
"""
function set_eigen_valueonly!(el::ElectronState, ham, xk)
    get_el_eigen_valueonly!(el.e_full, el.nw, ham, xk)

    # Reset window to a dummy value
    el.nband = 0
    el.rng = 1:0
    el
end

"""
    set_velocity_diag!(el::ElectronState, model, xk, mode)
Compute electron band velocity, only the band-diagonal part.
- `mode`: `:Direct` or `:BerryConnection`.
If `mode == :Direct`, `vel` interpolates the velocity operator.
If `mode == :BerryConnection`, `vel` interpolates the H(R) * R operator.
"""
function set_velocity_diag!(el::ElectronState{FT}, vel, xk, mode) where {FT}
    if mode === :Direct
        # For direct Wannier interpolation, there is no faster way to calculate only the diagonal part.
        # So we just calculate the full velocity matrix and set take the diagonal part.
        velocity = reshape(reinterpret(Complex{FT}, no_offset_view(el.v)), 3, el.nband, el.nband)
        get_el_velocity_direct!(velocity, el.nw, vel, xk, no_offset_view(el.u))
        for i in el.rng
            el.vdiag[i] = real.(el.v[i, i])
        end
    elseif mode === :BerryConnection
        # For Berry connection method, we ignore the Berry connection contribution which is
        # zero for the diagonal part.
        velocity_diag = reshape(reinterpret(FT, no_offset_view(el.vdiag)), 3, el.nband)
        get_el_velocity_diag_berry_connection!(velocity_diag, el.nw, vel, xk, no_offset_view(el.u))
    else
        throw(ArgumentError("mode must be :Direct or :BerryConnection, not $mode."))
    end
end

"""
    set_velocity(el::ElectronState, vel, xk, mode; skip_rbar=false)
Compute electron band velocity.
If `mode == :Direct`, `el.rbar` must be already set by calling `set_position!`.

- `mode`: `:Direct` or `:BerryConnection`.
If `mode == :Direct`, `vel` interpolates the velocity operator.
If `mode == :BerryConnection`, `vel` interpolates the H(R) * R operator.
"""
function set_velocity!(el::ElectronState{FT}, vel, xk, mode) where {FT}
    velocity = reshape(reinterpret(Complex{FT}, no_offset_view(el.v)), 3, el.nband, el.nband)
    if mode === :Direct
        get_el_velocity_direct!(velocity, el.nw, vel, xk, no_offset_view(el.u))
    elseif mode === :BerryConnection
        get_el_velocity_berry_connection!(velocity, el.nw, vel, no_offset_view(el.e),
            xk, no_offset_view(el.u), no_offset_view(el.rbar))
    else
        throw(ArgumentError("mode must be :Direct or :BerryConnection, not $mode."))
    end
end

"""
    set_position!(el::ElectronState, pos, xk)
Compute electron position matrix elements.
"""
function set_position!(el::ElectronState{FT}, pos, xk) where {FT}
    rbar = reshape(reinterpret(Complex{FT}, no_offset_view(el.rbar)), 3, el.nband, el.nband)
    get_el_velocity_direct!(rbar, el.nw, pos, xk, no_offset_view(el.u))
end

"""
    compute_symmetry_representation!(sym_H, el_k::ElectronState{FT}, el_sk::ElectronState{FT},
    xk, el_sym_op, is_tr) where FT

Compute the unitary matrix that represents the symmetry operation in the basis of electron
eigenstates.

# Time reversal symmetry
For symmetry operations with time reversal, the computed unitary matrix is the unitary part
``U`` of the anti-unitary representation ``S = UK``, where ``K`` denotes complex conjugation.
Correspondingly, different formula must be used for the basis transformation and matrix
element transfomation. Especially, when applying the symmetry operator to a matrix, one needs
to take the complex conjugate of the matrix.

For example, the velocity matrix transform as follows:
```jldoctest
if is_tr
    v_rotated = - Ref(Scart) .* (sym_H * conj.(el.v) * sym_H')
else
    v_rotated = Ref(Scart) .* (sym_H * els[ik].v * sym_H')
end
```
"""
function compute_symmetry_representation!(sym_H, el_k::ElectronState{FT}, el_sk::ElectronState{FT},
    xk, el_sym_op, is_tr) where FT
    uk = no_offset_view(el_k.u)
    usk = no_offset_view(el_sk.u)
    get_symmetry_representation_eigen!(sym_H, el_sym_op, xk, uk, usk, is_tr)
end
