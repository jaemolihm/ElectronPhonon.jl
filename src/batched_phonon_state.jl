# Phonon quantities over a q-point set, one array per requested quantity with q on the last axis

export BatchedPhononState

# The batched state containers: `BatchedPhononState` and `BatchedElectronState`.
abstract type AbstractBatchedState end

"""
    BatchedPhononState{T, KT, Q}

The phonons of a q-point set as dense stacks, the struct-of-arrays counterpart of a
`Vector{PhononState}`. Build one with [`compute_phonon_states_batched`](@ref).

Each requested quantity is one array in the `NamedTuple` field `qty`, named as the `PhononState`
field it holds and read as a property (`b.e`, `b.u`). The q point is the last axis of every array,
so `b.x[..., iq]` is `PhononState.x` at `b.qpts.vectors[iq]`. Reading a quantity that was not
requested throws. Index legend: `ν` mode, `a` displacement (atom × Cartesian), `d` Cartesian
direction, `q` q point.

| quantity | shape | |
|---|---|---|
| `e` | `(ν, q)` | frequency ω |
| `u` | `(a, ν, q)` | mass-scaled eigenmodes |
| `vdiag` | `(d, ν, q)` | diagonal velocity |
| `eph_dipole_coeff` | `(ν, q)` | dipole e-ph coefficients |
| `eph_r_coeff` | `(ν, d, q)` | dipole e-ph coefficients of the 2D dipole |

The arrays live on the backend that built them. `qpts` stays on the host, and is `nothing` for a
per-tile buffer (see [`alloc_tile`](@ref)), whose `n` is the tile width.
"""
struct BatchedPhononState{T, KT <: Union{Nothing, AbstractKpoints{T}}, Q <: NamedTuple} <:
        AbstractBatchedState
    nmodes :: Int
    n :: Int     # number of q points; the last-axis extent of every array in `qty`
    qpts :: KT   # host, or nothing for a tile
    qty :: Q

    function BatchedPhononState{T}(nmodes::Int, n::Int, qpts::KT, qty::Q) where
            {T, KT <: Union{Nothing, AbstractKpoints{T}}, Q <: NamedTuple}
        qpts === nothing || qpts.n == n || throw(ArgumentError(
            "qpts holds $(qpts.n) points, but n = $n"))
        for (name, x) in pairs(qty)
            dims = _phonon_quantity_dims(name, nmodes, n)
            eltype(x) === _phonon_quantity_eltype(name, T) && size(x) == dims ||
                throw(ArgumentError("quantity :$name must be a " *
                    "$(_phonon_quantity_eltype(name, T)) array of size $dims, got " *
                    "$(typeof(x)) of size $(size(x))"))
        end
        _check_quantity_names(BatchedPhononState, keys(qty))
        new{T, KT, Q}(nmodes, n, qpts, qty)
    end
end

# The shape and element type of each phonon quantity. A name with no branch here is not a phonon
# quantity, so the constructor (and through it every builder) rejects it.
function _phonon_quantity_dims(name::Symbol, nmodes, n)
    name === :e && return (nmodes, n)
    name === :u && return (nmodes, nmodes, n)
    name === :vdiag && return (3, nmodes, n)
    name === :eph_dipole_coeff && return (nmodes, n)
    name === :eph_r_coeff && return (nmodes, 3, n)
    throw(ArgumentError("unknown phonon quantity :$name"))
end

_phonon_quantity_eltype(name::Symbol, ::Type{T}) where {T} =
    name === :e || name === :vdiag ? T : Complex{T}

function Base.show(io::IO, b::BatchedPhononState{T}) where {T}
    print(io, "BatchedPhononState{$T}(nmodes = $(b.nmodes), n = $(b.n), " *
              "quantities = $(keys(b.qty)))")
end


# ---- shared by BatchedPhononState and BatchedElectronState ---------------------------------------

# A quantity name must not shadow a plain field, or `getproperty` could not tell them apart.
function _check_quantity_names(::Type{S}, names) where {S}
    for name in names
        name ∈ fieldnames(S) && throw(ArgumentError(
            "quantity name :$name is a field of $(nameof(S))"))
    end
end

# A plain field by `getfield`, any other name looked up in `qty`.
function Base.getproperty(b::AbstractBatchedState, name::Symbol)
    hasfield(typeof(b), name) && return getfield(b, name)
    qty = getfield(b, :qty)
    haskey(qty, name) || throw(ArgumentError("quantity :$name was not requested from this " *
        "$(nameof(typeof(b))); it holds $(keys(qty))"))
    getfield(qty, name)
end

Base.propertynames(b::AbstractBatchedState) =
    (fieldnames(typeof(b))..., keys(getfield(b, :qty))...)

"""
    quantity_arrays(b) -> Tuple

Every array of `b` with one entry per point on its last axis: the quantities, and for electrons
`iband_offset` and `nband`. Allocation of a tile, the gather `stage!` and byte counts map over it.
"""
quantity_arrays(b::BatchedPhononState) = values(b.qty)

"""
    alloc_tile(b, backend, width) -> typeof(b)-like

An uninitialised buffer of `width` points with the quantities, box width and element types of `b`,
on `backend`, and no point set. Filled by [`stage!`](@ref) or by a builder.
"""
function alloc_tile(b::BatchedPhononState{T}, backend, width::Integer) where {T}
    qty = map(x -> alloc(backend, eltype(x), Base.front(size(x))..., width), b.qty)
    BatchedPhononState{T}(b.nmodes, Int(width), nothing, qty)
end

"""
    stage!(tile, b, inds)

Copy the points `inds` of `b` into the first `length(inds)` points of `tile`, for every array of
[`quantity_arrays`](@ref), so a tile carries the window metadata with the states. Same-backend
arrays are gathered directly; a host `b` into a device `tile` is gathered on the host and uploaded
in one copy per array. Entries past `length(inds)` are left as they were.
"""
function stage!(tile::AbstractBatchedState, b::AbstractBatchedState, inds)
    keys(tile.qty) == keys(b.qty) || throw(ArgumentError(
        "tile holds $(keys(tile.qty)), the source $(keys(b.qty))"))
    _check_box(tile, b)
    length(inds) <= tile.n || throw(ArgumentError(
        "$(length(inds)) points do not fit a tile of width $(tile.n)"))
    foreach((dst, src) -> _gather_last!(dst, src, inds), quantity_arrays(tile), quantity_arrays(b))
    tile
end
_check_box(tile::BatchedPhononState, b::BatchedPhononState) = tile.nmodes == b.nmodes ||
    throw(ArgumentError("tile has nmodes = $(tile.nmodes), the source $(b.nmodes)"))
_check_box(tile, b) = throw(ArgumentError(
    "cannot stage a $(nameof(typeof(b))) into a $(nameof(typeof(tile)))"))

# Gather `src[..., inds]` into `dst[..., 1:length(inds)]`. A host source and a device destination
# are not one broadcast: gather on the host, then one contiguous upload.
function _gather_last!(dst::AbstractArray{T, N}, src::AbstractArray{T, N}, inds) where {T, N}
    d = selectdim(dst, N, 1:length(inds))
    if on_backend(CPUBackend(), src) && !on_backend(CPUBackend(), dst)
        copyto!(d, src[ntuple(_ -> Colon(), N - 1)..., inds])
    else
        d .= selectdim(src, N, inds)
    end
    dst
end
