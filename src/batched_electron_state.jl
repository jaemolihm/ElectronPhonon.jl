# Electron quantities over a k-point set in box storage, one array per requested quantity with k on
# the last axis.

export BatchedElectronState

"""
    BatchedElectronState{T, KT, ST, IT, Q}

The electron states of a k-point set as dense stacks, the struct-of-arrays counterpart of a
`Vector{ElectronState}`. Build one with [`compute_electron_states_batched`](@ref).

Each requested quantity is one array in the `NamedTuple` field `qty`, named as the `ElectronState`
field it holds and read as a property (`b.e`, `b.u`); the k point is the last axis of every array.
Reading a quantity that was not requested throws. Index legend: `a` Wannier function, `n`, `m`
local band, `d` Cartesian direction, `k` k point.

| quantity | shape | |
|---|---|---|
| `e` | `(n, k)` | band energy |
| `u` | `(a, n, k)` | eigenvector |
| `vdiag` | `(d, n, k)` | diagonal band velocity |
| `v` | `(d, m, n, k)` | velocity matrix |
| `rbar` | `(d, m, n, k)` | position matrix (Berry connection) |

**Box storage.** The energy window is applied once, by the builder. Local band `n ≤ nband[k]` is
physical band `iband_offset[k] + n`, with `iband_offset[k]` the first in-window band minus one,
never clamped. The band axes have the box width `nbox` (the largest `nband`, or `nw` when the window
is not known before the solve). Entries at `n > nband[k]` are undefined, including box columns past
physical band `nw`: reading them is a bug in the reader, and a reduction over the whole box selects
with `ifelse` on `n ≤ nband[k]` rather than multiplying by a 0/1 mask.

The arrays, `iband_offset` and `nband` live on the backend that built them. `kpts` stays on the
host; `sel` is the `FilteredBandStates` the states were built for, or `nothing` for a tuple window.
Both are `nothing` for a per-tile buffer (see [`alloc_tile`](@ref)), whose `n` is the tile width.
"""
struct BatchedElectronState{T, KT <: Union{Nothing, AbstractKpoints{T}},
        ST <: Union{Nothing, FilteredBandStates{T}}, IT <: AbstractVector{Int}, Q <: NamedTuple} <:
        AbstractBatchedState
    nw :: Int
    nbox :: Int        # width of every band axis
    n :: Int           # number of k points; the last-axis extent of every per-point array
    kpts :: KT         # host, or nothing for a tile
    sel :: ST          # host, or nothing
    iband_offset :: IT # (k,) physical band = iband_offset[k] + local band
    nband :: IT        # (k,) number of in-window bands
    qty :: Q

    function BatchedElectronState{T}(nw::Int, nbox::Int, n::Int, kpts::KT, sel::ST,
            iband_offset::IT, nband::IT, qty::Q) where {T, KT <: Union{Nothing, AbstractKpoints{T}},
            ST <: Union{Nothing, FilteredBandStates{T}}, IT <: AbstractVector{Int}, Q <: NamedTuple}
        kpts === nothing || kpts.n == n || throw(ArgumentError(
            "kpts holds $(kpts.n) points, but n = $n"))
        length(iband_offset) == length(nband) == n || throw(ArgumentError(
            "iband_offset and nband must have length n = $n"))
        for (name, x) in pairs(qty)
            dims = _electron_quantity_dims(name, nw, nbox, n)
            eltype(x) === _electron_quantity_eltype(name, T) && size(x) == dims ||
                throw(ArgumentError("quantity :$name must be a " *
                    "$(_electron_quantity_eltype(name, T)) array of size $dims, got " *
                    "$(typeof(x)) of size $(size(x))"))
        end
        _check_quantity_names(BatchedElectronState, keys(qty))
        new{T, KT, ST, IT, Q}(nw, nbox, n, kpts, sel, iband_offset, nband, qty)
    end
end

# The shape and element type of each electron quantity. A name with no branch here is not an
# electron quantity, so the constructor (and through it every builder) rejects it.
function _electron_quantity_dims(name::Symbol, nw, nbox, n)
    name === :e && return (nbox, n)
    name === :u && return (nw, nbox, n)
    name === :vdiag && return (3, nbox, n)
    (name === :v || name === :rbar) && return (3, nbox, nbox, n)
    throw(ArgumentError("unknown electron quantity :$name"))
end

_electron_quantity_eltype(name::Symbol, ::Type{T}) where {T} =
    name === :e || name === :vdiag ? T : Complex{T}

function Base.show(io::IO, b::BatchedElectronState{T}) where {T}
    print(io, "BatchedElectronState{$T}(nw = $(b.nw), nbox = $(b.nbox), n = $(b.n), " *
              "quantities = $(keys(b.qty)), $(typeof(b.nband)))")
end

"""
    unfold_rule(name::Symbol) -> Symbol

How the electron quantity `name` transforms under a symmetry operation that maps a k point to an
equivalent one: `:none` (copied), `:gauge` (rotated by the eigenvector representation of the
operation) or `:cartesian` (rotated as a Cartesian vector, `-Scart` with time reversal), as
`unfold_ElectronStates` applies them. Throws an `ArgumentError` for a quantity without a rule.
"""
function unfold_rule(name::Symbol)
    name === :e && return :none
    name === :u && return :gauge
    (name === :vdiag || name === :v) && return :cartesian
    throw(ArgumentError("electron quantity :$name has no symmetry unfolding rule"))
end

quantity_arrays(b::BatchedElectronState) = (values(b.qty)..., b.iband_offset, b.nband)

function alloc_tile(b::BatchedElectronState{T}, backend, width::Integer) where {T}
    qty = map(x -> alloc(backend, eltype(x), Base.front(size(x))..., width), b.qty)
    BatchedElectronState{T}(b.nw, b.nbox, Int(width), nothing, nothing,
        alloc(backend, Int, width), alloc(backend, Int, width), qty)
end

_check_box(tile::BatchedElectronState, b::BatchedElectronState) =
    tile.nw == b.nw && tile.nbox == b.nbox || throw(ArgumentError(
        "tile has (nw, nbox) = ($(tile.nw), $(tile.nbox)), the source ($(b.nw), $(b.nbox))"))
