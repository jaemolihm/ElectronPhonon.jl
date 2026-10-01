# Phonon quantities over a q-point set, one array per requested quantity with q on the last axis

using ChunkSplitters
using Base.Threads: nthreads, @threads

export BatchedPhononState

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


quantity_arrays(b::BatchedPhononState) = values(b.qty)

function alloc_tile(b::BatchedPhononState{T}, backend, width::Integer) where {T}
    qty = map(x -> alloc(backend, eltype(x), Base.front(size(x))..., width), b.qty)
    BatchedPhononState{T}(b.nmodes, Int(width), nothing, qty)
end

_check_box(tile::BatchedPhononState, b::BatchedPhononState) = tile.nmodes == b.nmodes ||
    throw(ArgumentError("tile has nmodes = $(tile.nmodes), the source $(b.nmodes)"))

"""
    Vector{PhononState{T}}(b::BatchedPhononState{T})

One `PhononState` per q point of `b`, on the host: `xq` from `b.qpts` and every quantity `b` holds;
a quantity it does not hold is left at `PhononState`'s zero.
"""
function Base.Vector{PhononState{T}}(b::BatchedPhononState{T}) where {T}
    b.qpts === nothing && throw(ArgumentError("a tile has no q points"))
    qty = map(Array, b.qty)
    states = [PhononState(b.nmodes, T) for _ in 1:b.n]
    @threads for iqs in chunks(1:b.n; n = nthreads())
        for iq in iqs
            ph = states[iq]
            ph.xq = b.qpts.vectors[iq]
            haskey(qty, :e) && (@views ph.e .= qty.e[:, iq])
            haskey(qty, :u) && (@views ph.u .= qty.u[:, :, iq])
            haskey(qty, :vdiag) && for i in 1:b.nmodes
                ph.vdiag[i] = Vec3{T}(qty.vdiag[1, i, iq], qty.vdiag[2, i, iq], qty.vdiag[3, i, iq])
            end
            haskey(qty, :eph_dipole_coeff) &&
                (@views ph.eph_dipole_coeff .= qty.eph_dipole_coeff[:, iq])
            haskey(qty, :eph_r_coeff) && (@views ph.eph_r_coeff .= qty.eph_r_coeff[:, :, iq])
        end
    end
    states
end
