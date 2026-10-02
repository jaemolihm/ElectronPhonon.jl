# Phonon quantities over a q-point set, one array per quantity with q on the last axis

using ChunkSplitters
using Base.Threads: nthreads, @threads

export BatchedPhononState

"""
    BatchedPhononState{T, KT, ET, UT, VT, DT, RT}

The phonons of a q-point set as dense stacks, the struct-of-arrays counterpart of a
`Vector{PhononState}`. Build one with [`compute_phonon_states_batched`](@ref).

One field per quantity, named as the `PhononState` field it holds, `nothing` when not requested.
The q point is the last axis, so `ph_states.x[..., iq]` is `PhononState.x` at
`ph_states.qpts.vectors[iq]`. Index legend: `ν` mode, `a` displacement (atom × Cartesian), `d`
Cartesian direction, `q` q point.

| field | shape | |
|---|---|---|
| `e` | `(ν, q)` | frequency ω |
| `u` | `(a, ν, q)` | mass-scaled eigenmodes |
| `vdiag` | `(d, ν, q)` | diagonal velocity |
| `eph_dipole_coeff` | `(ν, q)` | dipole e-ph coefficients |
| `eph_r_coeff` | `(ν, d, q)` | dipole e-ph coefficients of the 2D dipole |

The arrays live on the backend that built them. `qpts` stays on the host, and is `nothing` for a
per-batch buffer, whose `nq` is the buffer width.
"""
struct BatchedPhononState{T, KT <: Union{Nothing, AbstractKpoints{T}},
        ET <: Union{Nothing, AbstractMatrix{T}}, UT <: Union{Nothing, AbstractArray{Complex{T}, 3}},
        VT <: Union{Nothing, AbstractArray{T, 3}}, DT <: Union{Nothing, AbstractMatrix{Complex{T}}},
        RT <: Union{Nothing, AbstractArray{Complex{T}, 3}}}
    nmodes :: Int
    nq :: Int
    qpts :: KT
    e :: ET
    u :: UT
    vdiag :: VT
    eph_dipole_coeff :: DT
    eph_r_coeff :: RT

    function BatchedPhononState{T}(nmodes, nq, qpts::KT, e::ET, u::UT, vdiag::VT,
            eph_dipole_coeff::DT, eph_r_coeff::RT) where {T, KT, ET, UT, VT, DT, RT}
        qpts === nothing || qpts.n == nq || throw(ArgumentError("qpts holds $(qpts.n) points, nq = $nq"))
        e === nothing || size(e) == (nmodes, nq) || throw(ArgumentError("e must be (nmodes, nq)"))
        u === nothing || size(u) == (nmodes, nmodes, nq) || throw(ArgumentError("u must be (nmodes, nmodes, nq)"))
        vdiag === nothing || size(vdiag) == (3, nmodes, nq) || throw(ArgumentError("vdiag must be (3, nmodes, nq)"))
        eph_dipole_coeff === nothing || size(eph_dipole_coeff) == (nmodes, nq) || throw(ArgumentError("eph_dipole_coeff must be (nmodes, nq)"))
        eph_r_coeff === nothing || size(eph_r_coeff) == (nmodes, 3, nq) || throw(ArgumentError("eph_r_coeff must be (nmodes, 3, nq)"))
        new{T, KT, ET, UT, VT, DT, RT}(nmodes, nq, qpts, e, u, vdiag, eph_dipole_coeff, eph_r_coeff)
    end
end

# The empty container: zero-filled arrays on `backend` for the `quantities` requested, `nothing` for
# the others.
function BatchedPhononState(backend, nmodes, nq, quantities; qpts = nothing, FT = Float64)
    unknown = setdiff(quantities, (:e, :u, :vdiag, :eph_dipole_coeff, :eph_r_coeff))
    isempty(unknown) || throw(ArgumentError("unknown phonon quantities $unknown"))
    allunique(quantities) || throw(ArgumentError("quantities $quantities has duplicates"))
    alloc_if_required(name, T, dims...) = name ∈ quantities ? alloc_zeros(backend, T, dims...) : nothing
    BatchedPhononState{FT}(nmodes, nq, qpts,
        alloc_if_required(:e, FT, nmodes, nq),
        alloc_if_required(:u, Complex{FT}, nmodes, nmodes, nq),
        alloc_if_required(:vdiag, FT, 3, nmodes, nq),
        alloc_if_required(:eph_dipole_coeff, Complex{FT}, nmodes, nq),
        alloc_if_required(:eph_r_coeff, Complex{FT}, nmodes, 3, nq))
end

function Base.show(io::IO, ph_states::BatchedPhononState{T}) where {T}
    print(io, "BatchedPhononState{$T}(nmodes = $(ph_states.nmodes), nq = $(ph_states.nq))")
end

"""
    copy_batched_phonon_states!(dst, src, iqs)

Copy the q points `iqs` of `src` into the first `length(iqs)` points of `dst`, field by field (a
`nothing` field is skipped). `iqs` is a range or a host vector, checked here, or an index array
already on `src`'s device, used as it is: its caller has checked it. A host index vector for a
device `src` is uploaded once per call, so a hot loop keeps its own device index buffer instead.
"""
function copy_batched_phonon_states!(dst::BatchedPhononState, src::BatchedPhononState, iqs)
    dst.nmodes == src.nmodes && length(iqs) <= dst.nq || throw(ArgumentError(
        "cannot copy $(length(iqs)) q points of nmodes = $(src.nmodes) into $dst"))
    iqs = _copy_indices_on_backend(something(src.e, src.u, src.vdiag, src.eph_dipole_coeff, src.eph_r_coeff), iqs, src.nq)
    _copy_last_axis!(dst.e, src.e, iqs)
    _copy_last_axis!(dst.u, src.u, iqs)
    _copy_last_axis!(dst.vdiag, src.vdiag, iqs)
    _copy_last_axis!(dst.eph_dipole_coeff, src.eph_dipole_coeff, iqs)
    _copy_last_axis!(dst.eph_r_coeff, src.eph_r_coeff, iqs)
    dst
end

"""
    view_batched_phonon_states(ph_states, inds::AbstractUnitRange)

The points `inds` of `ph_states` as a `BatchedPhononState` of views (no copy), with `qpts = nothing`.
"""
@views function view_batched_phonon_states(ph_states::BatchedPhononState{T}, inds::AbstractUnitRange) where {T}
    # Colons rather than `selectdim`, so a device view stays a device array.
    view_points(x) = x === nothing ? nothing : x[ntuple(_ -> Colon(), ndims(x) - 1)..., inds]
    BatchedPhononState{T}(ph_states.nmodes, length(inds), nothing,
        view_points(ph_states.e),
        view_points(ph_states.u),
        view_points(ph_states.vdiag),
        view_points(ph_states.eph_dipole_coeff),
        view_points(ph_states.eph_r_coeff))
end

"""
    Vector{PhononState{T}}(ph_states::BatchedPhononState{T})

One `PhononState` per q point of `ph_states`, on the host: `xq` from `ph_states.qpts` and every
quantity `ph_states` holds; a quantity it does not hold is left at `PhononState`'s zero.
"""
function Base.Vector{PhononState{T}}(ph_states::BatchedPhononState{T}) where {T}
    ph_states.qpts === nothing && throw(ArgumentError("a per-batch buffer has no q points"))
    host(x) = x === nothing ? nothing : Array(x)
    e, u, vdiag, dip, rco = host(ph_states.e), host(ph_states.u), host(ph_states.vdiag), host(ph_states.eph_dipole_coeff),
        host(ph_states.eph_r_coeff)
    states = [PhononState(ph_states.nmodes, T) for _ in 1:ph_states.nq]
    @threads for iqs in chunks(1:ph_states.nq; n = nthreads())
        for iq in iqs
            ph = states[iq]
            ph.xq = ph_states.qpts.vectors[iq]
            e === nothing || (@views ph.e .= e[:, iq])
            u === nothing || (@views ph.u .= u[:, :, iq])
            vdiag === nothing || for i in 1:ph_states.nmodes
                ph.vdiag[i] = Vec3{T}(vdiag[1, i, iq], vdiag[2, i, iq], vdiag[3, i, iq])
            end
            dip === nothing || (@views ph.eph_dipole_coeff .= dip[:, iq])
            rco === nothing || (@views ph.eph_r_coeff .= rco[:, :, iq])
        end
    end
    states
end
