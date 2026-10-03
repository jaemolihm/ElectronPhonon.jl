# Phonon quantities over a q-point set, one array per quantity with q on the last axis

export BatchedPhononState

"""
    BatchedPhononState{T, KT, ET, UT, VT, DT, RT}

The phonons of a q-point set as dense stacks, the struct-of-arrays counterpart of a
`Vector{PhononState}`. Build one with [`compute_phonon_states_batched`](@ref).

One field per quantity, named as the `PhononState` field it holds, `nothing` when not requested.
The q point is the last axis, so `phs.x[..., iq]` is `PhononState.x` at
`phs.qpts.vectors[iq]`. Index legend: `ν` mode, `a` displacement (atom × Cartesian), `d`
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

function Base.show(io::IO, phs::BatchedPhononState{T}) where {T}
    print(io, "BatchedPhononState{$T}(nmodes = $(phs.nmodes), nq = $(phs.nq))")
end

"""
    copy_batched_phonon_states!(phs_dst, phs_src, iqs)

Copy the q points `iqs` of `phs_src` into the first `length(iqs)` points of `phs_dst`, field by field (a
`nothing` field is skipped). `iqs` is a range or a host vector, checked here, or an index array
already on `phs_src`'s device, used as it is: its caller has checked it. A host index vector for a
device `phs_src` is uploaded once per call, so a hot loop keeps its own device index buffer instead.
"""
function copy_batched_phonon_states!(phs_dst::BatchedPhononState, phs_src::BatchedPhononState, iqs)
    phs_dst.nmodes == phs_src.nmodes && length(iqs) <= phs_dst.nq || throw(ArgumentError(
        "cannot copy $(length(iqs)) q points of nmodes = $(phs_src.nmodes) into $phs_dst"))
    iqs = _copy_indices_on_backend(something(phs_src.e, phs_src.u, phs_src.vdiag,
        phs_src.eph_dipole_coeff, phs_src.eph_r_coeff), iqs, phs_src.nq)
    _copy_last_axis!(phs_dst.e, phs_src.e, iqs)
    _copy_last_axis!(phs_dst.u, phs_src.u, iqs)
    _copy_last_axis!(phs_dst.vdiag, phs_src.vdiag, iqs)
    _copy_last_axis!(phs_dst.eph_dipole_coeff, phs_src.eph_dipole_coeff, iqs)
    _copy_last_axis!(phs_dst.eph_r_coeff, phs_src.eph_r_coeff, iqs)
    phs_dst
end

"""
    view(phs::BatchedPhononState, inds::AbstractUnitRange)

The points `inds` of `phs` as a `BatchedPhononState` of views (no copy), with `qpts = nothing`.
"""
@views function Base.view(phs::BatchedPhononState{T}, inds::AbstractUnitRange) where {T}
    # Colons rather than `selectdim`, so a device view stays a device array.
    view_points(x) = x === nothing ? nothing : x[ntuple(_ -> Colon(), ndims(x) - 1)..., inds]
    BatchedPhononState{T}(phs.nmodes, length(inds), nothing,
        view_points(phs.e),
        view_points(phs.u),
        view_points(phs.vdiag),
        view_points(phs.eph_dipole_coeff),
        view_points(phs.eph_r_coeff))
end
