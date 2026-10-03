# Electron quantities over a k-point set in box storage, one array per quantity with k on the last
# axis.

export BatchedElectronState

"""
    BatchedElectronState{T, KT, IT, ET, UT, VDT, VT, RT}

The electron states of a k-point set as dense stacks, the struct-of-arrays counterpart of a
`Vector{ElectronState}`. Build one with [`compute_electron_states_batched`](@ref).

One field per quantity, named as the `ElectronState` field it holds, `nothing` when not requested;
the k point is the last axis. Index legend: `a` Wannier function, `n`, `m` local band, `d` Cartesian
direction, `k` k point.

| field | shape | |
|---|---|---|
| `e` | `(n, k)` | band energy |
| `u` | `(a, n, k)` | eigenvector |
| `vdiag` | `(d, n, k)` | diagonal band velocity |
| `v` | `(d, m, n, k)` | velocity matrix |
| `rbar` | `(d, m, n, k)` | position matrix (Berry connection) |

**Box storage.** The energy window is applied once, by the builder. Local band `n ≤ nband[k]` is
physical band `iband_offset[k] + n`, with `iband_offset[k]` the first in-window band minus one,
never clamped. The band axes have width `nband_max` (the largest `nband`, or `nw` when the window is
not known before the solve). Entries at `n > nband[k]` are undefined, including box columns past
physical band `nw`: reading them is a bug in the reader, and a reduction over the whole box selects
with `ifelse` on `n ≤ nband[k]` rather than multiplying by a 0/1 mask.

The arrays, `iband_offset` and `nband` live on the backend that built them. `kpts` stays on the
host, and is `nothing` for a per-batch buffer, whose `nk` is the buffer width.
"""
struct BatchedElectronState{T, KT <: Union{Nothing, AbstractKpoints{T}}, IT <: AbstractVector{Int},
        ET <: Union{Nothing, AbstractMatrix{T}}, UT <: Union{Nothing, AbstractArray{Complex{T}, 3}},
        VDT <: Union{Nothing, AbstractArray{T, 3}}, VT <: Union{Nothing, AbstractArray{Complex{T}, 4}},
        RT <: Union{Nothing, AbstractArray{Complex{T}, 4}}}
    nw :: Int
    nband_max :: Int
    nk :: Int
    kpts :: KT
    iband_offset :: IT  # (k,) physical band = iband_offset[k] + local band
    nband :: IT         # (k,) number of in-window bands
    e :: ET
    u :: UT
    vdiag :: VDT
    v :: VT
    rbar :: RT

    function BatchedElectronState{T}(nw, nband_max, nk, kpts::KT, iband_offset::IT, nband::IT, e::ET,
            u::UT, vdiag::VDT, v::VT, rbar::RT) where {T, KT, IT, ET, UT, VDT, VT, RT}
        kpts === nothing || kpts.n == nk || throw(ArgumentError("kpts holds $(kpts.n) points, nk = $nk"))
        length(iband_offset) == length(nband) == nk || throw(ArgumentError("iband_offset and nband must have length nk = $nk"))
        e === nothing || size(e) == (nband_max, nk) || throw(ArgumentError("e must be (nband_max, nk)"))
        u === nothing || size(u) == (nw, nband_max, nk) || throw(ArgumentError("u must be (nw, nband_max, nk)"))
        vdiag === nothing || size(vdiag) == (3, nband_max, nk) || throw(ArgumentError("vdiag must be (3, nband_max, nk)"))
        v === nothing || size(v) == (3, nband_max, nband_max, nk) || throw(ArgumentError("v must be (3, nband_max, nband_max, nk)"))
        rbar === nothing || size(rbar) == (3, nband_max, nband_max, nk) || throw(ArgumentError("rbar must be (3, nband_max, nband_max, nk)"))
        new{T, KT, IT, ET, UT, VDT, VT, RT}(nw, nband_max, nk, kpts, iband_offset, nband, e, u, vdiag,
                                            v, rbar)
    end
end

# The empty container: uninitialised arrays on `backend` for the `quantities` requested, `nothing`
# for the others, and zero `iband_offset`/`nband`.
function BatchedElectronState(backend, nw, nband_max, nk, quantities; kpts = nothing, FT = Float64)
    unknown = setdiff(quantities, (:e, :u, :vdiag, :v, :rbar))
    isempty(unknown) || throw(ArgumentError("unknown electron quantities $unknown"))
    allunique(quantities) || throw(ArgumentError("quantities $quantities has duplicates"))
    alloc_if_required(name, T, dims...) = name ∈ quantities ? alloc(backend, T, dims...) : nothing
    BatchedElectronState{FT}(nw, nband_max, nk, kpts,
        alloc_zeros(backend, Int, nk),                                     # iband_offset
        alloc_zeros(backend, Int, nk),                                     # nband
        alloc_if_required(:e, FT, nband_max, nk),
        alloc_if_required(:u, Complex{FT}, nw, nband_max, nk),
        alloc_if_required(:vdiag, FT, 3, nband_max, nk),
        alloc_if_required(:v, Complex{FT}, 3, nband_max, nband_max, nk),
        alloc_if_required(:rbar, Complex{FT}, 3, nband_max, nband_max, nk))
end

function Base.show(io::IO, els::BatchedElectronState{T}) where {T}
    print(io, "BatchedElectronState{$T}(nw = $(els.nw), nband_max = $(els.nband_max), " *
              "nk = $(els.nk), $(typeof(els.nband)))")
end

"""
    copy_batched_electron_states!(els_dst, els_src, inds)

Copy the k points `inds` of `els_src` into the first `length(inds)` points of `els_dst`, field by field (a
`nothing` field is skipped), `iband_offset` and `nband` included. `inds` is a range or a host vector,
checked here, or an index array already on `els_src`'s device, used as it is: its caller has checked
it. A host index vector for a device `els_src` is uploaded once per call.
"""
function copy_batched_electron_states!(els_dst::BatchedElectronState, els_src::BatchedElectronState,
        inds)
    els_dst.nw == els_src.nw && els_dst.nband_max == els_src.nband_max &&
        length(inds) <= els_dst.nk ||
        throw(ArgumentError("cannot copy $(length(inds)) points of $els_src into $els_dst"))
    inds = _copy_indices_on_backend(els_src.nband, inds, els_src.nk)
    _copy_last_axis!(els_dst.iband_offset, els_src.iband_offset, inds)
    _copy_last_axis!(els_dst.nband, els_src.nband, inds)
    _copy_last_axis!(els_dst.e, els_src.e, inds)
    _copy_last_axis!(els_dst.u, els_src.u, inds)
    _copy_last_axis!(els_dst.vdiag, els_src.vdiag, inds)
    _copy_last_axis!(els_dst.v, els_src.v, inds)
    _copy_last_axis!(els_dst.rbar, els_src.rbar, inds)
    els_dst
end

"""
    view(els::BatchedElectronState, inds::AbstractUnitRange)

The points `inds` of `els` as a `BatchedElectronState` of views (no copy), with `kpts = nothing`.
"""
@views function Base.view(els::BatchedElectronState{T}, inds::AbstractUnitRange) where {T}
    # Colons rather than `selectdim`, so a device view stays a device array.
    view_points(x) = x === nothing ? nothing : x[ntuple(_ -> Colon(), ndims(x) - 1)..., inds]
    BatchedElectronState{T}(els.nw, els.nband_max, length(inds), nothing,
        els.iband_offset[inds],
        els.nband[inds],
        view_points(els.e),
        view_points(els.u),
        view_points(els.vdiag),
        view_points(els.v),
        view_points(els.rbar))
end

"""
    reshape_view_batched_electron_states(els, nband_max, nk) -> BatchedElectronState

A container of box width `nband_max ≤ els.nband_max` and `nk ≤ els.nk` points on the memory of
`els`, as `Base.reshape` is for an array: each quantity is the dense leading elements of `els`'s
array taken at the new dimensions (`reshape_buffer_view`), so the contents are reinterpreted, not moved.
When `nband_max` shrinks, point `k` of the result is not point `k` of `els`: write the contents
through the result before reading them. `iband_offset` and `nband` are views of the first `nk`
points of `els`'s, and `kpts = nothing`.
"""
function reshape_view_batched_electron_states(els::BatchedElectronState{T}, nband_max, nk) where {T}
    nband_max <= els.nband_max && nk <= els.nk || throw(ArgumentError(
        "a box of $nband_max bands and $nk points does not fit $els"))
    prefix(x, dims...) = x === nothing ? nothing : reshape_buffer_view(x, dims...)
    BatchedElectronState{T}(els.nw, nband_max, nk, nothing,
        view(els.iband_offset, 1:nk),
        view(els.nband, 1:nk),
        prefix(els.e, nband_max, nk),
        prefix(els.u, els.nw, nband_max, nk),
        prefix(els.vdiag, 3, nband_max, nk),
        prefix(els.v, 3, nband_max, nband_max, nk),
        prefix(els.rbar, 3, nband_max, nband_max, nk))
end
