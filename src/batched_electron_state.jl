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
        length(iband_offset) == length(nband) == nk ||
            throw(ArgumentError("iband_offset and nband must have length nk = $nk"))
        e === nothing || size(e) == (nband_max, nk) || throw(ArgumentError("e must be (nband_max, nk)"))
        u === nothing || size(u) == (nw, nband_max, nk) ||
            throw(ArgumentError("u must be (nw, nband_max, nk)"))
        vdiag === nothing || size(vdiag) == (3, nband_max, nk) ||
            throw(ArgumentError("vdiag must be (3, nband_max, nk)"))
        v === nothing || size(v) == (3, nband_max, nband_max, nk) ||
            throw(ArgumentError("v must be (3, nband_max, nband_max, nk)"))
        rbar === nothing || size(rbar) == (3, nband_max, nband_max, nk) ||
            throw(ArgumentError("rbar must be (3, nband_max, nband_max, nk)"))
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
    a(name, T, dims...) = name ∈ quantities ? alloc(backend, T, dims...) : nothing
    BatchedElectronState{FT}(nw, nband_max, nk, kpts, alloc_zeros(backend, Int, nk),
        alloc_zeros(backend, Int, nk), a(:e, FT, nband_max, nk), a(:u, Complex{FT}, nw, nband_max, nk),
        a(:vdiag, FT, 3, nband_max, nk), a(:v, Complex{FT}, 3, nband_max, nband_max, nk),
        a(:rbar, Complex{FT}, 3, nband_max, nband_max, nk))
end

function Base.show(io::IO, b::BatchedElectronState{T}) where {T}
    print(io, "BatchedElectronState{$T}(nw = $(b.nw), nband_max = $(b.nband_max), nk = $(b.nk), " *
              "$(typeof(b.nband)))")
end

"""
    gather_batched_electron_states!(dst, src, inds)

Copy the k points `inds` of `src` into the first `length(inds)` points of `dst`, field by field (a
`nothing` field is skipped), `iband_offset` and `nband` included. `inds` is a range or a host vector,
checked here, or an index array already on `src`'s device, used as it is: its caller has checked
it. A host index vector for a device `src` is uploaded once per call.
"""
function gather_batched_electron_states!(dst::BatchedElectronState, src::BatchedElectronState, inds)
    dst.nw == src.nw && dst.nband_max == src.nband_max && length(inds) <= dst.nk ||
        throw(ArgumentError("cannot gather $(length(inds)) points of $src into $dst"))
    i = _gather_index(src.nband, inds, src.nk)
    _gather_last!(dst.iband_offset, src.iband_offset, i)
    _gather_last!(dst.nband, src.nband, i)
    _gather_last!(dst.e, src.e, i)
    _gather_last!(dst.u, src.u, i)
    _gather_last!(dst.vdiag, src.vdiag, i)
    _gather_last!(dst.v, src.v, i)
    _gather_last!(dst.rbar, src.rbar, i)
    dst
end
