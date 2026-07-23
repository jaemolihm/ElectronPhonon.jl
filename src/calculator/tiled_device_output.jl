# TiledDeviceOutput: the outer-k-tiled device-output machinery shared by the device-resident batched
# calculators (`BoltzmannCalculator`, MigdalEliashberg's `EliashbergCalculator`). It owns the parts
# both duplicated: the residency decision (full device-resident vs per-tile block), the outer-k tile
# ranges (via `ind_range_for_k_range`, including the ni-cap pre-scan and the contiguity error), the
# lazy device allocation from `ctx.backend`, the per-batch zeroing of the active tile region, and the
# per-tile device→host download (a contiguous copy that avoids scalar-indexing strided device→host
# copies). The calculator declares the output shape and which axis is tiled over outer-k states — any
# dims, any tiled-axis position — and one instance holds N arrays of identical tiling (e.g. g2 and ωq).
#
# Strictly opt-in and composable: the scatter itself stays calculator-side (each calculator calls its
# own `eph_window_scatter!` / `bte_window_accumulate!` into `device_array(t, k)` using `tile_offset`
# and `tile_stride`), and the individual building blocks (residency decision, tile ranges, download)
# are usable on their own so an exotic calculator can hand-roll instead of using the full object.
#
# Two residency modes, decided once on the first batch:
#   * full   — the whole output (all `n_full` outer states) is device-resident; the scatter writes
#              global state `i` at offset 0 with stride `n_full`; one device→host copy at the end.
#   * block  — only the current outer-k tile is device-resident (i-extent = the largest k-batch);
#              the scatter writes tile-local row `i − tile_offset`, and each batch is zeroed and then
#              downloaded to the host. Device memory is bounded by the tile regardless of grid size.
#
# Type stability: the lazily-allocated device/host buffers are held as `Vector{Any}` (their concrete
# device-array type is not known until the first batch). The hot path stays type-stable because the
# scatter kernel — `eph_window_scatter!` / `bte_window_accumulate!` — is the function-boundary type
# barrier: it specializes on the concrete array type at its call site. Everything the object itself
# does (alloc, zero, download) runs once per batch (cold), so the `Any` access there is immaterial.

# `narr` arrays of identical tiling. `dims` is the FULL output shape (`dims[i_axis] == n_full`);
# `i_axis` is the axis tiled over outer-k states. `el_i` supplies the outer-k → state-index tile
# ranges. `force_block` overrides the residency decision (`true` = always block, `false` = always
# full, `nothing` = decide from free device memory with `headroom` × the full-resident byte size).
mutable struct TiledDeviceOutput{FT}
    dims        :: Vector{Int}
    i_axis      :: Int
    n_full      :: Int
    narr        :: Int
    el_i        :: BandStates
    force_block :: Union{Nothing, Bool}
    headroom    :: Float64
    # Lazily filled on the first `tile_begin!` (device-array type unknown until then).
    dev         :: Vector{Any}   # narr device buffers (i-extent = n_full if full, ni_cap if block)
    host        :: Vector{Any}   # narr contiguous host mirrors (block mode only)
    decided     :: Bool
    block       :: Bool
    ni_cap      :: Int
    tile_i0     :: Int
    tile_ni     :: Int
end

"""
    residency_use_block(backend, full_bytes; headroom = 1.2) -> Bool

The residency heuristic (one of the composable building blocks): `true` if `full_bytes` of
full-device-resident output — summed over all arrays — should fall back to a per-tile block layout
because it would not fit free device memory with the given `headroom`, i.e.
`headroom * full_bytes > free_bytes(backend)`. This is exactly the decision `tile_begin!` makes on
its first batch (with `full_bytes = narr · sizeof(FT) · prod(dims)`); exposed standalone so an exotic
calculator that hand-rolls its buffers can make the same choice without a `TiledDeviceOutput`.
"""
residency_use_block(backend::AbstractBackend, full_bytes::Integer; headroom::Real = 1.2) =
    headroom * full_bytes > free_bytes(backend)

function TiledDeviceOutput{FT}(dims, i_axis::Integer, el_i::BandStates;
        narr::Integer = 1, force_block::Union{Nothing, Bool} = nothing,
        headroom::Real = 1.2) where {FT}
    d = collect(Int, dims)
    1 <= i_axis <= length(d) || throw(ArgumentError("i_axis $i_axis out of range for dims $d"))
    TiledDeviceOutput{FT}(d, Int(i_axis), d[i_axis], Int(narr), el_i, force_block, Float64(headroom),
        Any[], Any[], false, false, 0, 0, 0)
end

# True once the device buffers exist (i.e. at least one batch ran on this rank/window).
is_allocated(t::TiledDeviceOutput) = !isempty(t.dev)
is_block(t::TiledDeviceOutput) = t.block

# The k-th device output buffer (the scatter target); `1:narr`.
device_array(t::TiledDeviceOutput, k::Integer = 1) = t.dev[k]
# The k-th contiguous host mirror (block mode; valid after `tile_download!`).
host_array(t::TiledDeviceOutput, k::Integer = 1) = t.host[k]
# Global-i offset of the current tile (0 in full mode).
tile_offset(t::TiledDeviceOutput) = t.tile_i0
# Number of outer states in the current tile.
tile_length(t::TiledDeviceOutput) = t.tile_ni
# The buffer's extent along the tiled axis (n_full in full mode, ni_cap in block mode) — the stride a
# linear-index scatter uses.
tile_stride(t::TiledDeviceOutput) = size(t.dev[1], t.i_axis)

# Begin a batch: on the first batch, decide residency and allocate the device (and, for block mode,
# host-mirror) buffers from `ctx.backend`; every block-mode batch records this batch's outer-k tile
# range and zeros the tile's active region. `ctx` supplies `backend`, `batch` (the outer-k range) and
# `n_batch_max` (for the ni-cap pre-scan).
function tile_begin!(t::TiledDeviceOutput{FT}, ctx) where {FT}
    backend = ctx.backend
    if !t.decided
        if t.force_block === true
            t.block = true
        elseif t.force_block === false
            t.block = false
        else
            full_bytes = t.narr * sizeof(FT) * prod(t.dims)
            t.block = residency_use_block(backend, full_bytes; headroom = t.headroom)
            t.block && @warn "WARNING : full device-resident output " *
                "($(t.narr)×$(round(sizeof(FT) * prod(t.dims) / 1e9, digits = 2)) GB) does not fit free " *
                "GPU memory, falling back to per-tile block residency (per-batch device→host streaming) " *
                "in TiledDeviceOutput.tile_begin!. GPU performance may degrade compared to smaller calculations."
        end
        t.decided = true
        if t.block
            nk = t.el_i.kpts.n
            nb = ctx.n_batch_max
            t.ni_cap = maximum(length(ind_range_for_k_range(t.el_i, ks, min(ks + nb - 1, nk)))
                               for ks in 1:nb:nk)
            tdims = copy(t.dims); tdims[t.i_axis] = t.ni_cap
            t.dev  = Any[alloc(backend, FT, tdims...) for _ in 1:t.narr]
            t.host = Any[Array{FT}(undef, tdims...) for _ in 1:t.narr]
        else
            t.dev  = Any[fill!(alloc(backend, FT, t.dims...), zero(FT)) for _ in 1:t.narr]
        end
    end
    if t.block
        rng = ind_range_for_k_range(t.el_i, first(ctx.batch), last(ctx.batch))
        t.tile_i0 = first(rng) - 1
        t.tile_ni = length(rng)
        for A in t.dev
            fill!(selectdim(A, t.i_axis, 1:t.tile_ni), zero(FT))
        end
    end
    t
end

# Download the current tile (block mode): a single contiguous device→host copy per buffer into the
# host mirrors. The calculator then copies the mirror's leading `tile_length` slice into its output.
function tile_download!(t::TiledDeviceOutput)
    for k in 1:t.narr
        copyto!(t.host[k], t.dev[k])
    end
    t
end

# Free device (and host mirror) buffers and reset the residency decision, so the object can be reused
# on a different grid. Call from `postprocess_calculator!`.
function tile_free!(t::TiledDeviceOutput)
    t.dev = Any[]
    t.host = Any[]
    t.decided = false
    t.block = false
    t.ni_cap = 0
    t.tile_i0 = 0
    t.tile_ni = 0
    t
end
