# The two e-ph engines serve three drivers: outer k with inner k+q or q, and outer q with inner k.
# Each engine holds one run's device copy of `model.epmat`, its stage-1 output for one outer batch, and one tile of
# buffers per CPU thread chunk (`eng.tiles[chunk]`), and wraps the batched kernels of
# `wannier_to_bloch_batched.jl`:
#
#   stage1!  contract the outer R of `epmat` for an outer batch, apply the outer rotation;
#   stage2!  select the pairs to compute, gather their states, contract the other R, apply the
#            remaining rotations, and add the long-range (polar) term
#            (`eph_engine_add_longrange!`); call a function on each calculator-ready EPBlock.
#
# The resident containers (`eng.els_k`, `eng.els_kq`, `eng.phs`) are read-only. Stage 2 skips the
# pairs that need no e-ph matrix and gathers the others' states once, into the tile's buffers; a
# side whose whole tile is kept and resident is a view, with no copy. A pair is skipped when
# (i) its k+q has no state in the window (`_kqpairs_in_window!`), or (ii) it has no process inside
# `energy_conservation_tol` (`_kqpairs_conserving_energy!` on the host; on a device, the same test in
# `_select_kqpairs_on_device!` for an inner k+q grid). (i) by route:
#   outer k, inner k+q            never: the resident k+q set holds only points with a state;
#   outer k, inner q              no window band after the tile's k+q solve;
#   outer q, k+q solved per tile  no window band after the tile's k+q solve;
#   outer q, k+q precomputed      the k+q point is absent from the resident set.
#
# `engine_bytes` counts the buffers the constructors allocate, term for term, so `plan_batch` can
# choose the widths before the engine is built. Both engines are backend-generic: host arrays on
# `CPUBackend`, device arrays otherwise; the CUDA extension overrides only the fused rotation kernels.
#
# Model storage must match the loop order: outer k requires epmat_outer_momentum = "el"
# (column R_e), and outer q requires "ph" (column R_p). Both stage-1 Fourier transforms then
# use the batched interpolator directly.

function _require_epmat_layout(order::LoopTag, model)
    required = order isa OuterKLoop ? "el" : "ph"
    model.epmat_outer_momentum == required || throw(ArgumentError(
        "$(order isa OuterKLoop ? "outer-k" : "outer-q") loop requires a model loaded with " *
        "epmat_outer_momentum = \"$required\" (got \"$(model.epmat_outer_momentum)\")"))
    nothing
end

# Storage structs deliberately have no array-type parameters. At each batch/chunk boundary,
# _workspace_fields exposes concrete fields to explicitly named stage and chunk workers. The
# kernels then see concrete CPU/GPU arrays without encoding every buffer type in the engine.
function _workspace_fields(workspace)
    # function barrier for exposing concrete buffer types to specialized workers.
    NamedTuple{fieldnames(typeof(workspace))}(ntuple(i -> getfield(workspace, i), Val(fieldcount(typeof(workspace)))))
end

"""
    OuterKTileWorkspace

Scratch for one thread chunk's inner tiles. `ind_kept_kqpairs` lists the kept pairs of a tile, `ikqs` and
`iqs` their k+q and q indices on the host in tile order. The tile's pairs are split by the band count of
their k+q point (`class_key`, `class_order`, `class_start`, `_class_order!`): `ikqs_by_class` and
`iqs_by_class` hold their indices in that order on the host (`ikqs_dev`, `iqs_dev` on the backend), and
`els_kq`, `phs`, `wtqs`, `xkqs` and `P_kq` their gathered k+q states (one segment per class, at the class's
own band width), phonons, weights, k+q coordinates and Fourier phase. `pos_by_class` (`pos_dev` on the
backend) are the positions of the solved k+q points of the classes. `seg_host` and `seg_dev` are the
segment table of the tile. `itp_el_ham`, `hk` and `kqs` solve k+q per tile for an inner q grid, and are
`nothing` for an inner k+q grid.
`u_ph_id` holds the cartesian phonon basis, `dg` serves covariant derivatives, and
`uk_polar` and `mmat_buffer` are scratch of the polar term. Those features allocate buffers only
when requested.
"""
Base.@kwdef struct OuterKTileWorkspace
    ind_kept_kqpairs
    ikqs
    iqs
    ikqs_by_class
    iqs_by_class
    pos_by_class
    class_key
    class_order
    class_start
    ikqs_dev
    iqs_dev
    pos_dev
    slot_col
    seg_host
    seg_dev
    els_kq
    phs
    wtqs
    itp_el_ham
    hk
    kqs
    xkqs
    P_kq
    u_ph_id
    ep
    g
    tmp
    dg
    uk_polar
    mmat_buffer
end

"""
    OuterQTileWorkspace

Scratch for one thread chunk's k tiles in the outer-q loop. `ind_kept_kqpairs` lists the kept
pairs of a tile, `iks` and `ikqs` their indices (`iks_dev`, `ikqs_dev` on the backend), and `els_k`,
`els_kq`, `wtks` and `xks` hold their gathered states, weights and coordinates. `P_k` is their
stage-2 Fourier phase `exp(2πi R_e · x_k)`.
`kqs` holds the tile's k+q coordinates; `itp_el_ham` and `hk` solve k+q per tile, and are `nothing`
with precomputed k+q states.
`uk_rep` is scratch of the electron rotations (`eph_apply_rotations_rqkq!`), and `uk_polar` and
`mmat_buffer` of the polar term, allocated only for a polar model. `seg_host` and `seg_dev` are the
tile's segment table.
"""
Base.@kwdef struct OuterQTileWorkspace
    ind_kept_kqpairs
    iks
    ikqs
    iks_dev
    ikqs_dev
    els_k
    els_kq
    wtks
    xks
    P_k
    itp_el_ham
    hk
    kqs
    ep
    g
    tmp
    uk_rep
    uk_polar
    mmat_buffer
    seg_host
    seg_dev
end

"""
    DevicePairSelection

The buffers of `_select_kqpairs_on_device!`, which selects the (k, k+q) pairs of an outer-k run
over a k+q grid on the device: `ckq_dev` the k+q grid coordinates (`_pair_grid_coords`) and
`qtable_dev` the Int32 q index of each grid hash (0: none), for the whole run; per pair of a tile,
`keep_dev` its keep flag, `pos_dev` the running count and `iqs_tile_dev` its q index; `ikqs_dev`
the kept pairs' k+q indices and `n_kept_host` their number read back. One per engine: a device run
has one thread chunk. `compact = false` leaves `pos_dev` and `ikqs_dev` empty, for a selection that
only marks the kept q (`_mark_conserving_qs!`).
"""
struct DevicePairSelection{MI, VQ, VI}
    ckq_dev      :: MI
    qtable_dev   :: VQ
    keep_dev     :: VI
    pos_dev      :: VI
    iqs_tile_dev :: VI
    ikqs_dev     :: VI
    n_kept_host  :: Vector{Int}
end

function DevicePairSelection(backend, ckq, qpts, n_tile; compact = true)
    qtable = zeros(Int32, prod(qpts.ngrid))
    for (iq, xq) in enumerate(qpts.vectors)
        qtable[_hash_xk(xq, qpts.ngrid, qpts.shift) + 1] = iq
    end
    n_compact = compact ? n_tile : 0
    DevicePairSelection(to_device_copy(backend, ckq), to_device_copy(backend, qtable),
        alloc(backend, Int, n_tile), alloc(backend, Int, n_compact), alloc(backend, Int, n_tile),
        alloc(backend, Int, n_compact), Vector{Int}(undef, 1))
end

"""
    PhononsPerBatch

The phonons of an outer-k run built per outer batch, for the q of the batch's kept pairs only:
`ω_kept` the kept modes' frequencies of every q, which select the pairs; `iqs_per_batch[b]` the q of
batch b's kept pairs (`_select_qs_per_outer_batch`); `phs_capacity` the storage of the largest
batch's phonons; `iq_to_ph_dev` the Int32 index of each q in the current batch's phonons (0: none)
and `iqs_ph_dev` that of a tile's kept pairs.
"""
struct PhononsPerBatch{W, P, VQ, VI}
    ω_kept         :: W
    iqs_per_batch :: Vector{Vector{Int32}}
    phs_capacity  :: P
    iq_to_ph_dev  :: VQ
    iqs_ph_dev    :: VI
end

"""
    OuterKEngine

The `OuterKLoop` engine: g(k, R_p) for an outer-k batch (stage 1), then g(k, k+q) for one k and a
tile of k+q (stage 2), in the k+q convention of [`eph_rotate_kR_batched!`](@ref): stage 1 folds
`exp(-2πi R_p · x_k)` into g(k, R_p), so the stage-2 phase `exp(2πi R_p · x_{k+q})` of a tile is
shared by every k of the batch. Stage 1 orders the batch by band count and stores each class of
equal count `b` at width `b`, so a block's k side carries exactly the k's bands. With
`covariant_derivative_of_g`, the same two stages run on the
position-weighted `epmat_R` (`wannier_object_multiply_R` plus the tight-binding term
`im (r_j - r_i) g`) into `dg`. Without a k+q container (`run_eph_over_k_and_q`) the inner points are
q points, the k+q states are solved per (k, tile) into the tile's buffers and the phase is built at
x_k + x_q for each k.
"""
Base.@kwdef mutable struct OuterKEngine
    model        :: Model
    els_k        :: BatchedElectronState
    els_kq       :: Union{Nothing, BatchedElectronState}
    phs          :: BatchedPhononState  # every q, or the current outer batch's (`phonons_per_batch`)
    kpts         :: AbstractKpoints
    kqpts        :: Union{Nothing, AbstractKpoints}
    qpts         :: AbstractKpoints
    sel_k        :: Union{Nothing, FilteredBandStates}
    sel_kq       :: Union{Nothing, FilteredBandStates}
    imap_k                        # (nband_max_k, nk) state index of (box band, k) in `sel_k`, on the backend, or `nothing`
    imap_kq                       # the same for k+q and `sel_kq`
    window_kq
    energy_conservation_tol :: Float64
    eph_phonon_basis :: Symbol
    inner_loop_kq :: Bool         # inner points are k+q (resident states); false: q, k+q solved per tile
    iks_batch    :: UnitRange{Int} # the outer k points of the current stage 1
    backend      :: AbstractBackend
    epmat        :: WannierObject  # model.epmat on the backend
    itp_epmat                     # its batched R_e interpolator
    itp_epmat_R                   # interpolator of epmat_R (dg), or `nothing`
    irvecp_mat                    # (nr_p, 3) R_p
    mxks                          # (3, n_outer_batch) the batch's -x_k on the backend
    xkqs                          # (3, nkq) x_{k+q}, or x_q without a k+q grid
    wtkqs                         # (nkq,) their weights
    ck           :: Matrix{Int}     # (3, nk) k coordinates on the q grid (`_pair_grid_coords`)
    ckq          :: Matrix{Int}     # (3, nkq) k+q coordinates on the q grid
    pair_selection                # a `DevicePairSelection`, or `nothing`: the pairs are selected on the host
    phonons_per_batch             # a `PhononsPerBatch`, or `nothing`: `phs` holds every q
    P_mk                          # (nr_p, n_outer_batch) exp(-2πi R_p · x_k)
    ep_kR                         # stage-1 output, one (nw b nmodes, nr_p, n_b) segment per band class b
    dg_kR                         # the same, (nw b nmodes, nr_p, 3, n_b) segments, or `nothing`
    nband_k_host :: Vector{Int}     # (nk,) the k points' band counts, on the host
    nband_kq_host :: Vector{Int}    # (nkq,) the resident k+q points' band counts; empty without a k+q grid
    els_k_classes :: Vector         # [b] the batch's k states with b bands, at box width b
    iks_sorted   :: Vector{Int}     # the batch's k points ordered by band count
    slot_in_class :: Vector{Int}    # (n_outer_batch,) position of each batch k in its band class
    class_first  :: Vector{Int}     # (nband_max_k,) position of class b's first k in `iks_sorted`
    class_offsets :: Matrix{Int}    # (2, nband_max_k) element offsets of class b in ep_kR, dg_kR
    xks_host     :: Matrix{Float64} # (3, n_outer_batch) the batch's x_k, staged for `xks`
    xks                           # (3, n_outer_batch) the batch's x_k on the backend
    g_fourier                     # stage-1 Fourier output, `reshape_buffer_view` per use
    n_outer_batch :: Int
    n_inner_tile :: Int
    tiles        :: Vector{OuterKTileWorkspace}
end

"""
    OuterQEngine

The `OuterQLoop` engine: g(R_e, q) for an outer-q batch with the phonon basis applied (stage 1),
then g(k, k+q) for one q and a tile of k (stage 2), the k+q states solved per tile into the
buffers' leading `maximum(nband)` columns when they are not precomputed. Each chunk contracts
a read-only slice of the stage-1 output with its own Fourier phase.
"""
Base.@kwdef mutable struct OuterQEngine
    model        :: Model
    els_k        :: BatchedElectronState
    els_kq       :: Union{Nothing, BatchedElectronState}
    phs          :: BatchedPhononState
    kpts         :: AbstractKpoints
    kqpts        :: Union{Nothing, AbstractKpoints}
    qpts         :: AbstractKpoints
    sel_k        :: Union{Nothing, FilteredBandStates}
    sel_kq       :: Union{Nothing, FilteredBandStates}
    imap_k                        # (nband_max_k, nk) state index of (box band, k) in `sel_k`, on the backend, or `nothing`
    imap_kq                       # the same for k+q and `sel_kq`
    window_kq
    energy_conservation_tol :: Float64
    eph_phonon_basis :: Symbol
    precompute_el_kq :: Bool      # the k+q states are resident (`els_kq`); false: solved per tile
    iqs_batch    :: UnitRange{Int} # the outer q points of the current stage 1
    backend      :: AbstractBackend
    epmat        :: WannierObject  # model.epmat on the backend
    itp_epmat                     # its batched R_p interpolator
    irvece_mat                    # (nr_e, 3) R_e
    g_q                           # (nw² nmodes, nr_e, n_outer_batch) stage-1 Fourier output
    g_rot                         # (nw², nr_e, nmodes, n_outer_batch) basis scratch, or `nothing`
    ep_Rq                         # (nw² nmodes, nr_e, n_outer_batch) stage-1 output
    wtks                          # (nk,) k weights
    xqs_host     :: Matrix{Float64} # (3, n_outer_batch) the batch's x_q, staged for `xqs`
    xqs                           # (3, n_outer_batch) the batch's x_q on the backend
    n_outer_batch :: Int
    n_inner_tile :: Int
    tiles        :: Vector{OuterQTileWorkspace}
end

# The stable order of points by integer class key in `0:kmax` (a band count) by counting sort, into
# `order`; `start` has `kmax + 2` entries. Returns the classes `[(key, range)]` with `key >= 1`: the
# points of a class are `order[range]`, in their original order. The points of key 0 come first in
# `order` and belong to no class.
function _class_order!(order, start, keys)
    fill!(start, 0)
    for key in keys
        start[key + 2] += 1
    end
    start[1] = 1
    for i in 2:length(start)
        start[i] += start[i - 1]
    end
    # start[key + 1] is the first free slot of `key`; placing a point advances it to the end of the class.
    for (j, key) in enumerate(keys)
        order[start[key + 1]] = j
        start[key + 1] += 1
    end
    [(key, start[key]:start[key + 1] - 1) for key in 1:length(start) - 2 if start[key + 1] > start[key]]
end

# The segment table of a tile on the host (`seg_host`, one column per band class) and its device copy, as the
# views `EPBlock` carries: columns `1:nseg` of the chunk's `seg_host` / `seg_dev`.
function _upload_segments!(tile_workspace, nseg)
    copyto!(tile_workspace.seg_dev, 1, tile_workspace.seg_host, 1, SEG_ROWS * nseg)
    (view(tile_workspace.seg_dev, :, 1:nseg), view(tile_workspace.seg_host, :, 1:nseg))
end

# The source column of slot `s`, band `m` of pair `j` of a class segment (`slotrow`: the row of `seg` holding the
# class's slot offset, `brow` the row of its band count), for a source of columns `stride` apart: the
# column of band `m` (shifted by `bandoff[pt]` for a source indexed by physical band) of point `pts[j]`.
@inline function _slot_column(seg, nseg, slotrow, brow, s, pts, bandoff, stride)
    c = _segment_column(seg, nseg, slotrow, s)
    b = seg[brow, c]
    r = s - 1 - seg[slotrow, c]
    pt = pts[seg[SEG_PAIR, c] + r ÷ b + 1]
    _band_offset(bandoff, pt) + r % b + 1 + stride * (pt - 1)
end
_band_offset(::Nothing, pt) = 0
@inline _band_offset(bandoff, pt) = bandoff[pt]

# Gather the bands of the `n_slots` leading band slots of the tile's class segments (the k+q side:
# `slotrow = SEG_KQ_SLOT`, `brow = SEG_BKQ`; the k side: `SEG_K_SLOT`, `SEG_BK`) from the source arrays `e_src`,
# `u_src` and `vdiag_src` (`nothing` where `dst` holds no such field) into `dst`, one gather per field for
# the whole tile. `col` is scratch for the source columns.
@views function _gather_slots!(dst, e_src, u_src, vdiag_src, col, seg, slotrow, brow, pts, bandoff, stride, n_slots)
    (dst.v === nothing && dst.rbar === nothing) || throw(ArgumentError(
        "the engine's tile gathers e, u and vdiag only"))
    col = col[1:n_slots]
    col .= _slot_column.(Ref(seg), size(seg, 2), slotrow, brow, 1:n_slots, Ref(pts), Ref(bandoff), stride)
    # `col` is in range by construction (the segment table); the bounds check of a device index array is a
    # reduction and a device-to-host read per gather.
    gather!(d, src, ncomp) = d === nothing ||
        (@inbounds reshape(d, ncomp, :)[:, 1:n_slots] .= view(reshape(src, ncomp, :), :, col))
    gather!(dst.e, e_src, 1)
    gather!(dst.u, u_src, dst.nw)
    gather!(dst.vdiag, vdiag_src, 3)
    dst
end

# Where the next class segment of a container starts after one of `n` points with `b` bands, as the
# `offset` of `reshape_view_batched_electron_states`.
_after_segment(offset, n, b) = offset .+ (n, n * b, n * b^2)

# The integer coordinates `(ck, ckq)`, each `(3, n)`, of the points of `kpts` and `kqpts` on the q
# grid, reduced into `0:ng-1` once so that `_iq_of_pair` folds `ckq - ck` with a compare-and-add.
# Each list is measured from its own first point, which carries its grid shift (as in
# `combine_kpoint_grids`), so either grid may be shifted. Then `x_{k+q} - x_k` is the q-grid node
# `ckq - ck` plus the whole number of q-grid steps in `x_{k+q,1} - x_{k,1} - qpts.shift`, which is
# folded into `ckq`. Throws if a point is off its list's grid or that offset is not whole.
function _pair_grid_coords(kpts, kqpts, qpts)
    ng = qpts.ngrid
    ck = _grid_coords_checked(kpts, ng)
    ckq = _grid_coords_checked(kqpts, ng)
    (kpts.n == 0 || kqpts.n == 0) && return (ck, ckq)
    offset = (kqpts.vectors[1] - kpts.vectors[1] - qpts.shift) .* ng
    isapprox(round.(offset), offset; atol = sqrt(eps(Float64))) || throw(ArgumentError(
        "the k+q shift minus the k shift is not the q grid's shift $(qpts.shift) on the grid $ng"))
    ckq .= mod.(ckq .+ round.(Int, offset), ng)
    (ck, ckq)
end

# The `_grid_coords_reduced` of every point of `pts` on `ngrid`, measured from its first point.
function _grid_coords_checked(pts, ngrid)
    c = Matrix{Int}(undef, 3, pts.n)
    for i in axes(c, 2)
        xk = pts.vectors[i]
        x = (xk - pts.vectors[1]) .* ngrid
        isapprox(round.(x), x; atol = sqrt(eps(Float64))) || throw(ArgumentError(
            "k point $xk is not on the grid $ngrid through $(pts.vectors[1])"))
        c[:, i] .= _grid_coords_reduced(xk, ngrid, pts.vectors[1])
    end
    c
end

# The index into `qpts` of `x_{k+q} - x_k` for outer k `ik` and k+q point `ikq`, by integer grid
# hash on the coordinates of `_pair_grid_coords`. 0 if `qpts` does not hold it.
@inline function _iq_of_pair(qpts, ck, ckq, ik, ikq)
    ng1, ng2, ng3 = qpts.ngrid
    # Both operands are pre-reduced into 0:ng-1, so the fold is a compare-and-add.
    h1 = _wrap_reduced(ckq[1, ikq] - ck[1, ik], ng1)
    h2 = _wrap_reduced(ckq[2, ikq] - ck[2, ik], ng2)
    h3 = _wrap_reduced(ckq[3, ikq] - ck[3, ik], ng3)
    _ik_from_hash(qpts, (h1 * ng2 + h2) * ng3 + h3)
end

# Fill `iqs[1:nkq_tile]` with the q index of outer k `ik` and every k+q of the tile
# `ikq_first .+ (0:nkq_tile-1)`.
function _fill_iqs!(iqs, qpts, ck, ckq, ik, ikq_first, nkq_tile)
    for j in 1:nkq_tile
        iq = _iq_of_pair(qpts, ck, ckq, ik, ikq_first + j - 1)
        iq == 0 && throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
        iqs[j] = iq
    end
    iqs
end

"""
    q_index_table(kpts, kqpts, qpts) -> Matrix{Int32}

The index `iq_kk[ik, ikq]` into `qpts` of `q = x_{k+q} - x_k` for every outer k point `ik` of
`kpts` and inner point `ikq` of `kqpts`, `(kpts.n, kqpts.n)`. Every pair must have its q in
`qpts`, as for the set `combine_kpoint_grids(kqpts, kpts, -, ngrid_q)` of `run_eph_over_k_and_kq`;
a miss throws. Either grid may be shifted. The lookup is the outer-k loop's own (`_iq_of_pair`),
so `iq` is the index the loop used for that pair's phonons.
"""
function q_index_table(kpts, kqpts, qpts)
    qpts.n <= typemax(Int32) || throw(ArgumentError(
        "$(qpts.n) q points do not fit the Int32 q index"))
    ck, ckq = _pair_grid_coords(kpts, kqpts, qpts)
    iq_kk = Matrix{Int32}(undef, kpts.n, kqpts.n)
    @threads for ikq in axes(iq_kk, 2)
        for ik in axes(iq_kk, 1)
            iq_kk[ik, ikq] = _iq_of_pair(qpts, ck, ckq, ik, ikq)
        end
    end
    # A miss is 0, checked here rather than in the threaded loop, where it would surface wrapped.
    any(iszero, iq_kk) && throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
    iq_kk
end

# ---- OuterKEngine ----------------------------------------------------------------------------

# The device bytes the phonons per outer batch add to the outer-k engine (`engine_bytes`): the
# storage of the largest batch's phonons, the q index and upload, and the transients of one batch's
# phonon solve (`compute_phonon_states_batched` into the storage) in `persistent`; the pair's
# `iqs_ph_dev` entry in `per_pair`.
function _phonon_batches_bytes(model::Model{FT}, backend, phonon_batches) where {FT}
    phonon_batches === nothing && return (; persistent = 0, per_pair = 0)
    (; nmodes) = model
    cx, rl = sizeof(Complex{FT}), sizeof(FT)
    nq_max = maximum(length, phonon_batches.iqs_per_batch; init = 0)
    # The solve's transients: the dynamical matrices with their R vectors, the interpolator's Fourier
    # phase and output, the masses, and per Fourier block the D(q) stack, its eigenvalues and the
    # staged q.
    ph_solve = if nq_max == 0
        0
    else
        nr_ph, ndata_ph = length(model.ph_dyn.irvec), model.ph_dyn.ndata
        nblock = _default_batch_size(backend, nr_ph, ndata_ph; nk_hint = nq_max)
        cx * length(model.ph_dyn.op_r) + rl * 3nr_ph + cx * nr_ph * nblock + cx * ndata_ph + rl * nmodes +
            (cx * ndata_ph + rl * nmodes + rl * 3) * nblock
    end
    persistent = _phonon_state_bytes(FT, size(phonon_batches.ω_kept, 1), (:e, :u); ndisp = nmodes) * nq_max +
        sizeof(Int32) * (size(phonon_batches.ω_kept, 2) + nq_max) + ph_solve
    (; persistent, per_pair = sizeof(Int))
end

function engine_bytes(::Type{OuterKEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk, nkq,
        el_qty, ph_qty, covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq,
        device_pair_selection = false, nq_grid = 0, nmodes_kept = model.nmodes,
        phonon_batches = nothing, backend = nothing) where {FT}
    (; nw, nmodes) = model
    batch_ph = _phonon_batches_bytes(model, backend, phonon_batches)
    cx, rl, iz = sizeof(Complex{FT}), sizeof(FT), sizeof(Int)
    nr_p = length(model.epmat.irvec_next)
    nr_e = length(model.epmat.irvec)
    nepmat = length(model.epmat.op_r)
    ndata = nw * nband_max_k * nmodes
    nd = covariant_derivative_of_g ? 4 : 1                      # ep, plus three dg directions
    nrows = nw^2 * nmodes * nr_p                                # the epmat rows that stage 1 keeps
    nrows_max = nrows * (covariant_derivative_of_g ? 3 : 1)     # those of epmat_R with dg
    persistent =
        cx * nepmat * (covariant_derivative_of_g ? 4 : 1) +    # epmat (+ epmat_R)
        rl * 3 * (nr_p + nr_e) * (covariant_derivative_of_g ? 2 : 1) +  # R-vector matrices
        cx * nrows + (covariant_derivative_of_g ? cx * 3nrows : 0) +  # interpolator outputs
        rl * 3 * nkq + rl * nkq +                               # xkqs, wtkqs
        (inner_loop_kq ? 0 : cx * length(model.el_ham.op_r)) +    # el_ham
        (device_pair_selection ? iz * 3nkq + sizeof(Int32) * nq_grid : 0) +  # ckq_dev, qtable_dev
        iz * (nband_max_k * nk + (inner_loop_kq ? nband_max_kq * nkq : 0)) +  # imap_k, imap_kq
        batch_ph.persistent                                     # phonons per outer batch
    per_outer =
        cx * ndata * nr_p * nd +                                # ep_kR (+ dg_kR)
        cx * nr_p + 2rl * 3 +                                   # P_mk, xks, mxks
        cx * nr_e * (covariant_derivative_of_g ? 2 : 1) +       # Fourier phases
        cx * nrows_max +                                        # g_fourier
        cx * (nrows + 2 * nband_max_k * nrows ÷ nw) * nd +      # transients of eph_rotate_kR_batched!
        sum(b -> _electron_state_bytes(FT, nw, b, el_qty), 1:nband_max_k; init = 0)  # els_k_classes
    nbox = nband_max_kq * nband_max_k * nmodes_kept             # ep box, in the modes the run keeps
    per_pair =
        cx * nbox * (covariant_derivative_of_g ? 4 : 1) +       # ep (+ dg)
        cx * ndata + cx * nband_max_kq * nband_max_k * nmodes + # stage-2 scratch g, tmp
        cx * nr_p +                                             # P_kq
        _electron_state_bytes(FT, nw, nband_max_kq, el_qty) +   # k+q tile
        _phonon_state_bytes(FT, nmodes_kept, ph_qty; ndisp = nmodes) +  # phs tile
        (12 + nband_max_kq) * iz + rl + 3rl +                   # kept pairs, k+q and q indices (tile and class order, on the backend), slot columns, wtqs, x_{k+q}
        (device_pair_selection ? 4iz : 0) +                     # keep, pos, iqs_tile, ikqs on the device
        batch_ph.per_pair +                                     # iqs_ph_dev
        (inner_loop_kq ? 0 :
            cx * length(model.el_ham.irvec) + cx * nw^2 * 3) +  # k+q Fourier phase, H and eigensolve
        (eph_phonon_basis == :cartesian ? cx * nmodes^2 : 0) +  # identity basis
        (model.polar_eph.use ? cx * (nw * nband_max_k + nband_max_kq * nband_max_k) : 0)  # polar scratch
    (; persistent, per_outer, per_pair)
end

_electron_state_bytes(FT, nw, nb, qty) = 2sizeof(Int) +
    sizeof(FT) * ((:e ∈ qty) * nb + (:vdiag ∈ qty) * 3nb) +
    sizeof(Complex{FT}) * ((:u ∈ qty) * nw * nb + ((:v ∈ qty) + (:rbar ∈ qty)) * 3nb^2)
_phonon_state_bytes(FT, nm, qty; ndisp = nm) = sizeof(FT) * ((:e ∈ qty) * nm + (:vdiag ∈ qty) * 3nm) +
    sizeof(Complex{FT}) * ((:u ∈ qty) * ndisp * nm + (:eph_dipole_coeff ∈ qty) * nm + (:eph_r_coeff ∈ qty) * 3nm)

# Whether an outer-k run selects the pairs of a k+q tile on the device (`_select_kqpairs_on_device!`):
# over a k+q grid, on a device, with a finite `energy_conservation_tol`. With `Inf` the host's q index
# overlaps the device's work.
_uses_device_pair_selection(backend, inner_loop_kq, energy_conservation_tol) =
    inner_loop_kq && !(backend isa CPUBackend) && isfinite(energy_conservation_tol)

"""
    OuterKEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kwargs...)

Build the outer-k run's stage-1 buffers and one inner-tile workspace per thread chunk.
The model must use `epmat_outer_momentum = "el"`. The inner loop is k+q with
`inner_loop_kq = true` (`run_eph_over_k_and_kq`, resident states `els_kq`), or q with
`inner_loop_kq = false` (`run_eph_over_k_and_q`, `els_kq = nothing`), the k+q states solved per
tile. It defaults to `els_kq !== nothing`; a contradicting value is an `ArgumentError`.

`n_outer_batch` and `n_inner_tile` set the buffer capacities; `nchunks` sets the number of
independent thread workspaces. `el_qty` / `ph_qty` select stored state fields.
`energy_conservation_tol` skips point pairs (`_kqpairs_conserving_energy!`); it does not change any
band window. `phs` may hold only the lowest `phs.nmodes` modes (`size(phs.u, 1) == model.nmodes`
displacements), and the tiles then carry that many. `covariant_derivative_of_g` allocates the three derivative
components; `eph_phonon_basis` selects eigenmode or cartesian phonons.
With `phonon_batches = (; ω_kept, iqs_per_batch)` (`_setup_phonons`: the kept modes' frequencies of
every q, and the q of each outer batch), `stage1!` builds the phonons of each outer batch's q and
`phs` is an empty container with the run's modes; this needs the device pair selection.
"""
function OuterKEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, kqpts, qpts,
        inner_loop_kq = els_kq !== nothing, n_outer_batch, n_inner_tile, nchunks,
        covariant_derivative_of_g,
        eph_phonon_basis, sel_k = nothing, sel_kq = nothing, window_kq = (-Inf, Inf),
        energy_conservation_tol = Inf, phonon_batches = nothing) where {FT}
    # Validate the model layout and prepare the stage-1 interpolators for the selected inner loop.
    (; nw, nmodes) = model
    _require_epmat_layout(OuterKLoop(), model)
    # run_eph_over_k_and_q has no resident k+q container: solve k+q per inner q tile,
    # at box width nw. run_eph_over_k_and_kq instead gathers phonons for resident k+q states.
    inner_loop_kq == (els_kq !== nothing) || throw(ArgumentError(
        "inner_loop_kq = true takes the resident k+q states els_kq, and false takes els_kq = nothing"))
    nbk, nbkq = els_k.nband_max, inner_loop_kq ? els_kq.nband_max : nw
    inner_pts = inner_loop_kq ? kqpts : qpts
    epmat = to_device(backend, model.epmat)
    irvec_p = model.epmat.irvec_next
    nr_p = length(irvec_p)
    itp_epmat = BatchedWannierInterpolator(epmat; backend, batch_size = n_outer_batch)
    itp_epmat_R = if covariant_derivative_of_g
        # The position-weighted e-ph matrix `im R_e g(R_e, R_p)`.
        epmat_R = wannier_object_multiply_R(model.epmat, model.lattice)
        # Tight-binding approximation: dgᵃ_{ijν}(Rₑ, Rₚ) += im * (rᵃ_j - rᵃ_i) g_{ijν}(Rₑ, Rₚ)
        # epmat   : (i, j, nmodes, Rₚ, Rₑ)
        # epmat_R : (i, j, nmodes, Rₚ, 3, Rₑ)
        nrp = length(epmat_R.irvec_next)
        @views for ire in axes(epmat_R.op_r, 2)
            tmp_g  = Base.ReshapedArray(model.epmat.op_r[:, ire], (nw, nw, nmodes, nrp), ())
            tmp_gR = Base.ReshapedArray(epmat_R.op_r[:, ire], (nw, nw, nmodes, nrp, 3), ())
            for idir in 1:3, iw in 1:nw
                ri = model.wann_centers[iw][idir]
                tmp_gR[iw, :, :, :, idir] .-= im .* ri .* tmp_g[iw, :, :, :]
                tmp_gR[:, iw, :, :, idir] .+= im .* ri .* tmp_g[:, iw, :, :]
            end
        end
        BatchedWannierInterpolator(to_device(backend, epmat_R); backend, batch_size = n_outer_batch)
    end

    # Prepare run-wide coordinates, grid hashes, and the outer-batch phase buffer.
    # Grid coordinates as (3 × n) device matrices for the two phase builds. -x_k out of place: on
    # `CPUBackend` `_kpoints_to_device_matrix` is a view onto `kpts.vectors`.
    xkqs = _kpoints_to_device_matrix(backend, inner_pts)
    # The q index of a pair by integer grid hash (`_fill_iqs!`). Not needed without a k+q grid.
    ck, ckq = inner_loop_kq ? _pair_grid_coords(kpts, kqpts, qpts) :
        (Matrix{Int}(undef, 3, 0), Matrix{Int}(undef, 3, 0))
    device_pair_selection = _uses_device_pair_selection(backend, inner_loop_kq, energy_conservation_tol)
    device_pair_selection && nchunks > 1 &&
        throw(ArgumentError("the device pair selection runs on one thread chunk, got nchunks = $nchunks"))
    pair_selection = device_pair_selection ? DevicePairSelection(backend, ckq, qpts, n_inner_tile) : nothing
    (phonon_batches === nothing || device_pair_selection) ||
        throw(ArgumentError("phonons per outer batch need the device pair selection"))
    (phonon_batches === nothing || length(phonon_batches.iqs_per_batch) == cld(kpts.n, n_outer_batch)) ||
        throw(ArgumentError("iqs_per_batch holds $(length(phonon_batches.iqs_per_batch)) outer " *
            "batches, but n_outer_batch = $n_outer_batch makes $(cld(kpts.n, n_outer_batch)) of " *
            "$(kpts.n) outer k"))
    phonons_per_batch = if phonon_batches !== nothing
        # One storage for every batch, made before the loop's transients fragment the pool.
        nq_max = maximum(length, phonon_batches.iqs_per_batch; init = 0)
        PhononsPerBatch(phonon_batches.ω_kept, phonon_batches.iqs_per_batch,
            BatchedPhononState(backend, phs.nmodes, nq_max, [:e, :u]; FT, ndisp = nmodes),
            alloc(backend, Int32, qpts.n), alloc(backend, Int, n_inner_tile))
    end
    el_ham = inner_loop_kq ? nothing : to_device(backend, model.el_ham)
    # A partial last batch operates on views of the leading columns of the maximum-capacity buffers.
    P_mk = fill!(alloc(backend, Complex{FT}, nr_p, n_outer_batch), 1)
    ndata = nw * nbk * nmodes

    # Allocate independent inner-tile scratch for each thread chunk.
    tiles = map(1:nchunks) do _
        if inner_loop_kq
            itp_el_ham = hk = kqs = nothing
        else
            # Inner q points: solve k+q per (k, q tile).
            itp_el_ham = BatchedWannierInterpolator(el_ham; backend, batch_size = n_inner_tile)
            hk = alloc(backend, Complex{FT}, nw^2, n_inner_tile)
            kqs = Vector{Vec3{FT}}(undef, n_inner_tile)
        end
        OuterKTileWorkspace(;
            ind_kept_kqpairs = Vector{Int}(undef, n_inner_tile),
            ikqs = Vector{Int}(undef, n_inner_tile),
            iqs = Vector{Int}(undef, n_inner_tile),
            ikqs_by_class = Vector{Int}(undef, n_inner_tile),
            iqs_by_class = Vector{Int}(undef, n_inner_tile),
            pos_by_class = Vector{Int}(undef, n_inner_tile),
            class_key = Vector{Int}(undef, n_inner_tile),
            class_order = Vector{Int}(undef, n_inner_tile),
            class_start = zeros(Int, nbkq + 2),
            ikqs_dev = alloc(backend, Int, n_inner_tile),
            iqs_dev = alloc(backend, Int, n_inner_tile),
            pos_dev = alloc(backend, Int, n_inner_tile),
            slot_col = alloc(backend, Int, nbkq * n_inner_tile),
            seg_host = zeros(Int, SEG_ROWS, nbkq), seg_dev = alloc(backend, Int, SEG_ROWS, nbkq),
            els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
            phs = BatchedPhononState(backend, phs.nmodes, n_inner_tile, ph_qty; FT, ndisp = nmodes),
            wtqs = alloc(backend, FT, n_inner_tile),
            itp_el_ham, hk, kqs,
            xkqs = alloc(backend, FT, 3, n_inner_tile),
            P_kq = alloc(backend, Complex{FT}, nr_p, n_inner_tile),
            u_ph_id = eph_phonon_basis == :cartesian ? to_device_copy(backend,
                repeat(Matrix{Complex{FT}}(I, nmodes, nmodes), 1, 1, n_inner_tile)) : nothing,
            ep = alloc(backend, Complex{FT}, nbkq * nbk * phs.nmodes * n_inner_tile),
            g = alloc(backend, Complex{FT}, ndata, n_inner_tile),
            tmp = alloc(backend, Complex{FT}, nbkq * nbk * nmodes * n_inner_tile),
            dg = covariant_derivative_of_g ? alloc(backend, Complex{FT}, nbkq * nbk * phs.nmodes * 3 * n_inner_tile) : nothing,
            uk_polar = model.polar_eph.use ? alloc(backend, Complex{FT}, nw * nbk * n_inner_tile) : nothing,
            mmat_buffer = model.polar_eph.use ? alloc(backend, Complex{FT}, nbkq * nbk * n_inner_tile) : nothing,
        )
    end

    # Assemble the engine with maximum-capacity stage-1 outputs and the thread workspaces.
    nrows_max = nw^2 * nmodes * nr_p * (covariant_derivative_of_g ? 3 : 1)
    OuterKEngine(; model, els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq, window_kq,
        energy_conservation_tol, eph_phonon_basis, inner_loop_kq, iks_batch = 1:0,
        backend, epmat, itp_epmat, itp_epmat_R,
        irvecp_mat = _irvec_to_device_matrix(backend, irvec_p, FT), mxks = alloc(backend, FT, 3, n_outer_batch), xkqs,
        wtkqs = to_device_copy(backend, collect(FT, inner_pts.weights)), ck, ckq,
        pair_selection, phonons_per_batch, P_mk,
        ep_kR = alloc(backend, Complex{FT}, ndata, nr_p, n_outer_batch),
        dg_kR = covariant_derivative_of_g ? alloc(backend, Complex{FT}, ndata, nr_p, 3, n_outer_batch) : nothing,
        nband_k_host = Array(els_k.nband), nband_kq_host = inner_loop_kq ? Array(els_kq.nband) : Int[],
        imap_k = sel_k === nothing ? nothing : _indmap_to_device(backend, sel_k),
        imap_kq = sel_kq === nothing ? nothing : _indmap_to_device(backend, sel_kq),
        els_k_classes = [BatchedElectronState(backend, nw, b, n_outer_batch, el_qty; FT) for b in 1:nbk],
        iks_sorted = Int[], slot_in_class = zeros(Int, n_outer_batch), class_first = zeros(Int, nbk),
        class_offsets = zeros(Int, 2, nbk),
        xks_host = zeros(FT, 3, n_outer_batch), xks = alloc(backend, FT, 3, n_outer_batch),
        g_fourier = alloc(backend, Complex{FT}, nrows_max * n_outer_batch),
        n_outer_batch, n_inner_tile, tiles)
end

"""
    stage1!(eng::OuterKEngine, iks_batch)

g(k, R_p) for the outer k points `iks_batch`, rotated by `u_k` and multiplied by `exp(-2πi R_p · x_k)`,
into the first `length(iks_batch)` points of `eng.ep_kR` (and `eng.dg_kR`).

- `eng`: Outer-k engine owning the stage-1 buffers and interpolators.
- `iks_batch`: Indices of the active outer k points, up to the engine's batch capacity.
"""
function stage1!(eng::OuterKEngine, iks_batch::UnitRange{Int})
    isempty(iks_batch) && throw(ArgumentError("stage1! needs a nonempty outer batch"))
    _check_outer_batch(iks_batch, eng.kpts.n, eng.n_outer_batch)
    eng.iks_batch = iks_batch
    ppb = eng.phonons_per_batch
    if ppb !== nothing
        # Phonons per outer batch: those of the q the batch's kept pairs use, and the index of each
        # q in them.
        b, r = divrem(first(iks_batch) - 1, eng.n_outer_batch)
        (r == 0 && last(iks_batch) == min(first(iks_batch) + eng.n_outer_batch - 1, eng.kpts.n)) ||
            throw(ArgumentError("with phonons per outer batch, stage1! takes the batches of the loop"))
        iqs = ppb.iqs_per_batch[b + 1]
        eng.phs = if isempty(iqs)
            # No pair of the batch is kept.
            view(ppb.phs_capacity, 1:0)
        else
            compute_phonon_states_batched(eng.model, Kpoints(eng.qpts.vectors[iqs]), [:e, :u];
                backend = eng.backend, nmodes_kept = eng.phs.nmodes, phs_out = view(ppb.phs_capacity, 1:length(iqs)))
        end
        fill!(ppb.iq_to_ph_dev, 0)
        view(ppb.iq_to_ph_dev, to_device_copy(eng.backend, iqs)) .= Int32(1):Int32(length(iqs))
    end
    # function barrier for the outer-k stage-1 buffers.
    _stage1!(OuterKLoop(), _workspace_fields(eng), iks_batch)
    eng
end

@views function _stage1!(::OuterKLoop, eng_fields, iks_batch)
    # Order the batch by the number of bands at k, so that each band class is a contiguous range,
    # and stage the k coordinates on the backend in that order.
    nk_batch = length(iks_batch)
    (; iks_sorted, slot_in_class, class_first, class_offsets, nband_k_host) = eng_fields
    resize!(iks_sorted, nk_batch)
    nbk_max = length(class_first)
    order = Vector{Int}(undef, nk_batch)
    classes = _class_order!(order, zeros(Int, nbk_max + 2), nband_k_host[iks_batch])
    iks_sorted .= iks_batch[order]
    for (j, ik) in enumerate(iks_sorted)
        eng_fields.xks_host[:, j] .= eng_fields.kpts.vectors[ik]
    end
    xks = eng_fields.xks[:, 1:nk_batch]
    copyto!(eng_fields.xks, eng_fields.xks_host)

    # Build the k+q-convention phase.
    mxks = eng_fields.mxks[:, 1:nk_batch]
    mxks .= xks .* -1
    P_mk = eng_fields.P_mk[:, 1:nk_batch]
    build_fourier_phase!(P_mk, eng_fields.irvecp_mat, mxks)

    for (b, cols) in classes
        class_first[b] = first(cols)
        for (j, s) in enumerate(cols)
            slot_in_class[iks_sorted[s] - first(iks_batch) + 1] = j
        end
    end

    # Fourier-transform R_e, g_ija(R_e, R_p) at each k with i, j Wannier indices and a the atomic
    # displacement, and rotate each band class b by its u_k at width b, g_ina(k, R_p) with n ≤ b the
    # band of k, into its own segment of `ep_kR`, so stage 2 contracts no box padding. The same for
    # the covariant derivative, direction d: dg_ijad(R_e, R_p) -> dg_inad(k, R_p) into `dg_kR`.
    nw, nmodes, nr_p = eng_fields.model.nw, eng_fields.model.nmodes, size(eng_fields.P_mk, 1)
    for (b, cols) in classes
        # Class b holds the k with exactly b bands (`nband_k_host`), into a box of width b.
        copy_batched_electron_states!(eng_fields.els_k_classes[b], eng_fields.els_k, iks_sorted[cols])
    end
    for (itp, out, iout) in ((eng_fields.itp_epmat, eng_fields.ep_kR, 1),
                             (eng_fields.itp_epmat_R, eng_fields.dg_kR, 2))
        itp === nothing && continue
        ndata = itp.parent.ndata
        get_fourier_batched!(reshape_buffer_view(eng_fields.g_fourier, ndata, nk_batch), itp, xks)
        offset = 0
        for (b, cols) in classes
            class_offsets[iout, b] = offset
            dims = iout == 1 ? (nw * b * nmodes, nr_p, length(cols)) : (nw * b * nmodes, nr_p, 3, length(cols))
            out_c = reshape_buffer_view(out, dims...; offset)
            g = reshape_buffer_view(eng_fields.g_fourier, ndata, length(cols); offset = ndata * (first(cols) - 1))
            eph_rotate_kR_batched!(out_c, g, eng_fields.els_k_classes[b].u[:, :, 1:length(cols)];
                                   additional_phase = P_mk[:, cols])
            offset += length(out_c)
        end
    end
    eng_fields
end

"""
    stage2!(f, eng::OuterKEngine, ik, inner_indices; chunk=1)

Call `f(tile)` once on the calculator-ready [`EPBlock`](@ref) of outer k point `ik` and a tile of inner
points: gather the resident k+q states and their phonons (inner k+q points) or solve the k+q states (inner q
points) for the pairs to compute, split the pairs by the band count of their k+q point, compute their
e-ph matrix (and requested derivatives) class by class into the tile's flat item list, and add polar
corrections. `f` is not called if no pair is kept. Call `stage1!` first; `ik` must be in its current batch.

The tile borrows this chunk's buffers: its arrays are valid until that chunk's next `stage2!` or the next
`stage1!`. Consume it in `f` or copy the arrays you need to retain. Calls on different CPU chunks may run
concurrently, but `stage1!` must wait for all of them to finish.

- `ik`: Index into `eng.kpts`, not a batch-local index.
- `inner_indices`: Nonempty contiguous range into `eng.kqpts` (inner k+q) or `eng.qpts` (inner q).
- `chunk`: Independent workspace slot for CPU threading; use 1 for serial calls and GPU calls.
"""
function stage2!(f::F, eng::OuterKEngine, ik::Int, inner_indices::UnitRange{Int}; chunk::Int = 1) where {F}
    inner_pts = eng.inner_loop_kq ? eng.kqpts : eng.qpts
    _check_stage2(eng, ik, eng.iks_batch, inner_indices, chunk, inner_pts.n)
    # function barrier for the concrete resident states, engine buffers, and chunk scratch.
    _stage2!(f, OuterKLoop(), _workspace_fields(eng), _workspace_fields(eng.tiles[chunk]), ik, inner_indices)
    nothing
end

# The k+q side of a tile of resident k+q points in class order: the states of each class (the resident states
# themselves when the tile is one class at the resident width, else gathered at the class's width into
# disjoint segments of the tile's buffers), the weights, and the Fourier phase `exp(2πi R_p · x_{k+q})`.
# `ikqs_c[1:n]` are the gathered points on the host, in class order; `classes` the
# `(nband, range)` of the classes. The band counts, weights and coordinates are those of the resident points, so they depend on
# the tile only, not on the outer k. Fills the k+q rows of the tile's segment table.
@views function _gather_resident_kq!(eng_fields, tile_workspace, ikqs_tile, ikqs_c, n, classes, in_place)
    (; els_kq) = eng_fields
    _fill_kq_segments!(tile_workspace, classes, in_place)
    if in_place
        # One class at the resident width: the resident states as they are, in tile order.
        phase = tile_workspace.P_kq[:, 1:n]
        build_fourier_phase!(phase, eng_fields.irvecp_mat, eng_fields.xkqs[:, ikqs_tile])
        return (; wtqs = eng_fields.wtkqs[ikqs_tile], phase)
    end
    copyto!(tile_workspace.ikqs_dev, 1, ikqs_c, 1, n)
    ikqs_dev = tile_workspace.ikqs_dev[1:n]
    # The band counts of the gathered points, and their bands in one gather per field; the kept points
    # are in the resident set, so the gathers need no bounds check.
    _copy_last_axis!(tile_workspace.els_kq.iband_offset, els_kq.iband_offset, ikqs_dev)
    _copy_last_axis!(tile_workspace.els_kq.nband, els_kq.nband, ikqs_dev)
    seg_dev, _ = _upload_segments!(tile_workspace, length(classes))
    n_slots = sum(((b, r),) -> b * length(r), classes)
    _gather_slots!(tile_workspace.els_kq, els_kq.e, els_kq.u, els_kq.vdiag, tile_workspace.slot_col, seg_dev,
        SEG_KQ_SLOT, SEG_BKQ, ikqs_dev, nothing, els_kq.nband_max, n_slots)
    wtqs = _copy_last_axis!(tile_workspace.wtqs, eng_fields.wtkqs, ikqs_dev)[1:n]
    xkqs = _copy_last_axis!(tile_workspace.xkqs, eng_fields.xkqs, ikqs_dev)[:, 1:n]
    phase = tile_workspace.P_kq[:, 1:n]
    build_fourier_phase!(phase, eng_fields.irvecp_mat, xkqs)
    (; wtqs, phase)
end

# The k+q rows of the tile's segment table: first pair, number of pairs, band count and the segment's
# point and band-slot offsets in the gathered k+q states (0 where the resident states are used in place).
function _fill_kq_segments!(tile_workspace, classes, in_place)
    seg = tile_workspace.seg_host
    segment = (0, 0, 0)
    for (c, (b, r)) in enumerate(classes)
        seg[SEG_PAIR, c] = first(r) - 1
        seg[SEG_NPAIRS, c] = length(r)
        seg[SEG_BKQ, c] = b
        seg[SEG_KQ_PT, c] = in_place ? 0 : segment[1]
        seg[SEG_KQ_SLOT, c] = in_place ? 0 : segment[2]
        segment = _after_segment(segment, length(r), b)
    end
    nothing
end

# Split `n_kept` pairs by the band count `nbands(i)` of their k+q point: the classes, whether every pair has the
# same count (`uniform`: the class order is the pair order, so the sort and the reorder copies are skipped), and whether the tile
# is one class at the resident width with every pair of the tile kept (`in_place`: the resident k+q states are
# used as they are).
function _plan_classes!(tile_workspace, nbands, n_kept, n_tile, nband_max)
    (; class_key, class_order, class_start) = tile_workspace
    key_first = nbands(1)
    uniform = key_first > 0
    for i in 1:n_kept
        class_key[i] = nbands(i)
        uniform &= class_key[i] == key_first
    end
    if uniform
        class_order[1:n_kept] .= 1:n_kept
        classes = [(key_first, 1:n_kept)]
    else
        classes = _class_order!(class_order, class_start, view(class_key, 1:n_kept))
    end
    (classes, length(classes) == 1 && n_kept == n_tile && first(classes)[1] == nband_max, uniform)
end

# The k+q side and the pair order of a whole tile whose pairs are all kept (`tol = Inf`), for every outer k of
# a batch: depends on the tile only, so built once per tile.
function _prepare_shared_kq_tile!(eng_fields, tile_workspace, ikqs_tile)
    n_tile = length(ikqs_tile)
    (; nband_kq_host, els_kq) = eng_fields
    classes, in_place, uniform = _plan_classes!(tile_workspace, i -> nband_kq_host[ikqs_tile[i]], n_tile, n_tile,
                                                els_kq.nband_max)
    # The tile's pairs in class order.
    order = tile_workspace.class_order[1:n_tile]
    ikqs_c = tile_workspace.ikqs_by_class
    for i in 1:n_tile
        ikqs_c[i] = ikqs_tile[order[i]]
    end
    (; classes, in_place, uniform, order, _gather_resident_kq!(eng_fields, tile_workspace, ikqs_tile, ikqs_c, n_tile, classes,
                                                     in_place)...)
end

# `phase` is the Fourier phase of the whole tile, shared by the outer k points of a batch, or
# `nothing` to build it here for the kept pairs (always so when pairs can be skipped).
# The state containers are sliced with an explicit `view`: `@views` covers arrays only.
@views function _stage2!(f::F, ::OuterKLoop, eng_fields, tile_workspace, ik, inner_indices;
        shared = nothing) where {F}
    n_tile = length(inner_indices)
    # The stage-1 output of `ik`: its states at its own band count `b` (a one-point container of box
    # width `b`), and its `(nw b nmodes, nr_p)` slice of `ep_kR` (and `(…, 3)` slice of `dg_kR`).
    b = eng_fields.nband_k_host[ik]
    b == 0 && return nothing
    j = eng_fields.slot_in_class[ik - first(eng_fields.iks_batch) + 1]
    ik_sorted = eng_fields.class_first[b] + j - 1
    els_k = view(eng_fields.els_k_classes[b], j:j)
    nrows, nr_p = eng_fields.model.nw * b * eng_fields.model.nmodes, size(eng_fields.P_mk, 1)
    ep_kR = reshape_buffer_view(eng_fields.ep_kR, nrows, nr_p;
        offset = eng_fields.class_offsets[1, b] + nrows * nr_p * (j - 1))
    dg_kR = eng_fields.dg_kR === nothing ? nothing : reshape_buffer_view(eng_fields.dg_kR, nrows, nr_p, 3;
        offset = eng_fields.class_offsets[2, b] + 3 * nrows * nr_p * (j - 1))
    tol = eng_fields.energy_conservation_tol
    (; ind_kept_kqpairs) = tile_workspace
    (; phs, phonons_per_batch) = eng_fields

    if eng_fields.inner_loop_kq
        # Inner k+q points (run_eph_over_k_and_kq): the k+q states are resident, and each has a
        # state in the window. Find q = (k+q) - k of each pair, and keep the pairs with a process
        # inside the energy-conservation tolerance: their q indices end up in `iqs`, and their k+q
        # indices in `ikqs`, on the host in tile order.
        ikqs_tile = inner_indices
        (; els_kq, nband_kq_host) = eng_fields
        iqs_host, ikqs_host = tile_workspace.iqs, tile_workspace.ikqs
        if shared !== nothing
            # Every pair of the tile is kept, and the classes were prepared for the batch.
            _fill_iqs!(iqs_host, eng_fields.qpts, eng_fields.ck, eng_fields.ckq, ik, first(ikqs_tile), n_tile)
            n_kept = n_tile
        elseif eng_fields.pair_selection === nothing
            # On the host.
            _fill_iqs!(iqs_host, eng_fields.qpts, eng_fields.ck, eng_fields.ckq, ik, first(ikqs_tile), n_tile)
            n_kept = if isinf(tol)
                # Every pair of the tile is kept.
                n_tile
            else
                # Keep the pairs with a process inside the energy-conservation tolerance.
                ind_kept_kqpairs[1:n_tile] .= 1:n_tile
                _kqpairs_conserving_energy!(ind_kept_kqpairs, n_tile, tol,
                    els_k.e, els_k.nband, 1, els_kq.e, nothing, els_kq.nband, ikqs_tile, phs.e, iqs_host)
            end
            n_kept == 0 && return nothing
            for i in 1:n_kept
                jt = n_kept == n_tile ? i : ind_kept_kqpairs[i]
                ikqs_host[i] = ikqs_tile[jt]
                iqs_host[i] = iqs_host[jt]
            end
        else
            # On the device. The kept pairs' q and k+q indices come back for the classes and the `xq`.
            (; ck, pair_selection) = eng_fields
            ω_q = phonons_per_batch === nothing ? phs.e : phonons_per_batch.ω_kept
            n_kept = _select_kqpairs_on_device!(tile_workspace.iqs_dev, pair_selection,
                (ck[1, ik], ck[2, ik], ck[3, ik]), eng_fields.qpts.ngrid, els_k, els_kq,
                first(ikqs_tile), n_tile, ω_q, tol)
            n_kept == 0 && return nothing
            copyto!(iqs_host, 1, tile_workspace.iqs_dev, 1, n_kept)
            copyto!(ikqs_host, 1, pair_selection.ikqs_dev, 1, n_kept)
            all(>(0), iqs_host[1:n_kept]) ||
                throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
        end

        # The pairs by the band count of their k+q point, and the k+q side in that order.
        if shared === nothing
            classes, in_place, uniform = _plan_classes!(tile_workspace, i -> nband_kq_host[ikqs_host[i]], n_kept,
                                                        n_tile, els_kq.nband_max)
            order = tile_workspace.class_order[1:n_kept]
        else
            classes, in_place, uniform, order = shared.classes, shared.in_place, shared.uniform, shared.order
        end
        # The kept pairs' q and k+q indices in class order (the pair order if every pair has the same band count).
        if uniform
            iqs_c, ikqs_c = iqs_host, ikqs_host
        else
            iqs_c, ikqs_c = tile_workspace.iqs_by_class, tile_workspace.ikqs_by_class
            for i in 1:n_kept
                iqs_c[i] = iqs_host[order[i]]
                ikqs_c[i] = ikqs_host[order[i]]
            end
        end
        gathered = shared !== nothing ? shared :
            _gather_resident_kq!(eng_fields, tile_workspace, ikqs_tile, ikqs_c, n_kept, classes, in_place)
        # The q indices in class order on the backend, unless the device selection left them in that order.
        (uniform && eng_fields.pair_selection !== nothing) ||
            copyto!(tile_workspace.iqs_dev, 1, iqs_c, 1, n_kept)
        ikq = in_place ? ikqs_tile : tile_workspace.ikqs_dev[1:n_kept]
        wtqs, phase = gathered.wtqs, gathered.phase
        els_kq_tile = in_place ? view(els_kq, ikqs_tile) : tile_workspace.els_kq

        # Gather the kept pairs' phonons, by their index in `phs`: the q index, or the index in the
        # batch's phonons when those are built per outer batch.
        iqs = tile_workspace.iqs_dev[1:n_kept]
        iqs_ph = if phonons_per_batch === nothing
            # Every q is resident.
            iqs
        else
            # Phonons per outer batch: the kept pairs' indices into the batch's.
            (; iqs_ph_dev, iq_to_ph_dev) = phonons_per_batch
            iqs_ph = iqs_ph_dev[1:n_kept] .= getindex.(Ref(iq_to_ph_dev), iqs)   # a gather by an index array would bounds-check with a sync
            # Index 0 (a kept q missing from the batch's phonons) must not reach the rotation kernel.
            minimum(iqs_ph) > 0 || error("a kept pair's q is not among its outer batch's phonons")
            iqs_ph
        end
        phs_tile = view(copy_batched_phonon_states!(tile_workspace.phs, phs, iqs_ph), 1:n_kept)
        xqs = eng_fields.qpts.vectors[iqs_c[1:n_kept]]
        iq = iqs
    else
        # Inner q points (run_eph_over_k_and_q): the phonons are resident. Solve the electron
        # bands at k+q for this tile.
        iqs_tile = inner_indices
        for (jt, iq) in enumerate(iqs_tile)
            tile_workspace.kqs[jt] = eng_fields.kpts.vectors[ik] + eng_fields.qpts.vectors[iq]
        end
        bands_kq = solve_electron_bands_batched(tile_workspace.itp_el_ham, tile_workspace.hk,
            eng_fields.model, tile_workspace.kqs[1:n_tile], eng_fields.window_kq;
            eigenvectors = tile_workspace.els_kq.u !== nothing)

        # Keep the pairs whose k+q has a state in the window, and of those the pairs with a process
        # inside the energy-conservation tolerance.
        nbands_kq = Array(bands_kq.nband)
        n_kept = _kqpairs_in_window!(ind_kept_kqpairs, nbands_kq)
        if isfinite(tol)
            n_kept = _kqpairs_conserving_energy!(ind_kept_kqpairs, n_kept, tol,
                els_k.e, els_k.nband, 1, bands_kq.E, bands_kq.offset, nbands_kq, 1:n_tile, phs.e, iqs_tile)
        end
        n_kept == 0 && return nothing
        ikq = nothing

        # The pairs by the band count of their k+q point: positions `pos_c` in the solved tile, class order.
        classes, _, uniform = _plan_classes!(tile_workspace, i -> nbands_kq[ind_kept_kqpairs[i]], n_kept, n_tile, 0)
        in_place = false
        order = tile_workspace.class_order[1:n_kept]
        pos_c, iqs_c = tile_workspace.pos_by_class, tile_workspace.iqs_by_class
        for i in 1:n_kept
            pos_c[i] = ind_kept_kqpairs[order[i]]
            iqs_c[i] = iqs_tile[pos_c[i]]
        end
        # The kept q points are in the tile, so the gathers need no bounds check.
        copyto!(tile_workspace.iqs_dev, 1, iqs_c, 1, n_kept)
        iqs = tile_workspace.iqs_dev[1:n_kept]
        # The classes' window bands at k+q into the tile's segments; each box is exactly the class's band count.
        pos = length(classes) == 1 && n_kept == n_tile ? (1:n_tile) : begin
            copyto!(tile_workspace.pos_dev, 1, pos_c, 1, n_kept)
            tile_workspace.pos_dev[1:n_kept]
        end
        # The band counts of the class-ordered points and their bands, in one gather per field.
        _fill_kq_segments!(tile_workspace, classes, false)
        n_slots = sum(((nb, r),) -> nb * length(r), classes)
        nb_pts = view(tile_workspace.els_kq.nband, 1:n_kept)
        nb_pts .= getindex.(Ref(bands_kq.nband), pos)
        view(tile_workspace.els_kq.iband_offset, 1:n_kept) .= ifelse.(nb_pts .> 0, getindex.(Ref(bands_kq.offset), pos), 0)
        seg_dev, _ = _upload_segments!(tile_workspace, length(classes))
        _gather_slots!(tile_workspace.els_kq, bands_kq.E, bands_kq.U, nothing, tile_workspace.slot_col, seg_dev,
            SEG_KQ_SLOT, SEG_BKQ, pos, bands_kq.offset, eng_fields.model.nw, n_slots)
        els_kq_tile = tile_workspace.els_kq
        phs_tile = view(copy_batched_phonon_states!(tile_workspace.phs, phs, iqs), 1:n_kept)
        wtqs = _copy_last_axis!(tile_workspace.wtqs, eng_fields.wtkqs, iqs)[1:n_kept]
        xkqs = _copy_last_axis!(tile_workspace.xkqs, eng_fields.xkqs, iqs)[:, 1:n_kept]
        xqs = eng_fields.qpts.vectors[iqs_c[1:n_kept]]
        iq = iqs

        # Stage 1 includes exp(-2πi R_p·k), so stage 2 needs the phase at k+q, not q.
        xkqs .+= eng_fields.xks[:, ik_sorted]
        phase = tile_workspace.P_kq[:, 1:n_kept]
        build_fourier_phase!(phase, eng_fields.irvecp_mat, xkqs)
    end

    # The class segments of the tile: the e-ph matrix, the segment table, and the tile handed over.
    n_items = _compute_eph_for_classes!(eng_fields, tile_workspace, ep_kR, dg_kR, phase, classes, els_kq_tile,
        in_place, els_k, phs_tile, n_kept)
    seg = tile_workspace.seg_host
    item = 0
    for c in eachindex(classes)
        # the k+q rows were filled with the k+q side; the one shared k state starts at 0
        seg[SEG_ITEM, c] = item
        seg[SEG_BK, c] = b
        seg[SEG_K_PT, c] = 0
        seg[SEG_K_SLOT, c] = 0
        seg[SEG_K_SHARED, c] = 1
        item += seg[SEG_BKQ, c] * b * phs_tile.nmodes * seg[SEG_NPAIRS, c]
    end
    seg_dev, seg_host = _upload_segments!(tile_workspace, length(classes))
    tile = EPBlock{OuterKLoop}(; ep = reshape_buffer_view(tile_workspace.ep, n_items),
        dg = tile_workspace.dg === nothing ? nothing : reshape_buffer_view(tile_workspace.dg, n_items, 3),
        els_k, els_kq = els_kq_tile, phs = phs_tile, wtk = eng_fields.kpts.weights[ik], wtq = wtqs,
        xk = eng_fields.kpts.vectors[ik], xq = xqs, ik, ikq, iq, seg = seg_dev, seg_host,
        imap_k = eng_fields.imap_k, imap_kq = eng_fields.imap_kq)
    f(tile)
    nothing
end

# The e-ph matrix of a tile, class by class: one Fourier transform over R_p of all pairs into `g`
# (`ep_kR · phase`, and per direction of `dg_kR`), then per class the rotation by the class's k+q
# eigenvectors and the phonon basis into the class's items (`ep` and `dg`), the polar term added to `ep`.
# `classes` are the `(nband_kq, range of pairs)`, `els_kq_tile` the tile's k+q states (class segments, or the resident states if `in_place`), `els_k` the one
# outer k. Returns the number of items.
@views function _compute_eph_for_classes!(eng_fields, tile_workspace, ep_kR, dg_kR, phase, classes, els_kq_tile,
        in_place, els_k, phs, n_pairs)
    (; nw, nmodes) = eng_fields.model
    nm, nbk = phs.nmodes, els_k.nband_max
    ndata = size(ep_kR, 1)
    g = reshape_buffer_view(tile_workspace.g, ndata, n_pairs)
    u_ph_all = tile_workspace.u_ph_id === nothing ? phs.u : tile_workspace.u_ph_id
    n_items = sum(((nb, r),) -> nb * nbk * nm * length(r), classes)
    for d in 0:(dg_kR === nothing ? 0 : 3)
        # d = 0: the e-ph matrix; d = 1:3: its covariant derivative along direction d.
        mul!(g, d == 0 ? ep_kR : dg_kR[:, :, d], phase)
        item, segment = 0, (0, 0, 0)
        for (c, (nb, r)) in enumerate(classes)
            nc = length(r)
            u_ph = u_ph_all[:, :, r]
            g_c = reshape_buffer_view(g, ndata, nc; offset = ndata * (first(r) - 1))
            tmp = reshape_buffer_view(tile_workspace.tmp, nb, nbk * size(u_ph, 1), nc)
            ukq = in_place ? els_kq_tile.u : reshape_view_batched_electron_states(els_kq_tile, nb, nc; offset = segment).u
            segment = _after_segment(segment, nc, nb)
            len = nb * nbk * nm * nc
            if d == 0
                ep_c = reshape_buffer_view(tile_workspace.ep, nb, nbk, nm, nc; offset = item)
                eph_apply_rotations!(ep_c, reshape(g_c, nw, nbk, size(u_ph, 1), nc), ukq, u_ph, tmp)
                eng_fields.model.polar_eph.use && eph_engine_add_longrange!(ep_c, phs.eph_dipole_coeff[:, r], els_k.u, ukq,
                    tile_workspace, eng_fields.model)
            else
                dg_c = reshape_buffer_view(tile_workspace.dg, nb, nbk, nm, nc; offset = (d - 1) * n_items + item)
                eph_apply_rotations!(dg_c, reshape(g_c, nw, nbk, size(u_ph, 1), nc), ukq, u_ph, tmp)
            end
            item += len
        end
    end
    n_items
end


# ---- OuterQEngine ----------------------------------------------------------------------------

function engine_bytes(::Type{OuterQEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk,
        el_qty, ph_qty, precompute_el_kq, eph_phonon_basis) where {FT}
    (; nw, nmodes) = model
    cx, rl, iz = sizeof(Complex{FT}), sizeof(FT), sizeof(Int)
    nr_e = length(model.epmat.irvec_next)
    nr_p = length(model.epmat.irvec)
    ndata = nw^2 * nmodes
    nbkq = precompute_el_kq ? nband_max_kq : nw
    persistent =
        cx * length(model.epmat.op_r) +                         # epmat
        rl * 3 * (nr_e + nr_p) +                              # R-vector matrices
        cx * ndata * nr_e +                                     # interpolator output
        (precompute_el_kq ? 0 : cx * length(model.el_ham.op_r)) +  # el_ham
        rl * nk +                                               # wtks
        iz * nband_max_k * nk                                   # imap_k
    per_outer =
        cx * ndata * nr_e * (eph_phonon_basis == :cartesian ? 2 : 3) +   # g_q, ep_Rq (+ g_rot)
        cx * nr_p +                                             # Fourier phase
        rl * 3                                                  # xqs
    per_pair =
        cx * nbkq * nband_max_k * nmodes +                      # ep
        cx * ndata + cx * nbkq * nw * nmodes + cx * nw * nband_max_k * nmodes +   # g, tmp, uk_rep
        _electron_state_bytes(FT, nw, nband_max_k, el_qty) +    # k tile
        _electron_state_bytes(FT, nw, nbkq, el_qty) +           # k+q tile
        cx * nr_e +                                             # stage-2 Fourier phase
        5iz + rl + 3rl +                                        # kept pairs, iks, ikqs (+ on the backend), wtks, xks
        (precompute_el_kq ? 0 : cx * length(model.el_ham.irvec)) +   # k+q Fourier phase
        (precompute_el_kq ? 0 : cx * nw^2 * 3) +                # k+q Hamiltonian and its eigensolve
        (model.polar_eph.use ? cx * (nw * nband_max_k + nbkq * nband_max_k) : 0)   # polar scratch
    (; persistent, per_outer, per_pair)
end

"""
    OuterQEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kwargs...)

Build the outer-q run's buffers for a model with `epmat_outer_momentum = "ph"`.
The capacity, quantity, phonon-basis, and energy-conservation arguments have the meanings
documented for `OuterKEngine`. With `precompute_el_kq = true` the k+q states are gathered from the
resident container `els_kq`; with `false` (`els_kq = nothing`) they are solved per tile. It defaults
to `els_kq !== nothing`; a contradicting value is an `ArgumentError`.
"""
function OuterQEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, qpts,
        precompute_el_kq = els_kq !== nothing, n_outer_batch, n_inner_tile, nchunks, eph_phonon_basis,
        kqpts = nothing,
        sel_k = nothing, sel_kq = nothing, window_kq = (-Inf, Inf),
        energy_conservation_tol = Inf) where {FT}
    # Validate the model layout and determine the electron band-box dimensions.
    (; nw, nmodes) = model
    _require_epmat_layout(OuterQLoop(), model)
    precompute_el_kq == (els_kq !== nothing) || throw(ArgumentError(
        "precompute_el_kq = true takes the resident k+q states els_kq, and false takes els_kq = nothing"))
    nbk = els_k.nband_max
    nbkq = precompute_el_kq ? els_kq.nband_max : nw

    # Prepare the stage-1 Fourier interpolator and its read-only stage-2 outputs g(R_e, q).
    epmat = to_device(backend, model.epmat)
    irvec_e = model.epmat.irvec_next
    nr_e = length(irvec_e)
    ndata = nw^2 * nmodes
    itp_epmat = BatchedWannierInterpolator(epmat; backend, batch_size = n_outer_batch)
    el_ham = precompute_el_kq ? nothing : to_device(backend, model.el_ham)

    # Allocate independent k-tile states, interpolators, and scratch for each thread chunk.
    tiles = map(1:nchunks) do _
        # Each chunk owns its Fourier phase; the stage-1 output is read-only in stage 2.
        OuterQTileWorkspace(;
            ind_kept_kqpairs = Vector{Int}(undef, n_inner_tile),
            iks = Vector{Int}(undef, n_inner_tile),
            ikqs = Vector{Int}(undef, n_inner_tile),    # 0: k+q absent from the precomputed states
            iks_dev = alloc(backend, Int, n_inner_tile),
            ikqs_dev = alloc(backend, Int, n_inner_tile),
            els_k = BatchedElectronState(backend, nw, nbk, n_inner_tile, el_qty; FT),
            els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
            wtks = alloc(backend, FT, n_inner_tile),
            xks = Vector{Vec3{FT}}(undef, n_inner_tile),
            P_k = alloc(backend, Complex{FT}, nr_e, n_inner_tile),
            itp_el_ham = precompute_el_kq ? nothing :
                BatchedWannierInterpolator(el_ham; backend, batch_size = n_inner_tile),
            hk = precompute_el_kq ? nothing : alloc(backend, Complex{FT}, nw^2, n_inner_tile),
            kqs = Vector{Vec3{FT}}(undef, n_inner_tile),
            ep = alloc(backend, Complex{FT}, nbkq * nbk * nmodes * n_inner_tile),
            g = alloc(backend, Complex{FT}, ndata, n_inner_tile),
            tmp = alloc(backend, Complex{FT}, nbkq * nw * nmodes * n_inner_tile),
            uk_rep = alloc(backend, Complex{FT}, nw, nbk, nmodes * n_inner_tile),
            uk_polar = model.polar_eph.use ? alloc(backend, Complex{FT}, nw * nbk * n_inner_tile) : nothing,
            mmat_buffer = model.polar_eph.use ? alloc(backend, Complex{FT}, nbkq * nbk * n_inner_tile) : nothing,
            seg_host = zeros(Int, SEG_ROWS, 1), seg_dev = alloc(backend, Int, SEG_ROWS, 1),
        )
    end

    # Assemble the engine with maximum-capacity stage-1 outputs and coordinate buffers.
    OuterQEngine(; model, els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq, window_kq,
        imap_k = sel_k === nothing ? nothing : _indmap_to_device(backend, sel_k),
        imap_kq = sel_kq === nothing ? nothing : _indmap_to_device(backend, sel_kq),
        energy_conservation_tol, eph_phonon_basis, precompute_el_kq, iqs_batch = 1:0, backend, epmat,
        itp_epmat, irvece_mat = _irvec_to_device_matrix(backend, irvec_e, FT),
        g_q = alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch),
        g_rot = eph_phonon_basis == :cartesian ? nothing : alloc(backend, Complex{FT}, nw^2, nr_e, nmodes, n_outer_batch),
        ep_Rq = alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch),
        wtks = to_device_copy(backend, collect(FT, kpts.weights)),
        xqs_host = zeros(FT, 3, n_outer_batch), xqs = alloc(backend, FT, 3, n_outer_batch),
        n_outer_batch, n_inner_tile, tiles)
end

"""
    stage1!(eng::OuterQEngine, iqs_batch)

g(R_e, q) for the outer q points `iqs_batch` in `eng.eph_phonon_basis` (`:eigenmode`
rotates the modes by `eng.phs.u`, `:cartesian` leaves them), into `eng.ep_Rq[:, :, 1:length(iqs_batch)]`.

Both engines contract the outer R and rotate the outer states in stage 1. Here the rotation
mixes phonon modes only. Outer k instead rotates electronic bands, folds in the k+q-convention
phase, and optionally repeats the contraction and rotation for the three derivative components.

- `eng`: Outer-q engine owning the stage-1 buffers and Fourier interpolator.
- `iqs_batch`: Indices of the active outer q points, up to the engine's batch capacity.
"""
function stage1!(eng::OuterQEngine, iqs_batch::UnitRange{Int})
    isempty(iqs_batch) && throw(ArgumentError("stage1! needs a nonempty outer batch"))
    _check_outer_batch(iqs_batch, eng.qpts.n, eng.n_outer_batch)
    eng.iqs_batch = iqs_batch
    # function barrier for the outer-q stage-1 buffers.
    _stage1!(OuterQLoop(), _workspace_fields(eng), iqs_batch)
    eng
end

@views function _stage1!(::OuterQLoop, eng_fields, iqs_batch)
    # Stage the active q coordinates on the backend.
    nq_batch = length(iqs_batch)
    for (j, iq) in enumerate(iqs_batch)
        eng_fields.xqs_host[:, j] .= eng_fields.qpts.vectors[iq]
    end
    xqs = eng_fields.xqs[:, 1:nq_batch]
    copyto!(eng_fields.xqs, eng_fields.xqs_host)

    # Fourier-transform R_p for the active outer batch: g_ija(R_e, R_p) -> g_ija(R_e, q).
    ndata, nr_e = size(eng_fields.ep_Rq, 1), size(eng_fields.ep_Rq, 2)
    g = eng_fields.g_q[:, :, 1:nq_batch]
    get_fourier_batched!(reshape(g, ndata * nr_e, nq_batch), eng_fields.itp_epmat, xqs)

    # Rotate into the phonon eigenmodes, g_ija(R_e, q) -> g_ijν(R_e, q), or keep the cartesian
    # components.
    ep = eng_fields.ep_Rq[:, :, 1:nq_batch]
    if eng_fields.eph_phonon_basis == :cartesian
        # Cartesian basis: the displacement components are the modes.
        ep .= g
    else
        # Eigenmode basis: ep[ij, ν, R_e, q] = Σ_a g[ij, a, R_e, q] u_ph[a, ν, q], as one batched
        # GEMM over q on the (ij, R_e) × a layout.
        nw² = size(eng_fields.g_rot, 1)
        nmodes = size(eng_fields.g_rot, 3)
        g_rot = eng_fields.g_rot[:, :, :, 1:nq_batch]
        permutedims!(g_rot, reshape(g, nw², nmodes, nr_e, nq_batch), (1, 3, 2, 4))
        ep_rot = reshape(eng_fields.g_q[:, :, 1:nq_batch], nw² * nr_e, nmodes, nq_batch)   # g is consumed
        batched_gemm!('N', 'N', reshape(g_rot, nw² * nr_e, nmodes, nq_batch),
                      eng_fields.phs.u[:, :, iqs_batch], ep_rot)
        permutedims!(reshape(ep, nw², nmodes, nr_e, nq_batch), reshape(ep_rot, nw², nr_e, nmodes, nq_batch), (1, 3, 2, 4))
    end
    eng_fields
end

"""
    stage2!(f, eng::OuterQEngine, iq, iks_tile; chunk=1)

Call `f(tile)` once on the calculator-ready [`EPBlock`](@ref) of outer q point `iq` and a tile of k points: solve
the k+q states or look them up in the precomputed container, gather the states of the pairs to
compute, compute their e-ph matrix, and add polar corrections. `f` is not called if no pair is kept.
Call `stage1!` first; `iq` must be in its current batch.

The tile borrows this chunk's buffers: its arrays are valid until that chunk's next `stage2!`
or the next `stage1!`. Consume it in `f` or copy the arrays you need to retain. Calls on
different CPU chunks may run concurrently, but `stage1!` must wait for all of them to finish.

- `iq`: Index into `eng.qpts`, not a batch-local index.
- `iks_tile`: Nonempty contiguous range into `eng.kpts`.
- `chunk`: Independent workspace slot for CPU threading; use 1 for serial calls and GPU calls.
"""
function stage2!(f::F, eng::OuterQEngine, iq::Int, iks_tile::UnitRange{Int}; chunk::Int = 1) where {F}
    _check_stage2(eng, iq, eng.iqs_batch, iks_tile, chunk, eng.kpts.n)
    # function barrier for the concrete resident states, engine buffers, and chunk scratch.
    _stage2!(f, OuterQLoop(), _workspace_fields(eng), _workspace_fields(eng.tiles[chunk]), iq, iks_tile)
    nothing
end

# The state containers are sliced with an explicit `view`: `@views` covers arrays only.
@views function _stage2!(f::F, ::OuterQLoop, eng_fields, tile_workspace, iq, iks_tile) where {F}
    n_tile = length(iks_tile)
    tol = eng_fields.energy_conservation_tol
    (; ind_kept_kqpairs) = tile_workspace
    (; els_k, phs) = eng_fields
    for (j, ik) in enumerate(iks_tile)
        tile_workspace.kqs[j] = eng_fields.kpts.vectors[ik] + eng_fields.qpts.vectors[iq]
    end

    if eng_fields.precompute_el_kq
        # k+q precomputed: look up each k+q in the resident states. One with no state in the window
        # is absent from them (index 0).
        (; els_kq) = eng_fields
        ikqs_host = tile_workspace.ikqs
        for j in 1:n_tile
            ikqs_host[j] = something(xk_to_ik_unsafe(tile_workspace.kqs[j], eng_fields.kqpts), 0)
        end

        # Keep the pairs whose k+q has a state in the window, and of those the pairs with a process
        # inside the energy-conservation tolerance.
        n_kept = _kqpairs_in_window!(ind_kept_kqpairs, ikqs_host[1:n_tile])
        if isfinite(tol)
            n_kept = _kqpairs_conserving_energy!(ind_kept_kqpairs, n_kept, tol,
                els_k.e, els_k.nband, iks_tile, els_kq.e, nothing, els_kq.nband, ikqs_host, phs.e, iq)
        end
        n_kept == 0 && return nothing

        # Gather the kept pairs' k+q states, through their indices on the backend. The lookup
        # returned indices of the resident states, so the gather needs no bounds check.
        for (i, j) in enumerate(ind_kept_kqpairs[1:n_kept])
            ikqs_host[i] = ikqs_host[j]
        end
        ikqs = ikqs_host[1:n_kept]
        copyto!(tile_workspace.ikqs_dev, 1, ikqs_host, 1, n_kept)
        els_kq_block = view(copy_batched_electron_states!(tile_workspace.els_kq, els_kq,
            tile_workspace.ikqs_dev[1:n_kept]), 1:n_kept)
    else
        # k+q solved per tile: solve the electron bands at k+q for this q and k tile.
        bands_kq = solve_electron_bands_batched(tile_workspace.itp_el_ham, tile_workspace.hk,
            eng_fields.model, tile_workspace.kqs[1:n_tile], eng_fields.window_kq;
            eigenvectors = tile_workspace.els_kq.u !== nothing)

        # Keep the pairs whose k+q has a state in the window, and of those the pairs with a process
        # inside the energy-conservation tolerance.
        nbands_kq = Array(bands_kq.nband)
        n_kept = _kqpairs_in_window!(ind_kept_kqpairs, nbands_kq)
        if isfinite(tol)
            n_kept = _kqpairs_conserving_energy!(ind_kept_kqpairs, n_kept, tol,
                els_k.e, els_k.nband, iks_tile, bands_kq.E, bands_kq.offset, nbands_kq, 1:n_tile, phs.e, iq)
        end
        n_kept == 0 && return nothing

        # Move the kept pairs' window bands at k+q into the tile.
        els_kq_block = copy_window_bands!(tile_workspace.els_kq, bands_kq,
            n_kept == n_tile ? (1:n_tile) : ind_kept_kqpairs[1:n_kept]; nband_host = nbands_kq)
        ikqs = nothing
    end

    if n_kept == n_tile
        # Every pair is kept: the resident k states, weights and coordinates as they are. The
        # resident containers are read-only for the whole run, so the block may alias them.
        iks = iks_tile
        els_k_block = view(els_k, iks_tile)
        wtks = eng_fields.wtks[iks_tile]
        xks = eng_fields.kpts.vectors[iks_tile]
    else
        # Some pairs are skipped: gather the kept pairs' k states, weights and coordinates. The
        # kept k points are in the tile, so the gather needs no bounds check.
        for (i, j) in enumerate(ind_kept_kqpairs[1:n_kept])
            tile_workspace.iks[i] = iks_tile[j]
            tile_workspace.xks[i] = eng_fields.kpts.vectors[iks_tile[j]]
        end
        iks = tile_workspace.iks[1:n_kept]
        copyto!(tile_workspace.iks_dev, 1, tile_workspace.iks, 1, n_kept)
        iks_dev = tile_workspace.iks_dev[1:n_kept]
        els_k_block = view(copy_batched_electron_states!(tile_workspace.els_k, els_k, iks_dev), 1:n_kept)
        wtks = _copy_last_axis!(tile_workspace.wtks, eng_fields.wtks, iks_dev)[1:n_kept]
        xks = tile_workspace.xks[1:n_kept]
    end

    # function barrier for the concrete types of the kept pairs' states and indices.
    _compute_eph_for_pairs!(f, OuterQLoop(), eng_fields, tile_workspace, els_k_block, els_kq_block,
        iks, ikqs, iq, wtks, xks)
    nothing
end

# The outer-q tile is one segment at the box widths of its states (the k+q box is the largest window of
# the tile): the band padding of a smaller window is items of the list, which the rule of this engine
# excludes only on the outer-k side.
@views function _compute_eph_for_pairs!(f::F, ::OuterQLoop, eng_fields, tile_workspace, els_k, els_kq, iks, ikqs, iq,
        wtks, xks) where {F}
    xkmat = _kpoints_to_device_matrix(eng_fields.backend, xks)
    nbkq, nbk, nmodes = els_kq.nband_max, els_k.nband_max, eng_fields.phs.nmodes
    npairs = els_k.nk
    ep = reshape_buffer_view(tile_workspace.ep, nbkq, nbk, nmodes, npairs)

    # Fourier-transform R_e of this q's stage-1 output at the tile's k points:
    # g_ijν(R_e, q) -> g_ijν(k, q).
    iq_batch = iq - first(eng_fields.iqs_batch) + 1
    nw = els_k.nw
    g = tile_workspace.g[:, 1:npairs]
    phase = tile_workspace.P_k[:, 1:npairs]
    _fourier_batched!(g, eng_fields.ep_Rq[:, :, iq_batch], phase, eng_fields.irvece_mat, xkmat)

    # Rotate by u_k and u_{k+q}: g_ijν(k, q) -> g_mnν(k, q), with m, n the bands of k+q and k.
    tmp = reshape_buffer_view(tile_workspace.tmp, nbkq, nw * nmodes, npairs)
    uk_rep = reshape_buffer_view(tile_workspace.uk_rep, nw, nbk, nmodes * npairs)
    eph_apply_rotations_rqkq!(ep, g, els_k.u, els_kq.u, tmp, uk_rep)

    phs = view(eng_fields.phs, iq:iq)
    if eng_fields.model.polar_eph.use
        eph_engine_add_longrange!(ep, phs.eph_dipole_coeff, els_k.u, els_kq.u, tile_workspace, eng_fields.model)
    end
    seg = tile_workspace.seg_host
    seg[:, 1] .= (0, 0, npairs, nbkq, nbk, 0, 0, 0, 0, 0)
    seg_dev, seg_host = _upload_segments!(tile_workspace, 1)
    f(EPBlock{OuterQLoop}(; ep = reshape_buffer_view(tile_workspace.ep, length(ep)), dg = nothing,
        els_k, els_kq, phs, wtk = wtks, wtq = eng_fields.qpts.weights[iq], xk = xks, xkmat,
        xq = eng_fields.qpts.vectors[iq], ik = iks, ikq = ikqs, iq, seg = seg_dev, seg_host,
        imap_k = eng_fields.imap_k, imap_kq = eng_fields.imap_kq))
    nothing
end


# ---- Both orders -----------------------------------------------------------------------------

# A nonempty outer batch of either order: inside the `npoints` outer points and the capacity.
function _check_outer_batch(outer_batch, npoints, capacity)
    (first(outer_batch) >= 1 && last(outer_batch) <= npoints) ||
        throw(BoundsError(1:npoints, outer_batch))
    length(outer_batch) <= capacity ||
        throw(ArgumentError("outer batch exceeds the engine's capacity $capacity"))
    nothing
end

function _check_stage2(eng, outer_index, outer_batch, inner_indices, chunk, n_inner)
    outer_index ∈ outer_batch || throw(ArgumentError(
        "outer point $outer_index is not in the current stage1! batch $outer_batch"))
    1 <= chunk <= length(eng.tiles) || throw(BoundsError(eng.tiles, chunk))
    isempty(inner_indices) && throw(ArgumentError("stage2! needs a nonempty inner tile"))
    (first(inner_indices) >= 1 && last(inner_indices) <= n_inner) || throw(BoundsError(1:n_inner, inner_indices))
    length(inner_indices) <= eng.n_inner_tile || throw(ArgumentError(
        "inner tile exceeds the engine's capacity $(eng.n_inner_tile)"))
    nothing
end

"""
    OuterKContext(eng::OuterKEngine; iks_batch = eng.iks_batch, chunk = 1)
    OuterQContext(eng::OuterQEngine; iqs_batch = eng.iqs_batch, chunk = 1)

The calculator context for an outer batch and the workspace slot `chunk`. The default batch is the
engine's current stage-1 batch, so call `stage1!` first or pass the batch; the drivers pass it, so
that `calculator_begin_batch!` runs before the batch's stage 1.
"""
function OuterKContext(eng::OuterKEngine; iks_batch::UnitRange{Int} = eng.iks_batch, chunk::Int = 1)
    isempty(iks_batch) && throw(ArgumentError(
        "the engine's OuterKContext needs an outer batch: call stage1! first or pass `iks_batch`"))
    _check_outer_batch(iks_batch, eng.kpts.n, eng.n_outer_batch)
    1 <= chunk <= length(eng.tiles) || throw(BoundsError(eng.tiles, chunk))
    OuterKContext(eng.backend, iks_batch, chunk)
end

function OuterQContext(eng::OuterQEngine; iqs_batch::UnitRange{Int} = eng.iqs_batch, chunk::Int = 1)
    isempty(iqs_batch) && throw(ArgumentError(
        "the engine's OuterQContext needs an outer batch: call stage1! first or pass `iqs_batch`"))
    _check_outer_batch(iqs_batch, eng.qpts.n, eng.n_outer_batch)
    1 <= chunk <= length(eng.tiles) || throw(BoundsError(eng.tiles, chunk))
    OuterQContext(eng.backend, iqs_batch, chunk)
end

# The pairs of a tile whose k+q has a state in the window: writes their positions in the tile to
# the front of `ind_kept_kqpairs`, in order, and returns their number. `in_window[j]` is nonzero
# for such a pair: its number of window bands at k+q, or its index in the precomputed k+q states.
function _kqpairs_in_window!(ind_kept_kqpairs, in_window)
    n_kept = 0
    for (j, x) in enumerate(in_window)
        x == 0 && continue
        ind_kept_kqpairs[n_kept += 1] = j
    end
    n_kept
end

# Of the pairs `ind_kept_kqpairs[1:n_kept]`, keep those with a process inside the
# energy-conservation tolerance, |e_k - e_{k+q} ± ω_q| <= energy_conservation_tol for some bands
# and mode: compacts the list in place and returns the new number. Pair `j` of the tile has its k
# energies in column `iks[j]` of `e_k`, rows `1:nbands_k[iks[j]]`; its k+q energies in column
# `ikqs[j]` of `e_kq`, rows `offsets_kq[ikqs[j]] .+ (1:nbands_kq[ikqs[j]])` (from row 1 when
# `offsets_kq === nothing`); and its phonon energies in column `iqs[j]` of `ω_q`. The side shared by
# the tile passes its one index, `iks::Int` or `iqs::Int`.
@views function _kqpairs_conserving_energy!(ind_kept_kqpairs, n_kept, energy_conservation_tol,
        e_k, nbands_k, iks, e_kq, offsets_kq, nbands_kq, ikqs, ω_q, iqs)
    n_conserving = 0
    for j in ind_kept_kqpairs[1:n_kept]
        ik = iks isa Integer ? iks : iks[j]
        ikq = ikqs[j]
        iq = iqs isa Integer ? iqs : iqs[j]
        offset_kq = offsets_kq === nothing ? 0 : offsets_kq[ikq]
        e_nks = e_k[1:nbands_k[ik], ik]
        e_mkqs = e_kq[offset_kq .+ (1:nbands_kq[ikq]), ikq]
        any(abs(e_nk - e_mkq - sign_ph * ω) <= energy_conservation_tol
            for ω in ω_q[:, iq], e_mkq in e_mkqs, e_nk in e_nks, sign_ph in (-1, 1)) || continue
        ind_kept_kqpairs[n_conserving += 1] = j
    end
    n_conserving
end

"""
    _select_kqpairs_on_device!(iqs_dev, selection::DevicePairSelection, k_int, ngrid, els_k, els_kq,
                               ikq_first, n_tile, ω_q, tol) -> n_kept

The device counterpart of `_fill_iqs!` followed by `_kqpairs_conserving_energy!`, for one outer k
(`els_k` its one-point view, `k_int` its reduced grid coordinates) and the k+q tile
`ikq_first .+ (0:n_tile-1)` of `els_kq`, on the q grid `ngrid`: the q index of each pair from the
same integer grid hash, and the same test `|e_k - e_{k+q} ± ω_q| <= tol` over the window bands and
every mode of `ω_q` `(nmodes, nq)`. Writes the kept pairs' q and k+q indices, in tile order, to the
front of `iqs_dev` and `selection.ikqs_dev`, and returns their number, read back with one
device-to-host copy. A pair whose q is missing from the q set is kept with q index 0, for the caller
to refuse. Defined by the CUDA extension.
"""
function _select_kqpairs_on_device! end

"""
    _mark_conserving_qs!(q_marks, selection, k_int, ngrid, els_k, els_kq, ikq_first, n_tile, ω_q, tol)

Set `q_marks[iq] = true` for the q of every pair `_select_kqpairs_on_device!` keeps with the same
arguments. Defined by the CUDA extension.
"""
function _mark_conserving_qs! end

"""
    eph_engine_add_longrange!(ep, coeffs, uk_src, ukq, tile_workspace, model)

Add the long-range term on `ep` `(nbandkq, nbandk, nmodes, n)` after the two Fourier transforms, for either
loop order: the polar dipole term `coeff[ν] · u_{k+q}' u_k` of a polar model (unscreened, as
`epstate_compute_eph_dipole!`), with the coefficients `coeffs` `(nmodes, n)` in the phonon basis of `ep`, the
k+q eigenvectors `ukq` `(nw, nbandkq, n)` and the k eigenvectors `uk_src` `(nw, nbandk, 1 or n)` (the
shared side of `OuterKLoop` has extent 1). `tile_workspace` is for the scratch.
"""
function eph_engine_add_longrange!(ep, coeffs, uk_src, ukq, tile_workspace, model)
    nbkq, nbk, _, n = size(ep)
    nw = model.nw
    uk = reshape_buffer_view(tile_workspace.uk_polar, nw, nbk, n)
    uk .= uk_src
    mmat_buffer = reshape_buffer_view(tile_workspace.mmat_buffer, nbkq, nbk, n)
    add_eph_dipole_batched!(ep, coeffs, ukq, uk, mmat_buffer)
    ep
end
