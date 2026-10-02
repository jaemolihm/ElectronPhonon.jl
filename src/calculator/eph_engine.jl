# The two e-ph engines serve three drivers: outer k with inner k+q or q, and outer q with inner k.
# Each engine holds one run's device copy of `model.epmat`, its stage-1 output for one outer batch, and one tile of
# buffers per CPU thread chunk (`eng.tiles[chunk]`), and wraps the batched kernels of
# `wannier_to_bloch_batched.jl`:
#
#   stage1!  contract the outer R of `epmat` for an outer batch, apply the outer rotation;
#   stage2!  contract the other R for one block (outer point × inner tile), apply the remaining
#            rotations, the phonon basis fused in (`OuterKLoop`; `OuterQLoop` applies it in stage 1);
#   finish_ep!  the terms added on the block's `ep` (the polar dipole), order-agnostic.
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
# _with_workspace passes a concrete NamedTuple of their fields to a specialized worker. The
# kernels then see concrete CPU/GPU arrays without encoding every buffer type in the engine.
abstract type AbstractEphWorkspace end

function _workspace_fields(workspace::AbstractEphWorkspace)
    # function barrier for exposing concrete buffer types to specialized workers.
    NamedTuple{fieldnames(typeof(workspace))}(ntuple(i -> getfield(workspace, i), Val(fieldcount(typeof(workspace)))))
end

function _workspace_fields(workspace::NamedTuple)
    # Already-concrete workspace fields need no conversion.
    workspace
end

function _with_workspace(f::F, workspace) where {F}
    # function barrier for a thread chunk's concrete tile buffers.
    f(_workspace_fields(workspace))
end

"""
    OuterKTileWorkspace

Scratch for one thread chunk's inner tiles. For an inner k+q grid, `phs` and `iq` gather the
phonons; for an inner q grid, `els_kq`, `itp_el_ham`, `hk`, `kqs`, and `xkq` solve k+q per tile.
The unused route's fields are `nothing`. `u_ph_id` holds the cartesian phonon basis, `dg` and
`dg_d` serve covariant derivatives, `uk_polar` and `mmat` serve polar coupling, and `kept`
holds buffers for filtering pairs. Those features allocate buffers only when requested.
"""
Base.@kwdef struct OuterKTileWorkspace <: AbstractEphWorkspace
    phs
    iq
    iq_dev
    els_kq
    itp_el_ham
    hk
    kqs
    xkq
    P_kq
    u_ph_id
    ep
    g
    tmp
    dg
    dg_d
    uk_polar
    mmat
    kept
end

"""Scratch for one thread chunk's k tiles in the outer-q loop."""
Base.@kwdef struct OuterQTileWorkspace <: AbstractEphWorkspace
    itp_eRpq
    itp_el_ham
    els_k
    els_kq
    hk
    kqs
    ikq
    ikq_copy
    ep
    g
    tmp
    uk_rep
    uk_polar
    mmat
    kept
end

"""
    OuterKEngine

The `OuterKLoop` engine: g(k, R_p) for an outer-k batch (stage 1), then g(k, k+q) for one k and a
tile of k+q (stage 2), in the k+q convention of [`get_eph_RR_to_kR_batched!`](@ref): stage 1 folds
`exp(-2πi R_p · x_k)` into g(k, R_p), so the stage-2 phase `exp(2πi R_p · x_{k+q})` of a tile is
shared by every k of the batch. With `covariant_derivative_of_g`, the same two stages run on the
position-weighted `epmat_R` (`wannier_object_multiply_R` plus the tight-binding term
`im (r_j - r_i) g`) into `dg`. Without a k+q container (`run_eph_over_k_and_q`) the inner points are
q points, the k+q states are solved per (k, tile) into the tile's buffers and the phase is built at
x_k + x_q for each k.
"""
struct OuterKEngine <: AbstractEphWorkspace
    backend      :: AbstractBackend
    epmat        :: WannierObject  # model.epmat on the backend
    itp_epmat                     # its batched R_e interpolator
    itp_epmat_R                   # interpolator of epmat_R (dg), or `nothing`
    irvecp_mat                    # (nr_p, 3) R_p
    mxk                           # (3, nk) -x_k
    xkq                           # (3, nkq) x_{k+q}, or x_q without a k+q grid
    wtkq                          # (nkq,) their weights
    xks_int      :: Matrix{Int}     # (3, nk) k grid coordinates, reduced
    xkqs_int     :: Matrix{Int}     # (3, nkq) k+q grid coordinates, reduced, minus the q shift
    P_mk                          # (nr_p, n_outer_batch) exp(-2πi R_p · x_k)
    ep_kR                         # (nw nband_max_k nmodes, nr_p, n_outer_batch) stage-1 output
    dg_kR                         # (nw nband_max_k nmodes, nr_p, 3, n_outer_batch), or `nothing`
    els_k_batch  :: BatchedElectronState # the outer batch's k states
    xk_host      :: Matrix{Float64} # (3, n_outer_batch) the batch's x_k, staged for `xk`
    xk                            # (3, n_outer_batch) the batch's x_k on the backend
    g_fourier                     # stage-1 Fourier output, `reshape_buffer_view` per use
    n_inner_tile :: Int
    tiles        :: Vector{OuterKTileWorkspace}
end

"""
    OuterQEngine

The `OuterQLoop` engine: g(R_e, q) for an outer-q batch with the phonon basis applied (stage 1),
then g(k, k+q) for one q and a tile of k (stage 2), the k+q states solved per tile into the
buffers' leading `maximum(nband)` columns when they are not precomputed.
"""
struct OuterQEngine <: AbstractEphWorkspace
    backend      :: AbstractBackend
    epmat        :: WannierObject  # model.epmat on the backend
    itp_epmat                     # its batched R_p interpolator
    g_q                           # (nw² nmodes, nr_e, n_outer_batch) stage-1 Fourier output
    g_rot                         # (nw², nr_e, nmodes, n_outer_batch) basis scratch, or `nothing`
    ep_Rq                         # (nw² nmodes, nr_e, n_outer_batch) stage-1 output
    eRpq         :: WannierObject  # g(R_e, q) of the current q, the tiles' stage-2 parent
    wtk                           # (nk,) k weights
    xq_host      :: Matrix{Float64} # (3, n_outer_batch) the batch's x_q, staged for `xq`
    xq                            # (3, n_outer_batch) the batch's x_q on the backend
    n_inner_tile :: Int
    tiles        :: Vector{OuterQTileWorkspace}
end


# ---- OuterKEngine ----------------------------------------------------------------------------

function engine_bytes(::Type{OuterKEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk, nkq,
        el_qty, ph_qty, drop_pairs, covariant_derivative_of_g, eph_phonon_basis,
        inner_loop_kq = true) where {FT}
    (; nw, nmodes) = model
    cx, rl, iz = sizeof(Complex{FT}), sizeof(FT), sizeof(Int)
    _require_epmat_layout(OuterKLoop(), model)
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
        rl * 3 * (nk + nkq) + rl * nkq +                        # mxk, xkq, wtkq
        (inner_loop_kq ? 0 : cx * length(model.el_ham.op_r))      # el_ham
    per_outer =
        cx * ndata * nr_p * nd +                                # ep_kR (+ dg_kR)
        cx * nr_p + rl * 3 + iz +                               # P_mk, xk, a partial batch's k index
        cx * nr_e * (covariant_derivative_of_g ? 2 : 1) +       # Fourier phases
        cx * nrows_max +                                        # g_fourier
        cx * (nrows + 2 * nband_max_k * nrows ÷ nw) * nd +      # transients of eph_rotate_kR_batched!
        _electron_state_bytes(FT, nw, nband_max_k, el_qty)      # els_k_batch
    nbox = nband_max_kq * nband_max_k * nmodes
    per_pair =
        cx * nbox * (covariant_derivative_of_g ? 5 : 1) +       # ep (+ dg and its per-direction scratch)
        cx * ndata + cx * nbox +                                # stage-2 scratch g, tmp
        cx * nr_p +                                             # P_kq
        (!inner_loop_kq ?
            _electron_state_bytes(FT, nw, nw, el_qty) +         # k+q tile
            cx * length(model.el_ham.irvec) + cx * nw^2 * 3 + rl * 3 :   # its Fourier phase, H and eigensolve, x_{k+q}
            2iz + _phonon_state_bytes(FT, nmodes, ph_qty)) +    # iq, phs tile
        (eph_phonon_basis == :cartesian ? cx * nmodes^2 : 0) +  # identity basis
        (model.polar_eph.use ? cx * (nw * nband_max_k + nband_max_kq * nband_max_k) : 0)  # polar scratch
    drop_pairs && (per_pair += _electron_state_bytes(FT, nw, nband_max_kq, el_qty) +
        _phonon_state_bytes(FT, nmodes, ph_qty) + cx * nr_p + 5iz + rl + 3rl)   # the kept copy
    (; persistent, per_outer, per_pair)
end

_electron_state_bytes(FT, nw, nb, qty) = 2sizeof(Int) +
    sizeof(FT) * ((:e ∈ qty) * nb + (:vdiag ∈ qty) * 3nb) +
    sizeof(Complex{FT}) * ((:u ∈ qty) * nw * nb + ((:v ∈ qty) + (:rbar ∈ qty)) * 3nb^2)
_phonon_state_bytes(FT, nm, qty) = sizeof(FT) * ((:e ∈ qty) * nm + (:vdiag ∈ qty) * 3nm) +
    sizeof(Complex{FT}) * ((:u ∈ qty) * nm^2 + (:eph_dipole_coeff ∈ qty) * nm + (:eph_r_coeff ∈ qty) * 3nm)

"""
    OuterKEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kwargs...)

Build the outer-k run's stage-1 buffers and one inner-tile workspace per thread chunk.
The model must use epmat_outer_momentum = "el". The inner loop is k+q when els_kq is
provided (inner_loop_kq = true, run_eph_over_k_and_kq), or q when it is nothing
(inner_loop_kq = false, run_eph_over_k_and_q), with k+q states solved per tile.

n_outer_batch and n_inner_tile set the buffer capacities; nchunks sets the number of independent
thread workspaces. el_qty/ph_qty select stored state fields. drop_pairs allocates extra buffers
for compacting the surviving point pairs: it is enabled for energy-conservation filtering, or
for precomputed k+q states in the outer-q driver where a requested k+q point may be absent.
It does not change any band window or phonon-mode selection. covariant_derivative_of_g allocates
the three derivative components; eph_phonon_basis selects eigenmode or cartesian phonons.
"""
function OuterKEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, kqpts, qpts,
        n_outer_batch, n_inner_tile, nchunks, drop_pairs, covariant_derivative_of_g,
        eph_phonon_basis) where {FT}
    # Validate the model layout and prepare the stage-1 interpolators for the selected inner loop.
    (; nw, nmodes) = model
    # run_eph_over_k_and_q has no resident k+q container: solve k+q per inner q tile,
    # at box width nw. run_eph_over_k_and_kq instead gathers phonons for resident k+q states.
    _require_epmat_layout(OuterKLoop(), model)
    inner_loop_kq = els_kq !== nothing
    nbk, nbkq = els_k.nband_max, inner_loop_kq ? els_kq.nband_max : nw
    inner_pts = inner_loop_kq ? kqpts : qpts
    epmat = to_device(backend, model.epmat)
    irvec_p = model.epmat.irvec_next
    nr_p = length(irvec_p)
    itp_epmat = BatchedWannierInterpolator(epmat; backend, batch_size = n_outer_batch)
    itp_epmat_R = if covariant_derivative_of_g
        # The position-weighted e-ph matrix, `im R_e g(R_e, R_p)` plus the tight-binding term
        # `im (r_j - r_i) g`, rows `(i, j, ν, R_p, d)`.
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
    mxk = _kpoints_to_device_matrix(backend, kpts) .* -1
    xkq = _kpoints_to_device_matrix(backend, inner_pts)
    # The q index of a pair by integer grid hash (`_fill_iqs!`): both coordinate lists reduced into
    # `0:ng-1` once, the q-grid shift folded into the k+q side. Not needed without a k+q grid.
    xkqs_int = Matrix{Int}(undef, 3, inner_loop_kq ? kqpts.n : 0)
    xks_int = Matrix{Int}(undef, 3, inner_loop_kq ? kpts.n : 0)
    for ikq in axes(xkqs_int, 2)
        xkqs_int[:, ikq] .= _grid_coords_reduced(kqpts.vectors[ikq], qpts.ngrid, qpts.shift)
    end
    for ik in axes(xks_int, 2)
        xks_int[:, ik] .= _grid_coords_reduced(kpts.vectors[ik], qpts.ngrid, zero(Vec3{FT}))
    end
    el_ham = inner_loop_kq ? nothing : to_device(backend, model.el_ham)
    # A partial last batch operates on views of the leading columns of the maximum-capacity buffers.
    P_mk = fill!(alloc(backend, Complex{FT}, nr_p, n_outer_batch), 1)
    ndata = nw * nbk * nmodes

    # Allocate independent inner-tile scratch and optional filtering buffers for each thread chunk.
    tiles = map(1:nchunks) do _
        # The phonons and q indices of a k+q tile (k+q grid), or the per-tile k+q solve (q points).
        if inner_loop_kq
            # Resident k+q states; gather each (k, k+q tile)'s phonons by q index.
            phs_tile = BatchedPhononState(backend, nmodes, n_inner_tile, ph_qty; FT)
            iq = Vector{Int}(undef, n_inner_tile)
            iq_dev = alloc(backend, Int, n_inner_tile)
            els_kq_tile = itp_el_ham = hk = kqs = xkq_tile = nothing
        else
            # Inner q points; their phonons are resident, but solve k+q per (k, q tile).
            phs_tile = iq = iq_dev = nothing
            els_kq_tile = BatchedElectronState(backend, nw, nw, n_inner_tile, el_qty; FT)
            itp_el_ham = BatchedWannierInterpolator(el_ham; backend, batch_size = n_inner_tile)
            hk = alloc(backend, Complex{FT}, nw^2, n_inner_tile)
            kqs = Vector{Vec3{FT}}(undef, n_inner_tile)
            xkq_tile = alloc(backend, FT, 3, n_inner_tile)
        end
        OuterKTileWorkspace(;
            phs = phs_tile, iq, iq_dev, els_kq = els_kq_tile, itp_el_ham, hk, kqs, xkq = xkq_tile,
            P_kq = alloc(backend, Complex{FT}, nr_p, n_inner_tile),
            u_ph_id = eph_phonon_basis == :cartesian ? to_device_copy(backend,
                repeat(Matrix{Complex{FT}}(I, nmodes, nmodes), 1, 1, n_inner_tile)) : nothing,
            ep = alloc(backend, Complex{FT}, nbkq, nbk, nmodes, n_inner_tile),
            g = alloc(backend, Complex{FT}, ndata, n_inner_tile),
            tmp = alloc(backend, Complex{FT}, nbkq, nbk * nmodes, n_inner_tile),
            dg = covariant_derivative_of_g ? alloc(backend, Complex{FT}, nbkq, nbk, nmodes, 3, n_inner_tile) : nothing,
            dg_d = covariant_derivative_of_g ? alloc(backend, Complex{FT}, nbkq, nbk, nmodes, n_inner_tile) : nothing,
            uk_polar = model.polar_eph.use ? alloc(backend, Complex{FT}, nw * nbk * n_inner_tile) : nothing,
            mmat = model.polar_eph.use ? alloc(backend, Complex{FT}, nbkq * nbk * n_inner_tile) : nothing,
            kept = drop_pairs ? (;
                keep = Vector{Int}(undef, n_inner_tile),
                keep_dev = alloc(backend, Int, n_inner_tile),
                els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
                phs = BatchedPhononState(backend, nmodes, n_inner_tile, ph_qty; FT),
                P_kq = alloc(backend, Complex{FT}, nr_p, n_inner_tile),
                ikq = Vector{Int}(undef, n_inner_tile),
                iq = Vector{Int}(undef, n_inner_tile),
                iq_dev = alloc(backend, Int, n_inner_tile),
                wtq = alloc(backend, FT, n_inner_tile),
                xq = Vector{Vec3{FT}}(undef, n_inner_tile)) : nothing,
        )
    end

    # Assemble the engine with maximum-capacity stage-1 outputs and the thread workspaces.
    nrows_max = nw^2 * nmodes * nr_p * (covariant_derivative_of_g ? 3 : 1)
    OuterKEngine(backend, epmat, itp_epmat, itp_epmat_R,
        _irvec_to_device_matrix(backend, irvec_p, FT), mxk, xkq,
        to_device_copy(backend, collect(FT, inner_pts.weights)), xks_int, xkqs_int, P_mk,
        alloc(backend, Complex{FT}, ndata, nr_p, n_outer_batch),
        covariant_derivative_of_g ? alloc(backend, Complex{FT}, ndata, nr_p, 3, n_outer_batch) : nothing,
        BatchedElectronState(backend, nw, nbk, n_outer_batch, el_qty; FT),
        zeros(FT, 3, n_outer_batch), alloc(backend, FT, 3, n_outer_batch), alloc(backend, Complex{FT}, nrows_max * n_outer_batch),
        n_inner_tile, tiles)
end

"""
    stage1!(eng::OuterKEngine, els_k, kpts, iks_batch)

g(k, R_p) for the outer k points `iks_batch`, rotated by `u_k` and multiplied by `exp(-2πi R_p · x_k)`,
into the first `length(iks_batch)` points of `eng.ep_kR` (and `eng.dg_kR`).

- `eng`: Outer-k engine owning the stage-1 buffers and interpolators.
- `els_k`: Resident electron states from which the outer batch is gathered.
- `kpts`: Outer k-point coordinates indexed by `iks_batch`.
- `iks_batch`: Indices of the active outer k points, up to the engine's batch capacity.
"""
function stage1!(eng::OuterKEngine, els_k, kpts, iks_batch)
    # function barrier for the outer-k stage-1 buffers.
    _stage1!(OuterKLoop(), _workspace_fields(eng), els_k, kpts, iks_batch)
    eng
end

function _stage1!(::OuterKLoop, eng, els_k, kpts, iks_batch)
    # Gather the active electron states and stage their k coordinates on the backend.
    nk_batch = length(iks_batch)
    copy_batched_electron_states!(eng.els_k_batch, els_k, iks_batch)
    for (j, ik) in enumerate(iks_batch)
        eng.xk_host[:, j] .= kpts.vectors[ik]
    end
    xk = view(eng.xk, :, 1:nk_batch)
    copyto!(eng.xk, eng.xk_host)

    # Build the k+q-convention phase and select the active electronic rotations.
    P_mk = view(eng.P_mk, :, 1:nk_batch)
    @views build_fourier_phase!(P_mk, eng.irvecp_mat, eng.mxk[:, iks_batch])
    uks = view(eng.els_k_batch.u, :, :, 1:nk_batch)

    # Fourier-transform R_e and rotate the active batch into g(k, R_p).
    g = reshape_buffer_view(eng.g_fourier, eng.itp_epmat.parent.ndata, nk_batch)
    get_fourier_batched!(g, eng.itp_epmat, xk)
    ep_kR = view(eng.ep_kR, :, :, 1:nk_batch)
    eph_rotate_kR_batched!(ep_kR, g, uks; additional_phase = P_mk)

    # Repeat the Fourier transform and rotation for the optional covariant derivatives.
    if eng.itp_epmat_R !== nothing
        g = reshape_buffer_view(eng.g_fourier, eng.itp_epmat_R.parent.ndata, nk_batch)
        get_fourier_batched!(g, eng.itp_epmat_R, xk)
        dg_kR = view(eng.dg_kR, :, :, :, 1:nk_batch)
        eph_rotate_kR_batched!(dg_kR, g, uks; additional_phase = P_mk)
    end
    eng
end

"""
    stage2!(eng::OuterKEngine, tile_workspace, eph_inputs)

g(k, k+q) of the block `eph_inputs` (one k, `eph_inputs.n` k+q points) into the leading
`(eph_inputs.els_kq.nband_max, nband_max_k, nmodes, eph_inputs.n)` of `tile_workspace.ep` (and `tile_workspace.dg`):
`tile_workspace` owns the reusable arrays; `eph_inputs` contains the states, point indices,
weights, and phase for the current fixed k and inner tile. The kR→kq contraction uses the tile's phase `eph_inputs.phase`, the k+q rotation and the
phonon basis (`eph_inputs.phs.u`, or the identity for `:cartesian`).

- `eng`: Outer-k engine containing the stage-1 output for the current outer batch.
- `tile_workspace`: OuterKTileWorkspace or its concrete NamedTuple of reusable stage-2 buffers.
- `eph_inputs`: Active tile's states, phase, indices, weights, and outer-batch position `iouter`.
"""
function stage2!(eng::OuterKEngine, tile_workspace::Union{OuterKTileWorkspace, NamedTuple}, eph_inputs)
    # function barrier for the outer-k stage-2 engine and tile buffers.
    _stage2!(OuterKLoop(), _workspace_fields(eng), _workspace_fields(tile_workspace), eph_inputs)
end

function _stage2!(::OuterKLoop, eng, tile_workspace, eph_inputs)
    # Select output and scratch views at the active tile's band and point dimensions.
    (; n, iouter) = eph_inputs
    # The k+q box of the block: the container's, or the widest window of a tile solved per tile.
    nbkq, nbk = eph_inputs.els_kq.nband_max, eng.els_k_batch.nband_max
    nmodes = eph_inputs.phs.nmodes
    ep = reshape_buffer_view(tile_workspace.ep, nbkq, nbk, nmodes, n)
    ws = (; g = view(tile_workspace.g, :, 1:n), tmp = reshape_buffer_view(tile_workspace.tmp, nbkq, nbk * nmodes, n))

    # Contract R_p and apply the k+q electron rotation and the requested phonon basis.
    u_ph = tile_workspace.u_ph_id === nothing ? eph_inputs.phs.u : view(tile_workspace.u_ph_id, :, :, 1:n)   # the phonon basis
    get_eph_kR_to_kq_batched!(ep, view(eng.ep_kR, :, :, iouter), eph_inputs.phase, u_ph, eph_inputs.els_kq.u; ws...)

    # Apply the same contraction and rotations to each optional derivative component.
    tile_workspace.dg === nothing && return ep, nothing
    dg = reshape_buffer_view(tile_workspace.dg, nbkq, nbk, nmodes, 3, n)
    dg_d = reshape_buffer_view(tile_workspace.dg_d, nbkq, nbk, nmodes, n)
    for d in 1:3
        get_eph_kR_to_kq_batched!(dg_d, view(eng.dg_kR, :, :, d, iouter), eph_inputs.phase, u_ph,
                                  eph_inputs.els_kq.u; ws...)
        view(dg, :, :, :, d, :) .= dg_d
    end
    ep, dg
end


# ---- OuterQEngine ----------------------------------------------------------------------------

function engine_bytes(::Type{OuterQEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk,
        n_outer_batch, el_qty, ph_qty, drop_pairs, precompute_el_kq, eph_phonon_basis) where {FT}
    (; nw, nmodes) = model
    cx, rl, iz = sizeof(Complex{FT}), sizeof(FT), sizeof(Int)
    _require_epmat_layout(OuterQLoop(), model)
    nr_e = length(model.epmat.irvec_next)
    nr_p = length(model.epmat.irvec)
    ndata = nw^2 * nmodes
    nbkq = precompute_el_kq ? nband_max_kq : nw
    persistent =
        cx * length(model.epmat.op_r) +                         # epmat
        cx * ndata * nr_e + rl * 3 * (nr_e + nr_p) +            # eRpq, R-vector matrices
        cx * ndata * nr_e +                                     # interpolator output
        (precompute_el_kq ? 0 : cx * length(model.el_ham.op_r)) +  # el_ham
        rl * nk                                                 # wtk
    per_outer =
        cx * ndata * nr_e * (eph_phonon_basis == :cartesian ? 2 : 3) +   # g_q, ep_Rq (+ g_rot)
        cx * nr_p +                                             # Fourier phase
        rl * 3                                                  # xq
    per_pair =
        cx * nbkq * nband_max_k * nmodes +                      # ep
        cx * ndata + cx * nbkq * nw * nmodes + cx * nw * nband_max_k * nmodes +   # g, tmp, uk_rep
        _electron_state_bytes(FT, nw, nband_max_k, el_qty) +    # k tile
        _electron_state_bytes(FT, nw, nbkq, el_qty) + 2iz +     # k+q tile, ikq
        cx * nr_e + 2iz +                                       # stage-2 Fourier phase, ikq_copy
        (precompute_el_kq ? 0 : cx * length(model.el_ham.irvec)) +   # k+q Fourier phase
        (precompute_el_kq ? 0 : cx * nw^2 * 3) +                # k+q Hamiltonian and its eigensolve
        (model.polar_eph.use ? cx * (nw * nband_max_k + nbkq * nband_max_k) : 0)   # polar scratch
    drop_pairs && (per_pair += _electron_state_bytes(FT, nw, nband_max_k, el_qty) +
        _electron_state_bytes(FT, nw, nbkq, el_qty) + 4iz + rl + 3rl)   # the kept copy
    (; persistent, per_outer, per_pair)
end

"""
    OuterQEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kwargs...)

Build the outer-q run's buffers for a model with epmat_outer_momentum = "ph".
The capacity, quantity, phonon-basis, and drop_pairs arguments have the meanings documented for
OuterKEngine. Here els_kq = nothing selects a per-tile k+q solve; otherwise k+q states are copied
from the resident container. drop_pairs also handles requested k+q points absent from it.
"""
function OuterQEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, qpts,
        n_outer_batch, n_inner_tile, nchunks, drop_pairs, eph_phonon_basis) where {FT}
    # Validate the model layout and determine the electron band-box dimensions.
    (; nw, nmodes) = model
    nbk = els_k.nband_max
    _require_epmat_layout(OuterQLoop(), model)
    nbkq = els_kq === nothing ? nw : els_kq.nband_max

    # Prepare the stage-1 Fourier interpolator and the shared stage-2 parent g(R_e, q).
    epmat = to_device(backend, model.epmat)
    irvec_e = model.epmat.irvec_next
    nr_e = length(irvec_e)
    ndata = nw^2 * nmodes
    itp_epmat = BatchedWannierInterpolator(epmat; backend, batch_size = n_outer_batch)
    eRpq = WannierObject(irvec_e, alloc_zeros(backend, Complex{FT}, ndata, nr_e))
    el_ham = els_kq === nothing ? to_device(backend, model.el_ham) : nothing

    # Allocate independent k-tile states, interpolators, and scratch for each thread chunk.
    tiles = map(1:nchunks) do _
        # Each tile has its own interpolators: their phase scratch is written per call.
        OuterQTileWorkspace(;
            itp_eRpq = BatchedWannierInterpolator(eRpq; backend, batch_size = n_inner_tile),
            itp_el_ham = el_ham === nothing ? nothing :
                BatchedWannierInterpolator(el_ham; backend, batch_size = n_inner_tile),
            els_k = BatchedElectronState(backend, nw, nbk, n_inner_tile, el_qty; FT),
            els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
            hk = els_kq === nothing ? alloc(backend, Complex{FT}, nw^2, n_inner_tile) : nothing,
            kqs = Vector{Vec3{FT}}(undef, n_inner_tile),
            ikq = Vector{Int}(undef, n_inner_tile),     # 0: k+q absent from the precomputed states
            ikq_copy = Vector{Int}(undef, n_inner_tile),
            ep = alloc(backend, Complex{FT}, nbkq * nbk * nmodes * n_inner_tile),
            g = alloc(backend, Complex{FT}, ndata, n_inner_tile),
            tmp = alloc(backend, Complex{FT}, nbkq * nw * nmodes * n_inner_tile),
            uk_rep = alloc(backend, Complex{FT}, nw, nbk, nmodes * n_inner_tile),
            uk_polar = model.polar_eph.use ? alloc(backend, Complex{FT}, nw * nbk * n_inner_tile) : nothing,
            mmat = model.polar_eph.use ? alloc(backend, Complex{FT}, nbkq * nbk * n_inner_tile) : nothing,
            kept = drop_pairs ? (;
                keep = Vector{Int}(undef, n_inner_tile),
                keep_dev = alloc(backend, Int, n_inner_tile),
                els_k = BatchedElectronState(backend, nw, nbk, n_inner_tile, el_qty; FT),
                els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
                ik = Vector{Int}(undef, n_inner_tile),
                ikq = Vector{Int}(undef, n_inner_tile),
                wtk = alloc(backend, FT, n_inner_tile),
                xk = Vector{Vec3{FT}}(undef, n_inner_tile)) : nothing,
        )
    end

    # Assemble the engine with maximum-capacity stage-1 outputs and coordinate buffers.
    OuterQEngine(backend, epmat, itp_epmat,
        alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch),
        eph_phonon_basis == :cartesian ? nothing : alloc(backend, Complex{FT}, nw^2, nr_e, nmodes, n_outer_batch),
        alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch), eRpq,
        to_device_copy(backend, collect(FT, kpts.weights)), zeros(FT, 3, n_outer_batch),
        alloc(backend, FT, 3, n_outer_batch), n_inner_tile, tiles)
end

"""
    stage1!(eng::OuterQEngine, phs, qpts, iqs_batch, eph_phonon_basis)

g(R_e, q) for the outer q points `iqs_batch` in the phonon basis `eph_phonon_basis` (`:eigenmode`
rotates the modes by `phs.u`, `:cartesian` leaves them), into `eng.ep_Rq[:, :, 1:length(iqs_batch)]`.

Both engines contract the outer R and rotate the outer states in stage 1. Here the rotation
mixes phonon modes only. Outer k instead rotates electronic bands, folds in the k+q-convention
phase, and optionally repeats the contraction and rotation for the three derivative components.

- `eng`: Outer-q engine owning the stage-1 buffers and Fourier interpolator.
- `phs`: Resident phonon states supplying the rotations at the active q points.
- `qpts`: Outer q-point coordinates indexed by `iqs_batch`.
- `iqs_batch`: Indices of the active outer q points, up to the engine's batch capacity.
- `eph_phonon_basis`: `:eigenmode` to rotate by `phs.u`, or `:cartesian` to leave the modes unrotated.
"""
function stage1!(eng::OuterQEngine, phs, qpts, iqs_batch, eph_phonon_basis)
    # function barrier for the outer-q stage-1 buffers.
    _stage1!(OuterQLoop(), _workspace_fields(eng), phs, qpts, iqs_batch, eph_phonon_basis)
    eng
end

function _stage1!(::OuterQLoop, eng, phs, qpts, iqs_batch, eph_phonon_basis)
    # Stage the active q coordinates on the backend.
    nq_batch = length(iqs_batch)
    for (j, iq) in enumerate(iqs_batch)
        eng.xq_host[:, j] .= qpts.vectors[iq]
    end
    xq = view(eng.xq, :, 1:nq_batch)
    copyto!(eng.xq, eng.xq_host)

    # Fourier-transform R_p into g(R_e, q) for the active outer batch.
    ndata, nr_e = size(eng.ep_Rq, 1), size(eng.ep_Rq, 2)
    g = view(eng.g_q, :, :, 1:nq_batch)
    get_fourier_batched!(reshape(g, ndata * nr_e, nq_batch), eng.itp_epmat, xq)

    # Apply the phonon eigenmode rotation, or retain the cartesian components.
    ep = view(eng.ep_Rq, :, :, 1:nq_batch)
    if eph_phonon_basis == :cartesian
        ep .= g
    else
        # ep[(a, ν'), r, q] = Σ_ν g[(a, ν), r, q] u_ph[ν, ν', q], as one batched GEMM over q on the
        # (a, r) × ν layout.
        nw² = size(eng.g_rot, 1)
        nmodes = size(eng.g_rot, 3)
        g_rot = view(eng.g_rot, :, :, :, 1:nq_batch)
        permutedims!(g_rot, reshape(g, nw², nmodes, nr_e, nq_batch), (1, 3, 2, 4))
        ep_rot = reshape(view(eng.g_q, :, :, 1:nq_batch), nw² * nr_e, nmodes, nq_batch)   # g is consumed
        batched_gemm!('N', 'N', reshape(g_rot, nw² * nr_e, nmodes, nq_batch), view(phs.u, :, :, iqs_batch), ep_rot)
        permutedims!(reshape(ep, nw², nmodes, nr_e, nq_batch), reshape(ep_rot, nw², nr_e, nmodes, nq_batch), (1, 3, 2, 4))
    end
    eng
end

"""
    stage2!(eng::OuterQEngine, tile_workspace, eph_inputs)

g(k, k+q) of the block `eph_inputs` (one q, `eph_inputs.n` k points) into the leading
`(eph_inputs.els_kq.nband_max, eph_inputs.els_k.nband_max, nmodes, eph_inputs.n)` of `tile_workspace.ep`, from `eng.eRpq`
(the current q).

- `eng`: Outer-q engine whose shared `eRpq` parent holds the current q's stage-1 output.
- `tile_workspace`: OuterQTileWorkspace or its concrete NamedTuple of reusable stage-2 buffers.
- `eph_inputs`: Active k tile's electron/phonon states, coordinates, indices, and weights.
"""
function stage2!(eng::OuterQEngine, tile_workspace::Union{OuterQTileWorkspace, NamedTuple}, eph_inputs)
    # function barrier for the outer-q stage-2 engine and tile buffers.
    _stage2!(OuterQLoop(), _workspace_fields(eng), _workspace_fields(tile_workspace), eph_inputs)
end

function _stage2!(::OuterQLoop, eng, tile_workspace, eph_inputs)
    # Determine the active tile's electron band boxes and phonon-mode dimension.
    (; n) = eph_inputs
    nbkq, nbk = eph_inputs.els_kq.nband_max, eph_inputs.els_k.nband_max
    nw, nmodes = size(tile_workspace.uk_rep, 1), eph_inputs.phs.nmodes

    # Select the output view at the active tile's band and point dimensions.
    ep = reshape_buffer_view(tile_workspace.ep, nbkq, nbk, nmodes, n)

    # Contract R_e and apply both electron rotations using the preallocated tile scratch.
    get_eph_Rq_to_kq_batched!(ep, tile_workspace.itp_eRpq, eph_inputs.xk, eph_inputs.els_k.u, eph_inputs.els_kq.u;
        g = view(tile_workspace.g, :, 1:n), tmp = reshape_buffer_view(tile_workspace.tmp, nbkq, nw * nmodes, n),
        uk_rep = reshape_buffer_view(tile_workspace.uk_rep, nw, nbk, nmodes * n))
    ep, nothing
end


# ---- Both orders -----------------------------------------------------------------------------

"""
    finish_ep!(block, tile_workspace, model)

The terms added on a block's `ep` after the two stages, for either loop order: the polar dipole
term `coeff[ν] · u_{k+q}' u_k` of a polar model (unscreened, as `epstate_compute_eph_dipole!`),
with the coefficients in the block's phonon basis. `tile_workspace` is the block's tile, for the scratch.
"""
function finish_ep!(block, tile_workspace, model)
    model.polar_eph.use || return block
    nbkq, nbk, _, n = size(block.ep)
    nw = model.nw
    # u_k at the block's pair extent: the shared side of `OuterKLoop` has extent 1.
    uk = reshape_buffer_view(tile_workspace.uk_polar, nw, nbk, n)
    uk .= block.els_k.u
    add_eph_dipole_batched!(block.ep, block.phs.eph_dipole_coeff, block.els_kq.u, uk,
                            reshape_buffer_view(tile_workspace.mmat, nbkq, nbk, n))
    block
end
