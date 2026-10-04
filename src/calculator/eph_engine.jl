# The two e-ph engines serve three drivers: outer k with inner k+q or q, and outer q with inner k.
# Each engine holds one run's device copy of `model.epmat`, its stage-1 output for one outer batch, and one tile of
# buffers per CPU thread chunk (`eng.tiles[chunk]`), and wraps the batched kernels of
# `wannier_to_bloch_batched.jl`:
#
#   stage1!  contract the outer R of `epmat` for an outer batch, apply the outer rotation;
#   stage2!  select the pairs to compute, gather their states, contract the other R, apply the
#            remaining rotations, and add the long-range (polar) term
#            (`eph_engine_add_longrange!`); return a calculator-ready EPBlock.
#
# The resident containers (`eng.els_k`, `eng.els_kq`, `eng.phs`) are read-only. Stage 2 skips the
# pairs that need no e-ph matrix (`_select_pairs!`) and gathers the others' states once, into the
# tile's buffers; a side whose whole tile is selected and resident is a view, with no copy.
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
abstract type AbstractEphWorkspace end

function _workspace_fields(workspace::AbstractEphWorkspace)
    # function barrier for exposing concrete buffer types to specialized workers.
    NamedTuple{fieldnames(typeof(workspace))}(ntuple(i -> getfield(workspace, i), Val(fieldcount(typeof(workspace)))))
end

"""
    OuterKTileWorkspace

Scratch for one thread chunk's inner tiles. `keep` lists the selected pairs of a tile, `ikq` and
`iq` their indices (`iq_dev` on the backend), and `els_kq`, `phs`, `wtq`, `xkq` and `P_kq` hold
their gathered k+q states, phonons, weights, k+q coordinates and Fourier phase. `itp_el_ham`, `hk`
and `kqs` solve k+q per tile for an inner q grid, and are `nothing` for an inner k+q grid.
`u_ph_id` holds the cartesian phonon basis, `dg` and `dg_d` serve covariant derivatives, and
`uk_polar` and `mmat_buffer` are scratch of the polar term. Those features allocate buffers only
when requested.
"""
Base.@kwdef struct OuterKTileWorkspace <: AbstractEphWorkspace
    keep
    ikq
    iq
    iq_dev
    els_kq
    phs
    wtq
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
    mmat_buffer
end

"""
Scratch for one thread chunk's k tiles in the outer-q loop. `keep` lists the selected pairs of a
tile, `ik` and `ikq` their indices, and `els_k`, `els_kq`, `wtk` and `xk` hold their gathered
states, weights and coordinates. `P_k` is their stage-2 Fourier phase `exp(2πi R_e · x_k)`.
"""
Base.@kwdef struct OuterQTileWorkspace <: AbstractEphWorkspace
    keep
    ik
    ikq
    els_k
    els_kq
    wtk
    xk
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
end

"""
    OuterKEngine

The `OuterKLoop` engine: g(k, R_p) for an outer-k batch (stage 1), then g(k, k+q) for one k and a
tile of k+q (stage 2), in the k+q convention of [`eph_rotate_kR_batched!`](@ref): stage 1 folds
`exp(-2πi R_p · x_k)` into g(k, R_p), so the stage-2 phase `exp(2πi R_p · x_{k+q})` of a tile is
shared by every k of the batch. With `covariant_derivative_of_g`, the same two stages run on the
position-weighted `epmat_R` (`wannier_object_multiply_R` plus the tight-binding term
`im (r_j - r_i) g`) into `dg`. Without a k+q container (`run_eph_over_k_and_q`) the inner points are
q points, the k+q states are solved per (k, tile) into the tile's buffers and the phase is built at
x_k + x_q for each k.
"""
mutable struct OuterKEngine <: AbstractEphWorkspace
    model        :: Model
    els_k        :: BatchedElectronState
    els_kq       :: Union{Nothing, BatchedElectronState}
    phs          :: BatchedPhononState
    kpts         :: AbstractKpoints
    kqpts        :: Union{Nothing, AbstractKpoints}
    qpts         :: AbstractKpoints
    sel_k        :: Union{Nothing, FilteredBandStates}
    sel_kq       :: Union{Nothing, FilteredBandStates}
    window_kq
    energy_conservation_tol :: Float64
    eph_phonon_basis :: Symbol
    inner_loop_kq :: Bool         # inner points are k+q (resident states); false: q, k+q solved per tile
    n_outer      :: Int           # number of outer k points
    batch        :: UnitRange{Int}
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
mutable struct OuterQEngine <: AbstractEphWorkspace
    model        :: Model
    els_k        :: BatchedElectronState
    els_kq       :: Union{Nothing, BatchedElectronState}
    phs          :: BatchedPhononState
    kpts         :: AbstractKpoints
    kqpts        :: Union{Nothing, AbstractKpoints}
    qpts         :: AbstractKpoints
    sel_k        :: Union{Nothing, FilteredBandStates}
    sel_kq       :: Union{Nothing, FilteredBandStates}
    window_kq
    energy_conservation_tol :: Float64
    eph_phonon_basis :: Symbol
    n_outer      :: Int           # number of outer q points
    batch        :: UnitRange{Int}
    backend      :: AbstractBackend
    epmat        :: WannierObject  # model.epmat on the backend
    itp_epmat                     # its batched R_p interpolator
    irvece_mat                    # (nr_e, 3) R_e
    g_q                           # (nw² nmodes, nr_e, n_outer_batch) stage-1 Fourier output
    g_rot                         # (nw², nr_e, nmodes, n_outer_batch) basis scratch, or `nothing`
    ep_Rq                         # (nw² nmodes, nr_e, n_outer_batch) stage-1 output
    wtk                           # (nk,) k weights
    xq_host      :: Matrix{Float64} # (3, n_outer_batch) the batch's x_q, staged for `xq`
    xq                            # (3, n_outer_batch) the batch's x_q on the backend
    n_outer_batch :: Int
    n_inner_tile :: Int
    tiles        :: Vector{OuterQTileWorkspace}
end


# A block on a tile's output storage, at exactly the point and band extents of its states.
function EPBlock{O}(tile_workspace, els_k::BatchedElectronState, els_kq::BatchedElectronState,
        phs::BatchedPhononState; kwargs...) where {O <: LoopTag}
    n = O === OuterKLoop ? els_kq.nk : els_k.nk
    dims = (els_kq.nband_max, els_k.nband_max, phs.nmodes, n)
    ep = reshape_buffer_view(tile_workspace.ep, dims...)
    dg = O === OuterKLoop && tile_workspace.dg !== nothing ?
        reshape_buffer_view(tile_workspace.dg, dims[1:3]..., 3, n) : nothing
    EPBlock{O}(; ep, dg, els_k, els_kq, phs, kwargs...)
end

# Fill `iqs[1:nq]` with the index into `qpts` of `x_{k+q} - x_k` for outer k `ik` and every k+q of
# the tile `qstart .+ (0:nq-1)`, by integer grid hash on the coordinates the engine reduced once.
function _fill_iqs!(iqs, qpts, xkqs_int, xks_int, ik, qstart, nq)
    ng1, ng2, ng3 = qpts.ngrid
    k1, k2, k3 = xks_int[1, ik], xks_int[2, ik], xks_int[3, ik]
    for j in 1:nq
        ikq = qstart + j - 1
        # Both operands are pre-reduced into 0:ng-1, so the fold is a compare-and-add.
        h1 = _wrap_reduced(xkqs_int[1, ikq] - k1, ng1)
        h2 = _wrap_reduced(xkqs_int[2, ikq] - k2, ng2)
        h3 = _wrap_reduced(xkqs_int[3, ikq] - k3, ng3)
        hash = (h1 * ng2 + h2) * ng3 + h3
        iq = _ik_from_hash(qpts, hash)
        # 0 = miss on either index.
        (iq < 1 || iq > qpts.n) && throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
        iqs[j] = iq
    end
    iqs
end

# ---- OuterKEngine ----------------------------------------------------------------------------

function engine_bytes(::Type{OuterKEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk, nkq,
        el_qty, ph_qty, covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq) where {FT}
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
        _electron_state_bytes(FT, nw, nband_max_kq, el_qty) +   # k+q tile
        _phonon_state_bytes(FT, nmodes, ph_qty) +               # phs tile
        4iz + rl + 3rl +                                        # keep, ikq, iq, iq_dev, wtq, x_{k+q}
        (inner_loop_kq ? 0 :
            cx * length(model.el_ham.irvec) + cx * nw^2 * 3) +  # k+q Fourier phase, H and eigensolve
        (eph_phonon_basis == :cartesian ? cx * nmodes^2 : 0) +  # identity basis
        (model.polar_eph.use ? cx * (nw * nband_max_k + nband_max_kq * nband_max_k) : 0)  # polar scratch
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
thread workspaces. el_qty/ph_qty select stored state fields. energy_conservation_tol skips point
pairs (`_select_pairs!`); it does not change any band window or phonon-mode selection.
covariant_derivative_of_g allocates the three derivative components; eph_phonon_basis selects
eigenmode or cartesian phonons.
"""
function OuterKEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, kqpts, qpts,
        n_outer_batch, n_inner_tile, nchunks, covariant_derivative_of_g,
        eph_phonon_basis, sel_k = nothing, sel_kq = nothing, window_kq = (-Inf, Inf),
        energy_conservation_tol = Inf) where {FT}
    # Validate the model layout and prepare the stage-1 interpolators for the selected inner loop.
    (; nw, nmodes) = model
    _require_epmat_layout(OuterKLoop(), model)
    # run_eph_over_k_and_q has no resident k+q container: solve k+q per inner q tile,
    # at box width nw. run_eph_over_k_and_kq instead gathers phonons for resident k+q states.
    inner_loop_kq = els_kq !== nothing
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
            keep = Vector{Int}(undef, n_inner_tile),
            ikq = Vector{Int}(undef, n_inner_tile),
            iq = Vector{Int}(undef, n_inner_tile),
            iq_dev = alloc(backend, Int, n_inner_tile),
            els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
            phs = BatchedPhononState(backend, nmodes, n_inner_tile, ph_qty; FT),
            wtq = alloc(backend, FT, n_inner_tile),
            itp_el_ham, hk, kqs,
            xkq = alloc(backend, FT, 3, n_inner_tile),
            P_kq = alloc(backend, Complex{FT}, nr_p, n_inner_tile),
            u_ph_id = eph_phonon_basis == :cartesian ? to_device_copy(backend,
                repeat(Matrix{Complex{FT}}(I, nmodes, nmodes), 1, 1, n_inner_tile)) : nothing,
            ep = alloc(backend, Complex{FT}, nbkq, nbk, nmodes, n_inner_tile),
            g = alloc(backend, Complex{FT}, ndata, n_inner_tile),
            tmp = alloc(backend, Complex{FT}, nbkq, nbk * nmodes, n_inner_tile),
            dg = covariant_derivative_of_g ? alloc(backend, Complex{FT}, nbkq, nbk, nmodes, 3, n_inner_tile) : nothing,
            dg_d = covariant_derivative_of_g ? alloc(backend, Complex{FT}, nbkq, nbk, nmodes, n_inner_tile) : nothing,
            uk_polar = model.polar_eph.use ? alloc(backend, Complex{FT}, nw * nbk * n_inner_tile) : nothing,
            mmat_buffer = model.polar_eph.use ? alloc(backend, Complex{FT}, nbkq * nbk * n_inner_tile) : nothing,
        )
    end

    # Assemble the engine with maximum-capacity stage-1 outputs and the thread workspaces.
    nrows_max = nw^2 * nmodes * nr_p * (covariant_derivative_of_g ? 3 : 1)
    OuterKEngine(model, els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq, window_kq,
        energy_conservation_tol, eph_phonon_basis, inner_loop_kq, kpts.n, 1:0, backend, epmat, itp_epmat,
        itp_epmat_R,
        _irvec_to_device_matrix(backend, irvec_p, FT), mxk, xkq,
        to_device_copy(backend, collect(FT, inner_pts.weights)), xks_int, xkqs_int, P_mk,
        alloc(backend, Complex{FT}, ndata, nr_p, n_outer_batch),
        covariant_derivative_of_g ? alloc(backend, Complex{FT}, ndata, nr_p, 3, n_outer_batch) : nothing,
        BatchedElectronState(backend, nw, nbk, n_outer_batch, el_qty; FT),
        zeros(FT, 3, n_outer_batch), alloc(backend, FT, 3, n_outer_batch), alloc(backend, Complex{FT}, nrows_max * n_outer_batch),
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
    _check_stage1(eng, iks_batch)
    eng.batch = iks_batch
    # function barrier for the outer-k stage-1 buffers.
    _stage1!(OuterKLoop(), _workspace_fields(eng), iks_batch)
    eng
end

function _stage1!(::OuterKLoop, eng_fields, iks_batch)
    # Gather the active electron states and stage their k coordinates on the backend.
    nk_batch = length(iks_batch)
    copy_batched_electron_states!(eng_fields.els_k_batch, eng_fields.els_k, iks_batch)
    for (j, ik) in enumerate(iks_batch)
        eng_fields.xk_host[:, j] .= eng_fields.kpts.vectors[ik]
    end
    xk = view(eng_fields.xk, :, 1:nk_batch)
    copyto!(eng_fields.xk, 1, eng_fields.xk_host, 1, 3nk_batch)

    # Build the k+q-convention phase and select the active electronic rotations.
    P_mk = view(eng_fields.P_mk, :, 1:nk_batch)
    @views build_fourier_phase!(P_mk, eng_fields.irvecp_mat, eng_fields.mxk[:, iks_batch])
    uks = view(eng_fields.els_k_batch.u, :, :, 1:nk_batch)

    # Fourier-transform R_e and rotate by u_k: g_ija(R_e, R_p) -> g_ina(k, R_p), with i, j Wannier
    # indices, a the atomic displacement and n the band of k.
    g = reshape_buffer_view(eng_fields.g_fourier, eng_fields.itp_epmat.parent.ndata, nk_batch)
    get_fourier_batched!(g, eng_fields.itp_epmat, xk)
    ep_kR = view(eng_fields.ep_kR, :, :, 1:nk_batch)
    eph_rotate_kR_batched!(ep_kR, g, uks; additional_phase = P_mk)

    # The same for the covariant derivative, direction d: dg_ijad(R_e, R_p) -> dg_inad(k, R_p).
    if eng_fields.itp_epmat_R !== nothing
        g = reshape_buffer_view(eng_fields.g_fourier, eng_fields.itp_epmat_R.parent.ndata, nk_batch)
        get_fourier_batched!(g, eng_fields.itp_epmat_R, xk)
        dg_kR = view(eng_fields.dg_kR, :, :, :, 1:nk_batch)
        eph_rotate_kR_batched!(dg_kR, g, uks; additional_phase = P_mk)
    end
    eng_fields
end

"""
    stage2!(eng::OuterKEngine, ik, inner_indices; chunk=1)

Return the calculator-ready `EPBlock` of outer k point `ik` and a tile of inner points: gather the
resident k+q states and their phonons (inner k+q points) or solve the k+q states (inner q points)
for the pairs to compute (`_select_pairs!`), compute their e-ph matrix (and requested
derivatives), and add polar corrections. Return `nothing` if no pair is selected. Call `stage1!`
first; `ik` must be in its current batch.

The block borrows this chunk's buffers: its arrays are valid until that chunk's next `stage2!`
or the next `stage1!`. Consume it immediately or copy the arrays you need to retain. Calls on
different CPU chunks may run concurrently, but `stage1!` must wait for all of them to finish.

- `ik`: Index into `eng.kpts`, not a batch-local index.
- `inner_indices`: Nonempty contiguous range into `eng.kqpts` (inner k+q) or `eng.qpts` (inner q).
- `chunk`: Independent workspace slot; use 1 for serial calls and GPU calls.
"""
function stage2!(eng::OuterKEngine, ik::Int, tile::UnitRange{Int}; chunk::Int = 1)
    n_inner = eng.inner_loop_kq ? eng.kqpts.n : eng.qpts.n
    _check_stage2(eng, ik, tile, chunk, n_inner)
    # function barrier for the concrete resident states, engine buffers, and chunk scratch.
    _stage2!(OuterKLoop(), _workspace_fields(eng), _workspace_fields(eng.tiles[chunk]), ik, tile)
end

# `phase` is the Fourier phase of the whole tile, shared by the outer k points of a batch, or
# `nothing` to build it here for the selected pairs (always so when pairs can be skipped).
function _stage2!(::OuterKLoop, eng_fields, tile_workspace, ik, tile; phase = nothing)
    n = length(tile)
    iouter = ik - first(eng_fields.batch) + 1
    els_k = view(eng_fields.els_k_batch, iouter:iouter)
    tol = eng_fields.energy_conservation_tol
    (; keep) = tile_workspace
    (; phs) = eng_fields

    if eng_fields.inner_loop_kq
        # Inner k+q points (run_eph_over_k_and_kq): the k+q states are resident, and each has a
        # state in the window. Find q = (k+q) - k of each pair and select the pairs to compute.
        (; els_kq) = eng_fields
        iq_host = tile_workspace.iq
        _fill_iqs!(iq_host, eng_fields.qpts, eng_fields.xkqs_int, eng_fields.xks_int, ik, first(tile), n)
        nkeep = isinf(tol) ? n : _select_pairs!(keep, n, Returns(true), j -> _conserves_energy(
            view(els_k.e, 1:els_k.nband[1], 1), view(els_kq.e, 1:els_kq.nband[tile[j]], tile[j]),
            view(phs.e, :, iq_host[j]), tol))
        nkeep == 0 && return nothing

        if nkeep == n
            # The whole tile: the resident k+q states as they are.
            ikq = tile
            els_kq_block = view(els_kq, tile)
            wtq = view(eng_fields.wtkq, tile)
            if phase === nothing
                phase = view(tile_workspace.P_kq, :, 1:n)
                @views build_fourier_phase!(phase, eng_fields.irvecp_mat, eng_fields.xkq[:, tile])
            end
        else
            # Gather the selected pairs' k+q states, weights and coordinates.
            phase === nothing || throw(ArgumentError("a shared tile phase needs every pair of the tile"))
            for (i, j) in enumerate(view(keep, 1:nkeep))
                tile_workspace.ikq[i] = tile[j]
                iq_host[i] = iq_host[j]
            end
            ikq = view(tile_workspace.ikq, 1:nkeep)
            ikq_on_backend = _copy_indices_on_backend(eng_fields.xkq, ikq, els_kq.nk)
            els_kq_block = view(copy_batched_electron_states!(tile_workspace.els_kq, els_kq, ikq_on_backend), 1:nkeep)
            wtq = view(_copy_last_axis!(tile_workspace.wtq, eng_fields.wtkq, ikq_on_backend), 1:nkeep)
            xkq = view(_copy_last_axis!(tile_workspace.xkq, eng_fields.xkq, ikq_on_backend), :, 1:nkeep)
            phase = view(tile_workspace.P_kq, :, 1:nkeep)
            build_fourier_phase!(phase, eng_fields.irvecp_mat, xkq)
        end

        # Gather the selected pairs' phonons.
        copyto!(tile_workspace.iq_dev, 1, iq_host, 1, nkeep)
        iq = view(tile_workspace.iq_dev, 1:nkeep)
        phs_block = view(copy_batched_phonon_states!(tile_workspace.phs, phs, iq), 1:nkeep)
        xq = view(eng_fields.qpts.vectors, view(iq_host, 1:nkeep))
    else
        # Inner q points (run_eph_over_k_and_q): the phonons are resident. Solve the electron
        # bands at k+q for this tile and select the pairs to compute.
        for (j, iq_tile) in enumerate(tile)
            tile_workspace.kqs[j] = eng_fields.kpts.vectors[ik] + eng_fields.qpts.vectors[iq_tile]
        end
        bands_kq = solve_electron_bands_batched(tile_workspace.itp_el_ham, tile_workspace.hk,
            eng_fields.model, view(tile_workspace.kqs, 1:n), eng_fields.window_kq;
            eigenvectors = tile_workspace.els_kq.u !== nothing)
        nband_kq = Array(bands_kq.nband)
        nkeep = _select_pairs!(keep, n, j -> nband_kq[j] > 0, j -> isinf(tol) || _conserves_energy(
            view(els_k.e, 1:els_k.nband[1], 1),
            view(bands_kq.E, bands_kq.offset[j] .+ (1:nband_kq[j]), j), view(phs.e, :, tile[j]), tol))
        nkeep == 0 && return nothing
        ikq = nothing

        if nkeep == n
            # The whole tile: the resident phonons as they are.
            els_kq_block = copy_window_bands!(tile_workspace.els_kq, bands_kq, 1:n)
            iq = tile
            phs_block = view(phs, tile)
            wtq = view(eng_fields.wtkq, tile)
            xkq = view(tile_workspace.xkq, :, 1:n)
            @views xkq .= eng_fields.xkq[:, tile]
        else
            # Gather the selected pairs' k+q states, phonons, weights and coordinates.
            els_kq_block = copy_window_bands!(tile_workspace.els_kq, bands_kq, view(keep, 1:nkeep))
            for (i, j) in enumerate(view(keep, 1:nkeep))
                tile_workspace.iq[i] = tile[j]
            end
            iq = view(tile_workspace.iq, 1:nkeep)
            iq_on_backend = _copy_indices_on_backend(eng_fields.xkq, iq, phs.nq)
            phs_block = view(copy_batched_phonon_states!(tile_workspace.phs, phs, iq_on_backend), 1:nkeep)
            wtq = view(_copy_last_axis!(tile_workspace.wtq, eng_fields.wtkq, iq_on_backend), 1:nkeep)
            xkq = view(_copy_last_axis!(tile_workspace.xkq, eng_fields.xkq, iq_on_backend), :, 1:nkeep)
        end
        xq = view(eng_fields.qpts.vectors, iq)

        # Stage 1 includes exp(-2πi R_p·k), so stage 2 needs the phase at k+q, not q.
        xkq .+= view(eng_fields.xk, :, iouter)
        phase = view(tile_workspace.P_kq, :, 1:nkeep)
        build_fourier_phase!(phase, eng_fields.irvecp_mat, xkq)
    end

    # function barrier for the concrete types of the selected pairs' states and indices.
    _contract_pairs!(OuterKLoop(), eng_fields, tile_workspace, iouter, phase, els_k, els_kq_block,
        phs_block, ik, ikq, iq, wtq, xq)
end

function _contract_pairs!(::OuterKLoop, eng_fields, tile_workspace, iouter, phase, els_k, els_kq, phs,
        ik, ikq, iq, wtq, xq)
    # Borrow the tile's output storage for the block.
    block = EPBlock{OuterKLoop}(tile_workspace, els_k, els_kq, phs;
        ik, ikq, iq, wtk = eng_fields.kpts.weights[ik], wtq, xk = eng_fields.kpts.vectors[ik], xq)

    # Fourier-transform R_p and rotate by u_{k+q} and the phonon basis:
    # g_ina(k, R_p) -> g_mnν(k, q), with m the band of k+q and ν the phonon mode
    # (the displacement a itself under `:cartesian`).
    nbkq, nbk, nmodes, npair = size(block.ep)
    g = view(tile_workspace.g, :, 1:npair)
    tmp = reshape_buffer_view(tile_workspace.tmp, nbkq, nbk * nmodes, npair)
    u_ph = tile_workspace.u_ph_id === nothing ? block.phs.u : view(tile_workspace.u_ph_id, :, :, 1:npair)
    get_eph_kR_to_kq_batched!(block.ep, view(eng_fields.ep_kR, :, :, iouter), phase, u_ph, block.els_kq.u;
                              g, tmp)

    # The same for each direction d of the covariant derivative: dg_inad(k, R_p) -> dg_mnνd(k, q).
    if block.dg !== nothing
        dg_d = reshape_buffer_view(tile_workspace.dg_d, nbkq, nbk, nmodes, npair)
        for d in 1:3
            get_eph_kR_to_kq_batched!(dg_d, view(eng_fields.dg_kR, :, :, d, iouter), phase, u_ph,
                                      block.els_kq.u; g, tmp)
            view(block.dg, :, :, :, d, :) .= dg_d
        end
    end

    if eng_fields.model.polar_eph.use
        eph_engine_add_longrange!(block, tile_workspace, eng_fields.model)
    end
    block
end


# ---- OuterQEngine ----------------------------------------------------------------------------

function engine_bytes(::Type{OuterQEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk,
        n_outer_batch, el_qty, ph_qty, precompute_el_kq, eph_phonon_basis) where {FT}
    (; nw, nmodes) = model
    cx, rl, iz = sizeof(Complex{FT}), sizeof(FT), sizeof(Int)
    _require_epmat_layout(OuterQLoop(), model)
    nr_e = length(model.epmat.irvec_next)
    nr_p = length(model.epmat.irvec)
    ndata = nw^2 * nmodes
    nbkq = precompute_el_kq ? nband_max_kq : nw
    persistent =
        cx * length(model.epmat.op_r) +                         # epmat
        rl * 3 * (nr_e + nr_p) +                              # R-vector matrices
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
        _electron_state_bytes(FT, nw, nbkq, el_qty) +           # k+q tile
        cx * nr_e +                                             # stage-2 Fourier phase
        3iz + rl + 3rl +                                        # keep, ik, ikq, wtk, x_k
        (precompute_el_kq ? 0 : cx * length(model.el_ham.irvec)) +   # k+q Fourier phase
        (precompute_el_kq ? 0 : cx * nw^2 * 3) +                # k+q Hamiltonian and its eigensolve
        (model.polar_eph.use ? cx * (nw * nband_max_k + nbkq * nband_max_k) : 0)   # polar scratch
    (; persistent, per_outer, per_pair)
end

"""
    OuterQEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kwargs...)

Build the outer-q run's buffers for a model with epmat_outer_momentum = "ph".
The capacity, quantity, phonon-basis, and energy-conservation arguments have the meanings
documented for OuterKEngine. Here els_kq = nothing selects a per-tile k+q solve; otherwise k+q
states are copied from the resident container.
"""
function OuterQEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, qpts,
        n_outer_batch, n_inner_tile, nchunks, eph_phonon_basis, kqpts = nothing,
        sel_k = nothing, sel_kq = nothing, window_kq = (-Inf, Inf),
        energy_conservation_tol = Inf) where {FT}
    # Validate the model layout and determine the electron band-box dimensions.
    (; nw, nmodes) = model
    _require_epmat_layout(OuterQLoop(), model)
    nbk = els_k.nband_max
    nbkq = els_kq === nothing ? nw : els_kq.nband_max

    # Prepare the stage-1 Fourier interpolator and its read-only stage-2 outputs g(R_e, q).
    epmat = to_device(backend, model.epmat)
    irvec_e = model.epmat.irvec_next
    nr_e = length(irvec_e)
    ndata = nw^2 * nmodes
    itp_epmat = BatchedWannierInterpolator(epmat; backend, batch_size = n_outer_batch)
    el_ham = els_kq === nothing ? to_device(backend, model.el_ham) : nothing

    # Allocate independent k-tile states, interpolators, and scratch for each thread chunk.
    tiles = map(1:nchunks) do _
        # Each chunk owns its Fourier phase; the stage-1 output is read-only in stage 2.
        OuterQTileWorkspace(;
            keep = Vector{Int}(undef, n_inner_tile),
            ik = Vector{Int}(undef, n_inner_tile),
            ikq = Vector{Int}(undef, n_inner_tile),     # 0: k+q absent from the precomputed states
            els_k = BatchedElectronState(backend, nw, nbk, n_inner_tile, el_qty; FT),
            els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
            wtk = alloc(backend, FT, n_inner_tile),
            xk = Vector{Vec3{FT}}(undef, n_inner_tile),
            P_k = alloc(backend, Complex{FT}, nr_e, n_inner_tile),
            itp_el_ham = el_ham === nothing ? nothing :
                BatchedWannierInterpolator(el_ham; backend, batch_size = n_inner_tile),
            hk = els_kq === nothing ? alloc(backend, Complex{FT}, nw^2, n_inner_tile) : nothing,
            kqs = Vector{Vec3{FT}}(undef, n_inner_tile),
            ep = alloc(backend, Complex{FT}, nbkq * nbk * nmodes * n_inner_tile),
            g = alloc(backend, Complex{FT}, ndata, n_inner_tile),
            tmp = alloc(backend, Complex{FT}, nbkq * nw * nmodes * n_inner_tile),
            uk_rep = alloc(backend, Complex{FT}, nw, nbk, nmodes * n_inner_tile),
            uk_polar = model.polar_eph.use ? alloc(backend, Complex{FT}, nw * nbk * n_inner_tile) : nothing,
            mmat_buffer = model.polar_eph.use ? alloc(backend, Complex{FT}, nbkq * nbk * n_inner_tile) : nothing,
        )
    end

    # Assemble the engine with maximum-capacity stage-1 outputs and coordinate buffers.
    OuterQEngine(model, els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq, window_kq,
        energy_conservation_tol, eph_phonon_basis, qpts.n, 1:0, backend, epmat, itp_epmat,
        _irvec_to_device_matrix(backend, irvec_e, FT),
        alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch),
        eph_phonon_basis == :cartesian ? nothing : alloc(backend, Complex{FT}, nw^2, nr_e, nmodes, n_outer_batch),
        alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch),
        to_device_copy(backend, collect(FT, kpts.weights)), zeros(FT, 3, n_outer_batch),
        alloc(backend, FT, 3, n_outer_batch), n_outer_batch, n_inner_tile, tiles)
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
    _check_stage1(eng, iqs_batch)
    eng.batch = iqs_batch
    # function barrier for the outer-q stage-1 buffers.
    _stage1!(OuterQLoop(), _workspace_fields(eng), iqs_batch)
    eng
end

function _stage1!(::OuterQLoop, eng_fields, iqs_batch)
    # Stage the active q coordinates on the backend.
    nq_batch = length(iqs_batch)
    for (j, iq) in enumerate(iqs_batch)
        eng_fields.xq_host[:, j] .= eng_fields.qpts.vectors[iq]
    end
    xq = view(eng_fields.xq, :, 1:nq_batch)
    copyto!(eng_fields.xq, 1, eng_fields.xq_host, 1, 3nq_batch)

    # Fourier-transform R_p for the active outer batch: g_ija(R_e, R_p) -> g_ija(R_e, q).
    ndata, nr_e = size(eng_fields.ep_Rq, 1), size(eng_fields.ep_Rq, 2)
    g = view(eng_fields.g_q, :, :, 1:nq_batch)
    get_fourier_batched!(reshape(g, ndata * nr_e, nq_batch), eng_fields.itp_epmat, xq)

    # Rotate into the phonon eigenmodes, g_ija(R_e, q) -> g_ijν(R_e, q), or keep the cartesian
    # components.
    ep = view(eng_fields.ep_Rq, :, :, 1:nq_batch)
    if eng_fields.eph_phonon_basis == :cartesian
        ep .= g
    else
        # ep[ij, ν, R_e, q] = Σ_a g[ij, a, R_e, q] u_ph[a, ν, q], as one batched GEMM over q on the
        # (ij, R_e) × a layout.
        nw² = size(eng_fields.g_rot, 1)
        nmodes = size(eng_fields.g_rot, 3)
        g_rot = view(eng_fields.g_rot, :, :, :, 1:nq_batch)
        permutedims!(g_rot, reshape(g, nw², nmodes, nr_e, nq_batch), (1, 3, 2, 4))
        ep_rot = reshape(view(eng_fields.g_q, :, :, 1:nq_batch), nw² * nr_e, nmodes, nq_batch)   # g is consumed
        batched_gemm!('N', 'N', reshape(g_rot, nw² * nr_e, nmodes, nq_batch),
                      view(eng_fields.phs.u, :, :, iqs_batch), ep_rot)
        permutedims!(reshape(ep, nw², nmodes, nr_e, nq_batch), reshape(ep_rot, nw², nr_e, nmodes, nq_batch), (1, 3, 2, 4))
    end
    eng_fields
end

"""
    stage2!(eng::OuterQEngine, iq, k_indices; chunk=1)

Return the calculator-ready `EPBlock` of outer q point `iq` and a tile of k points: solve the k+q
states or look them up in the precomputed container, gather the states of the pairs to compute
(`_select_pairs!`), compute their e-ph matrix, and add polar corrections. Return `nothing` if no
pair is selected.
Call `stage1!` first; `iq` must be in its current batch.

The block borrows this chunk's buffers: its arrays are valid until that chunk's next `stage2!`
or the next `stage1!`. Consume it immediately or copy the arrays you need to retain. Calls on
different CPU chunks may run concurrently, but `stage1!` must wait for all of them to finish.

- `iq`: Index into `eng.qpts`, not a batch-local index.
- `k_indices`: Nonempty contiguous range into `eng.kpts`.
- `chunk`: Independent workspace slot; use 1 for serial calls and GPU calls.
"""
function stage2!(eng::OuterQEngine, iq::Int, tile::UnitRange{Int}; chunk::Int = 1)
    _check_stage2(eng, iq, tile, chunk, eng.kpts.n)
    # function barrier for the concrete resident states, engine buffers, and chunk scratch.
    _stage2!(OuterQLoop(), _workspace_fields(eng), _workspace_fields(eng.tiles[chunk]), iq, tile)
end

function _stage2!(::OuterQLoop, eng_fields, tile_workspace, iq, tile)
    n = length(tile)
    tol = eng_fields.energy_conservation_tol
    (; keep) = tile_workspace
    (; els_k, phs) = eng_fields
    for (j, ik) in enumerate(tile)
        tile_workspace.kqs[j] = eng_fields.kpts.vectors[ik] + eng_fields.qpts.vectors[iq]
    end

    # Solve or look up k+q for this q and k tile, select the pairs to compute and gather their
    # k+q states.
    if eng_fields.els_kq === nothing
        bands_kq = solve_electron_bands_batched(tile_workspace.itp_el_ham, tile_workspace.hk,
            eng_fields.model, view(tile_workspace.kqs, 1:n), eng_fields.window_kq;
            eigenvectors = tile_workspace.els_kq.u !== nothing)
        nband_kq = Array(bands_kq.nband)
        nkeep = _select_pairs!(keep, n, j -> nband_kq[j] > 0, j -> isinf(tol) || _conserves_energy(
            view(els_k.e, 1:els_k.nband[tile[j]], tile[j]),
            view(bands_kq.E, bands_kq.offset[j] .+ (1:nband_kq[j]), j), view(phs.e, :, iq), tol))
        nkeep == 0 && return nothing
        els_kq_block = copy_window_bands!(tile_workspace.els_kq, bands_kq, nkeep == n ? (1:n) : view(keep, 1:nkeep))
        ikq = nothing
    else
        # A k+q with no state in the window is absent from the precomputed states.
        els_kq = eng_fields.els_kq
        ikq_host = tile_workspace.ikq
        for j in 1:n
            ikq_host[j] = something(xk_to_ik_unsafe(tile_workspace.kqs[j], eng_fields.kqpts), 0)
        end
        nkeep = _select_pairs!(keep, n, j -> ikq_host[j] != 0, j -> isinf(tol) || _conserves_energy(
            view(els_k.e, 1:els_k.nband[tile[j]], tile[j]),
            view(els_kq.e, 1:els_kq.nband[ikq_host[j]], ikq_host[j]), view(phs.e, :, iq), tol))
        nkeep == 0 && return nothing
        for (i, j) in enumerate(view(keep, 1:nkeep))
            ikq_host[i] = ikq_host[j]
        end
        ikq = view(ikq_host, 1:nkeep)
        els_kq_block = view(copy_batched_electron_states!(tile_workspace.els_kq, els_kq, ikq), 1:nkeep)
    end

    # Gather the selected pairs' k states, weights and coordinates.
    if nkeep == n
        ik = tile
        els_k_block = view(copy_batched_electron_states!(tile_workspace.els_k, els_k, tile), 1:n)
        wtk = view(eng_fields.wtk, tile)
        xk = view(eng_fields.kpts.vectors, tile)
    else
        for (i, j) in enumerate(view(keep, 1:nkeep))
            tile_workspace.ik[i] = tile[j]
            tile_workspace.xk[i] = eng_fields.kpts.vectors[tile[j]]
        end
        ik = view(tile_workspace.ik, 1:nkeep)
        ik_on_backend = _copy_indices_on_backend(eng_fields.wtk, ik, els_k.nk)
        els_k_block = view(copy_batched_electron_states!(tile_workspace.els_k, els_k, ik_on_backend), 1:nkeep)
        wtk = view(_copy_last_axis!(tile_workspace.wtk, eng_fields.wtk, ik_on_backend), 1:nkeep)
        xk = view(tile_workspace.xk, 1:nkeep)
    end

    # function barrier for the concrete types of the selected pairs' states and indices.
    _contract_pairs!(OuterQLoop(), eng_fields, tile_workspace, els_k_block, els_kq_block, ik, ikq, iq, wtk, xk)
end

function _contract_pairs!(::OuterQLoop, eng_fields, tile_workspace, els_k, els_kq, ik, ikq, iq, wtk, xk)
    # Borrow the tile's output storage for the block.
    block = EPBlock{OuterQLoop}(tile_workspace, els_k, els_kq, view(eng_fields.phs, iq:iq);
        ik, ikq, iq, wtk, wtq = eng_fields.qpts.weights[iq], xk, xq = eng_fields.qpts.vectors[iq])

    # Fourier-transform R_e of this q's stage-1 output at the tile's k points:
    # g_ijν(R_e, q) -> g_ijν(k, q).
    iouter = iq - first(eng_fields.batch) + 1
    nbkq, nbk, nmodes, npair = size(block.ep)
    nw = block.els_k.nw
    g = view(tile_workspace.g, :, 1:npair)
    phase = view(tile_workspace.P_k, :, 1:npair)
    xkmat = _kpoints_to_device_matrix(eng_fields.backend, block.xk)
    _fourier_batched!(g, view(eng_fields.ep_Rq, :, :, iouter), phase, eng_fields.irvece_mat, xkmat)

    # Rotate by u_k and u_{k+q}: g_ijν(k, q) -> g_mnν(k, q), with m, n the bands of k+q and k.
    tmp = reshape_buffer_view(tile_workspace.tmp, nbkq, nw * nmodes, npair)
    uk_rep = reshape_buffer_view(tile_workspace.uk_rep, nw, nbk, nmodes * npair)
    eph_apply_rotations_rqkq!(block.ep, g, block.els_k.u, block.els_kq.u, tmp, uk_rep)

    if eng_fields.model.polar_eph.use
        eph_engine_add_longrange!(block, tile_workspace, eng_fields.model)
    end
    block
end


# ---- Both orders -----------------------------------------------------------------------------

function _check_stage1(eng, batch)
    isempty(batch) && throw(ArgumentError("stage1! needs a nonempty outer batch"))
    (first(batch) >= 1 && last(batch) <= eng.n_outer) || throw(BoundsError(1:eng.n_outer, batch))
    length(batch) <= eng.n_outer_batch ||
        throw(ArgumentError("outer batch exceeds the engine's capacity $(eng.n_outer_batch)"))
    nothing
end

function _check_stage2(eng, outer_index, tile, chunk, n_inner)
    outer_index ∈ eng.batch || throw(ArgumentError(
        "outer point $outer_index is not in the current stage1! batch $(eng.batch)"))
    1 <= chunk <= length(eng.tiles) || throw(BoundsError(eng.tiles, chunk))
    isempty(tile) && throw(ArgumentError("stage2! needs a nonempty inner tile"))
    (first(tile) >= 1 && last(tile) <= n_inner) || throw(BoundsError(1:n_inner, tile))
    length(tile) <= eng.n_inner_tile || throw(ArgumentError(
        "inner tile exceeds the engine's capacity $(eng.n_inner_tile)"))
    nothing
end

public OuterKEngine, OuterQEngine, stage1!, stage2!

"""
    LoopContext(eng::Union{OuterKEngine, OuterQEngine}; chunk=1)

The calculator context for the engine's current stage-1 batch and selected workspace slot.
Call `stage1!` before constructing it, and construct a new context after changing the batch.
"""
function LoopContext(eng::Union{OuterKEngine, OuterQEngine}; chunk::Int = 1)
    isempty(eng.batch) && throw(ArgumentError("call stage1! before constructing the engine's LoopContext"))
    1 <= chunk <= length(eng.tiles) || throw(BoundsError(eng.tiles, chunk))
    order = eng isa OuterKEngine ? OuterKLoop() : OuterQLoop()
    LoopContext(eng.backend, order, eng.batch, chunk)
end

# Select the pairs of a tile to compute, before any of their states is gathered: `keep[1:nkeep]`
# lists, in order, the pairs `j` of `1:n` (i) whose k+q has a state in the window
# (`has_kq_state(j)`), and (ii) that have a process inside the energy-conservation tolerance
# (`conserves(j)`). The other pairs need no e-ph matrix. Returns `nkeep`.
function _select_pairs!(keep, n, has_kq_state, conserves)
    nkeep = 0
    for j in 1:n
        has_kq_state(j) && conserves(j) || continue
        keep[nkeep += 1] = j
    end
    nkeep
end

# Whether a pair has a process inside the energy-conservation tolerance,
# |e_k - e_{k+q} ± ω_q| <= energy_conservation_tol for some bands and mode, given the window
# energies `e_k` and `e_kq` of its two electron states and its phonon energies `ω_q`.
function _conserves_energy(e_k, e_kq, ω_q, energy_conservation_tol)
    any(abs(e_nk - e_mkq - sign_ph * ω) <= energy_conservation_tol
        for ω in ω_q, e_mkq in e_kq, e_nk in e_k, sign_ph in (-1, 1))
end

"""
    eph_engine_add_longrange!(block, tile_workspace, model)

Add the long-range term on a block's `ep` after the two Fourier transforms, for either loop order:
the polar dipole term `coeff[ν] · u_{k+q}' u_k` of a polar model (unscreened, as
`epstate_compute_eph_dipole!`), with the coefficients in the block's phonon basis.
`tile_workspace` is the block's tile, for the scratch. A no-op for a non-polar model.
"""
function eph_engine_add_longrange!(block, tile_workspace, model)
    model.polar_eph.use || return block
    nbkq, nbk, _, n = size(block.ep)
    nw = model.nw
    # u_k at the block's pair extent: the shared side of `OuterKLoop` has extent 1.
    uk = reshape_buffer_view(tile_workspace.uk_polar, nw, nbk, n)
    uk .= block.els_k.u
    ukq = block.els_kq.u
    mmat_buffer = reshape_buffer_view(tile_workspace.mmat_buffer, nbkq, nbk, n)
    add_eph_dipole_batched!(block.ep, block.phs.eph_dipole_coeff, ukq, uk, mmat_buffer)
    block
end
