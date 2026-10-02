# The e-ph interpolation engines of the two loop orders (`_run_eph`, run_eph.jl). Each holds one
# run's device copy of `model.epmat`, its stage-1 output for one outer batch, and one tile of
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
# `epmat` layouts: `op_r` rows are `(a, ν, R_row)` (a = Wannier pair) and its columns are the other R,
# with `epmat_outer_momentum` naming the column R ("el": R_e, "ph": R_p). A stage that contracts the
# column R is one GEMM against `op_r` (`get_fourier_batched!`); a stage that contracts the row R is
# one strided-batched GEMM over the columns against a shared phase (`_fourier_rows_batched!`).

# `out[a, ic, j] = Σ_ir op_r[(a, ir), ic] · phase[ir, j]` for `op_r` `(nd · nr_row, nr_col)`: the
# Fourier transform over the row-block R of a two-R object, as one strided-batched GEMM over the
# columns with the phase shared by all of them (batch extent 1). `scratch` is `(nd, nj, nr_col)`.
function _fourier_rows_batched!(out, op_r, phase, scratch)
    nd, nj, nr_col = size(scratch)
    nr_row = size(phase, 1)
    @assert size(op_r) == (nd * nr_row, nr_col)
    @assert size(out) == (nd, nr_col, nj)
    batched_gemm!('N', 'N', reshape(op_r, nd, nr_row, nr_col), reshape(phase, nr_row, nj, 1), scratch)
    permutedims!(out, scratch, (1, 3, 2))
end


"""
    OuterKEngine

The `OuterKLoop` engine: g(k, R_p) for an outer-k batch (stage 1), then g(k, k+q) for one k and a
tile of k+q (stage 2), in the k+q convention of [`get_eph_RR_to_kR_batched!`](@ref): stage 1 folds
`exp(-2πi R_p · x_k)` into g(k, R_p), so the stage-2 phase `exp(2πi R_p · x_{k+q})` of a tile is
shared by every k of the batch. With `covariant_derivative_of_g`, the same two stages run on the
position-weighted `epmat_R` (`wannier_object_multiply_R` plus the tight-binding term
`im (r_j - r_i) g`) into `dg`.
"""
struct OuterKEngine{T_backend, T_epmat, T_itp_epmat, T_irvec_e_mat, T_itp_epmat_R, T_irvecp_mat,
        T_mxk, T_xkq, T_wtkq, T_P_mk, T_P_e, T_row_scratch, T_ep_kR, T_dg_kR, T_els_k_batch, T_xk,
        T_g_fourier, T_tiles}
    backend      :: T_backend
    epmat        :: T_epmat         # model.epmat on the backend
    itp_epmat    :: T_itp_epmat     # its interpolator ("el" layout), or `nothing` ("ph" layout)
    irvec_e_mat  :: T_irvec_e_mat   # (nr_e, 3) R_e of the row contraction ("ph" layout), or `nothing`
    itp_epmat_R  :: T_itp_epmat_R   # interpolator of epmat_R (dg), or `nothing`
    irvecp_mat   :: T_irvecp_mat    # (nr_p, 3) R_p
    mxk          :: T_mxk           # (3, nk) -x_k
    xkq          :: T_xkq           # (3, nkq) x_{k+q}
    wtkq         :: T_wtkq          # (nkq,) k+q weights
    xks_int      :: Matrix{Int}     # (3, nk) k grid coordinates, reduced
    xkqs_int     :: Matrix{Int}     # (3, nkq) k+q grid coordinates, reduced, minus the q shift
    P_mk         :: T_P_mk          # (nr_p, n_outer_batch) exp(-2πi R_p · x_k)
    P_e          :: T_P_e           # (nr_e, n_outer_batch) row-contraction phase, or `nothing`
    row_scratch  :: T_row_scratch   # (nw² nmodes, n_outer_batch, nr_p) for it, or `nothing`
    ep_kR        :: T_ep_kR         # (nw nband_max_k nmodes, nr_p, n_outer_batch) stage-1 output
    dg_kR        :: T_dg_kR         # (nw nband_max_k nmodes, nr_p, 3, n_outer_batch), or `nothing`
    els_k_batch   :: T_els_k_batch    # the outer batch's k states
    xk_host      :: Matrix{Float64} # (3, n_outer_batch) the batch's x_k, staged for `xk`
    xk           :: T_xk            # (3, n_outer_batch) the batch's x_k on the backend
    g_fourier    :: T_g_fourier     # stage-1 Fourier output, `dense_prefix` per use
    n_inner_tile :: Int
    tiles        :: T_tiles
end

"""
    OuterQEngine

The `OuterQLoop` engine: g(R_e, q) for an outer-q batch with the phonon basis applied (stage 1),
then g(k, k+q) for one q and a tile of k (stage 2), the k+q states solved per tile into the
buffers' leading `maximum(nband)` columns when they are not precomputed.
"""
struct OuterQEngine{T_backend, T_epmat, T_itp_epmat, T_irvec_p_mat, T_P_p, T_row_scratch, T_g_q,
        T_g_rot, T_ep_Rq, T_eRpq, T_wtk, T_xq, T_tiles}
    backend      :: T_backend
    epmat        :: T_epmat         # model.epmat on the backend
    itp_epmat    :: T_itp_epmat     # its interpolator ("ph" layout), or `nothing` ("el" layout)
    irvec_p_mat  :: T_irvec_p_mat   # (nr_p, 3) R_p of the row contraction ("el" layout), or `nothing`
    P_p          :: T_P_p           # (nr_p, n_outer_batch) row-contraction phase, or `nothing`
    row_scratch  :: T_row_scratch   # (nw² nmodes, n_outer_batch, nr_e) for it, or `nothing`
    g_q          :: T_g_q           # (nw² nmodes, nr_e, n_outer_batch) stage-1 Fourier output
    g_rot        :: T_g_rot         # (nw², nr_e, nmodes, n_outer_batch) basis scratch, or `nothing`
    ep_Rq        :: T_ep_Rq         # (nw² nmodes, nr_e, n_outer_batch) stage-1 output
    eRpq         :: T_eRpq          # g(R_e, q) of the current q, the tiles' stage-2 parent
    wtk          :: T_wtk           # (nk,) k weights
    xq_host      :: Matrix{Float64} # (3, n_outer_batch) the batch's x_q, staged for `xq`
    xq           :: T_xq            # (3, n_outer_batch) the batch's x_q on the backend
    n_inner_tile :: Int
    tiles        :: T_tiles
end


# ---- OuterKEngine ----------------------------------------------------------------------------

function engine_bytes(::Type{OuterKEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk, nkq,
        el_qty, ph_qty, drop_pairs, covariant_derivative_of_g, eph_phonon_basis) where {FT}
    (; nw, nmodes) = model
    cx, rl, iz = sizeof(Complex{FT}), sizeof(FT), sizeof(Int)
    el_layout = model.epmat_outer_momentum == "el"
    nr_p = length(el_layout ? model.epmat.irvec_next : model.epmat.irvec)
    nr_e = length(el_layout ? model.epmat.irvec : model.epmat.irvec_next)
    nepmat = length(model.epmat.op_r)
    ndata = nw * nband_max_k * nmodes
    nd = covariant_derivative_of_g ? 4 : 1                      # ep, plus three dg directions
    nrows = nw^2 * nmodes * nr_p                                # the epmat rows that stage 1 keeps
    nrows_max = nrows * (covariant_derivative_of_g ? 3 : 1)     # those of epmat_R with dg
    persistent =
        cx * nepmat * (covariant_derivative_of_g ? 4 : 1) +    # epmat (+ epmat_R)
        rl * 3 * (nr_p + nr_e) * (covariant_derivative_of_g ? 2 : 1) +  # R-vector matrices
        (el_layout ? cx * nrows : 0) + (covariant_derivative_of_g ? cx * 3nrows : 0) +  # interpolator outputs
        rl * 3 * (nk + nkq) + rl * nkq                          # mxk, xkq, wtkq
    per_outer =
        cx * ndata * nr_p * nd +                                # ep_kR (+ dg_kR)
        cx * nr_p + rl * 3 + iz +                               # P_mk, xk, a partial batch's k index
        cx * nr_e * (covariant_derivative_of_g ? 2 : 1) +       # Fourier phases
        (el_layout ? 0 : cx * nrows) +                          # row-contraction scratch
        cx * nrows_max +                                        # g_fourier
        cx * (nrows + 2 * nband_max_k * nrows ÷ nw) * nd +      # transients of eph_rotate_kR_batched!
        _electron_state_bytes(FT, nw, nband_max_k, el_qty)      # els_k_batch
    nbox = nband_max_kq * nband_max_k * nmodes
    per_pair =
        cx * nbox * (covariant_derivative_of_g ? 5 : 1) +       # ep (+ dg and its per-direction scratch)
        cx * ndata + cx * nbox +                                # stage-2 scratch g, tmp
        cx * nr_p + 2iz +                                       # P_kq, iq
        _phonon_state_bytes(FT, nmodes, ph_qty) +               # phs tile
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

function OuterKEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, kqpts, qpts,
        n_outer_batch, n_inner_tile, nchunks, drop_pairs, covariant_derivative_of_g,
        eph_phonon_basis) where {FT}
    (; nw, nmodes) = model
    nbk, nbkq = els_k.nband_max, els_kq.nband_max
    el_layout = model.epmat_outer_momentum == "el"
    epmat = to_device(backend, model.epmat)
    irvec_p = el_layout ? model.epmat.irvec_next : model.epmat.irvec
    nr_p = length(irvec_p)
    if el_layout
        itp_epmat = BatchedWannierInterpolator(epmat; backend, batch_size = n_outer_batch)
        irvec_e_mat = P_e = row_scratch = nothing
    else
        itp_epmat = nothing
        irvec_e_mat = _irvec_to_device_matrix(backend, model.epmat.irvec_next, FT)
        P_e = alloc(backend, Complex{FT}, length(model.epmat.irvec_next), n_outer_batch)
        row_scratch = alloc(backend, Complex{FT}, nw^2 * nmodes, n_outer_batch, nr_p)
    end
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

    # Grid coordinates as (3 × n) device matrices for the two phase builds. -x_k out of place: on
    # `CPUBackend` `_kpoints_to_device_matrix` is a view onto `kpts.vectors`.
    mxk = _kpoints_to_device_matrix(backend, kpts) .* -1
    xkq = _kpoints_to_device_matrix(backend, kqpts)
    # The q index of a pair by integer grid hash (`_fill_iqs!`): both coordinate lists reduced into
    # `0:ng-1` once, the q-grid shift folded into the k+q side.
    xkqs_int = Matrix{Int}(undef, 3, kqpts.n)
    xks_int = Matrix{Int}(undef, 3, kpts.n)
    for ikq in 1:kqpts.n
        xkqs_int[:, ikq] .= _grid_coords_reduced(kqpts.vectors[ikq], qpts.ngrid, qpts.shift)
    end
    for ik in 1:kpts.n
        xks_int[:, ik] .= _grid_coords_reduced(kpts.vectors[ik], qpts.ngrid, zero(Vec3{FT}))
    end
    # Only columns 1:nb of P_mk are rewritten for a partial last batch. The padded columns of ep_kR
    # (the repeated last k) are never read: they hold g(k, R_p) times 1 when the partial batch is
    # the first, otherwise times the phase a previous batch left in that column; finite either way.
    P_mk = fill!(alloc(backend, Complex{FT}, nr_p, n_outer_batch), 1)
    ndata = nw * nbk * nmodes

    tiles = map(1:nchunks) do _
        tile_bufs = (;
            phs = BatchedPhononState(backend, nmodes, n_inner_tile, ph_qty; FT),
            iq = Vector{Int}(undef, n_inner_tile),
            iq_dev = alloc(backend, Int, n_inner_tile),
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
        )
        merge(tile_bufs, (; kept = drop_pairs ? (;
            keep = Vector{Int}(undef, n_inner_tile),
            keep_dev = alloc(backend, Int, n_inner_tile),
            els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
            phs = BatchedPhononState(backend, nmodes, n_inner_tile, ph_qty; FT),
            P_kq = alloc(backend, Complex{FT}, nr_p, n_inner_tile),
            ikq = Vector{Int}(undef, n_inner_tile),
            iq = Vector{Int}(undef, n_inner_tile),
            iq_dev = alloc(backend, Int, n_inner_tile),
            wtq = alloc(backend, FT, n_inner_tile),
            xq = Vector{Vec3{FT}}(undef, n_inner_tile)) : nothing))
    end
    nrows_max = nw^2 * nmodes * nr_p * (covariant_derivative_of_g ? 3 : 1)
    OuterKEngine(backend, epmat, itp_epmat, irvec_e_mat, itp_epmat_R,
        _irvec_to_device_matrix(backend, irvec_p, FT), mxk, xkq, to_device_copy(backend, collect(FT, kqpts.weights)), xks_int, xkqs_int, P_mk, P_e,
        row_scratch, alloc(backend, Complex{FT}, ndata, nr_p, n_outer_batch),
        covariant_derivative_of_g ? alloc(backend, Complex{FT}, ndata, nr_p, 3, n_outer_batch) : nothing,
        BatchedElectronState(backend, nw, nbk, n_outer_batch, el_qty; FT),
        zeros(FT, 3, n_outer_batch), alloc(backend, FT, 3, n_outer_batch), alloc(backend, Complex{FT}, nrows_max * n_outer_batch),
        n_inner_tile, tiles)
end

"""
    stage1!(eng::OuterKEngine, els_k, kpts, batch)

g(k, R_p) for the outer k points `batch`, rotated by `u_k` and multiplied by `exp(-2πi R_p · x_k)`,
into `eng.ep_kR` (and `eng.dg_kR`). A partial last batch is padded with its last k, so the batched
kernels run on the full-width buffers.
"""
function stage1!(eng::OuterKEngine, els_k, kpts, batch)
    nb = length(batch)
    nmax = size(eng.xk_host, 2)
    iks = nb == nmax ? batch : [batch; fill(last(batch), nmax - nb)]
    copy_batched_electron_states!(eng.els_k_batch, els_k, iks)
    for (j, ik) in enumerate(iks)
        eng.xk_host[:, j] .= kpts.vectors[ik]
    end
    copyto!(eng.xk, eng.xk_host)
    @views build_fourier_phase!(eng.P_mk[:, 1:nb], eng.irvecp_mat, eng.mxk[:, batch])
    uks = eng.els_k_batch.u
    nr_p = size(eng.P_mk, 1)
    if eng.itp_epmat !== nothing
        g = dense_prefix(eng.g_fourier, eng.itp_epmat.parent.ndata, nmax)
        get_fourier_batched!(g, eng.itp_epmat, eng.xk)
    else
        build_fourier_phase!(eng.P_e, eng.irvec_e_mat, eng.xk)
        nd = size(eng.row_scratch, 1)
        g = dense_prefix(eng.g_fourier, nd * nr_p, nmax)
        _fourier_rows_batched!(reshape(g, nd, nr_p, nmax), eng.epmat.op_r, eng.P_e, eng.row_scratch)
    end
    eph_rotate_kR_batched!(eng.ep_kR, g, uks; additional_phase = eng.P_mk)
    if eng.itp_epmat_R !== nothing
        g = dense_prefix(eng.g_fourier, eng.itp_epmat_R.parent.ndata, nmax)
        get_fourier_batched!(g, eng.itp_epmat_R, eng.xk)
        eph_rotate_kR_batched!(eng.dg_kR, g, uks; additional_phase = eng.P_mk)
    end
    eng
end

"""
    stage2!(eng::OuterKEngine, tile_bufs, pairs)

g(k, k+q) of the block `pairs` (one k, `pairs.n` k+q points) into `tile_bufs.ep` (and
`tile_bufs.dg`): the kR→kq contraction on the tile's phase `pairs.phase`, the k+q rotation and the
phonon basis (`pairs.phs.u`, or the identity for `:cartesian`).
"""
function stage2!(eng::OuterKEngine, tile_bufs, pairs)
    (; n, iouter) = pairs
    ep = view(tile_bufs.ep, :, :, :, 1:n)
    ws = (; g = view(tile_bufs.g, :, 1:n), tmp = view(tile_bufs.tmp, :, :, 1:n))
    u_ph = tile_bufs.u_ph_id === nothing ? pairs.phs.u : view(tile_bufs.u_ph_id, :, :, 1:n)   # the phonon basis
    get_eph_kR_to_kq_batched!(ep, view(eng.ep_kR, :, :, iouter), pairs.phase, u_ph, pairs.els_kq.u; ws...)
    tile_bufs.dg === nothing && return ep, nothing
    dg = view(tile_bufs.dg, :, :, :, :, 1:n)
    dg_d = view(tile_bufs.dg_d, :, :, :, 1:n)
    for d in 1:3
        get_eph_kR_to_kq_batched!(dg_d, view(eng.dg_kR, :, :, d, iouter), pairs.phase, u_ph,
                                  pairs.els_kq.u; ws...)
        view(dg, :, :, :, d, :) .= dg_d
    end
    ep, dg
end


# ---- OuterQEngine ----------------------------------------------------------------------------

function engine_bytes(::Type{OuterQEngine}, model::Model{FT}; nband_max_k, nband_max_kq, nk,
        n_outer_batch, el_qty, ph_qty, drop_pairs, precompute_el_kq, eph_phonon_basis) where {FT}
    (; nw, nmodes) = model
    cx, rl, iz = sizeof(Complex{FT}), sizeof(FT), sizeof(Int)
    el_layout = model.epmat_outer_momentum == "el"
    nr_e = length(el_layout ? model.epmat.irvec : model.epmat.irvec_next)
    nr_p = length(el_layout ? model.epmat.irvec_next : model.epmat.irvec)
    ndata = nw^2 * nmodes
    nbkq = precompute_el_kq ? nband_max_kq : nw
    persistent =
        cx * length(model.epmat.op_r) +                         # epmat
        cx * ndata * nr_e + rl * 3 * (nr_e + nr_p) +            # eRpq, R-vector matrices
        (el_layout ? 0 : cx * ndata * nr_e) +                   # interpolator output
        (precompute_el_kq ? 0 : cx * length(model.el_ham.op_r)) +  # el_ham
        rl * nk                                                 # wtk
    per_outer =
        cx * ndata * nr_e * (eph_phonon_basis == :cartesian ? 2 : 3) +   # g_q, ep_Rq (+ g_rot)
        cx * nr_p + (el_layout ? cx * ndata * nr_e : 0) +       # Fourier phase / row scratch
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

function OuterQEngine(model::Model{FT}, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, qpts,
        n_outer_batch, n_inner_tile, nchunks, drop_pairs, eph_phonon_basis) where {FT}
    (; nw, nmodes) = model
    nbk = els_k.nband_max
    el_layout = model.epmat_outer_momentum == "el"
    epmat = to_device(backend, model.epmat)
    irvec_e = el_layout ? model.epmat.irvec : model.epmat.irvec_next
    nr_e = length(irvec_e)
    ndata = nw^2 * nmodes
    if el_layout
        itp_epmat = nothing
        irvec_p_mat = _irvec_to_device_matrix(backend, model.epmat.irvec_next, FT)
        P_p = alloc(backend, Complex{FT}, length(model.epmat.irvec_next), n_outer_batch)
        row_scratch = alloc(backend, Complex{FT}, ndata, n_outer_batch, nr_e)
    else
        itp_epmat = BatchedWannierInterpolator(epmat; backend, batch_size = n_outer_batch)
        irvec_p_mat = P_p = row_scratch = nothing
    end
    eRpq = WannierObject(irvec_e, alloc_zeros(backend, Complex{FT}, ndata, nr_e))
    el_ham = els_kq === nothing ? to_device(backend, model.el_ham) : nothing
    # The k+q box: the precomputed container's, or `nw` for the per-tile solve, whose blocks use its
    # leading `maximum(nband)` columns.
    nbkq = els_kq === nothing ? nw : els_kq.nband_max

    tiles = map(1:nchunks) do _
        # Each tile has its own interpolators: their phase scratch is written per call.
        tile_bufs = (;
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
        )
        merge(tile_bufs, (; kept = drop_pairs ? (;
            keep = Vector{Int}(undef, n_inner_tile),
            keep_dev = alloc(backend, Int, n_inner_tile),
            els_k = BatchedElectronState(backend, nw, nbk, n_inner_tile, el_qty; FT),
            els_kq = BatchedElectronState(backend, nw, nbkq, n_inner_tile, el_qty; FT),
            ik = Vector{Int}(undef, n_inner_tile),
            ikq = Vector{Int}(undef, n_inner_tile),
            wtk = alloc(backend, FT, n_inner_tile),
            xk = Vector{Vec3{FT}}(undef, n_inner_tile)) : nothing))
    end
    OuterQEngine(backend, epmat, itp_epmat, irvec_p_mat, P_p, row_scratch,
        alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch),
        eph_phonon_basis == :cartesian ? nothing : alloc(backend, Complex{FT}, nw^2, nr_e, nmodes, n_outer_batch),
        alloc(backend, Complex{FT}, ndata, nr_e, n_outer_batch), eRpq,
        to_device_copy(backend, collect(FT, kpts.weights)), zeros(FT, 3, n_outer_batch),
        alloc(backend, FT, 3, n_outer_batch), n_inner_tile, tiles)
end

"""
    stage1!(eng::OuterQEngine, phs, qpts, batch, eph_phonon_basis)

g(R_e, q) for the outer q points `batch` in the phonon basis `eph_phonon_basis` (`:eigenmode`
rotates the modes by `phs.u`, `:cartesian` leaves them), into `eng.ep_Rq[:, :, 1:length(batch)]`.
"""
function stage1!(eng::OuterQEngine, phs, qpts, batch, eph_phonon_basis)
    nb = length(batch)
    for (j, iq) in enumerate(batch)
        eng.xq_host[:, j] .= qpts.vectors[iq]
    end
    copyto!(eng.xq, eng.xq_host)
    xq = view(eng.xq, :, 1:nb)
    ndata, nr_e = size(eng.ep_Rq, 1), size(eng.ep_Rq, 2)
    g = view(eng.g_q, :, :, 1:nb)
    if eng.itp_epmat !== nothing
        get_fourier_batched!(reshape(g, ndata * nr_e, nb), eng.itp_epmat, xq)
    else
        P = view(eng.P_p, :, 1:nb)
        build_fourier_phase!(P, eng.irvec_p_mat, xq)
        _fourier_rows_batched!(g, eng.epmat.op_r, P, view(eng.row_scratch, :, 1:nb, :))
    end
    ep = view(eng.ep_Rq, :, :, 1:nb)
    if eph_phonon_basis == :cartesian
        ep .= g
    else
        # ep[(a, ν'), r, q] = Σ_ν g[(a, ν), r, q] u_ph[ν, ν', q], as one batched GEMM over q on the
        # (a, r) × ν layout.
        nw2 = size(eng.g_rot, 1)
        nmodes = size(eng.g_rot, 3)
        g_rot = view(eng.g_rot, :, :, :, 1:nb)
        permutedims!(g_rot, reshape(g, nw2, nmodes, nr_e, nb), (1, 3, 2, 4))
        ep_rot = reshape(view(eng.g_q, :, :, 1:nb), nw2 * nr_e, nmodes, nb)   # g is consumed
        batched_gemm!('N', 'N', reshape(g_rot, nw2 * nr_e, nmodes, nb), view(phs.u, :, :, batch), ep_rot)
        permutedims!(reshape(ep, nw2, nmodes, nr_e, nb), reshape(ep_rot, nw2, nr_e, nmodes, nb), (1, 3, 2, 4))
    end
    eng
end

"""
    stage2!(eng::OuterQEngine, tile_bufs, pairs)

g(k, k+q) of the block `pairs` (one q, `pairs.n` k points) into the leading
`(pairs.els_kq.nband_max, pairs.els_k.nband_max, nmodes, pairs.n)` of `tile_bufs.ep`, from `eng.eRpq`
(the current q).
"""
function stage2!(eng::OuterQEngine, tile_bufs, pairs)
    (; n) = pairs
    nbkq, nbk = pairs.els_kq.nband_max, pairs.els_k.nband_max
    nw, nmodes = size(tile_bufs.uk_rep, 1), pairs.phs.nmodes
    ep = dense_prefix(tile_bufs.ep, nbkq, nbk, nmodes, n)
    get_eph_Rq_to_kq_batched!(ep, tile_bufs.itp_eRpq, pairs.xk, pairs.els_k.u, pairs.els_kq.u;
        g = view(tile_bufs.g, :, 1:n), tmp = dense_prefix(tile_bufs.tmp, nbkq, nw * nmodes, n),
        uk_rep = dense_prefix(tile_bufs.uk_rep, nw, nbk, nmodes * n))
    ep, nothing
end


# ---- Both orders -----------------------------------------------------------------------------

"""
    finish_ep!(block, tile_bufs, model)

The terms added on a block's `ep` after the two stages, for either loop order: the polar dipole
term `coeff[ν] · u_{k+q}' u_k` of a polar model (unscreened, as `epstate_compute_eph_dipole!`),
with the coefficients in the block's phonon basis. `tile_bufs` is the block's tile, for the scratch.
"""
function finish_ep!(block, tile_bufs, model)
    model.polar_eph.use || return block
    nbkq, nbk, _, n = size(block.ep)
    nw = model.nw
    # u_k at the block's pair extent: the shared side of `OuterKLoop` has extent 1.
    uk = dense_prefix(tile_bufs.uk_polar, nw, nbk, n)
    uk .= block.els_k.u
    add_eph_dipole_batched!(block.ep, block.phs.eph_dipole_coeff, block.els_kq.u, uk,
                            dense_prefix(tile_bufs.mmat, nbkq, nbk, n))
    block
end
