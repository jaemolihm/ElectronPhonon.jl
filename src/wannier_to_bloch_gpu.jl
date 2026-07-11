using LinearAlgebra
using ElectronPhonon.AllocatedLAPACK: HermitianEigenWsSYEV, syev!

export eigvals_batched
export eigen_batched
export get_el_eigen_batched
export get_el_eigen_valueonly_batched
export get_el_velocity_direct_batched
export get_eph_RR_to_kR_batched!
export get_eph_kR_to_kq_batched!
export eph_apply_rotations!
export KRtoKQWorkspace
export batched_gemm!
export eph_window_scatter!
export device_free_bytes
export device_synchronize

# Batched Wannier -> Bloch diagonalization over a whole k-grid.
#
# These complement the per-k `get_el_eigen!` / `get_el_eigen_valueonly!` (wannier_to_bloch.jl):
# `get_fourier_batched!` interpolates `H(k)` for all k at once (one GEMM chain), then a
# batched Hermitian eigensolve diagonalizes the stack. Everything runs on the backend of
# `ham.op_r` (CPU or GPU); the CUDA extension provides the batched `heevjBatched!` methods.
#
# Naming mirrors the per-k routines:
#   get_el_eigen_batched           <-> get_el_eigen!            (eigenvalues + eigenvectors)
#   get_el_eigen_valueonly_batched <-> get_el_eigen_valueonly!  (eigenvalues only)

# =============================================================================
#  Batched Hermitian eigensolves (CPU methods; CUDA extension adds CuArray methods)

"""
    eigvals_batched(Hk) -> W

Eigenvalues of a stack of Hermitian matrices `Hk` of size `(nw, nw, nk)`, returned as
`(nw, nk)`. CPU method loops over LAPACK `syev!`; the CUDA extension provides a batched
`heevjBatched!` method for `CuArray`s.
"""
function eigvals_batched(Hk::AbstractArray{Complex{T},3}) where {T}
    nw, n2, nk = size(Hk)
    @assert nw == n2
    W = Matrix{T}(undef, nw, nk)
    ws = HermitianEigenWsSYEV{Complex{T},T}()
    A = Matrix{Complex{T}}(undef, nw, nw)
    @views for ik in 1:nk
        A .= Hk[:, :, ik]
        W[:, ik] .= syev!(ws, 'N', 'U', A)[1]
    end
    W
end

"""
    eigen_batched(Hk) -> (W, V)

Eigenvalues `W` `(nw, nk)` and eigenvectors `V` `(nw, nw, nk)` of a stack of Hermitian
matrices `Hk` of size `(nw, nw, nk)`. CPU method loops over LAPACK `syev!`; the CUDA
extension provides a batched `heevjBatched!` method for `CuArray`s.

Note: unlike the per-k `get_el_eigen!`, no EPW degeneracy gauge-fixing is applied, so for
degenerate bands the eigenvectors may differ from `get_el_eigen!` by a gauge (the
eigenvalues, and the eigen-decomposition, are unaffected).
"""
function eigen_batched(Hk::AbstractArray{Complex{T},3}) where {T}
    nw, n2, nk = size(Hk)
    @assert nw == n2
    W = Matrix{T}(undef, nw, nk)
    V = Array{Complex{T},3}(undef, nw, nw, nk)
    ws = HermitianEigenWsSYEV{Complex{T},T}()
    A = Matrix{Complex{T}}(undef, nw, nw)
    @views for ik in 1:nk
        A .= Hk[:, :, ik]
        W[:, ik] .= syev!(ws, 'V', 'U', A)[1]   # A is overwritten with the eigenvectors
        V[:, :, ik] .= A
    end
    W, V
end

# =============================================================================
#  Band-eigenvalue drivers over a k-grid

# Interpolate H(k) for all k into an (ndata, nk) array on `ham.op_r`'s backend.
function _fourier_hk_batched(ham::WannierObject{T}, xk_list; batch_size::Int) where {T}
    nw = isqrt(ham.ndata)
    nw^2 == ham.ndata || throw(ArgumentError(
        "ndata=$(ham.ndata) is not a perfect square; expected nw^2 for a Hamiltonian"))
    nk = length(xk_list)
    itp = BatchedWannierInterpolator(ham; batch_size)
    Hk = similar(ham.op_r, Complex{T}, ham.ndata, nk)
    get_fourier_batched!(Hk, itp, xk_list)
    reshape(Hk, nw, nw, nk)
end

"""
    get_el_eigen_valueonly_batched(ham::WannierObject, xk_list; batch_size=length(xk_list)) -> W

Electron band eigenvalues `(nw, nk)` at every k-point in `xk_list`. Batched counterpart of
[`get_el_eigen_valueonly!`](@ref). Runs on the backend of `ham.op_r`; the result is on that
same backend. `batch_size` controls the Fourier chunking (default: a single batch).
"""
function get_el_eigen_valueonly_batched(ham::WannierObject, xk_list; batch_size::Int=length(xk_list))
    eigvals_batched(_fourier_hk_batched(ham, xk_list; batch_size))
end

"""
    get_el_eigen_batched(ham::WannierObject, xk_list; batch_size=length(xk_list)) -> (W, V)

Electron band eigenvalues `(nw, nk)` and eigenvectors `(nw, nw, nk)` at every k-point in
`xk_list`. Batched counterpart of [`get_el_eigen!`](@ref). Runs on the backend of
`ham.op_r`; the results are on that same backend. See [`eigen_batched`](@ref) for the
eigenvector gauge caveat.
"""
function get_el_eigen_batched(ham::WannierObject, xk_list; batch_size::Int=length(xk_list))
    eigen_batched(_fourier_hk_batched(ham, xk_list; batch_size))
end

"""
    get_el_velocity_direct_batched(vel::WannierObject, xk_list, uks; batch_size=length(xk_list)) -> (nw, nw, 3, nk)

Batched counterpart of [`get_el_velocity_direct!`](@ref): for a 3-direction Wannier operator
`vel` (`ndata == nw^2 * 3`, e.g. `model.el_vel` (dH/dk) or `model.el_pos` (position A)), Fourier-
interpolate over all k in `xk_list` and apply the per-k gauge rotation `uk' * M[:,:,idir] * uk`
for each Cartesian direction. Runs on the backend of `vel.op_r`; `uks` is `(nw, nw, nk)` (full-band
eigenvectors, one `uk` per k, on the same backend). Returns the full-band `(nw, nw, 3, nk)` rotated
matrix; callers slice the in-window block (the in-window block equals the windowed-`uk` rotation).

Used for the electron position matrix `rbar` (`vel = el_pos`) and the `:Direct`-mode velocity
(`vel = el_vel`). The Fourier output is laid out `(nw, nw, 3, nk)` with `idir` the slowest of the
three operator dims, matching the per-k `get_fourier!`'s `reshape(out, (nw, nw, 3))` convention.
"""
function get_el_velocity_direct_batched(vel::WannierObject{T}, xk_list,
        uks::AbstractArray{Complex{T},3}; batch_size::Int=length(xk_list)) where {T}
    nw = size(uks, 1)
    @assert size(uks, 2) == nw
    nk = length(xk_list)
    @assert size(uks, 3) == nk
    @assert vel.ndata == nw^2 * 3 "expected ndata = nw^2*3 for a 3-direction operator, got $(vel.ndata)"

    itp = BatchedWannierInterpolator(vel; batch_size)
    Vk = similar(vel.op_r, Complex{T}, vel.ndata, nk)
    get_fourier_batched!(Vk, itp, xk_list)                  # (nw^2*3, nk)

    # Batch the rotation over b = (idir, k): stack the operator as (nw, nw, 3*nk) and replicate
    # uk across the three directions so each batch slice carries its own uk.
    Vb = reshape(Vk, nw, nw, 3 * nk)
    urep = similar(uks, nw, nw, 3, nk)
    urep .= reshape(uks, nw, nw, 1, nk)                     # broadcast uk over idir (non-scalar)
    ub = reshape(urep, nw, nw, 3 * nk)

    tmp = similar(Vb)
    batched_gemm!('N', 'N', Vb, ub, tmp)                    # tmp = M[:,:,idir] * uk
    out = similar(Vb)
    batched_gemm!('C', 'N', ub, tmp, out)                   # out = uk' * (M * uk)
    reshape(out, nw, nw, 3, nk)
end

# =============================================================================
#  Electron-phonon Wannier -> Bloch (backend-generic)
#
#  Counterparts of the per-k `get_eph_RR_to_kR!` / `get_eph_kR_to_kq!`
#  (wannier_to_bloch.jl). The Fourier step reuses `get_fourier_batched!`, and the gauge
#  rotations are plain GEMMs, so the same code runs on the CPU or GPU according to the
#  backend of the parent `op_r`. No CUDA-specific code is needed.

"""
    get_eph_RR_to_kR_batched!(epobj_ekpR::WannierObject, epmat_itp::BatchedWannierInterpolator, xk, uk)

Electron-phonon matrix in electron Bloch, phonon Wannier basis at `xk`. Counterpart of
[`get_eph_RR_to_kR!`](@ref) that runs on the backend of `epmat_itp.parent.op_r`.

The electron-k index of the Fourier-interpolated `g(k, R_ep)` is rotated by `uk` as a single
GEMM (`out[iw,ib,b] = Σ_jw g[iw,jw,b] uk[jw,ib]`, batched over `b = (imode, ir_ep)`),
implemented as `permutedims` + `transpose(uk) * g` + `permutedims`.
"""
function get_eph_RR_to_kR_batched!(epobj_ekpR::WannierObject{T}, epmat_itp::BatchedWannierInterpolator{T}, xk, uk) where {T}
    epmat = epmat_itp.parent
    nr_ep = length(epmat.irvec_next)
    nw, nband = size(uk)
    nmodes = div(epmat.ndata, nw^2 * nr_ep)
    ndata = nw * nband * nmodes
    @assert nmodes * nw^2 * nr_ep == epmat.ndata
    @assert epobj_ekpR.nr == nr_ep
    @assert size(epobj_ekpR.op_r, 1) >= ndata

    # Fourier over R_el at xk -> g(k, R_ep) in (nw, nw, nmodes*nr_ep)
    g_flat = similar(epmat.op_r, Complex{T}, epmat.ndata, 1)
    get_fourier_batched!(g_flat, epmat_itp, [xk])
    nbatch = nmodes * nr_ep
    g = reshape(g_flat, nw, nw, nbatch)

    # Rotate electron-k index by uk: one GEMM via permute. (uk is applied un-conjugated.)
    gp = permutedims(g, (2, 1, 3))                                   # (jw, iw, b)
    M  = transpose(uk) * reshape(gp, nw, nw * nbatch)                # (nband, nw*nbatch)
    out = permutedims(reshape(M, nband, nw, nbatch), (2, 1, 3))      # (nw, nband, nbatch)

    copyto!(view(epobj_ekpR.op_r, 1:ndata, :), reshape(out, ndata, nr_ep))
    epobj_ekpR.ndata = ndata
    epobj_ekpR._id += 1
    epobj_ekpR
end

"""
    get_eph_kR_to_kq_batched!(ep_kq, ep_ekpR_itp::BatchedWannierInterpolator, xq, u_ph, ukq)

Electron-phonon matrix in electron and phonon Bloch basis at `xq` (electron at `k` already
in the eigenstate basis). Counterpart of [`get_eph_kR_to_kq!`](@ref) that runs on the backend
of `ep_ekpR_itp.parent.op_r`. The two gauge rotations are the same reshaped GEMMs as the CPU
version (`ukq'` on the left, `u_ph` on the right).
"""
function get_eph_kR_to_kq_batched!(ep_kq, ep_ekpR_itp::BatchedWannierInterpolator{T}, xq, u_ph, ukq) where {T}
    nbandkq, nbandk, nmodes = size(ep_kq)
    nw = size(ukq, 1)
    @assert size(u_ph) == (nmodes, nmodes)
    @assert size(ukq, 2) == nbandkq
    parent = ep_ekpR_itp.parent
    @assert parent.ndata == nw * nbandk * nmodes

    # Fourier over R_ep at xq -> g(k+R_ep) in (nw, nbandk, nmodes)
    g_flat = similar(parent.op_r, Complex{T}, parent.ndata, 1)
    get_fourier_batched!(g_flat, ep_ekpR_itp, [xq])

    # ep_kq[ibkq, ibk, imode] = ukq'[ibkq, iw] * g[iw, ibk, jmode] * u_ph[jmode, imode]
    tmp = ukq' * reshape(g_flat, nw, nbandk * nmodes)               # (nbandkq, nbandk*nmodes)
    mul!(reshape(ep_kq, nbandkq * nbandk, nmodes),
         reshape(tmp, nbandkq * nbandk, nmodes), u_ph)
    ep_kq
end

# =============================================================================
#  Batched (strided) GEMM: C[:,:,b] = op(A[:,:,b]) * op(B[:,:,b]) for all b.
#  CPU method loops over `mul!`; the CUDA extension dispatches to
#  `CUBLAS.gemm_strided_batched!`. This is the one primitive the *list-batched* e-ph
#  drivers need beyond Fourier + plain GEMM (each k/q has its own rotation matrix).

@inline _batched_op(t::Char, X) = t == 'N' ? X : (t == 'T' ? transpose(X) : adjoint(X))

"""
    batched_gemm!(transA, transB, A, B, C)

`C[:,:,b] = op(transA, A[:,:,b]) * op(transB, B[:,:,b])` for every batch `b` (α=1, β=0),
where `op('N',X)=X`, `op('T',X)=transpose(X)`, `op('C',X)=adjoint(X)`. The CPU method loops
over `mul!`; the CUDA extension uses `CUBLAS.gemm_strided_batched!`.
"""
function batched_gemm!(transA::Char, transB::Char,
                       A::AbstractArray{T,3}, B::AbstractArray{T,3}, C::AbstractArray{T,3}) where {T}
    @assert size(A, 3) == size(B, 3) == size(C, 3)
    @views for b in axes(C, 3)
        mul!(C[:, :, b], _batched_op(transA, A[:, :, b]), _batched_op(transB, B[:, :, b]))
    end
    C
end

# =============================================================================
#  List-batched e-ph drivers: process many k (RR->kR) / many q (kR->kq) at once.
#  These collapse the per-k/q kernel launches into a few large kernels — the form that
#  wins on the GPU. Rotation matrices are stacked along the batch dimension.

"""
    get_eph_RR_to_kR_batched!(ep_ekpR_all, epmat_itp::BatchedWannierInterpolator, ks, uks)

Batched over a list of k-points `ks`. `uks` is `(nw, nband, nk)` (one `uk` per k).
Writes `ep_ekpR_all`, shape `(nw*nband*nmodes, nr_ep, nk)` — column `k` is the `op_r` of the
electron-Bloch / phonon-Wannier object at `ks[k]`.

One batched Fourier (`get_fourier_batched!`) over `R_el`, then one `batched_gemm!` for the
per-k rotation by `uk` (recast as `transpose(uk(k)) * permute(g(k))`).

Full-band only: `ep_ekpR_all` is sized exactly `(nw*nband*nmodes, …)`. Unlike the per-k
`get_eph_RR_to_kR!`, this list-batched path does not support an energy window
(`nband < nband_bound`) — all `nk` k-points must share the same `nband`.
"""
function get_eph_RR_to_kR_batched!(ep_ekpR_all::AbstractArray{Complex{T},3},
                                   epmat_itp::BatchedWannierInterpolator{T}, ks, uks) where {T}
    epmat = epmat_itp.parent
    nr_ep = length(epmat.irvec_next)
    nw, nband, nk = size(uks)
    nmodes = div(epmat.ndata, nw^2 * nr_ep)
    M = nmodes * nr_ep
    @assert nmodes * nw^2 * nr_ep == epmat.ndata
    @assert length(ks) == nk
    @assert size(ep_ekpR_all) == (nw * nband * nmodes, nr_ep, nk)

    g = similar(epmat.op_r, Complex{T}, epmat.ndata, nk)
    get_fourier_batched!(g, epmat_itp, ks)                              # (nw^2*nmodes*nr_ep, nk)

    gp = permutedims(reshape(g, nw, nw, M, nk), (2, 1, 3, 4))           # (jw, iw, M, k)
    C = similar(g, Complex{T}, nband, nw * M, nk)
    batched_gemm!('T', 'N', uks, reshape(gp, nw, nw * M, nk), C)        # C(k)=transpose(uk(k))*gp(k)
    out = permutedims(reshape(C, nband, nw, M, nk), (2, 1, 3, 4))       # (nw, nband, M, k)
    copyto!(ep_ekpR_all, reshape(out, nw * nband * nmodes, nr_ep, nk))
    ep_ekpR_all
end

"""
    KRtoKQWorkspace(proto, ndata, nbandkq, nbandk, nmodes, nq)

Preallocated scratch for [`get_eph_kR_to_kq_batched!`](@ref), reused across the per-k calls so the
driver does no per-call `similar`. `proto` is an array on the target backend (e.g. `parent.op_r`);
the buffers follow its backend and element type. `ndata = nw*nbandk*nmodes`.
"""
struct KRtoKQWorkspace{MT<:AbstractMatrix, AT<:AbstractArray}
    g::MT      # (ndata, nq)                  — Fourier output g(k+R_ep) for all q
    tmp::AT    # (nbandkq, nbandk*nmodes, nq) — after the ukq' rotation
end
function KRtoKQWorkspace(proto, ndata::Int, nbandkq::Int, nbandk::Int, nmodes::Int, nq::Int)
    T = real(eltype(proto))
    KRtoKQWorkspace(similar(proto, Complex{T}, ndata, nq),
                    similar(proto, Complex{T}, nbandkq, nbandk * nmodes, nq))
end

"""
    get_eph_kR_to_kq_batched!(ep_kq_all, ep_ekpR_itp::BatchedWannierInterpolator, qs, u_phs, ukqs; ws=nothing)

Batched over a list of q-points `qs` (for a fixed k). `ukqs` is `(nw, nbandkq, nq)` and
`u_phs` is `(nmodes, nmodes, nq)`. Writes `ep_kq_all`, shape `(nbandkq, nbandk, nmodes, nq)`.

One batched Fourier over `R_ep`, then two `batched_gemm!`s for the per-q rotations
(`ukq(q)'` on the left, `u_ph(q)` on the right).

Pass a [`KRtoKQWorkspace`](@ref) as `ws` (sized for at least this `nq`) to reuse the `g` / `tmp`
scratch across calls instead of allocating it each call — the per-k hot path in the GPU loop does
this, sizing `ws` for the max chunk width and passing `nq ≤` that for a partial final chunk.
"""
function get_eph_kR_to_kq_batched!(ep_kq_all::AbstractArray{Complex{T},4},
                                   ep_ekpR_itp::BatchedWannierInterpolator{T}, qs, u_phs, ukqs;
                                   ws::Union{Nothing,KRtoKQWorkspace}=nothing,
                                   g2_out=nothing, ωq=nothing) where {T}
    nbandkq, nbandk, nmodes, nq = size(ep_kq_all)
    nw = size(ukqs, 1)
    @assert size(ukqs) == (nw, nbandkq, nq)
    @assert size(u_phs) == (nmodes, nmodes, nq)
    parent = ep_ekpR_itp.parent
    @assert parent.ndata == nw * nbandk * nmodes

    if ws === nothing
        g   = similar(parent.op_r, Complex{T}, parent.ndata, nq)
        tmp = similar(parent.op_r, Complex{T}, nbandkq, nbandk * nmodes, nq)
    else
        # `ws` is sized for the max chunk width; use the first `nq` columns (a partial final chunk
        # passes nq < capacity), so the whole loop runs without padding the batch back up.
        @assert size(ws.g, 1) == parent.ndata && size(ws.g, 2) >= nq
        @assert size(ws.tmp, 1) == nbandkq && size(ws.tmp, 2) == nbandk * nmodes && size(ws.tmp, 3) >= nq
        g   = view(ws.g, :, 1:nq)
        tmp = view(ws.tmp, :, :, 1:nq)
    end

    get_fourier_batched!(g, ep_ekpR_itp, qs)                           # (nw*nbandk*nmodes, nq)
    eph_apply_rotations!(ep_kq_all, g, ukqs, u_phs, tmp; g2_out, ωq)
    ep_kq_all
end

"""
    device_free_bytes(proto) -> Int

Free memory (bytes) on the backend `proto` lives on, used to decide whether a large buffer fits
on the device. Generic fallback returns `typemax(Int)` (host memory: assume it always fits — the
caller's host allocation is governed by RAM, not this check). The CUDA extension returns
`CUDA.available_memory()` for a `CuArray` proto.
"""
device_free_bytes(proto) = typemax(Int)

"""
    device_synchronize(proto)

Block until queued device work on `proto`'s backend completes. Generic fallback is a no-op
(host work is synchronous); the CUDA extension calls `CUDA.synchronize()`. Used to bound the
host look-ahead in the GPU e-ph loop so per-tile scratch does not pile up in the memory pool.
"""
device_synchronize(proto) = nothing

"""
    eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                        nbandkq, nbandk, nm, nqc, n_i)

Device-resident scatter for a calculator that keeps `g2`/`ωq` on the device (no per-chunk
host streaming). For every `(m, n, ν, j)` entry of `g2vals` `(nbandkq, nbandk, nm, nqc)`, look
up the state indices `i = imap_i_col[n]` and `f = imap_f[m, ikqs[j]]`; if both are in-window
(`> 0`), write the value into the mode-fastest linear slot
`lin = ν + nm·(i−1) + nm·n_i·(f−1)` of the flat `g2_out` / `ωq_out`
(`ω = ωq[ν, j]`). The target `lin` indices are unique across the run (distinct k → distinct i,
distinct k+q → distinct f), so the writes never collide (no atomics needed). Generic
(CPU/fallback) method; the CUDA extension provides a one-kernel `CuArray` method.
"""
function eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, n_i::Int)
    @inbounds for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[j]]
        if i > 0 && f > 0
            lin = ν + nm * (i - 1) + nm * n_i * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, j]
            ωq_out[lin] = ωq[ν, j]
        end
    end
    nothing
end

"""
    eph_apply_rotations!(ep_kq_all, g, ukqs, u_phs, tmp; g2_out=nothing, ωq=nothing)

Apply the two e-ph gauge rotations to the Fourier-interpolated `g` (`(ndata, nq)`, reshaped to
`(nw, nbandk, nmodes, nq)`), writing the eigenbasis e-ph matrix `ep_kq_all`
`(nbandkq, nbandk, nmodes, nq)` = `ukq(q)' * g(q) * u_ph(q)`. If `g2_out !== nothing`, also write
`g2 = |ep_kq|² / (2 ωq)` (with `ωq` `(nmodes, nq)`) in the same pass.

Generic method: the two strided-batched GEMMs (`ukq'` on the left, `u_ph` on the right) plus an
optional `g2` broadcast — identical to the previous inline code, so any backend works. The CUDA
extension overrides this with a fused per-q kernel for small `nw*nmodes`, which avoids cuBLAS'
tiny-matmul inefficiency (the 4×4 / nmodes×nmodes strided-batched GEMMs run at ~2% of FP64 peak).
"""
function eph_apply_rotations!(ep_kq_all::AbstractArray{Complex{T},4}, g,
                              ukqs, u_phs, tmp; g2_out=nothing, ωq=nothing) where {T}
    nbandkq, nbandk, nmodes, nq = size(ep_kq_all)
    nw = size(ukqs, 1)
    batched_gemm!('C', 'N', ukqs, reshape(g, nw, nbandk * nmodes, nq), tmp)   # ukq(q)' * g(q)
    batched_gemm!('N', 'N', reshape(tmp, nbandkq * nbandk, nmodes, nq), u_phs,
                  reshape(ep_kq_all, nbandkq * nbandk, nmodes, nq))           # * u_ph(q)
    if g2_out !== nothing
        g2_out .= abs2.(ep_kq_all) ./ (2 .* reshape(ωq, 1, 1, nmodes, nq))
    end
    ep_kq_all
end
