using LinearAlgebra
using ElectronPhonon.AllocatedLAPACK: HermitianEigenWsSYEV, syev!

export eigvals_batched
export eigen_batched
export get_el_eigen_batched
export get_el_eigen_valueonly_batched
export get_eph_RR_to_kR_batched!
export get_eph_kR_to_kq_batched!
export KRtoKQWorkspace
export batched_gemm!

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

Pass a [`KRtoKQWorkspace`](@ref) as `ws` (sized for this `nq`) to reuse the `g` / `tmp` scratch
across calls instead of allocating it each call — the per-k hot path in the GPU loop does this.
"""
function get_eph_kR_to_kq_batched!(ep_kq_all::AbstractArray{Complex{T},4},
                                   ep_ekpR_itp::BatchedWannierInterpolator{T}, qs, u_phs, ukqs;
                                   ws::Union{Nothing,KRtoKQWorkspace}=nothing) where {T}
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
        g, tmp = ws.g, ws.tmp
        @assert size(g) == (parent.ndata, nq)
        @assert size(tmp) == (nbandkq, nbandk * nmodes, nq)
    end

    get_fourier_batched!(g, ep_ekpR_itp, qs)                           # (nw*nbandk*nmodes, nq)
    batched_gemm!('C', 'N', ukqs, reshape(g, nw, nbandk * nmodes, nq), tmp)   # ukq(q)' * g(q)
    batched_gemm!('N', 'N', reshape(tmp, nbandkq * nbandk, nmodes, nq), u_phs,
                  reshape(ep_kq_all, nbandkq * nbandk, nmodes, nq))           # * u_ph(q)
    ep_kq_all
end
