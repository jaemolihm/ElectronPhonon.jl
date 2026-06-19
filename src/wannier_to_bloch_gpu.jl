using LinearAlgebra
using ElectronPhonon.AllocatedLAPACK: HermitianEigenWsSYEV, syev!

export eigvals_batched
export eigen_batched
export get_el_eigen_batched
export get_el_eigen_valueonly_batched
export get_eph_RR_to_kR_batched!
export get_eph_kR_to_kq_batched!

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
