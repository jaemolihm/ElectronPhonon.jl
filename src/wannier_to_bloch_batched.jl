using LinearAlgebra
using ElectronPhonon.AllocatedLAPACK: HermitianEigenWsSYEV, syev!

# These are internal (used within the package and by the CUDA extension via `ElectronPhonon.`),
# so nothing here is exported; tests import the specific names they use.

# Batched Wannier -> Bloch diagonalization over a whole k-grid.
#
# These complement the per-k `compute_el_eigen!` / `compute_el_eigen_valueonly!`
# (wannier_to_bloch.jl): `get_fourier_batched!` interpolates `H(k)` for all k at once (one GEMM
# chain), then a batched Hermitian eigensolve diagonalizes the stack. Everything runs on the backend
# of `ham.op_r` (CPU or GPU); the CUDA extension provides the device eigensolve methods.
#
# Naming mirrors the per-k routines:
#   compute_el_eigen_batched           <-> compute_el_eigen!            (eigenvalues + eigenvectors)
#   compute_el_eigen_valueonly_batched <-> compute_el_eigen_valueonly!  (eigenvalues only)

# =============================================================================
#  Batched Hermitian eigensolves (CPU methods; CUDA extension adds CuArray methods)

"""
    eigvals_batched!(Hk) -> E

Eigenvalues of a stack of Hermitian matrices `Hk` of size `(nw, nw, nk)`, returned as
`(nw, nk)`. CPU method loops over LAPACK `syev!`; the CUDA extension provides a `CuArray` method:
one-thread-per-matrix Jacobi for `nw ≤ 12`, cuSOLVER `heevjBatched` above.

May overwrite (destroy) `Hk`; a caller that still needs `Hk` afterwards must copy it first.
"""
function eigvals_batched!(Hk::AbstractArray{Complex{T},3}) where {T}
    # CPU method; the CuArray method lives in ext/ElectronPhononCUDAExt.jl.
    nw, nw2, nk = size(Hk)
    @assert nw == nw2
    E = Matrix{T}(undef, nw, nk)
    ws = HermitianEigenWsSYEV{Complex{T},T}()
    @views for ik in 1:nk
        E[:, ik] .= syev!(ws, 'N', 'U', Hk[:, :, ik])[1]   # syev! overwrites the slice
    end
    E
end

"""
    eigen_batched!(Hk) -> (E, U)

Eigenvalues `E` `(nw, nk)` and eigenvectors `U` `(nw, nw, nk)` of a stack of Hermitian
matrices `Hk` of size `(nw, nw, nk)`. CPU method loops over LAPACK `syev!`; the CUDA
extension provides a `CuArray` method: one-thread-per-matrix Jacobi for `nw ≤ 12`, cuSOLVER
`heevjBatched` above.

Overwrites `Hk`: the returned `U` is `Hk` itself, overwritten in place with the eigenvectors.

Note: unlike the per-k `compute_el_eigen!`, no EPW degeneracy gauge-fixing is applied, so for
degenerate bands the eigenvectors may differ from `compute_el_eigen!` by a gauge (the
eigenvalues, and the eigen-decomposition, are unaffected).
"""
function eigen_batched!(Hk::AbstractArray{Complex{T},3}) where {T}
    # CPU method; the CuArray method lives in ext/ElectronPhononCUDAExt.jl.
    nw, nw2, nk = size(Hk)
    @assert nw == nw2
    E = Matrix{T}(undef, nw, nk)
    ws = HermitianEigenWsSYEV{Complex{T},T}()
    @views for ik in 1:nk
        E[:, ik] .= syev!(ws, 'V', 'U', Hk[:, :, ik])[1]
    end
    E, Hk   # eigenvectors overwritten into Hk
end

# =============================================================================
#  Band-eigenvalue drivers over a k-grid

# Like the per-k `wannier_to_bloch` drivers, the batched electron drivers take a Wannier
# interpolator (build it with `get_interpolator(ham; fourier_mode="batched", batch_size=…)`), not a
# raw `WannierObject`; `batch_size` is baked into the interpolator at construction.

# The k argument of every driver below is a host `Vector{Vec3}`. Staging it onto the interpolator's
# backend belongs one layer down, in `get_fourier_batched!`; nothing here handles a staged matrix.

# Interpolate H(k) for all k into an (ndata, nk) array on the interpolator's backend.
function _fourier_hk_batched(itp::BatchedWannierInterpolator{T}, xk_list) where {T}
    ham = itp.parent
    nw = isqrt(ham.ndata)
    nw^2 == ham.ndata || throw(ArgumentError(
        "ndata=$(ham.ndata) is not a perfect square; expected nw^2 for a Hamiltonian"))
    nk = length(xk_list)
    Hk = similar(ham.op_r, Complex{T}, ham.ndata, nk)
    get_fourier_batched!(Hk, itp, xk_list)
    reshape(Hk, nw, nw, nk)
end

"""
    compute_el_eigen_valueonly_batched(itp::BatchedWannierInterpolator, xk_list) -> E

Electron band eigenvalues `(nw, nk)` at every k-point in `xk_list`. Batched counterpart of
[`compute_el_eigen_valueonly!`](@ref). Runs on, and returns on, the backend of `itp.parent.op_r`.
"""
function compute_el_eigen_valueonly_batched(itp::BatchedWannierInterpolator, xk_list)
    eigvals_batched!(_fourier_hk_batched(itp, xk_list))
end

"""
    compute_el_eigen_batched(itp::BatchedWannierInterpolator, xk_list) -> (E, U)

Electron band eigenvalues `(nw, nk)` and eigenvectors `(nw, nw, nk)` at every k-point in
`xk_list`. Batched counterpart of [`compute_el_eigen!`](@ref). Runs on, and returns on, the backend
of `itp.parent.op_r`. See [`eigen_batched!`](@ref) for the eigenvector gauge caveat.
"""
function compute_el_eigen_batched(itp::BatchedWannierInterpolator, xk_list)
    eigen_batched!(_fourier_hk_batched(itp, xk_list))
end

"""
    compute_el_velocity_direct_batched(itp::BatchedWannierInterpolator, xk_list, uks)
        -> (nw, nw, 3, nk)

Batched counterpart of [`compute_el_velocity_direct!`](@ref): for a 3-direction Wannier operator
(`itp.parent.ndata == nw^2 * 3`, e.g. `model.el_vel` (dH/dk) or `model.el_pos` (position A)),
Fourier-interpolate over all k in `xk_list` and apply the per-k gauge rotation `uk' * M[:,:,idir] * uk`
for each Cartesian direction. Runs on the backend of `itp.parent.op_r`; `uks` is `(nw, nw, nk)`
(full-band eigenvectors, one `uk` per k, on the same backend). Returns the full-band `(nw, nw, 3, nk)`
rotated matrix; callers slice the in-window block (which equals the windowed-`uk` rotation).

Used for the electron position matrix `rbar` (`el_pos`) and the `:Direct`-mode velocity (`el_vel`).
The Fourier output is laid out `(nw, nw, 3, nk)` with `idir` the slowest of the three operator dims,
matching the per-k `get_fourier!`'s `reshape(out, (nw, nw, 3))` convention.
"""
function compute_el_velocity_direct_batched(itp::BatchedWannierInterpolator{T}, xk_list,
        uks::AbstractArray{Complex{T},3}) where {T}
    vel = itp.parent
    nw = size(uks, 1)
    @assert size(uks, 2) == nw
    nk = length(xk_list)
    @assert size(uks, 3) == nk
    @assert vel.ndata == nw^2 * 3 "expected ndata = nw^2*3 for a 3-direction operator, got $(vel.ndata)"

    Vk = similar(vel.op_r, Complex{T}, vel.ndata, nk)
    get_fourier_batched!(Vk, itp, xk_list)                  # (nw^2*3, nk)

    # Batch the rotation over b = (idir, k): stack the operator as (nw, nw, 3*nk) and replicate
    # uk across the three directions so each batch slice carries its own uk.
    # TODO: `urep`, `tmp`, and `out` each allocate a full (nw, nw, 3*nk) buffer here. Fine for the
    # one-shot setup use, but make this non-allocating (caller-provided scratch) if it moves to a
    # hot path.
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
#  List-batched e-ph drivers: process many k (RR->kR) / many q (kR->kq) at once.
#  These collapse the per-k/q kernel launches into a few large kernels — the form that
#  wins on the GPU. Rotation matrices are stacked along the batch dimension.

"""
    eph_rotate_kR_batched!(ep_ekpR_all, g, uks; additional_phase=nothing)

The k rotation of the e-ph matrix Fourier-transformed over `R_el` at a list of k points, `g`
`(nw^2 * M, nk)`, viewed as `g[iw, jw, M, k]` with `M` the remaining row axes (`nmodes`, `R_p`,
...): `ep_ekpR_all[iw, n, M, k] = Σ_jw g[iw, jw, M, k] uks[jw, n, k]`, recast as
`transpose(uk(k)) * permute(g(k))` in one `batched_gemm!`, times `additional_phase` along `R_p`
(the second axis of `ep_ekpR_all`). `uks` is `(nw, nband, nk)`, and `ep_ekpR_all` is
`(nw*nband*nmodes, nr_ep, ..., nk)`: slice `k` is the `op_r` of the electron-Bloch /
phonon-Wannier object at that k.

`additional_phase`, if given, is `(nr_ep × nk)` and multiplies the output, so the result is
`additional_phase[ip, k] · g(k, R_p)`. It is folded into the final copy, which already reads and
writes the whole array. The outer-k engine passes `conj(exp(2πi R_p · x_k))` to store `g` in the
k+q convention, which makes the following `R_p` Fourier a function of `x_{k+q}` alone and hence
independent of the outer `k`.

All `nk` points share one `nband` (unlike the per-k `compute_eph_RR_to_kR!`, which handles a per-k
window): a windowed run projects every k onto the same `nbandk_max`-wide eigenvector window.
"""
function eph_rotate_kR_batched!(ep_ekpR_all::AbstractArray{Complex{T}}, g, uks;
                                additional_phase=nothing) where {T}
    nw, nband, nk = size(uks)
    M = div(size(g, 1), nw^2)
    @assert M * nw^2 == size(g, 1)
    @assert size(g, 2) == nk
    @assert size(ep_ekpR_all, ndims(ep_ekpR_all)) == nk
    @assert length(ep_ekpR_all) == nw * nband * M * nk
    nr_ep = size(ep_ekpR_all, 2)

    gp = permutedims(reshape(g, nw, nw, M, nk), (2, 1, 3, 4))           # (jw, iw, M, k)
    C = similar(g, Complex{T}, nband, nw * M, nk)
    batched_gemm!('T', 'N', uks, reshape(gp, nw, nw * M, nk), C)        # C(k)=transpose(uk(k))*gp(k)
    out = reshape(permutedims(reshape(C, nband, nw, M, nk), (2, 1, 3, 4)), size(ep_ekpR_all))  # (nw, nband, M, k)
    if additional_phase === nothing
        copyto!(ep_ekpR_all, out)
    else
        @assert size(additional_phase) == (nr_ep, nk)
        ep_ekpR_all .= out .* reshape(additional_phase, 1, nr_ep, ntuple(_ -> 1, ndims(out) - 3)..., nk)
    end
    ep_ekpR_all
end

"""
    compute_eph_kR_to_kq_batched!(ep_kq_all, ep_kR::AbstractMatrix, phase::AbstractMatrix, u_phs,
                                  ukqs; g=nothing, tmp=nothing)

Batched over a list of q-points (for a fixed k). `ukqs` is `(nw, nbandkq, nq)` and
`u_phs` is `(ndisp, nmodes, nq)`: the phonon basis, `nmodes ≤ ndisp` of its columns. Writes
`ep_kq_all`, shape `(nbandkq, nbandk, nmodes, nq)`.

One batched Fourier over `R_ep`, then two `batched_gemm!`s for the per-q rotations
(`ukq(q)'` on the left, `u_ph(q)` on the right).

The three inputs are the kR intermediate `g(k, R_p)` as `ep_kR`, `(nw*nbandk*ndisp, nr)`; the
Fourier phase `exp(2πi R_p · x_q)` as `phase`, `(nr, nq)`; and the rotations. Taking the phase
rather than a q-list is what lets a caller build it once and reuse it over many `k` — the outer-k
engine does that via the k+q convention of [`eph_rotate_kR_batched!`](@ref).

`g` `(nw*nbandk*ndisp, nq)` and `tmp` `(nbandkq, nbandk*ndisp, nq)` are the scratch at exactly
this `nq`, reused across calls; `nothing` allocates them.
"""
function compute_eph_kR_to_kq_batched!(ep_kq_all::AbstractArray{Complex{T},4},
                                       ep_kR::AbstractMatrix, phase::AbstractMatrix, u_phs, ukqs;
                                   g=nothing, tmp=nothing) where {T}
    nbandkq, nbandk, nmodes, nq = size(ep_kq_all)
    nw = size(ukqs, 1)
    ndisp = size(u_phs, 1)
    ndata = nw * nbandk * ndisp
    @assert size(ukqs) == (nw, nbandkq, nq)
    @assert size(u_phs) == (ndisp, nmodes, nq)
    @assert size(ep_kR, 1) == ndata
    @assert size(phase) == (size(ep_kR, 2), nq)

    g = g === nothing ? similar(ep_kR, Complex{T}, ndata, nq) : g
    tmp = tmp === nothing ? similar(ep_kR, Complex{T}, nbandkq, nbandk * ndisp, nq) : tmp
    @assert size(g) == (ndata, nq)
    @assert size(tmp) == (nbandkq, nbandk * ndisp, nq)

    mul!(g, ep_kR, phase)                                              # (nw*nbandk*ndisp, nq)
    eph_apply_rotations!(ep_kq_all, reshape(g, nw, nbandk, ndisp, nq), ukqs, u_phs, tmp)
    ep_kq_all
end

"""
    eph_apply_rotations_rqkq!(ep_kq_all, g, uks, ukqs, tmp, uk_rep)

Apply the two per-k electron gauge rotations of the Rq→kq step (fixed q, a list of k; the batched
counterpart of [`compute_eph_Rq_to_kq!`](@ref)) to `g`, the electron-Wannier / phonon-Bloch e-ph
matrix Fourier-transformed over `R_el` at every k (`(nw²·nmodes, nk)`, viewed as `g[iw, jw, ν, k]`),
writing `ep_kq_all[m, n, ν, k] = Σ_{iw,jw} conj(ukqs[iw,m,k]) · g[iw,jw,ν,k] · uks[jw,n,k]`.
`uks` is `(nw, nbandk, nk)`, `ukqs` `(nw, nbandkq, nk)`; `tmp` `(nbandkq, nw*nmodes, nk)` and
`uk_rep` `(nw, nbandk, nmodes*nk)` are scratch at exactly this `nk`.

Generic method: two `batched_gemm!`s (`ukq(k)'` on the left over batch `k`; `uk(k)` on the right
over batch `(ν,k)` after replicating `uks` over the modes into `uk_rep`), so any backend works.
The CUDA extension overrides this with a fused per-(m,n,k) kernel for small `nw²·nmodes` — the
right-rotation GEMMs are `nw×nw` matmuls at an `nmodes·nk` batch count, deep inside cuBLAS'
tiny-batched-matmul overhead regime (cf. the outer-k `eph_apply_rotations!` fused kernel).
"""
function eph_apply_rotations_rqkq!(ep_kq_all::AbstractArray{Complex{T},4}, g,
                                   uks, ukqs, tmp, uk_rep) where {T}
    nbandkq, nbandk, nmodes, nk = size(ep_kq_all)
    nw = size(uks, 1)

    # 1. left rotation ukq(k)', batched over k: tmp[m, (jw,ν), k] = Σ_iw conj(ukqs[iw,m,k]) g[iw,(jw,ν),k]
    batched_gemm!('C', 'N', ukqs, reshape(g, nw, nw * nmodes, nk), tmp) # (nbandkq, nw*nmodes, nk)

    # 2. right rotation uk(k), batched over b = (ν, k): reshape tmp to (nbandkq, nw, nmodes*nk)
    #    (contiguous: splits the (jw,ν) axis into jw and ν, ν fastest in the batch), and replicate
    #    uk over the modes so each batch slice carries its own uk. ep_kq_all reshaped to
    #    (nbandkq, nbandk, nmodes*nk) receives ep[m,n,(ν,k)] = Σ_jw tmp[m,jw,(ν,k)] uks[jw,n,k].
    uk_rep4 = reshape(uk_rep, nw, nbandk, nmodes, nk)
    uk_rep4 .= reshape(uks, nw, nbandk, 1, nk)                         # broadcast uk over ν
    batched_gemm!('N', 'N', reshape(tmp, nbandkq, nw, nmodes * nk), uk_rep,
                  reshape(ep_kq_all, nbandkq, nbandk, nmodes * nk))
    ep_kq_all
end

"""
    add_eph_dipole_batched!(eps, coeffs, ukqs, uks, mmats)

Add the polar (long-range) e-ph dipole term to a batch of e-ph matrices `eps`
`(nbandkq, nbandk, nmodes, nk)`, the batched counterpart of the per-k `epstate_compute_eph_dipole!`
(unscreened, ϵ ≡ 1): `eps[m,n,ν,k] += coeffs[ν,k] · Σ_iw conj(ukqs[iw,m,k]) uks[iw,n,k]`, with
`ukqs` `(nw, nbandkq, nk)`, `uks` `(nw, nbandk, nk)` and `coeffs` `(nmodes, nk)`, or `(nmodes, 1)`
for one q shared by the batch. `mmats` is `(nbandkq, nbandk, nk)` scratch on the same backend. Runs on the backend of `eps` (the `batched_gemm!` + broadcast are backend-generic).
"""
function add_eph_dipole_batched!(eps, coeffs, ukqs, uks, mmats)
    nbandkq, nbandk, nmodes, nk = size(eps)
    batched_gemm!('C', 'N', ukqs, uks, mmats)   # mmats[m,n,k] = Σ_iw conj(ukqs[iw,m,k]) uks[iw,n,k]
    eps .+= reshape(coeffs, 1, 1, nmodes, :) .* reshape(mmats, nbandkq, nbandk, 1, nk)
    eps
end

# The CUDA extension's fused rotation kernel (`_fused_eph_rot_kernel!`) runs when
# `nw*nmodes ≤ _FUSED_ROT_MAX_NWNM`, `nw ≤ _FUSED_ROT_MAX_NW` and `nmodes ≤ _FUSED_ROT_MAX_NMODES`;
# above, the two rotation GEMMs are large enough that cuBLAS wins. The gate lives here rather than in
# the extension so the base package documents the crossover the generic `eph_apply_rotations!`
# docstring refers to. Measured on an A100 against the cuBLAS branch (2026-10-06), with
# nbandk = nbandkq = nw: the kernel wins 1.2-3x through nw*nmodes = 60 and up to nw = 14, and loses
# at nw = 16 with nmodes = 3 and at nw = 8 with nmodes = 15. `nmodes` bounds the per-thread
# register tuple (2 nmodes Float64). The two paths sum in different orders, so compare them with a
# tolerance, not bitwise.
const _FUSED_ROT_MAX_NWNM = 60
const _FUSED_ROT_MAX_NW = 12
const _FUSED_ROT_MAX_NMODES = 32

# Whether the fused rotation kernel takes `nw` Wannier functions and `ndisp` phonon displacements.
_fused_rotation_supported(nw, ndisp) =
    nw * ndisp <= _FUSED_ROT_MAX_NWNM && nw <= _FUSED_ROT_MAX_NW && ndisp <= _FUSED_ROT_MAX_NMODES

# The two-GEMM rotation paths merge `g`'s band and mode axes with a `reshape`, which needs `g` to be
# densely packed — a reshape of a strided view is a `ReshapedArray`, which the batched GEMMs reject.
# A non-strided array is `false`, not an error, so the caller's own assertion message is what the
# user sees. Axes of length 1 carry no meaningful stride, so they are skipped.
# Necessary but not sufficient on the device: a `view` that drops an axis can be dense and still
# reshape to a pointerless `ReshapedArray`, while a contiguous 2-D column slice reshapes back to a
# plain device array. Callers must hand over an operand that survives the reshape.
function _is_dense(a::AbstractArray)
    a isa StridedArray || return false
    st, sz = strides(a), size(a)
    expected = 1
    for d in eachindex(sz)
        (sz[d] == 1 || st[d] == expected) || return false
        expected *= sz[d]
    end
    true
end

"""
    eph_apply_rotations!(ep_kq_all, g, ukqs, u_phs, tmp)

Apply the two e-ph gauge rotations to the Fourier-interpolated `g` `(nw, nbandk, ndisp, nq)`,
writing the eigenbasis e-ph matrix `ep_kq_all`
`(nbandkq, nbandk, nmodes, nq)` = `ukq(q)' * g(q) * u_ph(q)`, with `u_phs` `(ndisp, nmodes, nq)`.

The two-GEMM paths merge `g`'s band and mode axes with a `reshape`, so they require a dense `g`
(asserted); the CUDA extension's fused path indexes `g` elementwise and takes any strided view.

Generic method: the two strided-batched GEMMs (`ukq'` on the left, `u_ph` on the right), so any
backend works. The CUDA extension overrides this with a fused kernel for small `nw*nmodes`, which
avoids cuBLAS' tiny-matmul inefficiency (the 4×4 / nmodes×nmodes strided-batched GEMMs run at ~2%
of FP64 peak).
"""
function eph_apply_rotations!(ep_kq_all::AbstractArray{Complex{T},4}, g::AbstractArray{Complex{T},4},
                              ukqs, u_phs, tmp) where {T}
    nbandkq, nbandk, nmodes, nq = size(ep_kq_all)
    nw = size(ukqs, 1)
    ndisp = size(u_phs, 1)
    @assert size(g) == (nw, nbandk, ndisp, nq)
    @assert _is_dense(g) "the two-GEMM rotation path needs a dense g; a strided g is only supported by the CUDA fused kernel"
    batched_gemm!('C', 'N', ukqs, reshape(g, nw, nbandk * ndisp, nq), tmp)    # ukq(q)' * g(q)
    batched_gemm!('N', 'N', reshape(tmp, nbandkq * nbandk, ndisp, nq), u_phs,
                  reshape(ep_kq_all, nbandkq * nbandk, nmodes, nq))           # * u_ph(q)
    ep_kq_all
end
