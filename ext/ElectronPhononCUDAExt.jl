module ElectronPhononCUDAExt

# CUDA (GPU) acceleration of the Wannier -> Bloch Fourier interpolation.
#
# Design: `WannierObject` is parameterized over its `op_r` array type, so a device
# `WannierObject` (op_r on the GPU) flows through the *generic* batched routines
# `get_fourier_batched!` / `get_el_eigen[_valueonly]_batched` defined in the base package —
# those use only `mul!`, broadcasting, `similar`, and `copyto!`, which CUDA.jl implements for
# `CuArray`. This extension only needs to provide:
#   1. `to_device`                       — move op_r to the GPU.
#   2. `eigvals_batched`/`eigen_batched`  — the batched Hermitian eigensolve (CUSOLVER), the
#                                           one piece with no generic AbstractArray fallback.

using ElectronPhonon
using ElectronPhonon: WannierObject
using CUDA
using CUDA.CUSOLVER: heevjBatched!
using CUDA.CUBLAS: gemm_strided_batched!

# Notes on `heevjBatched!` (cuSOLVER batched Jacobi eigensolver, `cusolverDn<t>heevjBatched`):
#   - It is *tuned* for small matrices (the often-quoted "n ≤ 32" is a performance figure,
#     not a correctness bound). Verified on cuSOLVER 13.3: it solves correctly well past 32
#     (tested to n=256, agreeing with LAPACK to ~1e-11); accuracy degrades gracefully with n.
#   - Some older cuSOLVER versions returned CUSOLVER_STATUS_INVALID_VALUE for large n. We do
#     not guard on size here — CUDA.jl/cuSOLVER raises its own error if a version cannot
#     handle the requested size.
#   - Being a Jacobi solver, it may differ from LAPACK at the level of `tol` (default
#     `eps(T)`) for clustered/degenerate spectra.
#   - EIGENVALUE vs EIGENVECTOR accuracy (ComplexF64/Float64 throughout — this is NOT a Float32
#     effect): the Jacobi sweeps converge eigenVALUES to ~machine eps (measured heevj-vs-LAPACK
#     2.9e-15 for the Pb Hamiltonian), but stop eigenVECTORS at the looser Jacobi tolerance —
#     measured relative residual ‖H·u − u·e‖/‖H‖ ≈ 9e-9, vs LAPACK QR ~1e-15. So `eigvals_batched`
#     (filter, eigenvalues only) is machine-precision, while `eigen_batched`'s eigenvectors carry a
#     ~1e-8 floor that propagates into anything using them (e-ph matrix / g2, band velocities).
#     Negligible for converged BZ-summed observables, but looser than the CPU path. To tighten it,
#     lower the Jacobi tolerance / raise max-sweeps (cuSOLVER `cusolverDnXsyevjSetTolerance` /
#     `SetMaxSweeps`, exposed via CUDA.jl's `heevjBatched!` info object) — not done here.

"""
    to_device(obj::WannierObject{T, <:Array}) -> WannierObject

Return a copy of a host `obj` with `op_r` moved to the GPU (`irvec` stays on the host).
`ndata` is preserved so partial-transform objects keep their semantics. The returned object
works with the generic `get_fourier_batched!` / `get_el_eigen[_valueonly]_batched`.

Restricted to host (`Array`-backed) objects — moving an already-device object is a no-op
that this method intentionally does not provide.
"""
function ElectronPhonon.to_device(obj::WannierObject{T, <:Array{Complex{T}}}) where {T}
    dev = WannierObject(obj.irvec, CuArray(obj.op_r); irvec_next = obj.irvec_next)
    dev.ndata = obj.ndata   # the outer constructor defaults ndata to size(op_r,1); restore it
    dev
end

"""
    eigvals_batched(Hk::CuArray{Complex{T},3}) -> CuMatrix

Eigenvalues `(nw, nk)` of a stack of Hermitian matrices `(nw, nw, nk)` in a single batched
Jacobi eigensolve, on the device. Best suited to small `nw` (see module notes).
"""
function ElectronPhonon.eigvals_batched(Hk::CuArray{Complex{T},3}) where {T}
    # heevjBatched! overwrites its argument; copy to keep Hk intact for the caller.
    heevjBatched!('N', 'U', copy(Hk))
end

"""
    eigen_batched(Hk::CuArray{Complex{T},3}) -> (CuMatrix, CuArray{_,3})

Eigenvalues `(nw, nk)` and eigenvectors `(nw, nw, nk)` of a stack of Hermitian matrices in a
single batched Jacobi eigensolve, on the device. Best suited to small `nw` (see module notes).
"""
function ElectronPhonon.eigen_batched(Hk::CuArray{Complex{T},3}) where {T}
    # heevjBatched!('V', ...) returns (W, V), with V the (copied) input overwritten with the
    # eigenvectors, so the caller's Hk is left intact.
    heevjBatched!('V', 'U', copy(Hk))
end

"""
    batched_gemm!(transA, transB, A::CuArray{T,3}, B, C) -> C

GPU strided-batched GEMM (`CUBLAS.gemm_strided_batched!`), `α=1`, `β=0`.
"""
function ElectronPhonon.batched_gemm!(transA::Char, transB::Char,
                                      A::CuArray{T,3}, B::CuArray{T,3}, C::CuArray{T,3}) where {T}
    gemm_strided_batched!(transA, transB, one(T), A, B, zero(T), C)
    C
end

# ---- fused e-ph gauge rotation (replaces the two tiny cuBLAS strided-batched GEMMs) -----------
#
# For small nw / nmodes the rotations `ep_kq = ukq' * g * u_ph` are nw×nw and nmodes×nmodes
# matmuls; cuBLAS strided-batched runs them at ~2% of FP64 peak (~11 GFLOP/s, flat ~135 ns/batch),
# making them ~82% of the kR->kq cost for materials like Pb (nw=4, nmodes=3). A single fused
# kernel — one thread per q, both rotations done from registers — is ~8× faster and bit-faithful
# (rel err ~3e-16). It also optionally writes g2 = |ep|²/(2ω) in the same pass (no separate abs2).
#
# Above the threshold (large nw/nmodes) the matmuls are big enough that cuBLAS is efficient and
# the per-thread loop would be slow, so we fall back to the two strided-batched GEMMs there.
# The kernel's per-thread work grows ~nw³·nmodes², so we gate on a single criterion — the PRODUCT
# nw·nmodes. A measured A6000 sweep (nq=8192, FP64) crosses over near nw·nmodes ≈ 24: (4,3),(6,3),
# (8,3),(6,4),(4,6) favour the fused kernel (1.3–2.9×); (8,4),(4,8),(6,6),(8,6),(10,3) favour cuBLAS
# (the old per-dim cap `nw≤8 && nmodes≤8` wrongly routed (8,8), where cuBLAS is ~5× faster). A
# separate per-dim cap is unnecessary: nmodes = 3·N_atoms ≥ 3 physically, so the product bounds the
# aspect ratio on its own. NOTE: assumes nbandk, nbandkq ≤ nw (true in the full-band loop: ep_kq is
# (nw,nw,nmodes,·)). THIS THRESHOLD IS A6000-TUNED — retune on other GPUs (on an H100 the crossover
# moves lower: better FP64 / batched-GEMM throughput makes cuBLAS competitive at smaller sizes).
const _FUSED_ROT_MAX_NWNM = 24

# g : (nw, nbandk, nmodes, nq) ; ukq : (nw, nbandkq, nq) ; uph : (nmodes, nmodes, nq)
# ep : (nbandkq, nbandk, nmodes, nq) ; g2 / ωq optional (g2 : same as ep ; ωq : (nmodes, nq)).
function _fused_eph_rot_kernel!(ep, g2, g, ukq, uph, ωq, nw, nbkq, nbk, nm, nq)
    q = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    q <= nq || return
    @inbounds for ibk in 1:nbk, im in 1:nm
        for ibkq in 1:nbkq
            acc = zero(eltype(ep))
            for jm in 1:nm
                tval = zero(eltype(ep))
                for iw in 1:nw
                    tval += conj(ukq[iw, ibkq, q]) * g[iw, ibk, jm, q]
                end
                acc += tval * uph[jm, im, q]
            end
            ep[ibkq, ibk, im, q] = acc
            if g2 !== nothing
                g2[ibkq, ibk, im, q] = abs2(acc) / (2 * ωq[im, q])
            end
        end
    end
    return
end

# `DenseCuArray` (not `CuArray`): the GPU e-ph loop passes contiguous device VIEWS
# (e.g. `view(epkq_dev, :,:,:, 1:nq_chunk)`) for a partial final q-chunk. The fused kernel takes
# them through `@cuda` (cudaconvert handles strided views) and the cuBLAS path takes their
# reshapes (strided), so no padding to a fixed batch width is needed.
function ElectronPhonon.eph_apply_rotations!(ep_kq_all::DenseCuArray{Complex{T},4}, g,
        ukqs::DenseCuArray, u_phs::DenseCuArray, tmp; g2_out=nothing, ωq=nothing) where {T}
    nbandkq, nbandk, nmodes, nq = size(ep_kq_all)
    nw = size(ukqs, 1)
    if nw * nmodes <= _FUSED_ROT_MAX_NWNM
        g4 = reshape(g, nw, nbandk, nmodes, nq)
        threads = 128
        blocks = cld(nq, threads)
        @cuda threads=threads blocks=blocks _fused_eph_rot_kernel!(
            ep_kq_all, g2_out, g4, ukqs, u_phs, ωq, nw, nbandkq, nbandk, nmodes, nq)
    else
        # Large nw/nmodes: cuBLAS strided-batched is efficient; keep the two-GEMM path.
        gemm_strided_batched!('C', 'N', one(Complex{T}), ukqs,
                              reshape(g, nw, nbandk * nmodes, nq), zero(Complex{T}), tmp)
        gemm_strided_batched!('N', 'N', one(Complex{T}),
                              reshape(tmp, nbandkq * nbandk, nmodes, nq), u_phs, zero(Complex{T}),
                              reshape(ep_kq_all, nbandkq * nbandk, nmodes, nq))
        if g2_out !== nothing
            g2_out .= abs2.(ep_kq_all) ./ (2 .* reshape(ωq, 1, 1, nmodes, nq))
        end
    end
    ep_kq_all
end

ElectronPhonon.device_free_bytes(::CuArray) = CUDA.available_memory()
ElectronPhonon.device_synchronize(::CuArray) = CUDA.synchronize()

# ---- device-resident scatter (calculator keeps g2/ωq on the device, no host streaming) --------
#
# One thread per (m,n,ν,j) entry: look up i = imap_i_col[n], f = imap_f[m, ikqs[j]]; if both
# in-window, write the value straight into the flat device g2_out / ωq_out at the mode-fastest
# linear slot. The target `lin` indices are unique across the whole run (distinct k → distinct i,
# distinct k+q → distinct f), so the writes never collide — no atomics, no compaction. Removes the
# per-chunk D2H + host scatter (the calculator's g2/ωq stay resident on the device).
function _window_scatter_kernel!(g2_out, ωq_out, g2vals, imap_i_col, imap_f,
                                 ikqs, ωq, nbandkq, nbandk, nm, nqc, n_i)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = nbandkq * nbandk * nm * nqc
    e <= N || return
    @inbounds begin
        m = (e - 1) % nbandkq + 1
        t = (e - 1) ÷ nbandkq
        n = t % nbandk + 1
        t2 = t ÷ nbandk
        ν = t2 % nm + 1
        j = t2 ÷ nm + 1
        i = imap_i_col[n]
        f = imap_f[m, ikqs[j]]
        if i > 0 && f > 0
            lin = ν + nm * (i - 1) + nm * n_i * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, j]
            ωq_out[lin] = ωq[ν, j]
        end
    end
    return
end

# Dispatch on the device-resident output arrays only: `g2vals` / `ωq` / `ikqs` / `imap_*` are
# strided device VIEWS (e.g. `view(g2_dev, :,:,:,1:nqc)`, `view(imap_i_dev,:,ik)`), i.e. SubArrays
# of CuArrays, not plain `CuArray`s — typing them `::CuArray` would miss this method. cudaconvert
# handles the contiguous/strided views inside the kernel.
function ElectronPhonon.eph_window_scatter!(g2_out::CuArray, ωq_out::CuArray, g2vals,
        imap_i_col, imap_f, ikqs, ωq,
        nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, n_i::Int)
    N = nbandkq * nbandk * nm * nqc
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _window_scatter_kernel!(
        g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
        nbandkq, nbandk, nm, nqc, n_i)
    nothing
end

end # module
