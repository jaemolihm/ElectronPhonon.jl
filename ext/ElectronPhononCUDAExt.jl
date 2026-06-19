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

end # module
