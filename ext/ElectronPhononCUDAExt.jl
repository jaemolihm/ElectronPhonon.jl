module ElectronPhononCUDAExt

# CUDA (GPU) acceleration of the Wannier -> Bloch Fourier interpolation.
#
# Design: `WannierObject` is parameterized over its `op_r` array type, so a device
# `WannierObject` (op_r on the GPU) flows through the *generic* batched routines
# `get_fourier_batched!` / `compute_el_eigen[_valueonly]_batched` defined in the base package —
# those use only `mul!`, broadcasting, `similar`, and `copyto!`, which CUDA.jl implements for
# `CuArray`. This extension only needs to provide:
#   1. `to_device`                       — move op_r to the GPU.
#   2. `eigvals_batched`/`eigen_batched`  — the batched Hermitian eigensolve (one-thread-per-matrix
#                                           Jacobi for small nw, cuSOLVER above), the one piece
#                                           with no generic AbstractArray fallback.

using ElectronPhonon
using ElectronPhonon: WannierObject
using CUDA
using CUDA: cuSOLVER
using CUDA.cuBLAS: gemm!, gemm_strided_batched!
using LinearAlgebra: LinearAlgebra
using CUDA.cuSPARSE: CuSparseMatrixCSR
using SparseArrays: SparseMatrixCSC

# Notes on the device batched eigensolve (`eigen_batched!` / `eigvals_batched!`):
#   - nw ≤ JACOBI_NW_MAX: one-thread-per-matrix cyclic Jacobi (`cuda_eigen_jacobi.jl`). Relative
#     residual ≤ 2e-15 and ‖UᴴU - I‖_F ≤ 2e-14 through nw = 16 on random, exactly-degenerate
#     and near-degenerate (1e-10 splitting) batches. Its eigenvalues are the same bit for bit with
#     and without eigenvectors, so the filter and the state build see the same energies.
#   - nw > JACOBI_NW_MAX: cuSOLVER `cusolverDnZheevjBatched` (`heevj_batched!`). The often-quoted
#     "n ≤ 32" is a performance figure, not a correctness bound, so there is no size guard.
#     Relative residual ≤ 3e-15 for nw ≤ 16 and ≤ 2.2e-14 at nw = 64 (same batches plus real
#     interpolated H(k)), with `tol = eps(Float64)` and `max_sweeps = 100`.
#   - Eigenvectors of degenerate bands differ from the per-k CPU path by a gauge (no EPW
#     gauge-fixing here). That is a basis choice, not an accuracy loss.

include("cuda_eigen_jacobi.jl")

# Largest nw solved by `jacobi_eigen_batched!`; one cutoff for both methods keeps their eigenvalues
# bitwise equal. Jacobi is faster than cuSOLVER through nw = 12 on A100, H100 and A6000, for
# eigenvalues only and with eigenvectors; at nw = 16 the H100 is slower.
const JACOBI_NW_MAX = 12

# GPU backend prototype: an empty `CuArray` carries only the device array TYPE, which is all `alloc`
# needs (`similar(proto, T, dims...)` ignores the prototype's element type and shape). This lets a
# `GPUBackend` be built before any real array is moved to the device.
ElectronPhonon.gpu_backend() = ElectronPhonon.GPUBackend(CuArray{ComplexF64}(undef, 0))

ElectronPhonon.free_bytes(::ElectronPhonon.GPUBackend) = CUDA.free_memory()
ElectronPhonon.synchronize(::ElectronPhonon.GPUBackend) = CUDA.synchronize()

# GPU-aware MPI hands the device pointer straight to the MPI library, so the kernels that filled the
# buffer must have finished first. `ElectronPhonon._sync_device` is a no-op on every other argument.
ElectronPhonon._sync_device(::CuArray) = CUDA.synchronize()

# `CUDA.reclaim()` drops task-local library state, runs a full GC, synchronizes, purges the cuBLAS
# and cuSOLVER handle caches and trims the memory pool. That is its behavior in CUDA.jl 6, which
# `[compat] CUDA = "6"` is what pins; 5.x's `reclaim` is a different, GC-less routine.
ElectronPhonon.reclaim_device_memory(::ElectronPhonon.GPUBackend) = (CUDA.reclaim(); nothing)

# `to_device(::GPUBackend, ::AbstractArray)` always densifies the given array. Add a specialization
# for sparse matrices that preserves sparsity on the device; CSR is what cuSPARSE's SpMM takes.
#
# `mul!` on the result runs cuSPARSE SpMM under `CUSPARSE_SPMM_ALG_DEFAULT`, whose summation order
# varies between runs (1-2 ulp). `cuSPARSE.mm!(…, CUSPARSE_SPMM_CSR_ALG3)` is reproducible if some
# caller ever needs bit-identical repeats.
ElectronPhonon.to_device(::ElectronPhonon.GPUBackend, A::SparseMatrixCSC) = CuSparseMatrixCSR(A)

"""
    to_device(::GPUBackend, obj::WannierObject{T, <:Array}) -> WannierObject

Return a copy of a host `obj` with `op_r` moved to the GPU (`irvec` stays on the host). The
source's `ndata` (partial-transform width) is preserved, so partial-transform objects keep their
semantics; the partial-transform entry point is `get_next_wannier_object`, which validates `ndata`
and hands back an already-partial object. The returned object works with the generic
`get_fourier_batched!` / `compute_el_eigen[_valueonly]_batched`.

Restricted to host (`Array`-backed) objects — moving an already-device object is a no-op
that this method intentionally does not provide.
"""
function ElectronPhonon.to_device(::ElectronPhonon.GPUBackend, obj::WannierObject{T, <:Array{Complex{T}}}) where {T}
    WannierObject(obj.irvec, CuArray(obj.op_r); obj.irvec_next, obj.ndata)
end
ElectronPhonon.to_device(::ElectronPhonon.GPUBackend, arr::AbstractArray) = CuArray(arr)

# `heevjBatched!` as cuSOLVER.jl's wrapper of `cusolverDnZheevjBatched` runs it (same `tol`,
# `max_sweeps`, handle, handle-cached workspace and `info` buffer, so bitwise the same `W` and `A`),
# but into a given `W` and without reading `info` back. An invalid argument still throws, through
# the returned status.
function heevj_batched!(jobz::Char, W::CuArray{Float64,2}, A::CuArray{ComplexF64,3})
    n, _, batchSize = size(A)
    lda = max(1, stride(A, 2))
    dh = cuSOLVER.dense_handle()
    resize!(dh.info, batchSize)
    params = Ref{cuSOLVER.syevjInfo_t}(C_NULL)
    cuSOLVER.cusolverDnCreateSyevjInfo(params)
    cuSOLVER.cusolverDnXsyevjSetTolerance(params[], eps(Float64))
    cuSOLVER.cusolverDnXsyevjSetMaxSweeps(params[], 100)
    function bufferSize()
        out = Ref{Cint}(0)
        cuSOLVER.cusolverDnZheevjBatched_bufferSize(dh, jobz, 'U', n, A, lda, W, out, params[], batchSize)
        out[] * sizeof(ComplexF64)
    end
    # `info` is not read: NVIDIA documents `info[i] = n+1` as non-convergence, but cuSOLVER 12.2.6
    # writes 0 even for `max_sweeps = 1` or a NaN input, so a readback would cost a host
    # synchronization and detect nothing.
    cuSOLVER.with_workspace(dh.workspace_gpu, bufferSize) do buffer
        cuSOLVER.cusolverDnZheevjBatched(dh, jobz, 'U', n, A, lda, W, buffer,
            sizeof(buffer) ÷ sizeof(ComplexF64), dh.info, params[], batchSize)
    end
    # The batched solver reads `params` at launch only (it has no residual or sweep count to report).
    cuSOLVER.cusolverDnDestroySyevjInfo(params[])
    W
end

# Two limits force a cuSOLVER (nw > JACOBI_NW_MAX) batch to be chunked:
#   1. `<t>heevjBatched` returns its workspace size as a 32-bit int, so a large enough batch
#      overflows the bufferSize query and throws CUSOLVER_STATUS_INVALID_VALUE (128³ = 2.097M
#      k-points first trips it at nw=4). The requirement grows as batchSize·nw².
#   2. The workspace is ~16 kB per matrix and the cuSOLVER handle caches it for the lifetime of the
#      process — neither GC nor `CUDA.reclaim()` returns it — so a 2^20 chunk would park 16 GB that
#      the device-resident e-ph tiles then cannot use.
# The cap below keeps the cached workspace under ~1.3 GB at every nw. It is chosen for (2), not (1):
# on 2.097M 4×4 solves (nk=128) it costs ~3% against an arbitrarily large chunk once the allocator
# is warm, and is faster on the first call, which has no 16 GB to allocate. The split is exact —
# results are batch-position independent — so all callers (filter, compute_states) are covered.
#
# The 64-bit-workspace `XsyevBatched!` does not help here: it wants ~1.05 MB per matrix almost
# independently of nw (capping one call at ~72k matrices on an 80 GB A100), and it grows the same
# shared handle cache past the 32-bit `lwork` that these Jacobi calls pass, which makes any later
# `heevjBatched!` in the process throw `InexactError`.
heevj_batch_max(nw::Int) = min(2^16, 2^24 ÷ nw^2)

# The cuSOLVER solve of `eigvals_batched!` / `eigen_batched!`, in chunks of `heevj_batch_max`: each
# chunk of `Hk` is overwritten in place (a trailing-range view of a `CuArray` is a `CuArray` on the
# same memory), with the eigenvectors for `jobz = 'V'`.
function heevj_chunked!(jobz::Char, Hk::CuArray{ComplexF64,3})
    nw, _, nk = size(Hk)
    E = similar(Hk, Float64, nw, nk)
    @views for c in Iterators.partition(1:nk, heevj_batch_max(nw))
        heevj_batched!(jobz, E[:, c], Hk[:, :, c])
    end
    E
end

"""
    eigvals_batched!(Hk::CuArray{ComplexF64,3}) -> CuMatrix

Eigenvalues `(nw, nk)` of a stack of Hermitian matrices `(nw, nw, nk)` on the device: Jacobi with
one thread per matrix for `nw ≤ JACOBI_NW_MAX` (`Hk` is only read), cuSOLVER `heevjBatched` above
(`Hk` is destroyed); see module notes. The Jacobi solve throws if a matrix did not converge.
"""
function ElectronPhonon.eigvals_batched!(Hk::CuArray{ComplexF64,3})
    nw, nw2, nk = size(Hk)
    nw == nw2 || throw(DimensionMismatch("Hk must be square in its first two dimensions, got $(size(Hk))"))
    nk == 0 && return similar(Hk, Float64, nw, 0)
    if nw <= JACOBI_NW_MAX
        jacobi_eigen_batched!(similar(Hk, Float64, nw, nk), nothing, Hk)
    else
        heevj_chunked!('N', Hk)
    end
end

"""
    eigen_batched!(Hk::CuArray{ComplexF64,3}) -> (CuMatrix, CuArray{_,3})

Eigenvalues `(nw, nk)` and eigenvectors `(nw, nw, nk)` of a stack of Hermitian matrices on the
device, by the solver of [`eigvals_batched!`](@ref), whose eigenvalues it reproduces bit for bit.
The eigenvectors overwrite `Hk`, which is the array returned.
"""
function ElectronPhonon.eigen_batched!(Hk::CuArray{ComplexF64,3})
    nw, nw2, nk = size(Hk)
    nw == nw2 || throw(DimensionMismatch("Hk must be square in its first two dimensions, got $(size(Hk))"))
    nk == 0 && return (similar(Hk, Float64, nw, 0), Hk)
    E = if nw <= JACOBI_NW_MAX
        jacobi_eigen_batched!(similar(Hk, Float64, nw, nk), Hk, Hk)
    else
        heevj_chunked!('V', Hk)
    end
    (E, Hk)
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

# Type piracy: GPUArrays ≥ 11.5.11 sends every product with a strided `CuArray` view operand, such
# as the row range `op_r[1:ndata, :]`, to its generic `gpu_coalesced_matmul_kernel` instead of
# cuBLAS (JuliaGPU/GPUArrays.jl#799), so a view and a contiguous copy give different bits. Remove
# once fixed upstream.
function LinearAlgebra._mul!(C::StridedCuMatrix{T}, A::StridedCuMatrix{T}, B::StridedCuMatrix{T},
                             α::Number, β::Number) where {T<:cuBLAS.CublasFloat}
    if C isa CuMatrix && A isa CuMatrix && B isa CuMatrix
        return invoke(LinearAlgebra._mul!,
                      Tuple{AbstractMatrix, AbstractVecOrMat, AbstractVecOrMat, Number, Number},
                      C, A, B, α, β)
    end
    gemm!('N', 'N', T(α), A, B, T(β), C)
end

# ---- fused e-ph gauge rotation (replaces the two tiny cuBLAS strided-batched GEMMs) -----------
#
# For small nw/nmodes the rotations `ep_kq = ukq' * g * u_ph` are tiny matmuls that cuBLAS
# strided-batched runs far below FP64 peak. A single fused kernel (both rotations from registers)
# is faster. Above a threshold the matmuls are large enough that cuBLAS wins, so we fall back to
# the two GEMMs.
# Per-thread work grows ~nw³·nmodes², so we gate on the single product nw·nmodes (nmodes = 3·N_atoms
# ≥ 3, so the product bounds the aspect ratio — no separate per-dim cap needed). Assumes nbandk,
# nbandkq ≤ nw (true in the full-band loop). The threshold itself is
# `ElectronPhonon._FUSED_ROT_MAX_NWNM`, in `src` next to the generic method it selects against.

# g : (nw, nbandk, nmodes, nq) ; ukq : (nw, nbandkq, nq) ; uph : (nmodes, nmodes, nq)
# ep : (nbandkq, nbandk, nmodes, nq).
# One thread per (ibkq, ibk, q) — NOT per q: a per-q thread leaves the GPU idle at production
# chunk sizes (nq ~ 2·10³-2·10⁴ threads is a handful of blocks on ~100 SMs; the (band², q) grid
# is nbandkq·nbandk× larger). Each thread accumulates its entry in a fixed order over (jm, iw), so
# the result does not depend on the launch configuration; the per-im re-read of g/ukq is L1-served
# (the kernel is occupancy-, not flop-bound).
function _fused_eph_rot_kernel!(ep, g, ukq, uph, nw, nbkq, nbk, nm, nq)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    t <= nbkq * nbk * nq || return
    @inbounds begin
        ibkq = (t - 1) % nbkq + 1
        r = (t - 1) ÷ nbkq
        ibk = r % nbk + 1
        q = r ÷ nbk + 1
        for im in 1:nm
            acc = zero(eltype(ep))
            for jm in 1:nm
                tval = zero(eltype(ep))
                for iw in 1:nw
                    tval += conj(ukq[iw, ibkq, q]) * g[iw, ibk, jm, q]
                end
                acc += tval * uph[jm, im, q]
            end
            ep[ibkq, ibk, im, q] = acc
        end
    end
    return
end

# `DenseCuArray` (not `CuArray`): the e-ph loop passes contiguous device VIEWS of a tile's
# buffers for a block narrower than the tile. The fused kernel takes
# them through `@cuda` (cudaconvert handles strided views) and the cuBLAS path takes their
# reshapes (strided), so no padding to a fixed batch width is needed. `g` alone may be any strided
# device array: only the fused branch accepts that, hence the density assert on the cuBLAS branch.
function ElectronPhonon.eph_apply_rotations!(ep_kq_all::DenseCuArray{Complex{T},4},
        g::AnyCuArray{Complex{T},4},
        ukqs::DenseCuArray, u_phs::DenseCuArray, tmp) where {T}
    nbandkq, nbandk, nmodes, nq = size(ep_kq_all)
    nw = size(ukqs, 1)
    @assert size(g) == (nw, nbandk, nmodes, nq)
    if nw * nmodes <= ElectronPhonon._FUSED_ROT_MAX_NWNM
        threads = 256
        blocks = cld(nbandkq * nbandk * nq, threads)
        @cuda threads=threads blocks=blocks _fused_eph_rot_kernel!(
            ep_kq_all, g, ukqs, u_phs, nw, nbandkq, nbandk, nmodes, nq)
    else
        # Large nw/nmodes: cuBLAS strided-batched is efficient; keep the two-GEMM path.
        @assert ElectronPhonon._is_dense(g)
        gemm_strided_batched!('C', 'N', one(Complex{T}), ukqs,
                              reshape(g, nw, nbandk * nmodes, nq), zero(Complex{T}), tmp)
        gemm_strided_batched!('N', 'N', one(Complex{T}),
                              reshape(tmp, nbandkq * nbandk, nmodes, nq), u_phs, zero(Complex{T}),
                              reshape(ep_kq_all, nbandkq * nbandk, nmodes, nq))
    end
    ep_kq_all
end

# ---- fused Rq→kq gauge rotation (outer-q loop) -------------------------------------------------
#
# Same tiny-GEMM pathology as the outer-k rotation above, but along the k batch: the right
# rotation of `eph_apply_rotations_rqkq!` is an nw×nw matmul strided-batched over nmodes·nk
# (≈ 2·10⁶ batches at nw=3, nmodes=21, nk ≈ 10⁵), where cuBLAS' flat per-batch overhead
# dominates. Following the outer-k `_fused_eph_rot_kernel!` precedent: one thread per (m, n, k)
# with the ν loop inside and the small iw/jw contractions in registers, replacing both GEMMs.
# The per-jw partial Σ_iw is re-read per n (L1-served), mirroring that kernel's redundancy note.
# Per-thread work grows as nmodes·nw², so gate on nw²·nmodes; above the threshold the matmuls are
# large enough that cuBLAS is efficient and the generic two-GEMM method is used. The threshold
# covers small-nw metals (e.g. nw=3, nmodes=21 → 189) and excludes e.g. nw=8, nmodes=12 → 768.
const _FUSED_RQKQ_MAX_NW2NM = 512

# g : (nw, nw, nmodes, nk) with legend g[iw, jw, ν, k] (iw = k+q leg, jw = k leg);
# uks : (nw, nbandk, nk); ukqs : (nw, nbandkq, nk); ep : (nbandkq, nbandk, nmodes, nk).
function _fused_rqkq_rot_kernel!(ep, g, uks, ukqs, nw, nbkq, nbk, nm, nk)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    t <= nbkq * nbk * nk || return
    @inbounds begin
        m = (t - 1) % nbkq + 1
        r = (t - 1) ÷ nbkq
        n = r % nbk + 1
        k = r ÷ nbk + 1
        for ν in 1:nm
            acc = zero(eltype(ep))
            for jw in 1:nw
                tval = zero(eltype(ep))
                for iw in 1:nw
                    tval += conj(ukqs[iw, m, k]) * g[iw, jw, ν, k]
                end
                acc += tval * uks[jw, n, k]
            end
            ep[m, n, ν, k] = acc
        end
    end
    return
end

function ElectronPhonon.eph_apply_rotations_rqkq!(ep_kq_all::CuArray{Complex{T},4}, g,
        uks::CuArray, ukqs::CuArray, tmp, uk_rep) where {T}
    nbandkq, nbandk, nmodes, nk = size(ep_kq_all)
    nw = size(uks, 1)
    if nw^2 * nmodes <= _FUSED_RQKQ_MAX_NW2NM
        g4 = reshape(g, nw, nw, nmodes, nk)
        threads = 256
        blocks = cld(nbandkq * nbandk * nk, threads)
        @cuda threads=threads blocks=blocks _fused_rqkq_rot_kernel!(
            ep_kq_all, g4, uks, ukqs, nw, nbandkq, nbandk, nmodes, nk)
        ep_kq_all
    else
        # Large nw²·nmodes: cuBLAS strided-batched is efficient; use the generic two-GEMM method.
        invoke(ElectronPhonon.eph_apply_rotations_rqkq!,
               Tuple{AbstractArray{Complex{T},4}, Any, Any, Any, Any, Any},
               ep_kq_all, g, uks, ukqs, tmp, uk_rep)
    end
end

# ---- device-resident scatter (calculator keeps g2/ωq on the device, no host streaming) --------
#
# One thread per (m,n,ν,j) entry: look up i = imap_i_col[n], f = imap_f[m, ikqs[j]]; if both
# in-window, write the value straight into the flat device g2_out / ωq_out at the mode-fastest
# linear slot. The target `lin` indices are unique across the whole run (distinct k → distinct i,
# distinct k+q → distinct f), so the writes never collide — no atomics, no compaction. Removes the
# per-batch D2H + host scatter (the calculator's g2/ωq stay resident on the device).
# `ni_stride` = the output buffer's outer-k (i) extent, `i0` = its global-i offset, so global state
# i writes to local row (i - i0): full buffer → ni_stride = n_i, i0 = 0; per-batch buffer →
# ni_stride = batch i-extent, i0 = batch offset. See `eph_window_scatter!` in calculator/calculator_utils.jl.

# Decode a 1-based flat index into its column-major subscripts, given the axis lengths. @inline and
# non-allocating (tuple recursion) so it is device-safe inside a kernel:
#   m, n, ν, ipair = _unroll_index(ind, (nbandkq, nbandk, nm, npairs))
@inline _unroll_index(ind::Integer, ::Tuple{}) = ()
@inline function _unroll_index(ind::Integer, dims::NTuple{N, Integer}) where {N}
    d = dims[1]
    ((ind - 1) % d + 1, _unroll_index((ind - 1) ÷ d + 1, Base.tail(dims))...)
end

function _window_scatter_kernel!(g2_out, ωq_out, g2vals, imap_i_col, imap_f,
                                 ikqs, ωq, nbandkq, nbandk, nm, npairs, ni_stride, i0)
    ind_mnνq = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = nbandkq * nbandk * nm * npairs
    ind_mnνq <= N || return
    @inbounds begin
        m, n, ν, ipair = _unroll_index(ind_mnνq, (nbandkq, nbandk, nm, npairs))
        i = imap_i_col[n]
        f = imap_f[m, ikqs[ipair]]
        if i > 0 && f > 0
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, ipair]
            ωq_out[lin] = ωq[ν, ipair]
        end
    end
    return
end

# Dispatch on the device-resident output arrays only. The inputs arrive as device VIEWS (e.g.
# `view(dev.g2, :, :, :, 1:npairs, chunk)`, `view(imap_i_dev, :, ik)`) whose type depends on the index
# pattern: a CONTIGUOUS view of a `CuArray` is a `CuArray`, a strided one is a `SubArray` of one,
# and both work here (cudaconvert handles either) only because they are left unannotated. So a host
# argument cannot be caught by an annotation; `CUDA.allowscalar(false)` instead makes an accidental
# host array a hard error inside the kernel. The outputs are annotated, so a strided output
# `SubArray` would miss this method and fall through to the generic one.
function ElectronPhonon.eph_window_scatter!(g2_out::CuArray, ωq_out::CuArray, g2vals,
        imap_i_col, imap_f, ikqs, ωq, ni_stride::Int, i0::Int)
    nbandkq, nbandk, nm, npairs = ElectronPhonon._scatter_extents(g2vals, ωq, ikqs, imap_i_col, imap_f)
    N = nbandkq * nbandk * nm * npairs
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _window_scatter_kernel!(
        g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
        nbandkq, nbandk, nm, npairs, ni_stride, i0)
    nothing
end

# Complex sibling of the above: writes Re/Im of the raw matrix element and, when `ωq_out` is not
# `nothing`, the frequency. `Nothing` is a singleton type, so that branch is resolved at compile
# time and the no-ωq launch carries no extra work. See `eph_window_scatter_reim!` in
# calculator/calculator_utils.jl.
function _window_scatter_reim_kernel!(re_out, im_out, ωq_out, epvals, imap_i_col, imap_f,
                                      ikqs, ωq, nbandkq, nbandk, nm, npairs, ni_stride, i0)
    ind_mnνq = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = nbandkq * nbandk * nm * npairs
    ind_mnνq <= N || return
    m, n, ν, ipair = _unroll_index(ind_mnνq, (nbandkq, nbandk, nm, npairs))
    i = imap_i_col[n]
    f = imap_f[m, ikqs[ipair]]
    if i > 0 && f > 0
        lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
        ep = epvals[m, n, ν, ipair]
        re_out[lin] = real(ep)
        im_out[lin] = imag(ep)
        ωq_out === nothing || (ωq_out[lin] = ωq[ν, ipair])
    end
    return
end

function ElectronPhonon.eph_window_scatter_reim!(re_out::CuArray, im_out::CuArray, ωq_out, epvals,
        imap_i_col, imap_f, ikqs, ωq, ni_stride::Int, i0::Int)
    nbandkq, nbandk, nm, npairs = ElectronPhonon._scatter_extents(epvals, ωq, ikqs, imap_i_col, imap_f)
    N = nbandkq * nbandk * nm * npairs
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _window_scatter_reim_kernel!(
        re_out, im_out, ωq_out, epvals, imap_i_col, imap_f, ikqs, ωq,
        nbandkq, nbandk, nm, npairs, ni_stride, i0)
    nothing
end

# ---- device BTE accumulate kernel (BoltzmannCalculator; CuArray method of bte_window_accumulate!) -
#
# GPU implementation of `bte_window_accumulate!`: one thread per (m, n, j) — m = k+q band, n = k
# band, j = q index within the batch. It looks up the outer/inner states i, f; sums the shared
# per-mode physics (`bte_scattering_increments` — the SAME function the CPU path calls) over the
# nmodes modes for each temperature; atomic-adds the scattering-out term into Sₒ (many (m,j) share an
# i) and writes the scattering-in term into Sᵢ (each (i,f) is hit by a unique thread across the whole
# run → no atomic). See the generic method's docstring for the full accumulation semantics.
function _bte_window_accumulate_kernel!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs,
        e_i, e_f, wf, μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nmodes, npairs, nT, i0)
    # Flat thread index ind_mnq ∈ 1:N over the (m, n, ipair) grid (N = nbandkq·nbandk·npairs).
    # TODO: the CUDA index intrinsics are Int32, so this overflows if N ≥ 2^31. Unreachable today (a
    # grid that large would exceed device memory), and systemic to all kernels in this extension;
    # widen to Int (or chunk the launch) if a case ever approaches 2^31 threads.
    ind_mnq = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = nbandkq * nbandk * npairs
    ind_mnq <= N || return
    @inbounds begin
        m, n, ipair = _unroll_index(ind_mnq, (nbandkq, nbandk, npairs))
        i = imap_i_at_k[n]         # outer (k) state index; 0 = out of window → skip
        i > 0 || return
        ikq = ikqs[ipair]       # k+q point of this pair
        f = imap_f[m, ikq]         # inner (k+q) state index; 0 = out of window → skip
        f > 0 || return
        ek = e_i[i]; ekq = e_f[f]; wtq = wf[f]   # per-final-state weight
        il = i - i0                # tile-local outer row (i0 = the current Sᵢ tile's global offset)
        for iT in 1:nT             # one entry per temperature
            μ = μs[iT]; T = Ts[iT]; η = ηs[iT]
            sₒ = zero(eltype(Sₒ_out)); sᵢ = sₒ
            for ν in 1:nmodes
                ωq = ωqmat[ν, ipair]
                ωq < ω_cutoff && continue
                sₒ_ν, sᵢ_ν = ElectronPhonon.bte_scattering_increments(
                    method, ek, ekq, ωq, g2vals[m, n, ν, ipair], wtq, μ, T, η)
                sₒ += sₒ_ν; sᵢ += sᵢ_ν
            end
            CUDA.@atomic Sₒ_out[i, iT] += sₒ
            Sᵢ_out[il, f, iT] = sᵢ
        end
    end
    return
end

# CuArray method of `bte_window_accumulate!` (generic method + full docstring in
# src/boltzmann/boltzmann_calculator.jl): launches `_bte_window_accumulate_kernel!` with one thread per
# (m, n, j) over the batch, accumulating this batch's Sₒ/Sᵢ contributions into the device buffers.
function ElectronPhonon.bte_window_accumulate!(Sₒ_out::CuArray, Sᵢ_out::CuArray, g2vals, ωqmat,
        imap_i_at_k, imap_f, ikqs, e_i, e_f, wf, μs, Ts, ηs, method::Int, ω_cutoff, i0::Int)
    nbandkq, nbandk, nmodes, npairs = ElectronPhonon._scatter_extents(g2vals, ωqmat, ikqs,
                                                                        imap_i_at_k, imap_f)
    nT = length(μs)
    N = nbandkq * nbandk * npairs
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _bte_window_accumulate_kernel!(
        Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs, e_i, e_f, wf,
        μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nmodes, npairs, nT, i0)
    nothing
end

end # module
