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
#   - The often-quoted "n ≤ 32" is a performance figure, not a correctness bound; it solves
#     correctly well past 32. We do not guard on size — cuSOLVER raises its own error if a
#     particular version cannot handle the requested n.
#   - EIGENVALUE vs EIGENVECTOR accuracy (ComplexF64/Float64 throughout — NOT a Float32 effect):
#     the Jacobi sweeps converge eigenVALUES to ~machine eps but stop eigenVECTORS at the looser
#     Jacobi tolerance (~1e-8 residual floor, vs LAPACK QR ~1e-15). So `eigvals_batched` (filter,
#     eigenvalues only) is machine-precision, while `eigen_batched`'s eigenvectors carry that floor
#     into anything using them (e-ph matrix / g2, band velocities) — negligible for converged
#     BZ-summed observables, but looser than the CPU path. Tightenable via the cuSOLVER Jacobi
#     tolerance / max-sweeps knobs; not done here.

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

ElectronPhonon.device_array_prototype(::Type{T}) where {T} = CuArray{T}(undef, 0)
ElectronPhonon.device_free_bytes(::CuArray) = CUDA.free_memory()
ElectronPhonon.device_synchronize(::CuArray) = CUDA.synchronize()

"""
    eigvals_batched(Hk::CuArray{Complex{T},3}) -> CuMatrix

Eigenvalues `(nw, nk)` of a stack of Hermitian matrices `(nw, nw, nk)` in a single batched
Jacobi eigensolve, on the device. Best suited to small `nw` (see module notes).
"""
# cuSOLVER's `<t>heevjBatched` returns its workspace size as a 32-bit int; the requirement grows as
# batchSize·nw², so above a batch-size threshold the bufferSize query overflows and throws
# CUSOLVER_STATUS_INVALID_VALUE (128³ = 2.097M k-points first trips it at nw=4). Cap the per-chunk
# batch to keep batchSize·nw² under budget (2^20 is validated safe at nw=4, and is the hard cap for
# small nw), then chunk above it. The split is exact — results are batch-position independent — so
# all callers (filter, compute_states) are transparently covered.
# TODO: switch to the 64-bit-workspace `XsyevBatched!` solver, which removes the need to chunk.
heevj_batch_max(nw::Int) = min(2^20, 2^24 ÷ nw^2)   # 2^24 = 2^20·4²: the nw=4 budget

function ElectronPhonon.eigvals_batched(Hk::CuArray{Complex{T},3}) where {T}
    nw, _, nk = size(Hk)
    batch_max = heevj_batch_max(nw)
    # heevjBatched! overwrites Hk (destroy-input contract; see the CPU method).
    nk <= batch_max && return heevjBatched!('N', 'U', Hk)
    E = similar(Hk, T, nw, nk)
    for c in Iterators.partition(1:nk, batch_max)
        # `Hk[:,:,c]` is a fresh getindex copy that heevjBatched! may overwrite — do NOT wrap the
        # source in @views (a CuArray trailing-range view aliases Hk). `E[:,c] .=` is already an
        # in-place broadcast assignment (dotview), so it needs no @views.
        E[:, c] .= heevjBatched!('N', 'U', Hk[:, :, c])
    end
    E
end

"""
    eigen_batched(Hk::CuArray{Complex{T},3}) -> (CuMatrix, CuArray{_,3})

Eigenvalues `(nw, nk)` and eigenvectors `(nw, nw, nk)` of a stack of Hermitian matrices in a
single batched Jacobi eigensolve, on the device. Best suited to small `nw` (see module notes).
"""
function ElectronPhonon.eigen_batched(Hk::CuArray{Complex{T},3}) where {T}
    nw, _, nk = size(Hk)
    batch_max = heevj_batch_max(nw)
    # heevjBatched!('V', ...) returns (E, U) with U being Hk overwritten with the eigenvectors
    # (destroy-input contract; see the CPU method).
    nk <= batch_max && return heevjBatched!('V', 'U', Hk)
    E = similar(Hk, T, nw, nk)
    U = similar(Hk, nw, nw, nk)
    for c in Iterators.partition(1:nk, batch_max)
        Ec, Uc = heevjBatched!('V', 'U', Hk[:, :, c])   # Hk[:,:,c]: fresh getindex copy ('V' overwrites it)
        E[:, c] .= Ec
        U[:, :, c] .= Uc
    end
    (E, U)
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
# For small nw/nmodes the rotations `ep_kq = ukq' * g * u_ph` are tiny matmuls that cuBLAS
# strided-batched runs far below FP64 peak. A single fused kernel (one thread per q, both rotations
# from registers) is faster and bit-faithful, and can also fold g2 = |ep|²/(2ω) into the same pass.
# Above a threshold the matmuls are large enough that cuBLAS wins, so we fall back to the two GEMMs.
# Per-thread work grows ~nw³·nmodes², so we gate on the single product nw·nmodes (nmodes = 3·N_atoms
# ≥ 3, so the product bounds the aspect ratio — no separate per-dim cap needed). Assumes nbandk,
# nbandkq ≤ nw (true in the full-band loop). The crossover is hardware-dependent: the value below
# was tuned on one GPU and should be retuned elsewhere.
const _FUSED_ROT_MAX_NWNM = 24

# g : (nw, nbandk, nmodes, nq) ; ukq : (nw, nbandkq, nq) ; uph : (nmodes, nmodes, nq)
# ep : (nbandkq, nbandk, nmodes, nq) ; g2 / ωq optional (g2 : same as ep ; ωq : (nmodes, nq)).
# One thread per (ibkq, ibk, q) — NOT per q: a per-q thread leaves the GPU idle at production
# chunk sizes (nq ~ 2·10³–2·10⁴ threads is a handful of blocks on ~100 SMs; the (band², q) grid
# is nbandkq·nbandk× larger). The scalar accumulation below is the previous per-q kernel body
# verbatim (with the ibk/ibkq loops hoisted into the thread index), so results are bit-identical;
# the redundant per-im re-read of g/ukq is L1-served (the kernel is occupancy-, not flop-bound).
function _fused_eph_rot_kernel!(ep, g2, g, ukq, uph, ωq, nw, nbkq, nbk, nm, nq)
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
            if g2 !== nothing
                g2[ibkq, ibk, im, q] = abs2(acc) / (2 * ωq[im, q])
            end
        end
    end
    return
end

# `DenseCuArray` (not `CuArray`): the GPU e-ph loop passes contiguous device VIEWS
# (e.g. `view(epkq_dev, :,:,:, 1:nq_batch)`) for a partial final q-batch. The fused kernel takes
# them through `@cuda` (cudaconvert handles strided views) and the cuBLAS path takes their
# reshapes (strided), so no padding to a fixed batch width is needed.
function ElectronPhonon.eph_apply_rotations!(ep_kq_all::DenseCuArray{Complex{T},4}, g,
        ukqs::DenseCuArray, u_phs::DenseCuArray, tmp; g2_out=nothing, ωq=nothing) where {T}
    nbandkq, nbandk, nmodes, nq = size(ep_kq_all)
    nw = size(ukqs, 1)
    if nw * nmodes <= _FUSED_ROT_MAX_NWNM
        g4 = reshape(g, nw, nbandk, nmodes, nq)
        threads = 256
        blocks = cld(nbandkq * nbandk * nq, threads)
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

# ---- device-resident scatter (calculator keeps g2/ωq on the device, no host streaming) --------
#
# One thread per (m,n,ν,j) entry: look up i = imap_i_col[n], f = imap_f[m, ikqs[j]]; if both
# in-window, write the value straight into the flat device g2_out / ωq_out at the mode-fastest
# linear slot. The target `lin` indices are unique across the whole run (distinct k → distinct i,
# distinct k+q → distinct f), so the writes never collide — no atomics, no compaction. Removes the
# per-batch D2H + host scatter (the calculator's g2/ωq stay resident on the device).
# `ni_stride` = the output buffer's outer-k (i) extent, `i0` = its global-i offset, so global state
# i writes to local row (i − i0): full buffer → ni_stride = n_i, i0 = 0; per-batch buffer →
# ni_stride = batch i-extent, i0 = batch offset. See `eph_window_scatter!` in calculator/calculator_utils.jl.
function _window_scatter_kernel!(g2_out, ωq_out, g2vals, imap_i_col, imap_f,
                                 ikqs, ωq, nbandkq, nbandk, nm, nqc, ni_stride, i0)
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
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
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
        nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, ni_stride::Int; i0::Int = 0)
    N = nbandkq * nbandk * nm * nqc
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _window_scatter_kernel!(
        g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
        nbandkq, nbandk, nm, nqc, ni_stride, i0)
    nothing
end

# ---- device BTE accumulate kernel (BoltzmannCalculator; CuArray method of bte_window_accumulate!) -
#
# GPU implementation of `bte_window_accumulate!`: one thread per (m, n, j) — m = k+q band, n = k
# band, j = q index within the chunk. It looks up the outer/inner states i, f; sums the shared
# per-mode physics (`bte_scattering_increments` — the SAME function the CPU path calls) over the nm
# modes for each temperature; atomic-adds the scattering-out term into Sₒ (many (m,j) share an i)
# and writes the scattering-in term into Sᵢ (each (i,f) is hit by a unique thread across the whole
# run → no atomic). See the generic method's docstring for the full accumulation semantics.
function _bte_window_accumulate_kernel!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_col, imap_f, ikqs,
        e_i, e_f, wq, μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nm, nqc, nT, i0)
    # Flat thread index e ∈ 1:N over the (m, n, j) grid (N = nbandkq·nbandk·nqc).
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = nbandkq * nbandk * nqc
    e <= N || return
    @inbounds begin
        # Unroll the composite index e-1 into (m, n, j), m fastest then n then j (column-major over
        # the (nbandkq, nbandk, nqc) grid): m ∈ 1:nbandkq, n ∈ 1:nbandk, j ∈ 1:nqc.
        m = (e - 1) % nbandkq + 1
        t = (e - 1) ÷ nbandkq
        n = t % nbandk + 1
        j = t ÷ nbandk + 1
        i = imap_i_col[n]          # outer (k) state index; 0 = out of window → skip
        i > 0 || return
        ikq = ikqs[j]              # k+q point of this q within the chunk
        f = imap_f[m, ikq]         # inner (k+q) state index; 0 = out of window → skip
        f > 0 || return
        ek = e_i[i]; ekq = e_f[f]; wtq = wq[ikq]
        il = i - i0                # tile-local outer row (i0 = 0 for the full-resident buffer)
        for iocc in 1:nT           # one entry per temperature
            μ = μs[iocc]; T = Ts[iocc]; η = ηs[iocc]
            sₒ = zero(eltype(Sₒ_out)); sᵢ = sₒ
            for ν in 1:nm
                ωq = ωqmat[ν, j]
                ωq < ω_cutoff && continue
                so, si = ElectronPhonon.bte_scattering_increments(
                    method, ek, ekq, ωq, g2vals[m, n, ν, j], wtq, μ, T, η)
                sₒ += so; sᵢ += si
            end
            CUDA.@atomic Sₒ_out[i, iocc] += sₒ
            Sᵢ_out[il, f, iocc] = sᵢ
        end
    end
    return
end

# CuArray method of `bte_window_accumulate!` (generic method + full docstring in
# src/boltzmann/boltzmann_calculator.jl): launches `_bte_window_accumulate_kernel!` with one thread per
# (m, n, j) over the chunk, accumulating this chunk's Sₒ/Sᵢ contributions into the device buffers.
function ElectronPhonon.bte_window_accumulate!(Sₒ_out::CuArray, Sᵢ_out::CuArray, g2vals, ωqmat,
        imap_i_col, imap_f, ikqs, e_i, e_f, wq, μs, Ts, ηs, method::Int, ω_cutoff,
        nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, nT::Int; i0::Int = 0)
    N = nbandkq * nbandk * nqc
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks _bte_window_accumulate_kernel!(
        Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_col, imap_f, ikqs, e_i, e_f, wq,
        μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nm, nqc, nT, i0)
    nothing
end

end # module
