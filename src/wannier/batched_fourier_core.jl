using LinearAlgebra

"""
    build_fourier_phase!(dest, irvec_mat, xkmat) -> dest

Wannier → Bloch phase matrix `dest[ip, j] = exp(2πi R_p · x_j)` for the R-vectors in `irvec_mat`
(`(nr × 3)` real, row `ip` = `R_p`) and the crystal-coordinate points in `xkmat` (`(3 × nk)` real).

Contract:
- all three arrays live on one backend, and the whole thing is a single broadcast (no scalar
  indexing, no scratch), so it runs unchanged on CPU and GPU;
- `dest` is `(nr, nk)` and may be a view;
- `dest` is caller-owned. That is what lets the GPU outer-k e-ph loop keep two phase tiles of
  different widths (`P_mk` at the k-batch width, `P_kq` at the q-tile width);
- stateless: the phase depends only on `(R_p, x)`, so a caller whose `x` list is loop-invariant
  builds it once and applies it many times. That hoist is the reason this is reachable at all
  (as `ElectronPhonon.build_fourier_phase!`) rather than being private to the Fourier engine.
"""
function build_fourier_phase!(dest, irvec_mat, xkmat)
    # `dest` is `(nr, nk)`: entry `[ir, ik]` is `exp(2πi R[ir] · x[ik])`, an outer product over the
    # `(nr × 3)` R-vectors and the `(3 × nk)` crystal coordinates. Verified allocation-free (`@allocated`
    # returns 0 on CPU), so it needs no `rdotk` temporary.
    # `xkmat[d:d, :]` is the `1 × nk` row of coordinate `d`, broadcast against the `nr` R-vectors.
    @views dest .= cispi.(2 .* (irvec_mat[:, 1] .* xkmat[1:1, :] .+
                                irvec_mat[:, 2] .* xkmat[2:2, :] .+
                                irvec_mat[:, 3] .* xkmat[3:3, :]))
    dest
end


"""
    BatchedFourierCore{T, WT, BT, MC, MR}

Stateless whole-batch Wannier → Bloch Fourier engine. Holds only the persistent GEMM scratch
needed to interpolate up to `batch_size` k-points in a single call; it has no notion of
registration order or cache position (that is [`SequentialQueryCache`](@ref)'s job).

Interpolates fresh from `parent.op_r` on every call and never consults `parent._id`, so it is
stateless with respect to the parent data (`_id` is a GridOpt-family concern).

The parent bound is `WannierObject`, not `AbstractWannierObject`: the engine is one GEMM against an
in-memory `op_r`, and a disk-backed parent is served by the per-k `"normal"` / `"gridopt"` modes
instead ([`get_interpolator`](@ref) rejects the combination).

# Type parameters
The buffer arrays are allocated on `backend`:
- `BT` — backend        : `CPUBackend` or `GPUBackend`
- `MC` — complex matrix : CPU `Matrix{Complex{T}}`, GPU `CuMatrix{Complex{T}}`
- `MR` — real matrix    : CPU `Matrix{T}`,          GPU `CuMatrix{T}`
"""
struct BatchedFourierCore{T, WT <: WannierObject, BT <: AbstractBackend, MC, MR}
    # Parent WannierObject to be interpolated
    parent::WT

    # Where `parent.op_r` and every buffer below live
    backend::BT

    # Maximum number of k-points a single `_fourier_batched!` call handles. A cap, not a size: the
    # last block of a longer k-list is shorter; named `batch_size` to match the public kwarg.
    batch_size::Int

    # R-vectors as an (nr × 3) real matrix on the backend, for the GEMM phase computation
    irvec_mat::MR

    # Scratch for the batched phase computation, on `backend`
    phase::MC                # (nr × batch_size) complex
end

function BatchedFourierCore(parent::WT; backend::AbstractBackend = CPUBackend(),
        batch_size::Int = 32) where {WT <: WannierObject{T}} where {T}
    # A device `op_r` reached with a forgotten `backend =` would otherwise run as a silent mixed
    # host/device broadcast; name the array instead.
    check_on_backend(backend, parent.op_r, "op_r")
    nr = length(parent.irvec)

    irvec_mat = _irvec_to_device_matrix(backend, parent.irvec, T)
    phase = alloc(backend, Complex{T}, nr, batch_size)

    BatchedFourierCore{T, WT, typeof(backend), typeof(phase), typeof(irvec_mat)}(
        parent, backend, batch_size, irvec_mat, phase)
end


"""
    _fourier_batched!(out, core::BatchedFourierCore, xkmat::AbstractMatrix)

Fourier-transform `core.parent` at the `(3 × nk)` crystal coordinates `xkmat`, writing into `out`
`(ndata, nk)`. One broadcast for the phases ([`build_fourier_phase!`](@ref)) and one GEMM for the
transform (`op_r * phase`), so it runs on any backend without scalar indexing.

PARTIAL, internal: `xkmat` and `out` must already live on `core.backend`, and `nk` must not exceed
`core.batch_size`. Staging a host k-list and splitting a longer one into blocks belong to the total
[`get_fourier_batched!`](@ref) on the interpolator.
"""
function _fourier_batched!(out, core::BatchedFourierCore, xkmat::AbstractMatrix)
    (; parent, phase) = core
    ndata = parent.ndata
    nk = size(xkmat, 2)
    @assert size(out) == (ndata, nk)
    @assert nk <= core.batch_size

    @views build_fourier_phase!(phase[:, 1:nk], core.irvec_mat, xkmat)
    # BLAS3 gemm: much faster than multiple BLAS2 gemv calls
    @views mul!(out, parent.op_r[1:ndata, :], phase[:, 1:nk])
    out
end


# Device byte budget for one interpolator's Fourier scratch. Fixed rather than `free_bytes`-derived,
# so the default stays deterministic and the device-byte formulas in `calculator/eph_device_staging.jl`
# stay static. 1 GiB is the smallest power of two that clears the measured launch/efficiency knee at
# both ends of the range of objects in use: at 512 MB a Cu-sized `el_ham` (nr = 2000, ndata = 49)
# gets 16 376 columns and runs 24% slower than at 65 536, while 1 GiB gives it 32 752 (+6%) and puts
# every Pb object past 10^5. `filter.jl` keeps its own 1 GiB constant; the two bound different
# quantities (an eigensolve stack there, Fourier scratch here) and are deliberately not shared.
const GPU_FOURIER_BATCH_BYTES = 2^30

"""
    _default_batch_size(backend, nr, ndata; nk_hint = typemax(Int), nbuffers = 1) -> Int

Default number of k-points one `_fourier_batched!` block handles, keyed on the backend because the
two sides want different things: the CPU wants a width that keeps the sequential per-k query API
responsive, the GPU one that amortizes kernel launches without an unbounded scratch buffer.

The `CPUBackend` value is a fixed 32, unchanged and untuned — the user's stated balance for the
per-k path, carried on their authority, with no measurement for it recorded anywhere in this repo.
It ignores `nk_hint` (32 cannot over-allocate) and `nbuffers` (there is no budget to divide).

The `GPUBackend` value spends [`GPU_FOURIER_BATCH_BYTES`](@ref) at `16·(nr + ndata)` bytes per
column — the `phase` buffer plus the `cached_results` the adapter would grow at this width if a
caller ever registered k-points. `nbuffers` splits the budget between simultaneously live copies
(see `get_interpolator_channel`), and `nk_hint` caps the scratch at the caller's grid size.
"""
_default_batch_size(::CPUBackend, nr, ndata; nk_hint = typemax(Int), nbuffers = 1) = 32

function _default_batch_size(::GPUBackend, nr, ndata; nk_hint = typemax(Int), nbuffers = 1)
    clamp(fld(GPU_FOURIER_BATCH_BYTES ÷ nbuffers, 16 * (nr + ndata)), 1, nk_hint)
end
