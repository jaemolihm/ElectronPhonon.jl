# =============================================================================
#  e-ph payload family — generalizations of `EPState` (src/EPState.jl).
#
#  A payload carries all per-call data in typed fields (self-describing), and the loop-level state
#  lives in `LoopContext` (src/calculator/AbstractCalculator.jl). `run_calculator!` dispatches on the
#  payload type, so the calculator interface grows by adding payload/scope *types*, never new hook
#  *names*. The payload TYPES live here (next to `EPState`, which the host payload wraps); the
#  interface FUNCTIONS (`run_calculator!`, `LoopContext`, `supports`, …) live in
#  src/calculator/AbstractCalculator.jl. Included right after `EPState.jl` so `EPData` can name the
#  `EPState` field, and before the calculator interface, which dispatches on these types.

abstract type AbstractElPhPayload end

"""
    EPData{FT, DGT} <: AbstractElPhPayload

Host per-(k, q) point payload — a light immutable wrapper of the reused `EPState` buffer plus the
per-point indices. Immutable and small, so constructing one per (k, q) is free (stack-allocated).

Fields:
- `epstate`   :: `EPState{FT}` — the thread's reused e-ph data buffer (states + matrix elements).
- `ik`       :: outer k-point index.
- `iq`       :: q-point index, or `nothing` when phonons are not precomputed.
- `ikq`      :: k+q-point index, or `nothing` when k+q states are computed on the fly.
- `xk`, `xq` :: the k / q vectors.
- `id_chunk` :: CPU thread-chunk id (selects the calculator's per-thread buffer).
- `epstate_dg`:: covariant-derivative dg (`OffsetArray`), or `nothing`.
"""
struct EPData{FT, DGT} <: AbstractElPhPayload
    epstate    :: EPState{FT}
    ik        :: Int
    iq        :: Union{Int, Nothing}
    ikq       :: Union{Int, Nothing}
    xk        :: Vec3{FT}
    xq        :: Vec3{FT}
    id_chunk  :: Int
    epstate_dg :: DGT
end

"""
    EPDataQBatched{AT4C, AT4R, AT2, VI} <: AbstractElPhPayload

Device payload of the GPU outer-k loop (`run_eph_over_k_and_kq`, `use_gpu`): one outer k-point with a
batch of its k+q points. Runs on the backend of `eps` (CPU or GPU); consumers stay backend-generic
(`similar`, `copyto!`, broadcasting, scatter) so no CUDA dependency leaks in.

The batched fields carry a batch of quantities (the trailing axis is the k+q batch), so they take the
plural names `eps`/`g2s`/`ωqs` (cf. `ikqs`). This payload differs from [`EPDataKBatched`](@ref)
because the two GPU loops feed different consumers: this outer-k loop pre-fuses `g2s` in its kernel
(what the BTE scatter needs), whereas the outer-q loop hands eigenvectors and lets the consumer form
its own contraction (so `EPDataKBatched` carries `uk`/`ukq` and no `g2s`).

Fields:
- `eps`           :: `(nbandkq, nbandk, nmodes, nq)` complex — eigenbasis e-ph matrices (raw elements).
- `g2s`           :: `(nbandkq, nbandk, nmodes, nq)` real — `|ep|²/(2ω)`, always formed by the loop's
  fused kernel (separate type param `AT4R` from `eps`'s `AT4C`: complex vs real element type).
- `ωqs`           :: `(nmodes, nq)` — phonon frequencies of this batch.
- `ik`            :: outer k index.
- `ikqs`          :: `(nq,)` k+q indices of this batch (device).
- `ibandk_offset` :: k-side window-projection band offset (0-based; 0 for full-band). `eps`'s
  band-of-k axis `n` (1-based) is PHYSICAL band `ibandk_offset + n`.
"""
struct EPDataQBatched{AT4C, AT4R, AT2, VI} <: AbstractElPhPayload
    eps           :: AT4C
    g2s           :: AT4R
    ωqs           :: AT2
    ik            :: Int
    ikqs          :: VI
    ibandk_offset :: Int
end

"""
    EPDataKBatched{AT4, AT3, AT2, AT1, VK} <: AbstractElPhPayload

Device payload of the GPU outer-q loop (`run_eph_over_q_and_k`, `use_gpu`): one phonon momentum q with
a batch of outer k-points. Runs on the backend of `eps`; consumers stay backend-generic.

Unlike [`EPDataQBatched`](@ref) it carries no fused `g2s`: the outer-q loop hands the raw e-ph
matrices `eps` plus the eigenvectors `uk`/`ukq`, and the consumer forms its own contraction (that is
why the fields differ between the two batched payloads). The per-batch quantity that is genuinely a
batch of matrices takes the plural name `eps`; `ek`/`ekq`/`uk`/`ukq`/`wtk` are named as the arrays
they are (the batch is just their trailing axis).

All fields are trimmed to the batch's actual width `nk` (the outer-k convention): the final partial
batch has `nk < n_batch_max`, and the loop hands width-`nk` views into its internally padded staging
buffers (a trailing-prefix device view, so it stays contiguous). A consumer therefore reads its own
size from any field (e.g. `size(eps, 4)`) and never sees a padded tail. (The loop still zeros the
internal `wtk` padding as defense-in-depth, but that padding is not exposed here.)

Fields (`m` = k+q band, `n` = k band, `k` = batch column):
- `eps` :: `(nw, nw, nmodes, nk)` — eigenbasis e-ph matrices. Out-of-window bands are already zeroed
  (the loop window-masks the eigenvector columns on both sides).
- `ek`  :: `(nw, nk)` — k-side band energies (all `nw` bands).
- `ekq` :: `(nw, nk)` — k+q-side band energies.
- `uk`  :: `(nw, nw, nk)` — k-side eigenvectors, zero-padded outside the window.
- `ukq` :: `(nw, nw, nk)` — k+q-side eigenvectors, zero-padded outside the window.
- `wtk` :: `(nk,)` — k-point integration weights.
- `xks` :: `(nk,)` — host `Vec3` list of the batch's k-vectors (for k-dependent phases).
- `iq`  :: q-point index.
"""
struct EPDataKBatched{AT4, AT3, AT2, AT1, VK} <: AbstractElPhPayload
    eps :: AT4
    ek  :: AT2
    ekq :: AT2
    uk  :: AT3
    ukq :: AT3
    wtk :: AT1
    xks :: VK
    iq  :: Int
end
