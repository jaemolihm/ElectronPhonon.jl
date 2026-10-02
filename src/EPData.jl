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
