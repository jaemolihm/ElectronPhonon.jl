# Utilities that calculators call from their GPU batched hooks. Not used inside ElectronPhonon.jl
# itself and not exported; downstream calculators reach them as `ElectronPhonon.<name>`. The
# backend/device primitives (`to_device`, `batched_gemm!`, …) live in `common/gpu_utils.jl`; this
# file holds the higher-level, calculator-facing helpers built on top of them.

"""
    eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                        nbandkq, nbandk, nm, nq_batch, ni_stride, i0)

Device-resident scatter for a calculator that keeps `g2`/`ωq` on the device (no per-batch
host streaming). For every `(m, n, ν, iq_batch)` entry of `g2vals` `(nbandkq, nbandk, nm, nq_batch)`, look
up the state indices `i = imap_i_col[n]` (in-window outer-k state) and `f = imap_f[m, ikqs[j]]`
(in-window k+q state); if both are in-window (`> 0`), write the value (`ω = ωq[ν, j]`) into the
mode-fastest linear slot `lin = ν + nm·(i−i0−1) + nm·ni_stride·(f−1)` of the flat `g2_out`/`ωq_out`.

The output buffer indexes outer-k states along `i`, and there are two ways to size it:
- **Full buffer** — holds all `n_i` outer states at once: pass `ni_stride = n_i`, `i0 = 0`.
- **Per-batch buffer** — the GPU e-ph loop walks the outer-k points in batches; a memory-bounded
  caller keeps only the CURRENT batch's outer states on the device (device use ∝ batch, not the
  whole grid) and flushes each batch to the host. Then the buffer's i-extent is the batch size, not
  `n_i`: pass `ni_stride =` that extent and `i0 =` the batch's global-i offset, so global state `i`
  writes to local row `i − i0`.

The target `lin` indices are unique across the run (distinct k → distinct i, distinct k+q →
distinct f), so the writes never collide (no atomics needed). Generic (CPU/fallback) method; the
CUDA extension provides a one-kernel `CuArray` method.

A helper for downstream device-resident calculators: from their `run_calculator!(calc,
::EPDataQBatched, ctx)` method they call this to scatter each e-ph batch's `g2`/`ωq` into their own window-mapped device
accumulators. The library itself stays agnostic to any particular calculator.

TODO: the non-collision invariant (unique `lin` indices across the run) has no in-repo test —
correctness currently rides on the downstream calculator's tests. Add a small scatter round-trip
test that checks the CPU and CUDA methods agree and that no two writes collide.
"""
function eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq::Int, nbandk::Int, nm::Int, nq_batch::Int, ni_stride::Int,
                             i0::Int)
    @inbounds for iq_batch in 1:nq_batch, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[iq_batch]]
        if i > 0 && f > 0
            # global outer state `i` lands at local row `i − i0` (see docstring for i0/ni_stride).
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, iq_batch]
            ωq_out[lin] = ωq[ν, iq_batch]
        end
    end
    nothing
end

# `eph_window_scatter!` above is used by device-resident calculators that copy g2/ωq (e.g. the
# MigdalEliashberg EliashbergCalculator), not by the BTE calculator. The BTE analogue
# `bte_window_accumulate!` lives next to its sole caller `BoltzmannCalculator`
# (src/boltzmann/boltzmann_calculator.jl); its CUDA method is in the extension.
