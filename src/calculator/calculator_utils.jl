# Utilities that calculators call from their GPU batched hooks. Not used inside ElectronPhonon.jl
# itself and not exported; downstream calculators reach them as `ElectronPhonon.<name>`. The
# backend/device primitives (`to_device`, `batched_gemm!`, …) live in `common/gpu_utils.jl`; this
# file holds the higher-level, calculator-facing helpers built on top of them.

"""
    eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                        nbandkq, nbandk, nm, nqc, ni_stride; i0 = 0)

Device-resident scatter for a calculator that keeps `g2`/`ωq` on the device (no per-chunk
host streaming). For every `(m, n, ν, j)` entry of `g2vals` `(nbandkq, nbandk, nm, nqc)`, look
up the state indices `i = imap_i_col[n]` (in-window outer-k state) and `f = imap_f[m, ikqs[j]]`
(in-window k+q state); if both are in-window (`> 0`), write the value (`ω = ωq[ν, j]`) into the
mode-fastest linear slot `lin = ν + nm·(i−i0−1) + nm·ni_stride·(f−1)` of the flat `g2_out`/`ωq_out`.

The output buffer indexes outer-k states along `i`, and there are two ways to size it:
- **Full buffer** — holds all `n_i` outer states at once: pass `ni_stride = n_i`, `i0 = 0`.
- **Block buffer** — the GPU e-ph loop walks the outer-k points in tiles; a memory-bounded caller
  keeps only the CURRENT tile's outer states on the device (device use ∝ tile, not the whole grid)
  and flushes each tile to the host. Then the buffer's i-extent is the tile size, not `n_i`: pass
  `ni_stride =` that extent and `i0 =` the tile's global-i offset, so global state `i` writes to
  local row `i − i0`.

The target `lin` indices are unique across the run (distinct k → distinct i, distinct k+q →
distinct f), so the writes never collide (no atomics needed). Generic (CPU/fallback) method; the
CUDA extension provides a one-kernel `CuArray` method.

A helper for downstream device-resident calculators: from their `run_calculator_batched!` hook
they call this to scatter each e-ph chunk's `g2`/`ωq` into their own window-mapped device
accumulators. The library itself stays agnostic to any particular calculator.

TODO: the non-collision invariant (unique `lin` indices across the run) has no in-repo test —
correctness currently rides on the downstream calculator's tests. Add a small scatter round-trip
test that checks the CPU and CUDA methods agree and that no two writes collide.
"""
function eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, ni_stride::Int;
                             i0::Int = 0)
    @inbounds for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[j]]
        if i > 0 && f > 0
            # global outer state `i` lands at local row `i − i0` (see docstring for i0/ni_stride).
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, j]
            ωq_out[lin] = ωq[ν, j]
        end
    end
    nothing
end
