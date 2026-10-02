# Utilities that calculators call from their GPU batched hooks. Not used inside ElectronPhonon.jl
# itself and not exported; downstream calculators reach them as `ElectronPhonon.<name>`. The
# backend/device primitives (`to_device`, `batched_gemm!`, …) live in `common/gpu_utils.jl`; this
# file holds the higher-level, calculator-facing helpers built on top of them.

# The block arrays a scatter reads must have exactly the extents it is given, so a full-width buffer
# passed with a smaller count fails here instead of being read past the block (the device kernels
# index unchecked). The index maps need only cover the box bands. `ωq === nothing` skips its check.
function _check_scatter_extents(vals, ωq, ikqs, imap_i_col, imap_f, nbandkq, nbandk, nm, nq_batch)
    size(vals) == (nbandkq, nbandk, nm, nq_batch) || throw(DimensionMismatch(
        "block values are $(size(vals)), the scatter was given ($nbandkq, $nbandk, $nm, $nq_batch)"))
    ωq === nothing || size(ωq) == (nm, nq_batch) ||
        throw(DimensionMismatch("ωq is $(size(ωq)), the scatter was given ($nm, $nq_batch)"))
    length(ikqs) == nq_batch ||
        throw(DimensionMismatch("$(length(ikqs)) k+q indices for $nq_batch block points"))
    length(imap_i_col) >= nbandk && size(imap_f, 1) >= nbandkq ||
        throw(DimensionMismatch("the index maps cover fewer bands than the block box"))
    nothing
end

"""
    eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                        nbandkq, nbandk, nm, nq_batch, ni_stride, i0)

Device-resident scatter for a calculator that keeps `g2`/`ωq` on the device (no per-batch
host streaming). For every `(m, n, ν, iq_batch)` entry of `g2vals` `(nbandkq, nbandk, nm, nq_batch)`, look
up the state indices `i = imap_i_col[n]` (in-window outer-k state) and `f = imap_f[m, ikqs[j]]`
(in-window k+q state); if both are in-window (`> 0`), write the value (`ω = ωq[ν, j]`) into the
mode-fastest linear slot `lin = ν + nm·(i-i0-1) + nm·ni_stride·(f-1)` of the flat `g2_out`/`ωq_out`.

The output buffer indexes outer-k states along `i`, and there are two ways to size it:
- **Full buffer** — holds all `n_i` outer states at once: pass `ni_stride = n_i`, `i0 = 0`.
- **Per-batch buffer** — the GPU e-ph loop walks the outer-k points in batches; a memory-bounded
  caller keeps only the CURRENT batch's outer states on the device (device use ∝ batch, not the
  whole grid) and flushes each batch to the host. Then the buffer's i-extent is the batch size, not
  `n_i`: pass `ni_stride =` that extent and `i0 =` the batch's global-i offset, so global state `i`
  writes to local row `i - i0`.

The target `lin` indices are unique across the run (distinct k → distinct i, distinct k+q →
distinct f), so the writes never collide (no atomics needed). Generic (CPU/fallback) method; the
CUDA extension provides a one-kernel `CuArray` method.

A helper for downstream device-resident calculators: from their `run_calculator!(calc,
::EPBlock{OuterKLoop}, ctx)` method they call this to scatter each block's `g2`/`ωq` into their own
window-mapped device accumulators, with `imap_i_col` / `imap_f` in the containers' box coordinates
(see `_indmap_to_device`). The library itself stays agnostic to any particular calculator.

TODO: the non-collision invariant (unique `lin` indices across the run) has no in-repo test —
correctness currently rides on the downstream calculator's tests. Add a small scatter round-trip
test that checks the CPU and CUDA methods agree and that no two writes collide.
"""
function eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq::Int, nbandk::Int, nm::Int, nq_batch::Int, ni_stride::Int,
                             i0::Int)
    _check_scatter_extents(g2vals, ωq, ikqs, imap_i_col, imap_f, nbandkq, nbandk, nm, nq_batch)
    @inbounds for iq_batch in 1:nq_batch, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[iq_batch]]
        if i > 0 && f > 0
            # global outer state `i` lands at local row `i - i0` (see docstring for i0/ni_stride).
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, iq_batch]
            ωq_out[lin] = ωq[ν, iq_batch]
        end
    end
    nothing
end

"""
    eph_window_scatter_reim!(re_out, im_out, ωq_out, epvals, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq, nbandk, nm, nq_batch, ni_stride, i0)

Complex sibling of [`eph_window_scatter!`](@ref): same window lookup and same linear slot, but it
writes `real(ep)` and `imag(ep)` of the raw matrix element `epvals` `(nbandkq, nbandk, nm,
nq_batch)` into two real arrays instead of `|g|²/(2ω)` into one. A calculator needs this when the
*phase* of `g` enters, not only its magnitude -- a four-channel Nambu vertex, where the `1/(2ω)` is
applied downstream, once, as the channels are combined.

`ωq_out === nothing` skips the frequency write and leaves `ωq` unread, which is what a caller that
sweeps twice over the same q points wants for the second sweep.

Generic (CPU/fallback) method; the CUDA extension provides a one-kernel `CuArray` method. See
[`eph_window_scatter!`](@ref) for `ni_stride`/`i0` and for why the target slots never collide.
"""
function eph_window_scatter_reim!(re_out, im_out, ωq_out, epvals, imap_i_col, imap_f, ikqs, ωq,
                                  nbandkq::Int, nbandk::Int, nm::Int, nq_batch::Int,
                                  ni_stride::Int, i0::Int)
    _check_scatter_extents(epvals, ωq, ikqs, imap_i_col, imap_f, nbandkq, nbandk, nm, nq_batch)
    for iq_batch in 1:nq_batch, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[iq_batch]]
        if i > 0 && f > 0
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            ep = epvals[m, n, ν, iq_batch]
            re_out[lin] = real(ep)
            im_out[lin] = imag(ep)
            ωq_out === nothing || (ωq_out[lin] = ωq[ν, iq_batch])
        end
    end
    nothing
end

# `eph_window_scatter!` above is used by device-resident calculators that copy g2/ωq (e.g. the
# MigdalEliashberg EliashbergCalculator), not by the BTE calculator. The BTE analogue
# `bte_window_accumulate!` lives next to its sole caller `BoltzmannCalculator`
# (src/boltzmann/boltzmann_calculator.jl); its CUDA method is in the extension.
