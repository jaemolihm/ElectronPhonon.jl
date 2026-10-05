# Utilities that calculators call from their `run_calculator!` methods, on any backend. Not used inside ElectronPhonon.jl
# itself and not exported; downstream calculators reach them as `ElectronPhonon.<name>`. The
# backend/device primitives (`to_device`, `batched_gemm!`, …) live in `common/gpu_utils.jl`; this
# file holds the higher-level, calculator-facing helpers built on top of them.

# The extents `(nbandkq, nbandk, nm, npairs)` of a scatter's block values `vals`, after checking
# that the other block arrays agree: `ωq` `(nm, npairs)` (unless `nothing`), `ikqs` of length
# `npairs`, and index maps with exactly the box rows `nbandk` / `nbandkq`. The scatters loop over
# these extents unchecked (the device kernels with no bounds checks at all), so a mismatched array
# fails here instead.
function _scatter_extents(vals, ωq, ikqs, imap_i_col, imap_f)
    nbandkq, nbandk, nm, npairs = size(vals)
    ωq === nothing || size(ωq) == (nm, npairs) ||
        throw(DimensionMismatch("ωq is $(size(ωq)), the block values $(size(vals))"))
    length(ikqs) == npairs ||
        throw(DimensionMismatch("$(length(ikqs)) k+q indices for $npairs block points"))
    length(imap_i_col) == nbandk ||
        throw(DimensionMismatch("the outer index map has $(length(imap_i_col)) rows, the box $nbandk"))
    size(imap_f, 1) == nbandkq ||
        throw(DimensionMismatch("the inner index map has $(size(imap_f, 1)) rows, the box $nbandkq"))
    (nbandkq, nbandk, nm, npairs)
end

"""
    eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq, ni_stride, i0)

Device-resident scatter for a calculator that keeps `g2`/`ωq` on the device (no per-batch
host streaming). For every `(m, n, ν, ipair)` entry of `g2vals` `(nbandkq, nbandk, nm, npairs)`, look
up the state indices `i = imap_i_col[n]` (in-window outer-k state) and `f = imap_f[m, ikqs[j]]`
(in-window k+q state); if both are in-window (`> 0`), write the value (`ω = ωq[ν, j]`) into the
mode-fastest linear slot `lin = ν + nm·(i-i0-1) + nm·ni_stride·(f-1)` of the flat `g2_out`/`ωq_out`.
The extents are those of `g2vals`; `ωq` `(nm, npairs)`, `ikqs` (length `npairs`) and the index
maps (`nbandk` / `nbandkq` rows) must agree with them, or a `DimensionMismatch` is thrown.

The output buffer indexes outer-k states along `i`, and there are two ways to size it:
- **Full buffer** — holds all `n_i` outer states at once: pass `ni_stride = n_i`, `i0 = 0`.
- **Per-batch buffer** — the outer-k e-ph loop walks the outer-k points in batches; a memory-bounded
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
                             ni_stride::Int, i0::Int)
    nbandkq, nbandk, nm, npairs = _scatter_extents(g2vals, ωq, ikqs, imap_i_col, imap_f)
    @inbounds for ipair in 1:npairs, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[ipair]]
        if i > 0 && f > 0
            # global outer state `i` lands at local row `i - i0` (see docstring for i0/ni_stride).
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, ipair]
            ωq_out[lin] = ωq[ν, ipair]
        end
    end
    nothing
end

"""
    eph_window_scatter_reim!(re_out, im_out, ωq_out, epvals, imap_i_col, imap_f, ikqs, ωq,
                             ni_stride, i0)

Complex sibling of [`eph_window_scatter!`](@ref): same window lookup and same linear slot, but it
writes `real(ep)` and `imag(ep)` of the raw matrix element `epvals` `(nbandkq, nbandk, nm,
npairs)` into two real arrays instead of `|g|²/(2ω)` into one. A calculator needs this when the
*phase* of `g` enters, not only its magnitude -- a four-channel Nambu vertex, where the `1/(2ω)` is
applied downstream, once, as the channels are combined.

`ωq_out === nothing` skips the frequency write and leaves `ωq` unread, which is what a caller that
sweeps twice over the same q points wants for the second sweep.

Generic (CPU/fallback) method; the CUDA extension provides a one-kernel `CuArray` method. See
[`eph_window_scatter!`](@ref) for `ni_stride`/`i0` and for why the target slots never collide.
"""
function eph_window_scatter_reim!(re_out, im_out, ωq_out, epvals, imap_i_col, imap_f, ikqs, ωq,
                                  ni_stride::Int, i0::Int)
    nbandkq, nbandk, nm, npairs = _scatter_extents(epvals, ωq, ikqs, imap_i_col, imap_f)
    for ipair in 1:npairs, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[ipair]]
        if i > 0 && f > 0
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            ep = epvals[m, n, ν, ipair]
            re_out[lin] = real(ep)
            im_out[lin] = imag(ep)
            ωq_out === nothing || (ωq_out[lin] = ωq[ν, ipair])
        end
    end
    nothing
end

# `eph_window_scatter!` above is used by device-resident calculators that copy g2/ωq (e.g. the
# MigdalEliashberg EliashbergCalculator), not by the BTE calculator. The BTE analogue
# `bte_window_accumulate!` lives next to its sole caller `BoltzmannCalculator`
# (src/boltzmann/boltzmann_calculator.jl); its CUDA method is in the extension.
