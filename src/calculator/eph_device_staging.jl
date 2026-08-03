# Device-memory accounting for the batched GPU e-ph loops. Folds the two loops' hand-counted byte
# formulas (formerly inline in `run_eph_over_k_and_kq.jl` and `run_eph_over_q_and_k.jl`) into one
# place: `_outer_k_staging_bytes` / `_outer_q_staging_bytes` return the loop's `(per_point, committed)`
# device-byte counts, and `plan_batch` turns those into a memory-adaptive batch width. The same byte
# functions feed `estimate_device_memory`, so the loop and the standalone estimate can never diverge.
#
# The historical byte formulas are treated as ground truth and reproduced verbatim (see the plan
# Caveat "byte-accounting formulas … treated as ground truth … not re-derived"): the term sums below
# equal the old expressions exactly, pinned by a transition test. These are plain functions — no
# registration object; the drivers keep their concrete-typed inline `alloc`/`similar` calls so the
# device-batched loops stay type-stable (the `_setup`/`_loop` rule), and only the byte arithmetic and
# the batch-sizing check are centralized here.

"""
    plan_batch(backend, per_point, committed, cap; headroom_num = 7, headroom_den = 10, what = "",
               warn = true) -> nbatch

Size a batched GPU e-ph loop's batch to free device memory:
`nbatch = min(cap, (free - committed) · headroom ÷ per_point)`, clamped to at least 1, where
`per_point` / `committed` are the per-(batched-inner index) and whole-run device-byte counts. On a
CPU backend `free_bytes` is `typemax(Int)`, so `nbatch = cap`. Errors if the whole-run commitments
alone exceed free device memory (a clear early failure instead of an OOM mid-loop); `what` names the
loop in that message. `headroom_num / headroom_den` is the usable fraction of free memory (default
`7/10`, i.e. a 30% headroom for the batched drivers' recycled temporaries), applied as
`x ÷ den * num` to match the integer arithmetic of the formulas this replaces. Pass `warn = false`
for a counterfactual query ("how wide would the batch be at a different `per_point`?"), which must
not tell the user their batch was reduced.
"""
function plan_batch(backend::AbstractBackend, per_point::Integer, committed::Integer, cap::Integer;
        headroom_num::Integer = 7, headroom_den::Integer = 10, what::AbstractString = "",
        warn::Bool = true)
    free = free_bytes(backend)
    if free != typemax(Int) && committed > free
        error("GPU $(what): committed device memory ($(round(committed / 1e9, digits = 2)) GB, " *
              "whole-run stacks) exceeds free device memory ($(round(free / 1e9, digits = 2)) GB). " *
              "Reduce the batch cap or the grid size.")
    end
    nb_mem = free == typemax(Int) ? Int(cap) :
        max(1, ((free - committed) ÷ headroom_den * headroom_num) ÷ per_point)
    nbatch = min(Int(cap), nb_mem)
    if warn && free != typemax(Int) && nb_mem < cap
        @warn "WARNING : GPU batch width reduced from the requested cap $(Int(cap)) to $nbatch to fit " *
            "free device memory in plan_batch$(isempty(what) ? "" : " ($what)"). GPU performance may " *
            "degrade compared to smaller calculations."
    end
    nbatch
end


# --- outer-k GPU loop device bytes (`run_eph_over_k_and_kq`) ---------------------------------------
#
# Returns `(per_point, committed)` in bytes. `ndata = nw·nbandk_max·nmodes` is the k-side projected
# e-ph data size (`= nw²·nmodes` full-band); `nr_ep` = number of R-vectors of g(k, R_ep); `nkq` /
# `nq_grid` = k+q / q grid sizes; `nk_batch_max` = the fixed outer-k batch width; `ndata_epmat` /
# `nr_epmat` = the parent e-ph object's (`epmat_dev`) data size / R-vector count, for the RR→kR
# interpolator's scratch. `nk_b` is the kR→kq GEMM strip width (`_krkq_strip_width`), which widens
# `kRkq_ws.g` and nothing else. The per-q term sums to `(40 + 16·nk_b)·nw·nbandk_max·nmodes +
# 16·nr_ep + 16·nmodes² + 8·nmodes + 8 + Σcalc`; the committed to the old hand-counted
# `16·nw²·nkq + (16·nmodes²+8·nmodes)·nq_grid +
# 16·nw·nbandk_max·(nmodes·nr_ep+1)·nk_batch_max` PLUS the `itp_epmat` Fourier scratch
# `(16·ndata_epmat + 16·nr_epmat + 24)·nk_batch_max` (added 2026-07-18; the parent RR→kR interpolator,
# built at `batch_size = nk_batch_max`, was omitted from the original hand-count — validated against a
# direct pool-stat measurement of `BatchedWannierInterpolator(epmat_dev)`) PLUS the k+q-convention
# terms `24·(nk + nkq) + 16·nr_ep·nk_batch_max`. All transition-pinned by test/test_gpu.jl.
# Not counted: the loop's `irvecp_mat` (`24·nr_ep`, 20 kB at Cu shapes), matching how the
# `BatchedFourierCore.irvec_mat` of the same shape has never been counted.
function _outer_k_staging_bytes(; nw, nbandk_max, nmodes, nr_ep, nk, nkq, nq_grid, nk_batch_max,
        calculators, ndata_epmat, nr_epmat, nk_b = 1, FT = Float64)
    cx = sizeof(Complex{FT})    # 16
    rl = sizeof(FT)             # 8
    iz = sizeof(Int)            # 8
    ndata = nw * nbandk_max * nmodes
    # Per-q device buffers (batched-inner axis = q): epkq/g2/uphs/ωq/iqs are the loop's own staging;
    # kRkq_ws.{g,tmp} and the kR→kq phase tile P_kq scale with the q-tile too.
    per_point =
        cx * ndata +                       # epkq_dev
        rl * ndata +                       # g2_dev
        cx * ndata * nk_b +                # kRkq_ws.g   (ndata_ekpR × the GEMM strip width)
        cx * nw * (nbandk_max * nmodes) +  # kRkq_ws.tmp (nbandkq=nw, nbandk·nmodes)
        cx * nr_ep +                       # P_kq (kR→kq phase tile)
        cx * nmodes * nmodes +             # uphs_dev
        rl * nmodes +                      # ωq_dev
        iz                                 # iqs_batch_dev
    for c in calculators
        per_point += eph_batched_bytes_per_point(c, EPDataQBatched; nw, nmodes)
    end
    # Whole-run + per-k-batch commitments (allocated after the sizing point; subtracted from free).
    # The `itp_epmat` term uses `nk_batch_max` (the outer-k batch width, a SEPARATE fixed cap from the
    # sized q-tile), so it belongs here in `committed`, not in the per-q term: cached_results
    # (16·ndata_epmat) + phase (16·nr_epmat) + xkmat (24), all × nk_batch_max.
    committed =
        cx * nw * nw * nkq +                                  # ukqs_all_dev
        cx * nmodes * nmodes * nq_grid +                      # uph_all_dev
        rl * nmodes * nq_grid +                               # ωq_all_dev
        cx * ndata * nr_ep * nk_batch_max +                   # ep_ekpR_all
        cx * nw * nbandk_max * nk_batch_max +                 # uks_dev
        (cx * ndata_epmat + cx * nr_epmat + rl * 3) * nk_batch_max + # itp_epmat Fourier scratch
        rl * 3 * (nk + nkq) +                                 # xk_dev + xkq_dev
        cx * nr_ep * nk_batch_max                             # P_k (k+q-convention phase)
    (per_point, committed)
end


# --- outer-q GPU loop device bytes (`run_eph_over_q_and_k`) ----------------------------------------
#
# Returns `(per_point, committed)` in bytes. The k side is streamed (no whole-grid device stack), so
# `committed == 0`; every device buffer scales with the k-batch. The per-k term reproduces the old
# formula verbatim (ground truth, not re-derived from the individual buffers), grouped by shape.
function _outer_q_staging_bytes(; nw, nmodes, nr_el_ham, nr_ep_eRpq, use_polar_eph, calculators,
        FT = Float64)
    cx = sizeof(Complex{FT})    # 16
    rl = sizeof(FT)             # 8
    per_point =
        cx * nw^2 * nmodes * 5 +              # ep_batch + RqToKQ ws.g/.tmp/.uk_rep + itp_ep_eRpq.cached_results
        cx * nw^2 * 8 +                       # Hkq_flat + Uk_batch + Ukq_batch + itp_el_ham.cached_results
                                             #   + eigen_batched (E,U) & rotation transients (historical margin)
        (use_polar_eph ? cx * nw^2 : 0) +    # mmats_batch (polar only)
        cx * (nr_el_ham + nr_ep_eRpq)        # interpolator core phase, both interpolators
    for c in calculators
        per_point += eph_batched_bytes_per_point(c, EPDataKBatched; nw, nmodes)
    end
    (per_point, 0)
end
