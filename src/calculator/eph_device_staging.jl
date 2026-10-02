# Device-memory accounting for the batched e-ph loops: `_outer_k_staging_bytes` /
# `_outer_q_staging_bytes` return the loop's `(per_point, committed)` device-byte counts, one term
# per buffer the loop allocates, and `plan_batch` turns those into a memory-adaptive batch width.
# The same byte functions feed `estimate_device_memory`; the resident state containers are counted
# by the estimate only (the `*_stack` arguments), since the loops build them before they size their
# batches.

"""
    plan_batch(backend, per_point, committed, cap; headroom_num = 7, headroom_den = 10, what = "",
               warn = true) -> nbatch

Size a batched e-ph loop's batch to free device memory:
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
        error("batched $(what): committed device memory ($(round(committed / 1e9, digits = 2)) GB, " *
              "whole-run stacks) exceeds free device memory ($(round(free / 1e9, digits = 2)) GB). " *
              "Reduce the batch cap or the grid size.")
    end
    nb_mem = free == typemax(Int) ? Int(cap) :
        max(1, ((free - committed) ÷ headroom_den * headroom_num) ÷ per_point)
    nbatch = min(Int(cap), nb_mem)
    if warn && free != typemax(Int) && nb_mem < cap
        @warn "WARNING : batch width reduced from the requested cap $(Int(cap)) to $nbatch to fit " *
            "free device memory in plan_batch$(isempty(what) ? "" : " ($what)"). Performance may " *
            "degrade compared to smaller calculations."
    end
    nbatch
end


# --- outer-k loop device bytes (`run_eph_over_k_and_kq`) -------------------------------------------
#
# Returns `(per_point, committed)` in bytes: per k+q point of a q-tile, and for the whole run plus one
# outer-k batch. `nbandk_max` / `nbandkq_max` are the two containers' box widths; `nr_ep` = number of
# R-vectors of g(k, R_ep); `nr_epmat` = that of the parent `epmat_dev`, for the RR→kR interpolator's
# phase scratch; `nk_stack` / `nkq_stack` / `nq_grid` = the resident k, k+q and phonon containers to
# count (the loop passes 0: they are built before it sizes its tiles; the estimate passes the grid
# sizes); `nk_batch_max` = the outer-k batch width. Each calculator adds its
# `eph_batched_bytes_per_point` triple (given the run's containers `els_k`, `els_kq`, `phs`, `nothing` in
# the estimate): `per_pair` per k+q point, `persistent` and `per_outer · nk_batch_max` to the
# committed bytes. Not counted: the loop's `irvecp_mat`
# (`24·nr_ep`, 20 kB at Cu shapes) and the interpolator's `cached_results`, never allocated because
# `itp_epmat` is driven only through `get_fourier_batched!`.
function _outer_k_staging_bytes(; nw, nbandk_max, nbandkq_max, nmodes, nr_ep, nk, nkq, nk_stack,
        nkq_stack, nq_grid, nk_batch_max, calculators, nr_epmat, FT = Float64, els_k = nothing,
        els_kq = nothing, phs = nothing)
    cx = sizeof(Complex{FT})    # 16
    rl = sizeof(FT)             # 8
    iz = sizeof(Int)            # 8
    ndata = nw * nbandk_max * nmodes
    per_point =
        cx * nbandkq_max * nbandk_max * nmodes +   # epkq_dev
        cx * ndata +                               # kRkq_ws.g
        cx * nbandkq_max * nbandk_max * nmodes +   # kRkq_ws.tmp
        cx * nr_ep +                               # P_kq (kR→kq phase tile)
        cx * nmodes * nmodes + rl * nmodes +       # phonon tile (u, e)
        iz                                         # iqs_batch_dev
    committed =
        (cx * nw * nbandk_max + rl * nbandk_max + 2iz) * nk_stack +     # k container (u, e, offsets, nband)
        (cx * nw * nbandkq_max + rl * nbandkq_max + 2iz) * nkq_stack +  # k+q container
        (cx * nmodes * nmodes + rl * nmodes) * nq_grid +                # phonon container
        cx * ndata * nr_ep * nk_batch_max +                             # ep_ekpR_all
        (cx * nw * nbandk_max + rl * nbandk_max) * nk_batch_max +       # k-side tile
        (cx * nr_epmat + rl * 3) * nk_batch_max +                       # itp_epmat phase + k staging
        rl * 3 * (nk + nkq) + rl * nkq +                                # mxk_dev, xkq_dev, wtkq_dev
        cx * nr_ep * nk_batch_max                                       # P_mk (k+q-convention phase)
    for c in calculators
        b = eph_batched_bytes_per_point(c, EPBlock{OuterKLoop}; nw, nmodes, nband_max_k = nbandk_max,
                                        nband_max_kq = nbandkq_max, els_k, els_kq, phs)
        per_point += b.per_pair
        committed += b.persistent + b.per_outer * nk_batch_max
    end
    (per_point, committed)
end


# --- outer-q loop device bytes (`run_eph_over_q_and_k`) --------------------------------------------
#
# Returns `(per_point, committed)` in bytes: per k point of a k-batch, and for the whole run. The k
# container (box width `nbandk_max`) is resident: `nk_stack` points of it are counted (0 from the
# loop, which builds it before sizing the batch; `nk` from the estimate), plus its weights. The k+q
# states are solved per batch at the full width `nw`. The `cached_results` of the two interpolators
# are not counted: both are driven only through `get_fourier_batched!`, which never registers a
# k-point. Each calculator adds its `eph_batched_bytes_per_point` triple (one outer q per batch).
function _outer_q_staging_bytes(; nw, nbandk_max, nmodes, nr_el_ham, nr_ep_eRpq, use_polar_eph,
        calculators, nk, nk_stack, FT = Float64, els_k = nothing, phs = nothing)
    cx = sizeof(Complex{FT})    # 16
    rl = sizeof(FT)             # 8
    iz = sizeof(Int)            # 8
    per_point =
        cx * nw * nbandk_max * nmodes +                  # ep_batch
        cx * nw^2 * nmodes +                             # RqToKQ ws.g
        cx * nw^2 * nmodes +                             # RqToKQ ws.tmp
        cx * nw * nbandk_max * nmodes +                  # RqToKQ ws.uk_rep
        (cx * nw * nbandk_max + rl * nbandk_max + 2iz) + # k tile
        (cx * nw * nw + rl * nw + 2iz) +                 # k+q tile
        cx * nw^2 * 5 +                                  # Hkq_flat + eigen_batched (E, U) and
                                                         #   rotation transients (historical margin)
        (use_polar_eph ? cx * nw * nbandk_max : 0) +     # mmats_batch (polar only)
        cx * (nr_el_ham + nr_ep_eRpq)                    # interpolator core phase, both interpolators
    committed =
        (cx * nw * nbandk_max + rl * nbandk_max + 2iz) * nk_stack +   # k container
        rl * nk                                                       # weights
    for c in calculators
        b = eph_batched_bytes_per_point(c, EPBlock{OuterQLoop}; nw, nmodes, nband_max_k = nbandk_max,
                                        nband_max_kq = nw, els_k, els_kq = nothing, phs)
        per_point += b.per_pair
        committed += b.persistent + b.per_outer
    end
    (per_point, committed)
end
