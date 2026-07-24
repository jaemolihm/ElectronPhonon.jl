
"Functions for filtering k points and bands outside a certain energy window"

# TODO: Threads

using Base.Threads
using ChunkSplitters
using Folds
using MPI
using ElectronPhonon: Kpoints
using Interpolations: AbstractInterpolation

export filter_kpoints
export filter_electron_states
export filter_electron_states_multigrid
export filter_qpoints

# range of bands inside the window. Assume e is sorted in ascending order.
inside_window(e, window_min, window_max) = searchsortedfirst(e, window_min):searchsortedlast(e, window_max)

"""
    filter_kpoints(kpts_input, nw, el_ham, window[, mpi_comm]; kwargs...)
        -> (kpts, band_min, band_max, nelec_below_window)

DEPRECATED backward-compat alias for `filter_electron_states`. New code should call
`filter_electron_states(...) -> FilteredStates` and read `sel.kpts`, `band_range(sel)`,
`sel.nstates_base`. This wrapper unpacks those into the legacy tuple. `mpi_comm` (optional 5th
positional) is forwarded as the `mpi_comm` keyword.
"""
function filter_kpoints(kpts_input, nw, el_ham, window; kwargs...)
    @warn "filter_kpoints is deprecated; use filter_electron_states(...) -> FilteredStates " *
          "(sel.kpts, band_range(sel), sel.nstates_base)." maxlog=1
    sel = filter_electron_states(kpts_input, nw, el_ham, window; kwargs...)
    br = band_range(sel)
    sel.kpts, first(br), last(br), sel.nstates_base
end
filter_kpoints(kpts_input, nw, el_ham, window, mpi_comm; kwargs...) =
    filter_kpoints(kpts_input, nw, el_ham, window; mpi_comm, kwargs...)

function _filter_kpoints(nw, kpoints, el_ham, window; fourier_mode="normal", use_gpu=false)
    ik_keep = zeros(Bool, kpoints.n)
    nelec_below_window_ = zeros(eltype(window), kpoints.n)
    band_min_ = zeros(Int, kpoints.n)
    band_max_ = zeros(Int, kpoints.n)

    if use_gpu
        # GPU: batched valueonly eigensolve (the band-eigenvalues are the expensive part of
        # filtering); the cheap window test stays on the host. `to_device` /
        # `get_el_eigen_valueonly_batched` come from the base + CUDA extension. Chunk over k so the
        # per-chunk device H(k) stack (nw*nw*kchunk complex) stays bounded — a single all-nk solve
        # can exhaust GPU memory on large grids. kchunk caps that stack at ~1 GiB (nk if smaller).
        kchunk = clamp(fld(2^30, nw * nw * 16), 1, kpoints.n)
        itp_elham = get_interpolator(to_device(el_ham); fourier_mode="batched", batch_size=kchunk)
        kstart = 1
        while kstart <= kpoints.n
            kstop = min(kstart + kchunk - 1, kpoints.n)
            E = Array(get_el_eigen_valueonly_batched(itp_elham, view(kpoints.vectors, kstart:kstop)))  # (nw, kchunk)
            @views for (jl, ik) in enumerate(kstart:kstop)
                bands_in_window = inside_window(E[:, jl], window...)
                nelec_below_window_[ik] = (bands_in_window.start - 1) * kpoints.weights[ik]
                if !isempty(bands_in_window)
                    ik_keep[ik] = true
                    band_min_[ik] = bands_in_window[1]
                    band_max_[ik] = bands_in_window[end]
                end
            end
            kstart = kstop + 1
        end
        nelec_below_window = sum(nelec_below_window_)
        return (; ik_keep, band_min = minimum(band_min_), band_max = maximum(band_max_),
                nelec_below_window, band_min_per_k = band_min_, band_max_per_k = band_max_)
    end

    @threads for iks in chunks(kpoints.vectors; n=2*nthreads())
        ham = get_interpolator(el_ham; fourier_mode)
        register_kpoints!(ham, view(kpoints.vectors, iks))
        eigenvalues = zeros(real(eltype(el_ham)), nw)

        for ik in iks
            xk = kpoints.vectors[ik]
            get_el_eigen_valueonly!(eigenvalues, nw, ham, xk)
            bands_in_window = inside_window(eigenvalues, window...)

            nelec_below_window_[ik] = (bands_in_window.start - 1) * kpoints.weights[ik]
            if ! isempty(bands_in_window)
                ik_keep[ik] = true
                band_min_[ik] = bands_in_window[1]
                band_max_[ik] = bands_in_window[end]
            end
        end
    end

    nelec_below_window = sum(nelec_below_window_)
    band_min = minimum(band_min_)
    band_max = maximum(band_max_)
    # `band_min_per_k`/`band_max_per_k` are per-k in-window band ranges (0 where the point has no
    # in-window band); the multigrid / selection generators need them, so they are surfaced here
    # rather than collapsed to the global min/max alone.
    (; ik_keep, band_min, band_max, nelec_below_window, band_min_per_k = band_min_, band_max_per_k = band_max_)
end


"""
    filter_qpoints(qpoints, kpoints, nw, el_ham, window; fourier_mode="gridopt")

Filter only q points which have k point such that the energy at k+q is inside
the window.
"""
function filter_qpoints(qpoints, kpoints, nw, el_ham, window; fourier_mode="gridopt")
    iq_keep = zeros(Bool, qpoints.n)

    @threads for iqs in chunks(qpoints.vectors; n=2*nthreads())
        ham = get_interpolator(el_ham; fourier_mode)
        eigenvalues = zeros(real(eltype(el_ham)), nw)

        for iq in iqs
            xq = qpoints.vectors[iq]
            register_kpoints!(ham, kpoints.vectors .+ Ref(xq))

            for xk in kpoints.vectors
                xkq = xq + xk
                get_el_eigen_valueonly!(eigenvalues, nw, ham, xkq)

                # If k+q is inside window, use this q point
                if ! isempty(inside_window(eigenvalues, window...))
                    iq_keep[iq] = true
                    break
                end
            end
        end
    end
    get_filtered_kpoints(qpoints, iq_keep)
end

function filter_qpoints(qpoints, kpoints, itp_el::Dict{Int, <: AbstractInterpolation}, window)
    iq_keep = Folds.map(1:qpoints.n) do iq
        xq = qpoints.vectors[iq]
        for xk in kpoints.vectors, iw in keys(itp_el)
            xkq = xq + xk
            ekq = itp_el[iw](xkq...)
            if window[1] <= ekq <= window[2]
                # If k+q is inside window, use this q point
                return true
            end
        end
        return false
    end
    get_filtered_kpoints(qpoints, iq_keep)
end



# Filter `kpts_input` to `window` and return the filtered `GridKpoints` together with the PER-k
# in-window band ranges (`band_min[ik]:band_max[ik]`) and the below-window carrier count. This is
# the state-selection building block: `_filter_kpoints` already computes the per-k band ranges, so
# they are surfaced here (kept for the retained points, in order) rather than collapsed to a global
# min/max. Non-MPI (the selection generators are single-node); a trivial window keeps the whole grid
# with all bands `1:nw`.
function _filter_with_band_ranges(kpts_input, nw, el_ham, window;
                                  symmetry=nothing, fourier_mode="gridopt", use_gpu=false, shift=(0, 0, 0))
    if kpts_input isa NTuple{3,Integer}
        (symmetry === nothing || all(shift .== 0)) ||
            error("nonzero shift and symmetry incompatible (not implemented)")
        kpoints = kpoints_grid(kpts_input; symmetry, shift)
    else
        kpoints = kpts_input
    end
    if window == (-Inf, Inf)
        gkpts = kpoints isa GridKpoints ? kpoints : GridKpoints(kpoints)
        return gkpts, fill(1, gkpts.n), fill(nw, gkpts.n), zero(eltype(window))
    end
    r = _filter_kpoints(nw, kpoints, el_ham, window; fourier_mode, use_gpu)
    gkpts = GridKpoints(get_filtered_kpoints(kpoints, r.ik_keep))
    gkpts, r.band_min_per_k[r.ik_keep], r.band_max_per_k[r.ik_keep], r.nelec_below_window
end

"""
    filter_electron_states(kpts_input, nw, el_ham, window; symmetry, fourier_mode, use_gpu,
                           mpi_comm, shift) -> FilteredStates
    filter_electron_states(kpts_input, model::Model, window; kwargs...) -> FilteredStates

The unified electron-state filtering primitive (Generator 1). Filters a k-grid spec (an
`NTuple{3,Int}`, `Kpoints`, or `GridKpoints`) to the energy `window` — optionally IBZ-reducing with
`symmetry` and offsetting an `NTuple` grid by `shift` — and returns the selected `(k, band)` states
as a `FilteredStates`: `sel.kpts` is the filtered `GridKpoints`, `band_range(sel)` the global band
range, `sel.nstates_base` the below-window carrier count, `state_weights(sel)` the per-state (uniform
per-k) BZ weights. Supersedes the legacy tuple-returning `filter_kpoints`.

`shift` applies only to the `NTuple` grid path (ignored for a prebuilt `Kpoints`/`GridKpoints`) and
is mutually exclusive with `symmetry`. Under `mpi_comm` the k-points are split across ranks: the grid
is built distributed, each rank filters its slice once (single eigensolve pass), then the kept
k-points and per-k band ranges are redistributed together through the same gather/scatter so they
stay aligned, and the local below-window counts are summed.
"""
function filter_electron_states(kpts_input, nw::Integer, el_ham, window;
        symmetry=nothing, fourier_mode="gridopt", use_gpu=false, mpi_comm=nothing, shift=(0, 0, 0))
    if mpi_comm === nothing
        gkpts, ibmin, ibmax, nelec = _filter_with_band_ranges(kpts_input, nw, el_ham, window;
            symmetry, fourier_mode, use_gpu, shift)
    else
        kpts_input isa NTuple{3,Integer} ||
            throw(ArgumentError("filter_electron_states with mpi_comm requires an NTuple grid spec"))
        (symmetry === nothing || all(shift .== 0)) ||
            error("nonzero shift and symmetry incompatible (not implemented)")
        # Build the grid distributed and filter each rank's slice once (single eigensolve pass).
        kpoints = kpoints_grid(kpts_input, mpi_comm; shift, symmetry)
        gkpts_l, ibmin_l, ibmax_l, nelec_l = _filter_with_band_ranges(kpoints, nw, el_ham, window;
            fourier_mode, use_gpu)
        # Redistribute the kept k-points via the shared GridKpoints wrapper (rank-concatenate +
        # even-split, no reorder; preserves the global ngrid). The per-k band ranges follow with the
        # SAME gather/scatter, so they stay aligned to `gkpts`. Sum the local below-window counts.
        gkpts = mpi_gather_and_scatter(gkpts_l, mpi_comm)
        ibmin = mpi_scatter(mpi_gather(ibmin_l, mpi_comm), mpi_comm)
        ibmax = mpi_scatter(mpi_gather(ibmax_l, mpi_comm), mpi_comm)
        nelec = mpi_sum(nelec_l, mpi_comm)
    end
    iks = Int[]; ibands = Int[]
    for ik in 1:gkpts.n
        ibmax[ik] >= ibmin[ik] || continue
        for b in ibmin[ik]:ibmax[ik]
            push!(iks, ik); push!(ibands, b)
        end
    end
    FilteredStates(gkpts, iks, ibands; nw, nstates_base=nelec)
end

# Convenience: pull nw/el_ham off a Model. Kept kpts-first to match the core signature; dispatches on
# the 2nd argument being a `Model` (no ambiguity with the `(kpts_input, nw::Integer, el_ham, window)`
# core method).
filter_electron_states(kpts_input, model::Model, window; kwargs...) =
    filter_electron_states(kpts_input, model.nw, model.el_ham, window; kwargs...)

"""
    filter_electron_states_multigrid(nks_f, nks_c, window_f, window_c, nw, el_ham;
                                     fourier_mode="gridopt", symmetry=nothing, use_gpu=false)
        -> FilteredStates

Generator 2 (double-grid): build a `FilteredStates` sampling a FINE grid `nks_f` in the narrow
window `window_f` merged with a COARSE grid `nks_c` in the wide window `window_c` (`nks_f` a multiple
of `nks_c`, `window_f` contained in `window_c`). The per-`(k, band)` weight follows the clean
double-grid partition:

  * a fine-grid state `(k, b)` with `b` in the narrow window gets the fine BZ weight
    `kpts_fine.weights[k]`;
  * a coarse-grid state `(k, b)` with `b` in the wide window but NOT in the narrow window gets the
    coarse BZ weight `kpts_coarse.weights[k]`.

Concretely, at a fine-grid k point `ik`, a band `ib1` inside the narrow window gets the fine-grid
weight, while a band `ib2` outside the narrow window but inside the wide window gets the coarse-grid
weight — so a single physical node carries two different weights across its bands, which a merged
single-window k-set cannot express. Every coarse node coincides with a fine node (coarse ⊂ fine), so
the two grids merge on the shared node. `nstates_base` is the coarse full-grid below-`window_c`
carrier count. The merged grid is stamped with the FINE `ngrid` (`nks_f`) so every k lies on a common
grid for the q-lookup; its per-point `weights` are informational (the per-state `weights` on the
selection are the authoritative BZ weights).
"""
function filter_electron_states_multigrid(nks_f, nks_c, window_f, window_c, nw, el_ham;
                                  fourier_mode="gridopt", symmetry=nothing, use_gpu=false)

    all(mod.(nks_f, nks_c) .== 0) || throw(ArgumentError("nks_f must be a multiple of nks_c"))
    (window_f[1] >= window_c[1] && window_f[2] <= window_c[2]) ||
        throw(ArgumentError("the fine window_f must be contained in the coarse window_c"))

    kpts_f, ibmin_f, ibmax_f, _            = _filter_with_band_ranges(nks_f, nw, el_ham, window_f; symmetry, fourier_mode, use_gpu)  # fine, narrow
    kpts_c, ibmin_c, ibmax_c, nstates_base = _filter_with_band_ranges(nks_c, nw, el_ham, window_c; symmetry, fourier_mode, use_gpu)  # coarse, wide

    T = eltype(kpts_f.weights)

    # Merge the two grids into one on the FINE ngrid: the fine points first (in order), then the coarse
    # points not already on the (filtered) fine grid. A coarse node coincides exactly with a fine node,
    # but the fine grid was filtered to window_f, so a coarse node whose fine counterpart has no
    # narrow-window band is absent from it.
    xks = collect(kpts_f.vectors)
    k_weights = collect(kpts_f.weights)
    for ik_c in 1:kpts_c.n
        if xk_to_ik(kpts_c.vectors[ik_c], kpts_f) === nothing
            # coarse k point ik_c not part of fine grid
            push!(xks, kpts_c.vectors[ik_c]); push!(k_weights, kpts_c.weights[ik_c])
        end
    end
    merged = GridKpoints(Kpoints(length(xks), xks, k_weights, nks_f))

    iks = Int[]
    ibands = Int[]
    weights = T[]
    # (1) fine points: narrow-window bands at the fine BZ weight.
    for ik_f in 1:kpts_f.n, ib in ibmin_f[ik_f]:ibmax_f[ik_f]
        push!(iks, ik_f); push!(ibands, ib); push!(weights, kpts_f.weights[ik_f])
    end
    # (2) coarse points: wide-window bands OUTSIDE the narrow window, at the coarse BZ weight. A band
    # is inside the narrow window iff it lies in the coincident fine point's narrow range (the shared
    # node has identical eigenvalues, so this band-index test equals testing e_kb against window_f).
    for ik_c in 1:kpts_c.n
        ik = xk_to_ik(kpts_c.vectors[ik_c], merged)
        ik_f = xk_to_ik(kpts_c.vectors[ik_c], kpts_f)      # coincident fine point (nothing if none)
        rng_narrow_ik = ik_f === nothing ? (1:0) : (ibmin_f[ik_f]:ibmax_f[ik_f])
        for ib in ibmin_c[ik_c]:ibmax_c[ik_c]
            ib in rng_narrow_ik && continue                # inside narrow window -> already at fine weight
            push!(iks, ik); push!(ibands, ib); push!(weights, kpts_c.weights[ik_c])
        end
    end
    # (3) order states so each k point's states form a contiguous, band-ascending block (the GPU
    # outer-k tiling maps a contiguous k-range to a contiguous state range; see ind_range_for_k_range).
    p = sortperm(collect(zip(iks, ibands)))
    FilteredStates(merged, iks[p], ibands[p]; nw, weights=weights[p], nstates_base)
end
