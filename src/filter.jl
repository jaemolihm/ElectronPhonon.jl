
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
    filter_electron_states_multigrid(nks1, nks2, window1, window2, nw, el_ham;
                                     fourier_mode="gridopt", symmetry=nothing, use_gpu=false)
        -> FilteredStates

Generator 2 (double-grid): build a `FilteredStates` sampling a FINE grid `nks1` in the narrow
`window1` merged with a COARSE grid `nks2` in the wide `window2` (`nks1` a multiple of `nks2`). The
per-`(k, band)` weight follows the clean double-grid partition:

  * a fine-grid state `(k, b)` with `b` in the narrow window gets the fine BZ weight
    `kpts_fine.weights[k]`;
  * a coarse-grid state `(k, b)` with `b` in the wide window but NOT in the narrow window gets the
    coarse BZ weight `kpts_coarse.weights[k]`.

Every coarse node coincides with a fine node (coarse ⊂ fine), so the k-points dedup on the fine
grid: a coarse node coincident with a kept fine point shares its shared-grid index, contributing its
wide-only bands at the coarse weight while the fine point's narrow bands stay at the fine weight — a
single physical node carrying two weights across its bands, which a merged single-window k-set
cannot express. `nstates_base` is the coarse full-grid below-`window2` carrier count. The shared
grid is stamped with the FINE `ngrid` (`nks1`) so every k lies on a common grid for the q-lookup;
its per-point `weights` are informational (the per-state `weights` on the selection are the
authoritative BZ weights).
"""
function filter_electron_states_multigrid(nks1, nks2, window1, window2, nw, el_ham;
                                  fourier_mode="gridopt", symmetry=nothing, use_gpu=false)

    all(mod.(nks1, nks2) .== 0) || throw(ArgumentError("nks1 must be a multiple of nks2"))
    (window1[1] >= window2[1] && window1[2] <= window2[2]) ||
        throw(ArgumentError("the fine window1 must be contained in the coarse window2"))

    kpts1, ibmin1, ibmax1, _            = _filter_with_band_ranges(nks1, nw, el_ham, window1; symmetry, fourier_mode, use_gpu)  # fine, narrow
    kpts2, ibmin2, ibmax2, nstates_base = _filter_with_band_ranges(nks2, nw, el_ham, window2; symmetry, fourier_mode, use_gpu)  # coarse, wide

    T = eltype(kpts1.weights)

    # Shared grid points 1..kpts1.n are the fine points; coarse-only points are appended. A coarse node
    # coincides exactly with a fine node, and window1 is contained in window2 (asserted above), so at a
    # coincident point the narrow in-window bands `ibmin1:ibmax1` are a contiguous subset of the wide
    # bands `ibmin2:ibmax2`. Invert the coarse->fine node map: for each fine point that a coarse node
    # coincides with, record that wide band range and coarse weight (`wide_lo`/`wide_hi`/`wide_w`, 0 =
    # no coincident coarse node); coarse nodes with no surviving fine point become appended points
    # (`extra_*`). No dedup or per-band sort is needed since the ranges are contiguous and nested.
    xks = collect(kpts1.vectors)
    k_weights = collect(kpts1.weights)
    wide_lo = zeros(Int, kpts1.n); wide_hi = zeros(Int, kpts1.n); wide_w = zeros(T, kpts1.n)
    extra_lo = Int[]; extra_hi = Int[]; extra_w = T[]
    for i2 in 1:kpts2.n
        i1 = xk_to_ik(kpts2.vectors[i2], kpts1)   # exact: a coarse node lies on the fine grid
        if i1 === nothing
            push!(xks, kpts2.vectors[i2]); push!(k_weights, kpts2.weights[i2])
            push!(extra_lo, ibmin2[i2]); push!(extra_hi, ibmax2[i2]); push!(extra_w, kpts2.weights[i2])
        else
            wide_lo[i1] = ibmin2[i2]; wide_hi[i1] = ibmax2[i2]; wide_w[i1] = kpts2.weights[i2]
        end
    end

    # Emit per-state arrays directly, points in order (fine points first, then coarse-only), bands
    # ascending: narrow bands take the fine weight, the wide-only bands take the coarse weight.
    iks = Int[]; ibands = Int[]; wstate = T[]
    for i1 in 1:kpts1.n
        nlo, nhi = ibmin1[i1], ibmax1[i1]
        blo, bhi = wide_lo[i1] == 0 ? (nlo, nhi) : (wide_lo[i1], wide_hi[i1])
        for b in blo:bhi
            push!(iks, i1); push!(ibands, b)
            push!(wstate, nlo <= b <= nhi ? kpts1.weights[i1] : wide_w[i1])
        end
    end
    for j in eachindex(extra_lo), b in extra_lo[j]:extra_hi[j]
        push!(iks, kpts1.n + j); push!(ibands, b); push!(wstate, extra_w[j])
    end
    gkpts = GridKpoints(Kpoints(length(xks), xks, k_weights, nks1))
    FilteredStates(gkpts, iks, ibands; nw, weights=wstate, nstates_base)
end
