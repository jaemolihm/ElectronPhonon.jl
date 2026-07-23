# BandStates: a set of single-particle states indexed by (k-point, band).
#
# Designed to supersede `BTStates`. Differences, by design:
#   * The k-grid is embedded as a `kpts::AbstractKpoints` (carrying vectors, weights, ngrid,
#     and — for `GridKpoints` — the integer-grid hash). No more carrying `xks` + `ngrid`
#     separately, and per-state weights are `kpts.weights[ik]` rather than a `k_weight` field.
#   * A per-state k-index `ik` into `kpts` is stored (so the k-point of a state is known
#     directly, never recovered via `xk_to_ik`). Pure k-properties (k-vector, weight) are
#     NOT cached per-state — they are derived via `kpts.*[ik]`; only intrinsic per-state
#     quantities (`e`, `v`) are stored as length-n arrays. Use `state_xks`/`state_weights`
#     to materialize a dense length-n array on demand for a hot/GPU loop.
#   * An eager `indmap` gives O(1) `(ik, iband) → state` reverse lookup (no O(n) scans).
#   * A full iterator: `for st in states` yields a non-allocating per-state NamedTuple.
#   * Generic over `Kpoints`/`GridKpoints`; only k-vector→state queries need the hash.

export AbstractBandStates, BandStates, FilteredStates
export state_index, state_weights, state_xks, band_range, electron_states_to_BandStates, unfold_band_states

"""
    AbstractBandStates{T, KT<:AbstractKpoints{T}}

A selection of single-particle `(k-point, band)` states over a shared k-grid `kpts`. Two concrete
subtypes:

  * `FilteredStates` — a lean selection (which `(k, band)` pairs, per-state weights, per-k band
    extent, `nstates_base`), what the k-point/band generators emit and `compute_electron_states`
    consumes;
  * `BandStates` — the rich, velocity-complete form (adds per-state energies `es` and velocities
    `vs`), used for the scatter, transport, and the δf feedback.

The `(k, band)`-selection machinery (`state_index`, `_build_indmap`, `state_weights`, `state_xks`,
`ind_range_for_k_range`, `_indmap_to_device`, `find_unfolding_indices`, the length/index interface,
and the `bt_*` accessors) dispatches on `AbstractBandStates`, so both subtypes share it. The
`es`/`vs`-dependent methods (transport, δf feedback, the per-state iterator) stay on `BandStates`.

Every subtype carries the fields the shared machinery reads: `n`, `nband`, `nband_ignore`, `nw`,
`kpts`, `iks`, `ibands`, `weights` (per-state; empty ⇒ derived from `kpts.weights[iks]`),
`nstates_base`, `indmap`, and `band_extent` (per-k `UnitRange`, length `kpts.n`).
"""
abstract type AbstractBandStates{T, KT <: AbstractKpoints{T}} end

"""
    BandStates{T, KT<:AbstractKpoints{T}} <: AbstractBandStates{T, KT}

Single-particle states indexed by `(k-point, band)`, `i = 1…n`. Aligned per-state arrays
`ik`, `iband`, `e`, (optionally `v`) all have length `n`; the k-grid `kpts` (its `vectors`,
`weights`, `ngrid`) is shared. The same k-point repeats once per band.

State `i` is `(kpts.vectors[ik[i]], iband[i])`, with energy `e[i]`, velocity `v[i]` (if `v`
is non-empty), and BZ weight `weights[i]` (or `kpts.weights[ik[i]]` if `weights` is empty). For
phonons `iband` is the mode index.

Pure k-properties (k-vector, weight) are derived through `ik` rather than cached per-state;
`state_xks`/`state_weights` materialize a dense length-n array when a hot/GPU loop needs one.
"""
struct BandStates{T, KT <: AbstractKpoints{T}} <: AbstractBandStates{T, KT}
    n::Int                  # number of states
    nband::Int              # number of distinct bands indexed = maximum(iband) - nband_ignore
                            # (the band extent of `indmap`)
    nband_ignore::Int       # bands below the lowest indexed one = minimum(iband) - 1, subtracted
                            # so band `iband` maps to `indmap` row `iband - nband_ignore ∈ 1:nband`
    nw::Int                 # full (Wannier) band count of the model these states came from; the
                            # physical-band extent (≥ nband_ignore + nband). Informational — the
                            # device index-map pad width is passed explicitly to `_indmap_to_device`.
    kpts::KT                # k-grid: vectors, weights, ngrid (+ hash if GridKpoints)
    iks::Vector{Int}        # per-state k index into kpts (k-vector/weight derived via this)
    ibands::Vector{Int}     # per-state band index (mode index for phonons)
    es::Vector{T}           # per-state energy (intrinsic: depends on band + k)
    vs::Vector{Vec3{T}}     # per-state band velocity (intrinsic); empty if not computed
    weights::Vector{T}      # per-state BZ weight; empty ⇒ derive from kpts.weights[iks]
    nstates_base::T         # occupied states per cell below the window (electron counting)
    indmap::Matrix{Int}     # (iband - nband_ignore, ik) → state index; 0 where absent
    band_extent::Vector{UnitRange{Int}}  # per-k band extent (one range per kpts point; 1:0 if none)
end

"""
    FilteredStates{T, KT<:AbstractKpoints{T}} <: AbstractBandStates{T, KT}

Lean selection of `(k-point, band)` states over a shared `kpts`, before eigenvectors/velocities are
computed. Mirrors `BandStates` minus `es`/`vs`. Emitted by the k-point/band generators and consumed
by `compute_electron_states(model, sel, …)`, which computes eigenvectors/velocities for exactly the
per-k `band_extent` bands. `electron_states_to_BandStates(el_states, sel)` then attaches `es`/`vs`.
"""
struct FilteredStates{T, KT <: AbstractKpoints{T}} <: AbstractBandStates{T, KT}
    n::Int
    nband::Int
    nband_ignore::Int
    nw::Int
    kpts::KT
    iks::Vector{Int}
    ibands::Vector{Int}
    weights::Vector{T}
    nstates_base::T
    indmap::Matrix{Int}
    band_extent::Vector{UnitRange{Int}}
end

function _build_indmap(n, nk, nband, nband_ignore, ik, iband)
    indmap = zeros(Int, nband, nk)
    @inbounds for i in 1:n
        indmap[iband[i] - nband_ignore, ik[i]] = i
    end
    indmap
end

# Per-k band extent: for each of the `nk` k-points, the `min:max` band range of the states at that
# k (1:0 if none). Bands are contiguous per k in every generator (an in-window range), so this
# span is exactly the set of selected bands there.
function _build_band_extent(nk, ik, iband)
    lo = fill(typemax(Int), nk)
    hi = zeros(Int, nk)
    @inbounds for i in eachindex(ik)
        k = ik[i]; b = iband[i]
        lo[k] = min(lo[k], b); hi[k] = max(hi[k], b)
    end
    [hi[k] >= lo[k] ? (lo[k]:hi[k]) : (1:0) for k in 1:nk]
end

"""
    BandStates(kpts, ik, iband, e; nw, v, weights, nstates_base, band_extent)

Primary constructor: `kpts` is the (shared) k-grid, `ik[i]` the k-index of state `i`,
`iband[i]` its band, `e[i]` its energy. `nw` is the model's full Wannier band count. `v`
(per-state velocity) and `weights` (per-state BZ weight; empty ⇒ derived from `kpts.weights`)
default to empty. `band_extent` defaults to the per-k span of `ik`/`iband`.
"""
function BandStates(kpts::AbstractKpoints{T}, ik::AbstractVector{<:Integer},
        iband::AbstractVector{<:Integer}, e::AbstractVector;
        nw::Integer, v::AbstractVector = Vec3{T}[], weights::AbstractVector = T[],
        nstates_base = zero(T), band_extent = nothing) where {T}
    n = length(ik)
    n == length(iband) == length(e) || error("BandStates: ik, iband, e must have equal length")
    (isempty(v) || length(v) == n) || error("BandStates: v must be empty or length n")
    (isempty(weights) || length(weights) == n) || error("BandStates: weights must be empty or length n")
    # Empty selection (no in-window state, e.g. an empty MPI rank): nband_ignore/nband = 0 so
    # band_range = 1:0 and indmap has no rows.
    nband_ignore = isempty(iband) ? 0 : minimum(iband) - 1
    nband = isempty(iband) ? 0 : maximum(iband) - nband_ignore
    indmap = _build_indmap(n, kpts.n, nband, nband_ignore, ik, iband)
    be = band_extent === nothing ? _build_band_extent(kpts.n, ik, iband) :
        collect(UnitRange{Int}, band_extent)
    BandStates{T, typeof(kpts)}(n, nband, nband_ignore, Int(nw), kpts, collect(Int, ik),
        collect(Int, iband), collect(T, e), collect(Vec3{T}, v), collect(T, weights),
        T(nstates_base), indmap, be)
end

"""
    FilteredStates(kpts, ik, iband; nw, weights, nstates_base, band_extent)

Build a `FilteredStates` over the shared grid `kpts` from the selected `(ik[i], iband[i])` pairs.
`weights` (per-state BZ weight; empty ⇒ derived from `kpts.weights`) and `band_extent` (per-k band
range; defaults to the per-k span of `ik`/`iband`) default as for `BandStates`.
"""
function FilteredStates(kpts::AbstractKpoints{T}, ik::AbstractVector{<:Integer},
        iband::AbstractVector{<:Integer};
        nw::Integer, weights::AbstractVector = T[], nstates_base = zero(T),
        band_extent = nothing) where {T}
    n = length(ik)
    n == length(iband) || error("FilteredStates: ik, iband must have equal length")
    (isempty(weights) || length(weights) == n) || error("FilteredStates: weights must be empty or length n")
    # Empty selection (no in-window state, e.g. an empty MPI rank): band_range = 1:0, indmap no rows.
    nband_ignore = isempty(iband) ? 0 : minimum(iband) - 1
    nband = isempty(iband) ? 0 : maximum(iband) - nband_ignore
    indmap = _build_indmap(n, kpts.n, nband, nband_ignore, ik, iband)
    be = band_extent === nothing ? _build_band_extent(kpts.n, ik, iband) :
        collect(UnitRange{Int}, band_extent)
    FilteredStates{T, typeof(kpts)}(n, nband, nband_ignore, Int(nw), kpts, collect(Int, ik),
        collect(Int, iband), collect(T, weights), T(nstates_base), indmap, be)
end

"""
    electron_states_to_BandStates(el_states, kpts, nstates_base=0) -> (BandStates, imap)

Flatten a per-k vector of `ElectronState` onto `kpts` into a `BandStates`, and return it together
with `imap[iband, ik]` = state index — an `OffsetMatrix` over the physical band range, 0 outside
the window. The model's full Wannier band count `nw` is read from the `ElectronState`s (they all
carry it) and stored on the `BandStates`. The per-state
k-index `ik` is stored directly (no deduplication: `kpts` already holds the distinct k-points).
`kpts` must be a `GridKpoints` (its k-vector→index hash is needed for the e-ph loop and
`state_index(xk, …)` queries); callers holding a plain `Kpoints` promote it first.

The same `(iband, ik) → state` information also lives in the returned `BandStates` as `indmap`
(query it with `state_index`, or build a device copy for the GPU scatter with `_indmap_to_device`);
the returned `imap` is the physical-band `OffsetMatrix` form that CPU calculator loops index directly.

This is the `BandStates` replacement for `electron_states_to_BTStates`.
"""
function electron_states_to_BandStates(el_states::Vector{ElectronState{T}},
        kpts::GridKpoints{T}, nstates_base = zero(T)) where {T}
    nk = length(el_states)
    nw = first(el_states).nw     # full Wannier band count (same on every ElectronState)
    n = sum(el.nband for el in el_states)
    iband_min = minimum(el.rng.start for el in el_states if el.nband > 0)
    iband_max = maximum(el.rng.stop for el in el_states if el.nband > 0)
    imap = OffsetArray(zeros(Int, iband_max - iband_min + 1, nk), iband_min:iband_max, :)
    ik = zeros(Int, n)
    iband = zeros(Int, n)
    e = zeros(T, n)
    v = zeros(Vec3{T}, n)
    istate = 0
    for jk in 1:nk
        el = el_states[jk]
        el.nband == 0 && continue
        for ib in el.rng
            istate += 1
            ik[istate] = jk
            iband[istate] = ib
            e[istate] = el.e[ib]
            v[istate] = el.vdiag[ib]
            imap[ib, jk] = istate
        end
    end
    BandStates(kpts, ik, iband, e; nw, v, nstates_base), imap
end

"""
    _selection_from_computed_states(kpts, el_states, nstates_base; nw) -> FilteredStates

Build a `FilteredStates` from already-computed `ElectronState`s: each state's per-k band extent is
`el.rng`, per-state weights are left empty (uniform ⇒ derived from `kpts.weights`). Used by the
driver's sugar path to wrap the filter+compute result into a selection, so the calculator consumes a
selection on both the sugar and prebuilt-selection paths. Promotes `kpts` to `GridKpoints`.
"""
function _selection_from_computed_states(kpts, el_states, nstates_base; nw)
    gkpts = kpts isa GridKpoints ? kpts : GridKpoints(kpts)
    iks = Int[]; ibands = Int[]
    for (ik, el) in enumerate(el_states)
        for b in el.rng
            push!(iks, ik); push!(ibands, b)
        end
    end
    FilteredStates(gkpts, iks, ibands; nw, nstates_base)
end

"""
    electron_states_to_BandStates(el_states, sel::FilteredStates) -> BandStates

Attach per-state energies/velocities to a prebuilt `FilteredStates`, gathering `es[i]`/`vs[i]` from
`el_states[sel.iks[i]]` at band `sel.ibands[i]`, and carrying over the selection's `kpts`, `iks`,
`ibands`, per-state `weights`, `nstates_base`, `indmap`, and `band_extent`. `el_states` must have been
computed with `compute_electron_states(model, sel, …)` so each `el.rng` covers the selected bands.
This is the selection-path variant used by the driver to build `calc.el_i`/`el_f` (it bypasses the
uniform per-k flatten, preserving the multigrid's per-`(k, band)` weights).
"""
function electron_states_to_BandStates(el_states::Vector{ElectronState{T}},
        sel::FilteredStates{T}) where {T}
    n = sel.n
    es = zeros(T, n)
    vs = zeros(Vec3{T}, n)
    @inbounds for i in 1:n
        el = el_states[sel.iks[i]]
        b = sel.ibands[i]
        es[i] = el.e_full[b]
        vs[i] = el.vdiag[b]
    end
    # Materialize the per-state weights (the selection's own, or derived from kpts.weights when the
    # selection left them empty) so `es`/`vs`-side consumers can index `el.weights` directly, O(1)
    # and non-allocating, in a hot loop (the BTE scatter's per-final-state weight). Same values as
    # `state_weights` derives lazily.
    weights = collect(T, state_weights(sel))
    BandStates{T, typeof(sel.kpts)}(n, sel.nband, sel.nband_ignore, sel.nw, sel.kpts,
        copy(sel.iks), copy(sel.ibands), es, vs, weights, sel.nstates_base,
        copy(sel.indmap), copy(sel.band_extent))
end

# Build a device `(nband_physical, nk)` integer index map addressable by PHYSICAL band: row `iband` ∈
# 1:nband_physical holds the flattened state index for `(iband, ik)`, and 0 where that band is absent
# / out of the energy window, so a device kernel can look a state up directly from its physical band
# index. (`s.indmap` is stored band-offset by `nband_ignore`; this places its rows at their
# physical-band positions `nband_ignore+1 : nband_ignore+nband`.)
#
# `nband_physical` is the physical-band row count (row stride) to pad to — a property of the
# CONSUMER's index space, not of `s`, so it is passed explicitly rather than read from `s.nw`: the
# two callers choose it differently (Boltzmann passes `model.nw`; ME passes `nbandkq`).
#
# Why the full physical-band rows and not the (smaller) in-window / projected band count:
#   * k+q map: the scatter indexes it by the physical k+q band `m`. The k+q band axis is NOT
#     window-projected (all bands are kept; out-of-window ones are the 0 entries), so it needs a row
#     per physical band.
#   * k map: the scatter reads it as a per-k *shifted* window `view(·, ibandk_offset+1 : +nbandk, ik)`
#     (the k side IS projected to nbandk). It could be stored as just `nbandk` rows by baking each k's
#     `ibandk_offset` into its column, but this Int map is tiny next to the streamed Sᵢ (GBs), so both
#     maps share the one physical-band layout and the k side simply offsets at read time.
function _indmap_to_device(backend::AbstractBackend, s::AbstractBandStates, nband_physical::Integer)
    host = zeros(Int, nband_physical, s.kpts.n)
    @views host[s.nband_ignore+1 : s.nband_ignore+s.nband, :] .= s.indmap
    copyto!(alloc(backend, Int, nband_physical, s.kpts.n), host)
end

"""
    find_unfolding_indices(el_i::BandStates, el_f::BandStates, symmetry) -> Vector{NTuple{2,Int}}

For each inner (full-BZ) state `f`, find the outer (IBZ) state `i` and symmetry index `isym`
such that `S_isym · k_i ≡ k_f` (mod reciprocal lattice) with the same band. Runs once at
kernel assembly (not a hot loop). Errors if any inner state has no representative.
`BandStates` replacement for the `BTStates` method (same semantics).
"""
function find_unfolding_indices(el_i::AbstractBandStates, el_f::AbstractBandStates, symmetry)
    xks_i = state_xks(el_i)   # dense gather once (setup, not a hot loop)
    xks_f = state_xks(el_f)
    ind_and_isym = fill((0, 0), el_f.n)
    for f in 1:el_f.n
        xk_f = xks_f[f]
        ib = el_f.ibands[f]
        found = false
        for (isym, S) in enumerate(symmetry)
            for j in 1:el_i.n
                el_i.ibands[j] == ib || continue
                Sk = apply_symop(S, xks_i[j], :momentum)
                dk = Sk - xk_f
                if all(abs.(dk .- round.(dk)) .< 1e-10)
                    ind_and_isym[f] = (j, isym)
                    found = true
                    break
                end
            end
            found && break
        end
        found || error("find_unfolding_indices: no IBZ representative for inner state $f " *
                       "(k = $xk_f, band = $ib)")
    end
    ind_and_isym
end

"""
    unfold_band_states(sel::FilteredStates, symmetry) -> FilteredStates

Unfold an IBZ `FilteredStates` to the full Brillouin zone: the k-points are unfolded
with `unfold_kpoints`, and each IBZ state `(k, band)` is copied to every point of its symmetry star,
carrying the same band and a per-state weight divided by the star size (so the star's total BZ weight
equals the IBZ state's — the full-BZ per-state weight is `1/N` for a uniform level, `1/N_fine` /
`1/N_coarse` for the multigrid double grid). Runs on the lean selection (before eigenvectors exist),
so the caller builds the full-BZ k+q selection explicitly and passes it to `run_eph_over_k_and_kq`,
which then consumes it as-is. `el_f` is then the exact symmetry unfolding of `el_i`, which the
`interpolate=false` δf feedback map relies on. `symmetry === nothing` returns a copy unchanged.
"""
function unfold_band_states(sel::FilteredStates{T}, symmetry) where {T}
    symmetry === nothing && return deepcopy(sel)
    kpts_u, ik_to_ikirr_isym = unfold_kpoints(sel.kpts, symmetry)
    # Star size of each IBZ point = number of full-BZ points mapping back to it.
    starsize = zeros(Int, sel.kpts.n)
    for (ibz, _) in ik_to_ikirr_isym
        starsize[ibz] += 1
    end
    w_sel = state_weights(sel)
    states_by_ibz = [Int[] for _ in 1:sel.kpts.n]
    for i in 1:sel.n
        push!(states_by_ibz[sel.iks[i]], i)
    end
    iks_u = Int[]; ibands_u = Int[]; weights_u = T[]
    for full_ik in 1:kpts_u.n
        ibz, _ = ik_to_ikirr_isym[full_ik]
        for i in states_by_ibz[ibz]
            push!(iks_u, full_ik)
            push!(ibands_u, sel.ibands[i])
            push!(weights_u, w_sel[i] / starsize[ibz])
        end
    end
    FilteredStates(kpts_u, iks_u, ibands_u; nw=sel.nw, weights=weights_u, nstates_base=sel.nstates_base)
end

# --- iteration / indexing (non-allocating; only plain array indexing, no hash lookup) ---
# length/first/lastindex only need `n`, so they are shared on the abstract type; the per-state
# iterator (getindex/iterate/eltype) yields the state energy `es`, so it stays on `BandStates`
# (a `FilteredStates` has no energies and is not meant to be iterated as states).
Base.length(s::AbstractBandStates) = s.n
Base.firstindex(::AbstractBandStates) = 1
Base.lastindex(s::AbstractBandStates) = s.n
@inline Base.getindex(s::BandStates, i::Int) =
    (; ik = s.iks[i], iband = s.ibands[i], xk = s.kpts.vectors[s.iks[i]], e = s.es[i],
       weight = isempty(s.weights) ? s.kpts.weights[s.iks[i]] : s.weights[i])
Base.iterate(s::BandStates, i::Int = 1) = i > s.n ? nothing : (s[i], i + 1)
Base.eltype(::Type{<:BandStates{T}}) where {T} =
    NamedTuple{(:ik, :iband, :xk, :e, :weight), Tuple{Int, Int, Vec3{T}, T, T}}

"""Per-state BZ weights (dense length-`n` array): the stored per-state `weights` when non-empty
(e.g. the multigrid double-grid partition), else gathered from `kpts.weights[iks]`."""
state_weights(s::AbstractBandStates) = isempty(s.weights) ? s.kpts.weights[s.iks] : s.weights

"Per-state k-vectors, gathered from `kpts.vectors` (dense length-`n` array)."
state_xks(s::AbstractBandStates) = s.kpts.vectors[s.iks]

"""
    band_range(s::AbstractBandStates) -> UnitRange

Global physical-band range spanned by the selection, `band_min:band_max`. O(1): the constructor sets
`nband_ignore = minimum(iband)-1` and `nband = maximum(iband)-nband_ignore`, so this is
`nband_ignore+1 : nband_ignore+nband`. Reproduces the `(band_min, band_max)` the legacy
`filter_kpoints` tuple carried.
"""
band_range(s::AbstractBandStates) = (s.nband_ignore + 1):(s.nband_ignore + s.nband)

"""
    state_index(s, ik::Int, iband::Int) -> Int
    state_index(s, xk, iband::Int) -> Int   # GridKpoints only (uses the k-grid hash)

O(1) reverse lookup of the state index for `(ik, iband)`, or `0` if absent. The k-vector
form resolves `ik = xk_to_ik(xk, s.kpts)` first and requires `kpts isa GridKpoints`.
"""
@inline function state_index(s::AbstractBandStates, ik::Int, iband::Int)
    b = iband - s.nband_ignore
    (1 <= b <= s.nband && 1 <= ik <= s.kpts.n) || return 0
    @inbounds s.indmap[b, ik]
end
function state_index(s::AbstractBandStates{T, <:GridKpoints},
        xk::Vec3, iband::Int) where {T}
    ik = xk_to_ik(xk, s.kpts)
    ik === nothing ? 0 : state_index(s, ik, iband)
end

"""
    ind_range_for_k_range(s::BandStates, kstart::Integer, kend::Integer) -> UnitRange

State-index range of the states whose k-point lies in `kstart:kend`. States are enumerated in
k order (see `electron_states_to_BandStates`), so a contiguous k-block maps to a contiguous
state block; errors if it does not. Empty range (`1:0`) if no state falls in the k-range.
"""
function ind_range_for_k_range(s::AbstractBandStates, kstart::Integer, kend::Integer)
    # TODO: O(n_i) scan per call (once per outer-k batch — negligible today). Since states are in k
    # order, this could be an O(1) lookup from a per-k state-offset prefix if it ever matters.
    imin = typemax(Int); imax = 0; count = 0
    @inbounds for i in 1:s.n
        (kstart <= s.iks[i] <= kend) || continue
        imin = min(imin, i); imax = max(imax, i); count += 1
    end
    count == 0 && return 1:0
    imax - imin + 1 == count || error("ind_range_for_k_range: k-range $kstart:$kend maps to a " *
        "non-contiguous state range ($count states span $(imax - imin + 1) indices).")
    imin:imax
end
