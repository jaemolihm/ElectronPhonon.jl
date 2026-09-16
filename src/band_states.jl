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

export AbstractBandStates, BandStates, FilteredBandStates
export state_index, state_weights, state_xks, band_range, electron_states_to_BandStates,
    electron_states_to_FilteredBandStates, unfold_band_states, filter_states,
    state_index_in_star, state_indices_full_star

"""
    AbstractBandStates{T, KT<:AbstractKpoints{T}}

A selection of single-particle `(k-point, band)` states over a shared k-grid `kpts`. Two concrete
subtypes:

  * `FilteredBandStates` — a lean selection (which `(k, band)` pairs, per-state weights, per-k band
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
`nstates_base`, `indmap`, and `band_extent` (per-k `UnitRange`, length `kpts.n`; a contiguous
superset of the bands there, not the band set — see `BandStates`).
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

`band_extent` is the minimal contiguous superset of the bands present at each k, NOT the band set —
`iks`/`ibands` are the authority. It is a range because its consumer is
`set_window!(::ElectronState, ::UnitRange)`. Window-based selections are contiguous per k;
`filter_states` on an arbitrary set need not be, and then `electron_states_to_BandStates(el_states,
kpts, …)`, which emits one state per band in the extent, returns more states than the
`FilteredBandStates` method, which emits the selection's own list (`{2, 5}` at a k: 4 states vs 2).
Match states across two sets with `state_index`, never by position.
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
    band_extent::Vector{UnitRange{Int}}  # per-k contiguous superset of the bands present, NOT the
                            # band set (one range per kpts point; 1:0 if none)
end

"""
    FilteredBandStates{T, KT<:AbstractKpoints{T}} <: AbstractBandStates{T, KT}

Lean selection of `(k-point, band)` states over a shared `kpts`, before eigenvectors/velocities are
computed. Mirrors `BandStates` minus `es`/`vs`. Emitted by the k-point/band generators and consumed
by `compute_electron_states(model, sel, …)`, which computes eigenvectors/velocities for exactly the
per-k `band_extent` bands. `electron_states_to_BandStates(el_states, sel)` then attaches `es`/`vs`.
"""
struct FilteredBandStates{T, KT <: AbstractKpoints{T}} <: AbstractBandStates{T, KT}
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

# Per-k `min:max` band range of the states at that k (1:0 if none) — the minimal contiguous
# superset of the bands there, which equals them only if those are contiguous: true for a
# window-based selection, not for `filter_states` on an arbitrary set.
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
    BandStates(kpts, ik, iband, e; nw, v, weights, nstates_base)

Primary constructor: `kpts` is the (shared) k-grid, `ik[i]` the k-index of state `i`,
`iband[i]` its band, `e[i]` its energy. `nw` is the model's full Wannier band count. `v`
(per-state velocity) defaults to empty. `weights` (per-state BZ weight) is always stored
length-`n`: when the caller passes an empty `weights`, it is materialized from `kpts.weights[ik]`.
`band_extent` is the contiguous superset of `iband` per k (see `BandStates`).
"""
function BandStates(kpts::AbstractKpoints{T}, ik::AbstractVector{<:Integer},
        iband::AbstractVector{<:Integer}, e::AbstractVector;
        nw::Integer, v::AbstractVector = Vec3{T}[], weights::AbstractVector = T[],
        nstates_base = zero(T)) where {T}
    n = length(ik)
    n == length(iband) == length(e) || error("BandStates: ik, iband, e must have equal length")
    (isempty(v) || length(v) == n) || error("BandStates: v must be empty or length n")
    (isempty(weights) || length(weights) == n) || error("BandStates: weights must be empty or length n")
    # Empty selection (no in-window state, e.g. an empty MPI rank): nband_ignore/nband = 0 so
    # band_range = 1:0 and indmap has no rows.
    nband_ignore = isempty(iband) ? 0 : minimum(iband) - 1
    nband = isempty(iband) ? 0 : maximum(iband) - nband_ignore
    indmap = _build_indmap(n, kpts.n, nband, nband_ignore, ik, iband)
    be = _build_band_extent(kpts.n, ik, iband)
    w = isempty(weights) ? kpts.weights[collect(Int, ik)] : collect(T, weights)
    BandStates{T, typeof(kpts)}(n, nband, nband_ignore, Int(nw), kpts, collect(Int, ik),
        collect(Int, iband), collect(T, e), collect(Vec3{T}, v), w,
        T(nstates_base), indmap, be)
end

"""
    FilteredBandStates(kpts, iks, ibands; nw, weights, nstates_base)

Build a `FilteredBandStates` over the shared grid `kpts` from the selected `(iks[i], ibands[i])` pairs.
`weights` (per-state BZ weight) is always stored length-`n`: an empty `weights` is materialized from
`kpts.weights[iks]`. `band_extent` is the contiguous superset of `ibands` per k (see `BandStates`).
"""
function FilteredBandStates(kpts::AbstractKpoints{T}, iks::AbstractVector{<:Integer},
        ibands::AbstractVector{<:Integer};
        nw::Integer, weights::AbstractVector = T[], nstates_base = zero(T)) where {T}
    n = length(iks)
    n == length(ibands) || error("FilteredBandStates: iks, ibands must have equal length")
    (isempty(weights) || length(weights) == n) || error("FilteredBandStates: weights must be empty or length n")
    # Empty selection (no in-window state, e.g. an empty MPI rank): band_range = 1:0, indmap no rows.
    nband_ignore = isempty(ibands) ? 0 : minimum(ibands) - 1
    nband = isempty(ibands) ? 0 : maximum(ibands) - nband_ignore
    indmap = _build_indmap(n, kpts.n, nband, nband_ignore, iks, ibands)
    be = _build_band_extent(kpts.n, iks, ibands)
    w = isempty(weights) ? kpts.weights[collect(Int, iks)] : collect(T, weights)
    FilteredBandStates{T, typeof(kpts)}(n, nband, nband_ignore, Int(nw), kpts, collect(Int, iks),
        collect(Int, ibands), w, T(nstates_base), indmap, be)
end

"""
    electron_states_to_BandStates(el_states, kpts, nstates_base=0) -> (BandStates, imap)

Flatten a per-k vector of `ElectronState` onto `kpts` into a `BandStates`, and return it together
with `imap[iband, ik]` = state index — an `OffsetMatrix` over the physical band range, 0 outside
the window, the form CPU calculator loops index directly. The
`electron_states_to_BandStates(el_states, sel)` method returns the same pair, so the two are
interchangeable at a call site. The model's full Wannier band count `nw` is read from the
`ElectronState`s (they all carry it) and stored on the `BandStates`. The per-state
k-index `ik` is stored directly (no deduplication: `kpts` already holds the distinct k-points).
`kpts` must be a `GridKpoints` (its k-vector→index hash is needed for the e-ph loop and
`state_index(xk, …)` queries); callers holding a plain `Kpoints` promote it first.

`imap` is a view of the `BandStates`' own `indmap` with its rows offset by `nband_ignore`, not a
second copy — query the same map with `state_index`, or build a device copy for the GPU scatter with
`_indmap_to_device`.

This is the `BandStates` replacement for `electron_states_to_BTStates`.
"""
function electron_states_to_BandStates(el_states::Vector{ElectronState{T}},
        kpts::GridKpoints{T}, nstates_base = zero(T)) where {T}
    nk = length(el_states)
    nw = first(el_states).nw     # full Wannier band count (same on every ElectronState)
    n = sum(el.nband for el in el_states)
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
        end
    end
    bs = BandStates(kpts, ik, iband, e; nw, v, nstates_base)
    bs, OffsetArray(bs.indmap, band_range(bs), 1:kpts.n)
end

"""
    electron_states_to_FilteredBandStates(kpts, el_states, nstates_base; nw) -> FilteredBandStates

Build a `FilteredBandStates` from already-computed `ElectronState`s: each state's per-k band extent is
`el.rng`, per-state weights are left empty (uniform ⇒ derived from `kpts.weights`). Used by the
driver's sugar path to wrap the filter+compute result into a selection, so the calculator consumes a
selection on both the sugar and prebuilt-selection paths. Promotes `kpts` to `GridKpoints`.
"""
function electron_states_to_FilteredBandStates(kpts, el_states, nstates_base; nw)
    gkpts = kpts isa GridKpoints ? kpts : GridKpoints(kpts)
    iks = Int[]; ibands = Int[]
    for (ik, el) in enumerate(el_states)
        for b in el.rng
            push!(iks, ik); push!(ibands, b)
        end
    end
    FilteredBandStates(gkpts, iks, ibands; nw, nstates_base)
end

"""
    electron_states_to_BandStates(el_states, sel::FilteredBandStates) -> (BandStates, imap)

Attach per-state energies/velocities to a prebuilt `FilteredBandStates`, gathering `es[i]`/`vs[i]` from
`el_states[sel.iks[i]]` at band `sel.ibands[i]`, and carrying over the selection's `kpts`, `iks`,
`ibands`, per-state `weights`, `nstates_base`, `indmap`, and `band_extent`. `el_states` must have been
computed with `compute_electron_states(model, sel, …)` so each `el.rng` covers the selected bands.
This is the selection-path variant used by the driver to build `calc.el_i`/`el_f` (it bypasses the
uniform per-k flatten, preserving the multigrid's per-`(k, band)` weights). It returns the same
`(BandStates, imap)` pair as the `kpts` method, so the two are interchangeable at a call site.
"""
function electron_states_to_BandStates(el_states::Vector{ElectronState{T}},
        sel::FilteredBandStates{T}) where {T}
    n = sel.n
    es = zeros(T, n)
    vs = zeros(Vec3{T}, n)
    @inbounds for i in 1:n
        el = el_states[sel.iks[i]]
        b = sel.ibands[i]
        es[i] = el.e_full[b]
        vs[i] = el.vdiag[b]
    end
    # Carry over the selection's per-state weights (always materialized) so `es`/`vs`-side consumers
    # index `el.weights` directly, O(1) and non-allocating, in a hot loop (the BTE scatter's
    # per-final-state weight).
    bs = BandStates{T, typeof(sel.kpts)}(n, sel.nband, sel.nband_ignore, sel.nw, sel.kpts,
        copy(sel.iks), copy(sel.ibands), es, vs, copy(sel.weights), sel.nstates_base,
        copy(sel.indmap), copy(sel.band_extent))
    bs, OffsetArray(bs.indmap, band_range(bs), 1:sel.kpts.n)
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
    indmap_host = zeros(Int, nband_physical, s.kpts.n)
    @views indmap_host[s.nband_ignore+1 : s.nband_ignore+s.nband, :] .= s.indmap
    to_device(backend, indmap_host)
end

"""
    mpi_allgather(s::BandStates, comm::MPI.Comm) -> BandStates

Rank-concatenate a distributed `BandStates` so every rank holds the whole set: rank 0's states
first, then rank 1's, and so on. The `BandStates` counterpart of `mpi_gather(::BTStates, …)`,
which returns the gathered set on the root only.

Each rank's `kpts` must be a DISJOINT slice of one grid, which is what the k-splitters produce
(`filter_electron_states` redistributes with `mpi_gather_and_scatter`: rank-concatenate then
even-split, no reorder). The global k-index of a local state is then its local one shifted by the
k-point count of the preceding ranks — no k-vector lookup is needed, and the per-k weights survive.

Velocities are carried only if EVERY rank has them; if any rank's `vs` is empty the result's is
too, since a partly-filled `vs` would be indexed as though it were complete. `nw` and
`nstates_base` are taken from the local slice: both are global properties already equal on every
rank (`filter_electron_states` `mpi_sum`s the below-window count before constructing).
"""
function mpi_allgather(s::BandStates{FT}, comm::MPI.Comm) where {FT}
    rank = mpi_myrank(comm)
    kcounts = mpi_allgather([s.kpts.n], comm)
    kpts = mpi_allgather(s.kpts, comm)
    iks = mpi_allgather(s.iks .+ sum(@view kcounts[1:rank]), comm)
    ibands = mpi_allgather(collect(s.ibands), comm)
    es = mpi_allgather(collect(s.es), comm)
    weights = mpi_allgather(collect(s.weights), comm)
    # Collective, so the decision must be taken identically on every rank before any rank branches.
    v = all(isone, mpi_allgather([Int(!isempty(s.vs))], comm)) ?
        mpi_allgather(collect(s.vs), comm) : Vec3{FT}[]
    BandStates(kpts, iks, ibands, es; s.nw, v, weights, s.nstates_base)
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
    unfold_band_states(sel::FilteredBandStates, symmetry) -> FilteredBandStates

Unfold an IBZ `FilteredBandStates` to the full Brillouin zone: the k-points are unfolded
with `unfold_kpoints`, and each IBZ state `(k, band)` is copied to every point of its symmetry star,
carrying the same band and a per-state weight divided by the star size (so the star's total BZ weight
equals the IBZ state's — the full-BZ per-state weight is `1/N` for a uniform level, `1/N_fine` /
`1/N_coarse` for the multigrid double grid). Runs on the lean selection (before eigenvectors exist),
so the caller builds the full-BZ k+q selection explicitly and passes it to `run_eph_over_k_and_kq`,
which then consumes it as-is. `el_f` is then the exact symmetry unfolding of `el_i`, which the
`interpolate=false` δf feedback map relies on. `symmetry === nothing` returns a copy unchanged.
"""
function unfold_band_states(sel::FilteredBandStates{T}, symmetry) where {T}
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
    FilteredBandStates(kpts_u, iks_u, ibands_u; nw=sel.nw, weights=weights_u, nstates_base=sel.nstates_base)
end

# --- iteration / indexing (non-allocating; only plain array indexing, no hash lookup) ---
# length/first/lastindex only need `n`, so they are shared on the abstract type. `getindex`
# yields a per-state NamedTuple, one method per concrete type: the `BandStates` one carries the
# state energy `es[i]`, which a `FilteredBandStates` does not have. Both carry the `(xk, iband)`
# identity, which is what `state_index(other, s[i])` needs, so a state can be looked up in
# another selection from either type.
Base.length(s::AbstractBandStates) = s.n
Base.firstindex(::AbstractBandStates) = 1
Base.lastindex(s::AbstractBandStates) = s.n
@inline Base.getindex(s::BandStates, i::Int) =
    (; ik = s.iks[i], iband = s.ibands[i], xk = s.kpts.vectors[s.iks[i]], e = s.es[i],
       weight = s.weights[i])
@inline Base.getindex(s::FilteredBandStates, i::Int) =
    (; ik = s.iks[i], iband = s.ibands[i], xk = s.kpts.vectors[s.iks[i]],
       weight = s.weights[i])
Base.iterate(s::AbstractBandStates, i::Int = 1) = i > s.n ? nothing : (s[i], i + 1)
Base.eltype(::Type{<:BandStates{T}}) where {T} =
    NamedTuple{(:ik, :iband, :xk, :e, :weight), Tuple{Int, Int, Vec3{T}, T, T}}
Base.eltype(::Type{<:FilteredBandStates{T}}) where {T} =
    NamedTuple{(:ik, :iband, :xk, :weight), Tuple{Int, Int, Vec3{T}, T}}

"Per-state BZ weights (the stored length-`n` `weights`, always materialized at construction)."
state_weights(s::AbstractBandStates) = s.weights

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
    state_index(s, st) -> Int               # `st` a per-state item, e.g. `other[i]`

O(1) reverse lookup of the state index for `(ik, iband)`, or `0` if absent. The k-vector
form resolves `ik = xk_to_ik_unsafe(xk, s.kpts)` first and requires `kpts isa GridKpoints`.

The item form takes anything carrying `xk` and `iband` — in particular `other[i]`, the NamedTuple
`getindex` yields on either subtype — so a state of one selection is located in another with
`state_index(s, other[i])`. It goes through `xk`, not `ik`: the two selections' k-grids are
independent, so only the k-vector is a shared address.
"""
@inline function state_index(s::AbstractBandStates, ik::Int, iband::Int)
    b = iband - s.nband_ignore
    (1 <= b <= s.nband && 1 <= ik <= s.kpts.n) || return 0
    @inbounds s.indmap[b, ik]
end
function state_index(s::AbstractBandStates{T, <:GridKpoints},
        xk::Vec3, iband::Int) where {T}
    # `_unsafe` keeps today's behaviour, which is wrong for an `xk` off this selection's grid:
    # it aliases onto a neighbouring node instead of reporting absence. See issue #26.
    ik = xk_to_ik_unsafe(xk, s.kpts)
    ik === nothing ? 0 : state_index(s, ik, iband)
end
state_index(s::AbstractBandStates, st::NamedTuple) = state_index(s, st.xk, st.iband)

"""
    state_index_in_star(s, xk, iband, symmetry) -> Int

State index of `(xk, iband)` in `s`, searching the symmetry star of `xk` when the exact k-vector
is absent: the irreducible representative of a k-point on one grid need not be the representative
chosen on another. The band index is preserved by the point group (ε_{n,Sk} = ε_{n,k}), the
assumption `find_unfolding_indices` already makes. Returns 0 if no image of `xk` carries band
`iband` in `s`.
"""
function state_index_in_star(s::AbstractBandStates, xk, iband::Integer, symmetry)
    j = state_index(s, xk, Int(iband))
    j != 0 && return j
    for S in symmetry
        j = state_index(s, apply_symop(S, xk, :momentum), Int(iband))
        j != 0 && return j
    end
    0
end

"""
    state_indices_full_star(s, xk, iband, symmetry) -> Vector{Int}
    state_indices_full_star(s, st, symmetry) -> Vector{Int}   # `st` a per-state item

Indices in `s` of every image `(S·xk, iband)` of `(xk, iband)` under `symmetry`, sorted and
deduplicated. Images absent from `s` (out of window, or at a k-point `s.kpts` does not hold) are
dropped, so the result can be shorter than the group order and empty. Unlike `unfold_band_states`,
which unfolds a whole selection into a NEW `FilteredBandStates` carrying its own k-grid, this
returns indices into an EXISTING selection, for one state at a time.

The item form takes a state as `states[i]` yields it, like `state_index(s, st)`.
"""
function state_indices_full_star(s::AbstractBandStates, xk, iband::Integer, symmetry)
    J = Int[]
    for S in symmetry
        j = state_index(s, apply_symop(S, xk, :momentum), Int(iband))
        j != 0 && push!(J, j)
    end
    sort!(unique!(J))
end

state_indices_full_star(s::AbstractBandStates, st, symmetry) =
    state_indices_full_star(s, st.xk, st.iband, symmetry)

"""
    filter_states(s::AbstractBandStates, keep) -> AbstractBandStates

Subset of `s` keeping the states `keep` (state indices into `s`), of the same concrete type.
k-points carrying no kept state are DROPPED, because consumers iterate `kpts.n` (an outer-k e-ph
loop visits every k-point of the grid, kept states or not); `get_filtered_kpoints` preserves
`ngrid`, so the subset stays commensurate with the grid it was cut from and `precompute_ph` still
fires. Per-state weights, `nw` and `nstates_base` are carried over unchanged, so the subset's
weights no longer sum to the full BZ. States are emitted sorted by `(iks, ibands)`, the order
`ind_range_for_k_range` requires.

`filter_electron_states` is not an alternative: it rebuilds a selection from an energy window and
ignores a prebuilt k-set's selection.
"""
function filter_states(s::AbstractBandStates, keep::AbstractVector{<:Integer})
    ks = sort(unique(collect(Int, keep)); by = i -> (s.iks[i], s.ibands[i]))
    ik_keep = falses(s.kpts.n)
    ik_keep[s.iks[ks]] .= true
    kpts_new = get_filtered_kpoints(s.kpts, ik_keep)
    # `get_filtered_kpoints` keeps the kept k-points in their original order, so a kept k-point's
    # new index is its rank among them.
    ik_new = zeros(Int, s.kpts.n)
    ik_new[ik_keep] .= 1:kpts_new.n
    _rebuild_states(s, kpts_new, ik_new[s.iks[ks]], ks)
end

# States `ks` of `s`, re-gridded onto `kpts_new` with the new per-state k-indices `iks_new`,
# rebuilt as the same concrete type as `s`. Split off `filter_states` so the k-point re-gridding
# is written once: only this last step differs per subtype (energies are positional on the
# `BandStates` constructor and absent from `FilteredBandStates`, and `es`/`vs` need subsetting
# too). Dispatch rather than an `if s isa BandStates` branch, which would put `s.es`/`s.vs`
# accesses — fields a `FilteredBandStates` does not have — in the shared method body.
_rebuild_states(s::FilteredBandStates, kpts_new, iks_new, ks) =
    FilteredBandStates(kpts_new, iks_new, s.ibands[ks];
        s.nw, weights = s.weights[ks], s.nstates_base)

# `vs` is empty when velocities were never computed, and stays empty in the subset.
_rebuild_states(s::BandStates, kpts_new, iks_new, ks) =
    BandStates(kpts_new, iks_new, s.ibands[ks], s.es[ks];
        s.nw, v = isempty(s.vs) ? s.vs : s.vs[ks],
        weights = s.weights[ks], s.nstates_base)

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
