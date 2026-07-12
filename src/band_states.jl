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

export BandStates, state_index, state_weights, state_xks, electron_states_to_BandStates

"""
    BandStates{T, KT<:AbstractKpoints{T}}

Single-particle states indexed by `(k-point, band)`, `i = 1…n`. Aligned per-state arrays
`ik`, `iband`, `e`, (optionally `v`) all have length `n`; the k-grid `kpts` (its `vectors`,
`weights`, `ngrid`) is shared. The same k-point repeats once per band.

State `i` is `(kpts.vectors[ik[i]], iband[i])`, with energy `e[i]`, velocity `v[i]` (if `v`
is non-empty), and BZ weight `kpts.weights[ik[i]]`. For phonons `iband` is the mode index.

Pure k-properties (k-vector, weight) are derived through `ik` rather than cached per-state;
`state_xks`/`state_weights` materialize a dense length-n array when a hot/GPU loop needs one.
"""
struct BandStates{T, KT <: AbstractKpoints{T}}
    n::Int                  # number of states
    nband::Int              # number of distinct bands indexed = maximum(iband) − nband_ignore
                            # (the band extent of `indmap`)
    nband_ignore::Int       # bands below the lowest indexed one = minimum(iband) − 1, subtracted
                            # so band `iband` maps to `indmap` row `iband − nband_ignore ∈ 1:nband`
    kpts::KT                # k-grid: vectors, weights, ngrid (+ hash if GridKpoints)
    iks::Vector{Int}        # per-state k index into kpts (k-vector/weight derived via this)
    ibands::Vector{Int}     # per-state band index (mode index for phonons)
    es::Vector{T}           # per-state energy (intrinsic: depends on band + k)
    vs::Vector{Vec3{T}}     # per-state band velocity (intrinsic); empty if not computed
    nstates_base::T         # occupied states per cell below the window (electron counting)
    indmap::Matrix{Int}     # (iband − nband_ignore, ik) → state index; 0 where absent
end

function _build_indmap(n, nk, nband, nband_ignore, ik, iband)
    indmap = zeros(Int, nband, nk)
    @inbounds for i in 1:n
        indmap[iband[i] - nband_ignore, ik[i]] = i
    end
    indmap
end

"""
    BandStates(kpts, ik, iband, e; v, nstates_base)

Primary constructor: `kpts` is the (shared) k-grid, `ik[i]` the k-index of state `i`,
`iband[i]` its band, `e[i]` its energy. `v` (per-state velocity) defaults to empty.
"""
function BandStates(kpts::AbstractKpoints{T}, ik::AbstractVector{<:Integer},
        iband::AbstractVector{<:Integer}, e::AbstractVector;
        v::AbstractVector = Vec3{T}[], nstates_base = zero(T)) where {T}
    n = length(ik)
    n == length(iband) == length(e) || error("BandStates: ik, iband, e must have equal length")
    (isempty(v) || length(v) == n) || error("BandStates: v must be empty or length n")
    nband_ignore = minimum(iband) - 1
    nband = maximum(iband) - nband_ignore
    indmap = _build_indmap(n, kpts.n, nband, nband_ignore, ik, iband)
    BandStates{T, typeof(kpts)}(n, nband, nband_ignore, kpts, collect(Int, ik),
        collect(Int, iband), collect(T, e), collect(Vec3{T}, v),
        T(nstates_base), indmap)
end

"""
    electron_states_to_BandStates(el_states, kpts, nstates_base=0) -> BandStates

Flatten a per-k vector of `ElectronState` onto `kpts` into a `BandStates`, storing the
per-state k-index `ik` directly (no deduplication: `kpts` already holds the distinct
k-points). `kpts` must be a `GridKpoints` (its k-vector→index hash is needed for the e-ph loop
and `state_index(xk, …)` queries). The `(iband, ik) → state` reverse map lives in the returned
`BandStates` as `indmap` — query it with `state_index`, or build a device copy for the GPU
scatter with `_indmap_to_device`.

This is the `BandStates` replacement for `electron_states_to_BTStates`.
"""
function electron_states_to_BandStates(el_states::Vector{ElectronState{T}},
        kpts::GridKpoints{T}, nstates_base = zero(T)) where {T}
    nk = length(el_states)
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
    BandStates(kpts, ik, iband, e; v, nstates_base)
end

# Build a device `(nw, nk)` integer index map addressable by PHYSICAL band: row `iband` ∈ 1:nw holds
# the flattened state index for `(iband, ik)`, and 0 where that band is absent / out of the energy
# window. `nw` is the full (Wannier) band count — the band axis the e-ph matrix `ep_kq` is indexed by
# — so a device kernel can look a state up directly from its physical band index. `proto` is a device
# array used only as a `similar` prototype. (`s.indmap` is stored band-offset by `nband_ignore`; this
# places its rows at their physical-band positions `nband_ignore+1 : nband_ignore+nband`.)
#
# Why the full `nw` rows and not the (smaller) in-window / projected band count:
#   * k+q map: the scatter indexes it by the physical k+q band `m ∈ 1:nw`. The k+q band axis is NOT
#     window-projected (all nw k+q bands are kept; out-of-window ones are the 0 entries), so it needs
#     a row per physical band — nw is required here.
#   * k map: the scatter reads it as a per-k *shifted* window `view(·, ibandk_offset+1 : +nbandk, ik)`
#     (the k side IS projected to nbandk). It could be stored as just `nbandk` rows by baking each k's
#     `ibandk_offset` into its column, but this Int map is tiny next to the streamed Sᵢ (GBs), so both
#     maps share the one physical-band (nw) layout and the k side simply offsets at read time.
function _indmap_to_device(proto, s::BandStates, nw::Integer)
    host = zeros(Int, nw, s.kpts.n)
    @views host[s.nband_ignore+1 : s.nband_ignore+s.nband, :] .= s.indmap
    copyto!(similar(proto, Int, nw, s.kpts.n), host)
end

"""
    find_unfolding_indices(el_i::BandStates, el_f::BandStates, symmetry) -> Vector{NTuple{2,Int}}

For each inner (full-BZ) state `f`, find the outer (IBZ) state `i` and symmetry index `isym`
such that `S_isym · k_i ≡ k_f` (mod reciprocal lattice) with the same band. Runs once at
kernel assembly (not a hot loop). Errors if any inner state has no representative.
`BandStates` replacement for the `BTStates` method (same semantics).
"""
function find_unfolding_indices(el_i::BandStates, el_f::BandStates, symmetry)
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

# --- iteration / indexing (non-allocating; only plain array indexing, no hash lookup) ---
Base.length(s::BandStates) = s.n
Base.firstindex(::BandStates) = 1
Base.lastindex(s::BandStates) = s.n
@inline Base.getindex(s::BandStates, i::Int) =
    (; ik = s.iks[i], iband = s.ibands[i], xk = s.kpts.vectors[s.iks[i]], e = s.es[i],
       weight = s.kpts.weights[s.iks[i]])
Base.iterate(s::BandStates, i::Int = 1) = i > s.n ? nothing : (s[i], i + 1)
Base.eltype(::Type{<:BandStates{T}}) where {T} =
    NamedTuple{(:ik, :iband, :xk, :e, :weight), Tuple{Int, Int, Vec3{T}, T, T}}

"Per-state BZ weights, gathered from `kpts.weights` (dense length-`n` array)."
state_weights(s::BandStates) = s.kpts.weights[s.iks]

"Per-state k-vectors, gathered from `kpts.vectors` (dense length-`n` array)."
state_xks(s::BandStates) = s.kpts.vectors[s.iks]

"""
    state_index(s, ik::Int, iband::Int) -> Int
    state_index(s, xk, iband::Int) -> Int   # GridKpoints only (uses the k-grid hash)

O(1) reverse lookup of the state index for `(ik, iband)`, or `0` if absent. The k-vector
form resolves `ik = xk_to_ik(xk, s.kpts)` first and requires `kpts isa GridKpoints`.
"""
@inline function state_index(s::BandStates, ik::Int, iband::Int)
    b = iband - s.nband_ignore
    (1 <= b <= s.nband && 1 <= ik <= s.kpts.n) || return 0
    @inbounds s.indmap[b, ik]
end
function state_index(s::BandStates{T, <:GridKpoints},
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
function ind_range_for_k_range(s::BandStates, kstart::Integer, kend::Integer)
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
