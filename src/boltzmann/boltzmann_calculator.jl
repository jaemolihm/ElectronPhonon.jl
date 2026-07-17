# BoltzmannCalculator: an AbstractCalculator that accumulates the BTE scattering-out (Sₒ) and
# scattering-in (Sᵢ) matrices during a single pass of `run_eph_over_k_and_kq`. It uses BandStates /
# imap addressing; rather than copying g2/ωq it folds the temperature-dependent occupation physics
# into Sₒ/Sᵢ via the shared `bte_scattering_increments` (see src/boltzmann/bte_scattering_core.jl).
#
# One calculator, both backends: the same calculator instance is used on CPU or GPU — the backend is
# chosen by `run_eph_over_k_and_kq`'s `use_gpu`, which dispatches `run_calculator!` on the payload
# type: `ElPhDataPoint` (host loop, per (ik,iq,ikq)) or `ElPhDataOuterKBatched` (device, per k-batch).
# Both fold the identical `bte_scattering_increments`, so they compute the same scattering (to
# round-off); the CPU path also serves as the validation reference for the GPU path.
#
# Output layout is what the transport solver (`solve_electron_bte` / `solve_thermoelectric_bte`)
# consumes unchanged:
#   Sₒ :: Vector{Vector}  — Sₒ[iT][i]      (inverse SERTA lifetime γ_{nk})
#   Sᵢ :: Vector{Matrix}  — Sᵢ[iT][i, f]   (scattering-in kernel)
#
# Supported configuration (asserted at setup): FermiDirac occupation + Gaussian smearing — the
# configuration `bte_scattering_increments` implements and the one used for transport.
#
# GPU device memory (Sᵢ): the scattering-in matrix Sᵢ (n_i·n_f·nT) is the large object. On the GPU it
# is never held whole on the device — it is tiled over outer k, each tile filled by one k-batch and
# streamed to the host (OuterIterationBatch begin/end brackets), so only one tile (≈ one k-batch of rows)
# is device-resident. This bounds device memory to the tile regardless of grid size at no measurable
# speed cost: streaming is within ~2% of a single whole-Sᵢ copy even at 1.1 GB Sᵢ, because the D2H
# bytes moved are identical either way. There is deliberately no full-device-resident Sᵢ path. (Sₒ is
# small — n_i·nT — and stays device-resident, streamed once at the end.) See benchmark/README.md for
# the profile (and why the scatter kernel is NOT a negligible fraction of GPU time).

export BoltzmannCalculator

# Device buffers for the GPU batched path, built once in the first `calculator_begin!(…,
# OuterIterationBatch(), ctx)` from `alloc(ctx.backend, …)`. Held behind `dev::Union{Nothing, …}` on
# the calculator; touched only at hook granularity (one kernel launch per call), so the function
# boundary keeps the hot code type-stable. Sᵢ is never held whole here — `Sᵢ_tile` is one outer-k
# tile (i-extent = the largest k-batch), streamed to the host per batch.
struct BoltzmannDeviceBuffers{MT, MI, VT, AT3}
    Sₒ      :: MT      # (n_i, nT)  — small, device-resident
    imap_i  :: MI      # (nw, n_k)  physical-band → outer state index
    imap_f  :: MI      # (nw, n_kq) physical-band → inner state index
    e_i     :: VT      # (n_i,) outer energies
    e_f     :: VT      # (n_f,) inner energies
    wq      :: VT      # (n_kq,) inner k+q weights
    μ       :: VT      # (nT,)
    T       :: VT      # (nT,)
    η       :: VT      # (nT,)
    Sᵢ_tile :: AT3     # (ni_cap, n_f, nT)
end

Base.@kwdef mutable struct BoltzmannCalculator{FT} <: AbstractCalculator
    # --- Parameters ---
    const occ::ElectronOccupationParams
    const smearing::Vector{Tuple{Symbol, Float64}}        # one (type, η) per temperature
    # Occupation-factor convention, an integer 1..6; the six conventions are defined in
    # `bte_scattering_increments` (src/boltzmann/bte_scattering_core.jl).
    const occupation_method::Int = 5
    # :SERTA or :BTE. Both Sₒ and Sᵢ are always computed here; this only selects what
    # `solve_electron_bte` does (SERTA uses Sₒ alone; BTE also uses the Sᵢ scattering-in kernel).
    # TODO: for :SERTA, skip allocating/streaming Sᵢ entirely — it is unused there, so this would
    # save the (large) Sᵢ host + tile storage.
    const scattering_method::Symbol = :BTE
    const omega_cutoff::FT = FT(omega_acoustic)           # skip modes below this (e.g. acoustic modes at Γ)

    # Number of CPU thread-chunks for the CPU-path buffers; set at setup (0 = not set yet).
    nchunks::Int = 0
    # Physical-band range (iband_min:iband_max) spanning the in-window bands; set at setup. The CPU
    # per-chunk Sₒ/Sᵢ buffers are OffsetArrays indexed over this band range. (1:0 = not set yet.)
    rng_band::UnitRange{Int} = 1:0
    # Full (Wannier) band count (= model.nw); set at setup. GPU path only: sizes the physical-band
    # device index maps built in the first OuterIterationBatch begin. (0 = not set yet.)
    nw::Int = 0

    # --- State (BandStates) --- the (iband, ik) → state reverse map is `el_*.indmap` (via state_index)
    el_i::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing
    el_f::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing

    # --- Host outputs (solver-facing) ---
    Sₒ::Vector{Vector{FT}} = Vector{Vector{FT}}()         # per iT, length n_i
    Sᵢ::Vector{Matrix{FT}} = Vector{Matrix{FT}}()         # per iT, (n_i, n_f)

    # --- CPU-path thread buffers (run_calculator!) ---
    # Allocated lazily on the first CPU `calculator_begin!(…, OuterIteration(), ctx)`; the GPU path
    # never allocates them (Sᵢ_buffer would be nchunks·nT·rng_band·n_f — prohibitive on production grids).
    Sₒ_buffer::Vector{Vector{OffsetVector{FT, Vector{FT}}}} = Vector{Vector{OffsetVector{FT, Vector{FT}}}}()
    Sᵢ_buffer::Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}} = Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}}()

    # --- Device buffers (GPU batched path) ---
    # Built once in the first `calculator_begin!(…, OuterIterationBatch(), ctx)`; nothing until then
    # (and on the CPU path). See `BoltzmannDeviceBuffers`.
    dev::Union{Nothing, BoltzmannDeviceBuffers} = nothing

    # --- Sᵢ tile bookkeeping (streamed to the host per outer-k batch) ---
    # Sᵢ is never held whole on the device: each outer-k batch fills `dev.Sᵢ_tile` and streams it to
    # the host (OuterIterationBatch begin/end), so only one tile is device-resident at a time.
    Sᵢ_tile_host::Array{FT, 3} = zeros(FT, 0, 0, 0)
    tile_i0::Int = 0
    tile_ni::Int = 0
end

ElectronPhonon.supports(::BoltzmannCalculator, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::BoltzmannCalculator, ::Type{ElPhDataPoint}) = true
ElectronPhonon.supports(::BoltzmannCalculator, ::Type{ElPhDataOuterKBatched}) = true

function ElectronPhonon.setup_calculator!(calc::BoltzmannCalculator{FT}, kpts, qpts, el_states;
        el_states_kq, kqpts, nelec_below_window_k, nelec_below_window_kq, nchunks_threads, rng_band,
        nw, kwargs...) where {FT}
    mpi_isroot() && println("Setting up BoltzmannCalculator")
    calc.scattering_method === :MRTA &&
        throw(ArgumentError("scattering_method :MRTA not implemented"))
    calc.occ.occ_type === :FermiDirac ||
        throw(ArgumentError("BoltzmannCalculator supports occ_type = :FermiDirac only (got $(calc.occ.occ_type))"))
    all(s -> s[1] === :Gaussian, calc.smearing) ||
        throw(ArgumentError("BoltzmannCalculator supports :Gaussian smearing only"))
    FT === Float64 ||
        throw(ArgumentError("BoltzmannCalculator requires FT = Float64: FP32 is not tested and " *
                            "FP32 support is not planned (transport accuracy)."))
    1 <= calc.occupation_method <= 6 ||
        throw(ArgumentError("occupation_method must be an integer in 1:6, got $(calc.occupation_method)"))
    calc.nchunks = nchunks_threads
    calc.rng_band = rng_band
    calc.nw = nw

    # Chemical potential: computed on a temporary BTStates exactly as the CPU reference, so μ is
    # identical (bte_compute_μ! reads BTStates fields directly).
    if !chemical_potential_is_computed(calc.occ)
        el, _ = electron_states_to_BTStates(el_states, kpts, nelec_below_window_k)
        bte_compute_μ!(calc.occ, el; do_print=true)
    end

    # electron_states_to_BandStates requires GridKpoints (its k-vector hash is needed by the loop);
    # the non-symmetry path hands us a plain Kpoints, so promote here.
    # TODO: make run_eph_over_k_and_kq pass GridKpoints directly; if that turns out not to be
    # possible, make setup fail early on a plain Kpoints instead of promoting here.
    gkpts   = kpts isa GridKpoints   ? kpts   : GridKpoints(kpts)
    gkqpts  = kqpts isa GridKpoints  ? kqpts  : GridKpoints(kqpts)
    # (imap discarded: the BoltzmannCalculator scatter reads `el_*.indmap` via `_indmap_to_device`.)
    calc.el_i, _ = electron_states_to_BandStates(el_states, gkpts, nelec_below_window_k)
    calc.el_f, _ = electron_states_to_BandStates(el_states_kq, gkqpts, nelec_below_window_kq)
    n_i = calc.el_i.n
    n_f = calc.el_f.n
    nT = length(calc.occ)

    calc.Sₒ = [zeros(FT, n_i) for _ in 1:nT]
    calc.Sᵢ = [zeros(FT, n_i, n_f) for _ in 1:nT]
    # The device buffers (GPU) and the per-chunk CPU thread buffers are both allocated lazily — the
    # device ones in the first OuterIterationBatch begin, the CPU ones in the first CPU OuterIteration
    # begin — so setup does not need to know the backend. Reset them here so a reused calculator on a
    # different grid re-allocates correctly-sized buffers.
    calc.dev = nothing
    calc.Sₒ_buffer = empty(calc.Sₒ_buffer)
    calc.Sᵢ_buffer = empty(calc.Sᵢ_buffer)
    calc
end

# --- CPU (non-batched, ElPhDataPoint) path --------------------------------------------------

# CPU path: allocate the per-chunk thread buffers on the first outer iteration (this runs once,
# single-threaded, so no lock is needed), then zero them for the new outer k. Dispatched on
# `PointMode`; in the batched loop (`BatchedMode`) the default no-op runs (the batched loop fires
# per-k OuterIteration brackets too, but the device buffers live in OuterIterationBatch).
function ElectronPhonon.calculator_begin!(calc::BoltzmannCalculator{FT}, ::OuterIteration,
        ctx::LoopContext{<:AbstractBackend, PointMode}) where {FT}
    if isempty(calc.Sₒ_buffer)
        rb = calc.rng_band
        n_f = calc.el_f.n
        nT = length(calc.occ)
        calc.Sₒ_buffer = [[OffsetArray(zeros(FT, length(rb)), rb) for _ in 1:nT] for _ in 1:calc.nchunks]
        calc.Sᵢ_buffer = [[OffsetArray(zeros(FT, length(rb), n_f), rb, :) for _ in 1:nT] for _ in 1:calc.nchunks]
    end
    for c in eachindex(calc.Sₒ_buffer)
        for x in calc.Sₒ_buffer[c]; x .= 0; end
        for x in calc.Sᵢ_buffer[c]; x .= 0; end
    end
    calc
end

# CPU path: called per (ik, iq, ikq) by the host e-ph loop; accumulates into the per-chunk thread
# buffers, reduced into Sₒ/Sᵢ by the OuterIteration end bracket.
function ElectronPhonon.run_calculator!(calc::BoltzmannCalculator{FT}, p::ElPhDataPoint, ctx) where {FT}
    (; epdata, ikq, id_chunk) = p
    Sₒ = calc.Sₒ_buffer[id_chunk]
    Sᵢ = calc.Sᵢ_buffer[id_chunk]
    (; el_k, el_kq, ph, wtq) = epdata
    method = calc.occupation_method
    @inbounds for n in el_k.rng
        ek = el_k.e[n]
        for m in el_kq.rng
            ind_el_f = state_index(calc.el_f, ikq, m)
            ind_el_f == 0 && continue
            ekq = el_kq.e[m]
            for (iocc, (; μ, T)) in enumerate(calc.occ)
                η = calc.smearing[iocc][2]
                sₒ = zero(FT); sᵢ = zero(FT)
                for imode in 1:ph.nmodes
                    ωq = ph.e[imode]
                    ωq < calc.omega_cutoff && continue
                    so, si = bte_scattering_increments(method, ek, ekq, ωq,
                        epdata.g2[m, n, imode], wtq, μ, T, η)
                    sₒ += so; sᵢ += si
                end
                Sₒ[iocc][n] += sₒ
                Sᵢ[iocc][n, ind_el_f] += sᵢ
            end
        end
    end
    calc
end

# CPU path: reduce the per-chunk buffers into the global Sₒ/Sᵢ. Dispatched on `PointMode`; the
# batched loop (`BatchedMode`) runs the default no-op.
function ElectronPhonon.calculator_end!(calc::BoltzmannCalculator, ::OuterIteration,
        ctx::LoopContext{<:AbstractBackend, PointMode})
    ik = ctx.outer_index
    @inbounds @views for n in calc.rng_band
        ind_el_i = state_index(calc.el_i, ik, n)
        ind_el_i == 0 && continue
        for c in eachindex(calc.Sₒ_buffer)
            for iT in eachindex(calc.Sₒ)
                calc.Sₒ[iT][ind_el_i] += calc.Sₒ_buffer[c][iT][n]
                calc.Sᵢ[iT][ind_el_i, :] .+= calc.Sᵢ_buffer[c][iT][n, :]
            end
        end
    end
    calc
end

# --- GPU batched path (ElPhDataOuterKBatched) -----------------------------------------------

# Batched path: called by the GPU e-ph loop once per outer-k batch, before its k iterations.
# `ctx.backend` (a `GPUBackend`) allocates from the e-ph matrix backend. On the first batch it builds
# the run's one-time device buffers into `calc.dev` (a `BoltzmannDeviceBuffers`); every batch it
# records this batch's Sᵢ tile range and zeros the tile's active region. Dispatched on `BatchedMode`;
# the per-point (`PointMode`) path runs the default no-op.
function ElectronPhonon.calculator_begin!(calc::BoltzmannCalculator{FT}, ::OuterIterationBatch,
        ctx::LoopContext{<:AbstractBackend, BatchedMode}) where {FT}
    backend = ctx.backend
    n_i, n_f, nT = calc.el_i.n, calc.el_f.n, length(calc.occ)
    kstart, kend = first(ctx.batch), last(ctx.batch)

    # One-time device buffers, built on the first batch and reused across all k in the run. `nw` (the
    # full Wannier band count, from setup) sizes the physical-band index maps; see `_indmap_to_device`.
    # `Sᵢ_tile` is sized to the LARGEST outer-k tile so it never grows mid-run (immutable buffer).
    if calc.dev === nothing
        nk_outer = calc.el_i.kpts.n
        ni_cap = maximum(length(ind_range_for_k_range(calc.el_i, ks, min(ks + ctx.n_batch_max - 1, nk_outer)))
                         for ks in 1:ctx.n_batch_max:nk_outer)
        calc.dev = BoltzmannDeviceBuffers(
            fill!(alloc(backend, FT, n_i, nT), zero(FT)),                                  # Sₒ
            _indmap_to_device(backend, calc.el_i, calc.nw),                                # imap_i
            _indmap_to_device(backend, calc.el_f, calc.nw),                                # imap_f
            copyto!(alloc(backend, FT, n_i), calc.el_i.es),                                # e_i
            copyto!(alloc(backend, FT, n_f), calc.el_f.es),                                # e_f
            copyto!(alloc(backend, FT, calc.el_f.kpts.n), calc.el_f.kpts.weights),         # wq
            copyto!(alloc(backend, FT, nT), collect(FT, calc.occ.μlist)),                  # μ
            copyto!(alloc(backend, FT, nT), collect(FT, calc.occ.Tlist)),                  # T
            copyto!(alloc(backend, FT, nT), FT[s[2] for s in calc.smearing]),              # η
            alloc(backend, FT, ni_cap, n_f, nT),                                           # Sᵢ_tile
        )
        calc.Sᵢ_tile_host = Array{FT, 3}(undef, ni_cap, n_f, nT)
    end

    # Sᵢ tile for this batch (zeroed every batch — Sᵢ is streamed per tile).
    rng = ind_range_for_k_range(calc.el_i, kstart, kend)   # contiguous outer-state block for this k-batch
    calc.tile_i0 = first(rng) - 1; calc.tile_ni = length(rng)
    fill!(view(calc.dev.Sᵢ_tile, 1:calc.tile_ni, :, :), zero(FT))
    calc
end

# Batched path: stream this batch's Sᵢ tile from device to the host output (called once per
# outer-k batch, after its k iterations). Dispatched on `BatchedMode`.
function ElectronPhonon.calculator_end!(calc::BoltzmannCalculator, ::OuterIterationBatch,
        ctx::LoopContext{<:AbstractBackend, BatchedMode})
    if calc.tile_ni > 0
        ni = calc.tile_ni
        i0 = calc.tile_i0
        copyto!(calc.Sᵢ_tile_host, calc.dev.Sᵢ_tile)
        @inbounds for iT in 1:length(calc.occ)
            @views calc.Sᵢ[iT][i0+1:i0+ni, :] .= calc.Sᵢ_tile_host[1:ni, :, iT]
        end
    end
    calc
end

"""
    bte_window_accumulate!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs, e_i, e_f, wq,
                           μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nmodes, nqc, nT; i0=0)

Accumulate the BTE scattering-out (Sₒ) and scattering-in (Sᵢ) contributions of one e-ph chunk into
the (in-energy-window) device buffers — the transport analogue of `eph_window_scatter!`, and the
device-resident work of `run_calculator!(::BoltzmannCalculator, ::ElPhDataOuterKBatched, ctx)` (its sole caller, so it lives here). For every
`(m, n, j)` of the chunk look up the outer/inner states `i = imap_i_at_k[n]`,
`f = imap_f[m, ikqs[j]]` (skip if either is out-of-window, `== 0`), then for each temperature
`iocc` sum the shared per-mode physics (`bte_scattering_increments`) over the `nmodes` phonon modes
(`ωqmat[ν,j] ≥ ω_cutoff`) and:

  * `Sₒ_out[i, iocc] += Σ_ν sₒ`      — scattering-out, added over `(m, ν, j)` (many `(m,j)` map to
    the same outer `i`), so the device method uses an atomic add here. `Sₒ` is small (`n_i × nT`)
    and device-resident, so it is indexed by the GLOBAL outer state `i`;
  * `Sᵢ_out[i−i0, f, iocc] = Σ_ν sᵢ` — scattering-in; each `(i, f)` pair is produced by a unique
    `(n, m, j)` across the whole run (distinct k → distinct i, distinct k+q → distinct f), so this
    is a collision-free plain write (no atomics needed). `Sᵢ` is streamed to the host one tile per
    outer-k batch, so `Sᵢ_out` is the current tile and the row is the tile-local `i − i0`.

`imap_i_at_k` is `imap_i[:, ik]` — the outer-state indices at the batch's fixed outer k `ik`.
`i0` is the global-i offset of the current `Sᵢ` tile. Generic (CPU/fallback) method; the CUDA
extension provides the `CuArray` kernel. The physics lives entirely in `bte_scattering_increments`
so the two paths agree.
"""
function bte_window_accumulate!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs,
        e_i, e_f, wq, μs, Ts, ηs, method::Int, ω_cutoff,
        nbandkq::Int, nbandk::Int, nmodes::Int, nqc::Int, nT::Int; i0::Int = 0)
    @inbounds for j in 1:nqc, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_at_k[n]
        i > 0 || continue
        ikq = ikqs[j]
        f = imap_f[m, ikq]
        f > 0 || continue
        ek = e_i[i]; ekq = e_f[f]; wtq = wq[ikq]
        il = i - i0
        for iocc in 1:nT
            μ = μs[iocc]; T = Ts[iocc]; η = ηs[iocc]
            sₒ = zero(eltype(Sₒ_out)); sᵢ = sₒ
            for ν in 1:nmodes
                ωq = ωqmat[ν, j]
                ωq < ω_cutoff && continue
                so, si = bte_scattering_increments(method, ek, ekq, ωq, g2vals[m, n, ν, j], wtq, μ, T, η)
                sₒ += so; sᵢ += si
            end
            Sₒ_out[i, iocc] += sₒ
            Sᵢ_out[il, f, iocc] = sᵢ
        end
    end
    nothing
end

function ElectronPhonon.run_calculator!(calc::BoltzmannCalculator{FT}, p::ElPhDataOuterKBatched, ctx) where {FT}
    (; ep_kq, g2, ωq, ik, ikqs, ibandk_offset) = p
    dev = calc.dev
    nbandkq, nbandk, nmodes, nqc = size(ep_kq)
    nT = length(calc.occ)
    # Device buffers (imap/energies/Sₒ/Sᵢ_tile) were built once in the first OuterIterationBatch begin.
    # `g2 = |ep|²/(2ω)` is always folded by the loop's fused kernel and carried in the payload.

    # k-side window projection: ep_kq's band-n axis covers nbandk bands starting at physical band
    # ibandk_offset+1, so shift into the physical-band imap_i by that offset (full-band runs have
    # ibandk_offset = 0, nbandk = nw). The k+q axis (m ∈ 1:nbandkq = nw) is not projected.
    imap_i_at_k = view(dev.imap_i, ibandk_offset+1:ibandk_offset+nbandk, ik)

    # Scatter into this batch's Sᵢ tile (streamed to the host by the OuterIterationBatch end bracket).
    ElectronPhonon.bte_window_accumulate!(dev.Sₒ, dev.Sᵢ_tile, g2, ωq,
        imap_i_at_k, dev.imap_f, ikqs, dev.e_i, dev.e_f, dev.wq,
        dev.μ, dev.T, dev.η, calc.occupation_method, calc.omega_cutoff,
        nbandkq, nbandk, nmodes, nqc, nT; i0 = calc.tile_i0)
    calc
end

function ElectronPhonon.postprocess_calculator!(calc::BoltzmannCalculator{FT}; kwargs...) where {FT}
    # CPU path (dev === nothing): no device buffers to stream/free. This also covers the GPU no-batch
    # corner: if this rank owns no outer k (empty MPI slice / empty window) the OuterIterationBatch
    # begin never ran, so `dev` is still nothing and Sₒ stays the setup zeros.
    calc.dev === nothing && return calc
    # GPU path: Sₒ is kept device-resident, so stream it to the host here. (Sᵢ was already streamed
    # one tile per outer-k batch in the OuterIterationBatch end bracket.)
    Sₒ_host = Array(calc.dev.Sₒ)        # (n_i, nT)
    @inbounds for iT in 1:length(calc.occ)
        @views calc.Sₒ[iT] .= Sₒ_host[:, iT]
    end
    # Free device buffers so the calc can be reused.
    calc.dev = nothing
    calc.Sᵢ_tile_host = zeros(FT, 0, 0, 0)
    calc
end
