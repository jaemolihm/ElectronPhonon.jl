# BoltzmannCalculator: an AbstractCalculator that accumulates the BTE scattering-out (Sₒ) and
# scattering-in (Sᵢ) matrices during a single pass of `run_eph_over_k_and_kq`. It uses BandStates /
# imap addressing; rather than copying g2/ωq it folds the temperature-dependent occupation physics
# into Sₒ/Sᵢ via the shared `bte_scattering_increments` (see src/boltzmann/bte_scattering_core.jl).
#
# One calculator, both backends: the same calculator instance is used on CPU or GPU — the backend is
# chosen by `run_eph_over_k_and_kq`'s `use_gpu`, which dispatches `run_calculator!` on the payload
# type: `EPData` (host loop, per (ik,iq,ikq)) or `EPDataQBatched` (device, per k-batch).
# Both fold the identical `bte_scattering_increments`, so they compute the same scattering (to
# round-off); the CPU path also serves as the validation reference for the GPU path.
#
# Output layout is what the transport solver (`solve_electron_bte` / `solve_thermoelectric_bte`)
# consumes unchanged:
#   Sₒ :: Vector{Vector}  — Sₒ[iT][i]      (inverse SERTA lifetime γ_{nk})
#   Sᵢ :: Vector{Matrix}  — Sᵢ[iT][i, f]   (scattering-in kernel)
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

# Naming note: two subscript systems coexist here. `Sₒ`/`Sᵢ` = scattering-OUT / scattering-IN (the
# subscript is out/in), whereas the `_i`/`_f` suffix (`imap_i`, `e_i`, `el_i`, `n_i`) = initial/outer
# (k) vs final/inner (k+q). The Latin `ᵢ` in `Sᵢ` looks like the `_i` suffix but means "in", not
# "initial".

# Device buffers for the GPU batched path, built once in the first `calculator_begin!(…,
# OuterIterationBatch(), ctx)` from `alloc(ctx.backend, …)`. Held behind `dev::Union{Nothing, …}` on
# the calculator; touched only at hook granularity (one kernel launch per call), so the function
# boundary keeps the hot code type-stable. The tiled Sᵢ output lives in `calc.tiled` (a
# `TiledDeviceOutput`); the energies/weights are gathered from the run's shared `ctx.el_k_stacks`.
struct BoltzmannDeviceBuffers{MT, MI, VT, ST}
    Sₒ       :: MT      # (n_i, nT)  — small, device-resident
    imap_i   :: MI      # (nw, n_k)  physical-band → outer state index
    imap_f   :: MI      # (nw, n_kq) physical-band → inner state index
    e_i      :: VT      # (n_i,) outer energies
    e_f      :: VT      # (n_f,) inner energies
    wq       :: VT      # (n_kq,) inner k+q weights
    μ        :: VT      # (nT,)
    T        :: VT      # (nT,)
    smearing :: ST      # (nT,)
end

Base.@kwdef mutable struct BoltzmannCalculator{FT} <: AbstractCalculator
    # --- Parameters ---
    const occ::ElectronOccupationParams
    const smearing_list::Vector{SmearingType{FT}}        # One per temperature
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

    # --- Tiled Sᵢ device output (GPU batched path) ---
    # Sᵢ is never held whole on the device: `TiledDeviceOutput` (always block mode) keeps one outer-k
    # tile (i-extent = the largest k-batch) resident and streams it to `calc.Sᵢ` per batch. Built at
    # setup; device buffers allocated lazily on the first batch.
    tiled::Union{Nothing, TiledDeviceOutput{FT}} = nothing

    # Set by `postprocess_calculator!`; `setup_calculator!` errors if already `true`. A calculator
    # instance is single-use — reconstruct it rather than re-running it on a new grid.
    done::Bool = false
end

ElectronPhonon.supports(::BoltzmannCalculator, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::BoltzmannCalculator, ::Type{EPData}) = true
ElectronPhonon.supports(::BoltzmannCalculator, ::Type{EPDataQBatched}) = true

# Consume the run's shared device stacks: outer/inner band energies and inner k+q weights (the
# `bte_window_accumulate!` scatter reads e_i[i], e_f[f], wq[ikq]). The loop builds them once and the
# begin bracket gathers the per-state views, so the calculator no longer hand-uploads them.
ElectronPhonon.required_el_k_device_stacks(::BoltzmannCalculator) = [:e_k, :e_kq, :wtq]

function ElectronPhonon.setup_calculator!(calc::BoltzmannCalculator{FT}, kpts, qpts, el_states;
        el_states_kq, kqpts, nelec_below_window_k, nelec_below_window_kq, nchunks_threads, rng_band,
        nw, kwargs...) where {FT}
    mpi_isroot() && println("Setting up BoltzmannCalculator")
    calc.done &&
        throw(ArgumentError("this BoltzmannCalculator has already been run; reconstruct the " *
                            "calculator, reuse is not supported"))
    calc.scattering_method === :MRTA &&
        throw(ArgumentError("scattering_method :MRTA not implemented"))
    calc.occ.occ_type === :FermiDirac ||
        throw(ArgumentError("BoltzmannCalculator supports occ_type = :FermiDirac only (got $(calc.occ.occ_type))"))
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
    calc.el_i, _ = electron_states_to_BandStates(el_states, gkpts, nelec_below_window_k; nw)
    calc.el_f, _ = electron_states_to_BandStates(el_states_kq, gkqpts, nelec_below_window_kq; nw)
    n_i = calc.el_i.n
    n_f = calc.el_f.n
    nT = length(calc.occ)

    calc.Sₒ = [zeros(FT, n_i) for _ in 1:nT]
    calc.Sᵢ = [zeros(FT, n_i, n_f) for _ in 1:nT]
    # Tiled Sᵢ device output: shape (n_i, n_f, nT), tiled over the outer-k state axis (axis 1), always
    # block mode (there is deliberately no full-device-resident Sᵢ path). Metadata only at setup; the
    # device/host tile buffers are lazy in `tile_begin!` (GPU only, first batch), so this is cheap on
    # the CPU path where `tile_begin!` never runs.
    calc.tiled = TiledDeviceOutput{FT}((n_i, n_f, nT), 1, calc.el_i; narr = 1, force_block = true)
    calc
end

# --- CPU (non-batched, EPData) path --------------------------------------------------

# CPU path: allocate the per-chunk thread buffers on the first outer iteration (this runs once,
# single-threaded, so no lock is needed), then zero them for the new outer k. Dispatched on
# `SingleMode`; in the batched loop (`BatchedMode`) the default no-op runs (the batched loop fires
# per-k OuterIteration brackets too, but the device buffers live in OuterIterationBatch).
function ElectronPhonon.calculator_begin!(calc::BoltzmannCalculator{FT}, ::OuterIteration,
        ctx::LoopContext{CPUBackend, SingleMode}) where {FT}
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
# buffers, reduced into Sₒ/Sᵢ by the OuterIteration end bracket. The `(m, n, iocc, imode)` loop below
# mirrors, term for term (`ek`/`ekq`/`ωq`/`so`/`si`/`sₒ`/`sᵢ`), the device work in
# `bte_window_accumulate!` (the EPDataQBatched path); both fold the identical `bte_scattering_increments`.
function ElectronPhonon.run_calculator!(calc::BoltzmannCalculator{FT}, p::EPData, ctx::LoopContext{CPUBackend, SingleMode}) where {FT}
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
            for (iT, (; μ, T)) in enumerate(calc.occ)
                smearing = calc.smearing_list[iT]
                sₒ = zero(FT); sᵢ = zero(FT)
                for imode in 1:ph.nmodes
                    ωq = ph.e[imode]
                    ωq < calc.omega_cutoff && continue
                    so, si = bte_scattering_increments(method, ek, ekq, ωq,
                        epdata.g2[m, n, imode], wtq, μ, T, smearing)
                    sₒ += so; sᵢ += si
                end
                Sₒ[iT][n] += sₒ
                Sᵢ[iT][n, ind_el_f] += sᵢ
            end
        end
    end
    calc
end

# CPU path: reduce the per-chunk buffers into the global Sₒ/Sᵢ. Dispatched on `SingleMode`; the
# batched loop (`BatchedMode`) runs the default no-op.
function ElectronPhonon.calculator_end!(calc::BoltzmannCalculator, ::OuterIteration,
        ctx::LoopContext{CPUBackend, SingleMode})
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

# --- GPU batched path (EPDataQBatched) -----------------------------------------------

# Flatten a per-(physical band, k) device energy grid `(nw, nk)` to the per-state device vector
# `(s.n,)` aligned with `BandStates` `s`, by gathering at each state's (iband, ik). The gather target
# indices are built on the host once (first batch) and uploaded; the device gather is vectorized (no
# scalar indexing). `e_grid[iband(state), ik(state)] == s.es[state]` by construction, so this
# reproduces the former direct upload of `s.es` exactly.
function _gather_state_energies(backend, e_grid, s::BandStates)
    nw = size(e_grid, 1)
    li = s.ibands .+ nw .* (s.iks .- 1)      # column-major linear index into (nw, nk)
    li_dev = copyto!(alloc(backend, Int, length(li)), li)
    vec(e_grid)[li_dev]
end

# Batched path: called by the GPU e-ph loop once per outer-k batch, before its k iterations.
# `ctx.backend` (a `GPUBackend`) allocates from the e-ph matrix backend. On the first batch it builds
# the run's one-time device buffers into `calc.dev` (a `BoltzmannDeviceBuffers`), sourcing the band
# energies / k+q weights from the run's shared `ctx.el_k_stacks`; every batch it records this batch's
# Sᵢ tile range and zeros the tile's active region (via `calc.tiled`). Dispatched on `BatchedMode`;
# the per-point (`SingleMode`) path runs the default no-op.
function ElectronPhonon.calculator_begin!(calc::BoltzmannCalculator{FT}, ::OuterIterationBatch,
        ctx::LoopContext{<:AbstractBackend, BatchedMode}) where {FT}
    backend = ctx.backend
    nT = length(calc.occ)

    # One-time device buffers, built on the first batch and reused across all k in the run. `nw` (the
    # full Wannier band count, from setup) sizes the physical-band index maps; see `_indmap_to_device`.
    if calc.dev === nothing
        stacks = ctx.el_k_stacks     # shared device stacks (this calculator declared :e_k/:e_kq/:wtq)
        calc.dev = BoltzmannDeviceBuffers(
            fill!(alloc(backend, FT, calc.el_i.n, nT), zero(FT)),               # Sₒ
            _indmap_to_device(backend, calc.el_i, calc.nw),                     # imap_i
            _indmap_to_device(backend, calc.el_f, calc.nw),                     # imap_f
            _gather_state_energies(backend, stacks.e_k, calc.el_i),             # e_i (from shared stack)
            _gather_state_energies(backend, stacks.e_kq, calc.el_f),           # e_f (from shared stack)
            stacks.wtq,                                                         # wq  (shared k+q weights)
            to_device(backend, collect(FT, calc.occ.μlist)),                    # μ
            to_device(backend, collect(FT, calc.occ.Tlist)),                    # T
            to_device(backend, calc.smearing_list),                             # smearing (one per T)
        )
    end

    # Sᵢ tile for this batch (block mode: zeroed and its range recorded by the helper).
    tile_begin!(calc.tiled, ctx)
    calc
end

# Batched path: stream this batch's Sᵢ tile from device to the host output (called once per
# outer-k batch, after its k iterations). Dispatched on `BatchedMode`.
function ElectronPhonon.calculator_end!(calc::BoltzmannCalculator, ::OuterIterationBatch,
        ctx::LoopContext{<:AbstractBackend, BatchedMode})
    t = calc.tiled
    ni = tile_length(t)
    if ni > 0
        i0 = tile_offset(t)
        tile_download!(t)             # contiguous device→host copy into the tile's host mirror
        host = host_array(t, 1)
        @inbounds for iT in 1:length(calc.occ)
            @views calc.Sᵢ[iT][i0+1:i0+ni, :] .= host[1:ni, :, iT]
        end
    end
    calc
end

"""
    bte_window_accumulate!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs, e_i, e_f, wq,
                           μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nmodes, nqc; i0=0)

Accumulate the BTE scattering-out (Sₒ) and scattering-in (Sᵢ) contributions of one e-ph chunk into
the (in-energy-window) device buffers — the transport analogue of `eph_window_scatter!`, and the
device-resident work of `run_calculator!(::BoltzmannCalculator, ::EPDataQBatched, ctx)` (its sole caller, so it lives here). For every
`(m, n, j)` of the chunk look up the outer/inner states `i = imap_i_at_k[n]`,
`f = imap_f[m, ikqs[j]]` (skip if either is out-of-window, `== 0`), then for each temperature
`iT` sum the shared per-mode physics (`bte_scattering_increments`) over the `nmodes` phonon modes
(`ωqmat[ν,j] ≥ ω_cutoff`) and:

  * `Sₒ_out[i, iT] += Σ_ν sₒ`      — scattering-out, added over `(m, ν, j)` (many `(m,j)` map to
    the same outer `i`), so the device method uses an atomic add here. `Sₒ` is small (`n_i × nT`)
    and device-resident, so it is indexed by the GLOBAL outer state `i`;
  * `Sᵢ_out[i−i0, f, iT] = Σ_ν sᵢ` — scattering-in; each `(i, f)` pair is produced by a unique
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
        nbandkq::Int, nbandk::Int, nmodes::Int, nqc::Int; i0::Int = 0)
    nT = length(μs)
    @inbounds for j in 1:nqc, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_at_k[n]
        i > 0 || continue
        ikq = ikqs[j]
        f = imap_f[m, ikq]
        f > 0 || continue
        ek = e_i[i]; ekq = e_f[f]; wtq = wq[ikq]
        il = i - i0
        for iT in 1:nT
            μ = μs[iT]; T = Ts[iT]; η = ηs[iT]
            sₒ = zero(eltype(Sₒ_out)); sᵢ = sₒ
            for ν in 1:nmodes
                ωq = ωqmat[ν, j]
                ωq < ω_cutoff && continue
                so, si = bte_scattering_increments(method, ek, ekq, ωq, g2vals[m, n, ν, j], wtq, μ, T, η)
                sₒ += so; sᵢ += si
            end
            Sₒ_out[i, iT] += sₒ
            Sᵢ_out[il, f, iT] = sᵢ
        end
    end
    nothing
end

# GPU path: called once per outer-k batch by the device e-ph loop; scatters into the device Sₒ and the
# current Sᵢ tile via `bte_window_accumulate!` (the device analogue of the CPU EPData loop above —
# same `bte_scattering_increments`). No per-chunk reduction: the scatter writes the global buffers.
function ElectronPhonon.run_calculator!(calc::BoltzmannCalculator{FT}, p::EPDataQBatched, ctx) where {FT}
    (; ep_kq, g2, ωq, ik, ikqs, ibandk_offset) = p
    dev = calc.dev
    nbandkq, nbandk, nmodes, nqc = size(ep_kq)
    # Device buffers (imap/energies/Sₒ in `dev`, the Sᵢ tile in `calc.tiled`) were built once in the
    # first OuterIterationBatch begin. `g2 = |ep|²/(2ω)` is folded by the loop's fused kernel (payload).

    # k-side window projection: ep_kq's band-n axis covers nbandk bands starting at physical band
    # ibandk_offset+1, so shift into the physical-band imap_i by that offset (full-band runs have
    # ibandk_offset = 0, nbandk = nw). The k+q axis (m ∈ 1:nbandkq = nw) is not projected.
    imap_i_at_k = view(dev.imap_i, ibandk_offset+1:ibandk_offset+nbandk, ik)

    # Scatter into this batch's Sᵢ tile (streamed to the host by the OuterIterationBatch end bracket).
    t = calc.tiled
    ElectronPhonon.bte_window_accumulate!(dev.Sₒ, device_array(t, 1), g2, ωq,
        imap_i_at_k, dev.imap_f, ikqs, dev.e_i, dev.e_f, dev.wq,
        dev.μ, dev.T, dev.smearing, calc.occupation_method, calc.omega_cutoff,
        nbandkq, nbandk, nmodes, nqc; i0 = tile_offset(t))
    calc
end

function ElectronPhonon.postprocess_calculator!(calc::BoltzmannCalculator{FT}; kwargs...) where {FT}
    calc.done = true    # single-use: `setup_calculator!` errors on a re-run
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
    # Free device buffers (the calc is single-use; `done` blocks a re-run in `setup_calculator!`).
    calc.dev = nothing
    calc.tiled === nothing || tile_free!(calc.tiled)
    calc
end
