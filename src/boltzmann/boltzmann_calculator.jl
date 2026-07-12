# BoltzmannCalculator: an AbstractCalculator that accumulates the BTE scattering-out (Sₒ) and
# scattering-in (Sᵢ) matrices during a single pass of `run_eph_over_k_and_kq`. It uses BandStates /
# imap addressing; rather than copying g2/ωq it folds the temperature-dependent occupation physics
# into Sₒ/Sᵢ via the shared `bte_scattering_increments` (see src/boltzmann/bte_scattering_core.jl).
#
# One calculator, both backends: the same calculator instance is used on CPU or GPU — the backend is
# chosen by `run_eph_over_k_and_kq`'s `use_gpu`, which dispatches to `run_calculator!` (host loop,
# per (ik,iq,ikq)) or `run_calculator_outer_k_batched!` (device, per k-batch). Both fold the identical
# `bte_scattering_increments`, so they compute the same scattering (to round-off); the CPU path also
# serves as the validation reference for the GPU path.
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
# streamed to the host (setup/flush_calculator_outer_batch!), so only one tile (≈ one k-batch of rows)
# is device-resident. This bounds device memory to the tile regardless of grid size at no measurable
# speed cost: streaming is within ~2% of a single whole-Sᵢ copy even at 1.1 GB Sᵢ, because the D2H
# bytes moved are identical either way. There is deliberately no full-device-resident Sᵢ path. (Sₒ is
# small — n_i·nT — and stays device-resident, streamed once at the end.) See benchmark/README.md for
# the profile (and why the scatter kernel is NOT a negligible fraction of GPU time).

export BoltzmannCalculator

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
    # device index maps built in setup_calculator_outer_batch!. (0 = not set yet.)
    nw::Int = 0
    # Backend selected by run_eph_over_k_and_kq (set at setup): false = CPU (run_calculator!),
    # true = GPU (run_calculator_outer_k_batched!). Each per-path hook errors if called on the wrong backend.
    use_gpu::Bool = false

    # --- State (BandStates) --- the (iband, ik) → state reverse map is `el_*.indmap` (via state_index)
    el_i::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing
    el_f::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing

    # --- Host outputs (solver-facing) ---
    Sₒ::Vector{Vector{FT}} = Vector{Vector{FT}}()         # per iT, length n_i
    Sᵢ::Vector{Matrix{FT}} = Vector{Matrix{FT}}()         # per iT, (n_i, n_f)

    # --- CPU-path thread buffers (run_calculator!) ---
    # Allocated at setup when use_gpu = false; the GPU path never allocates them (Sᵢ_buffer would be
    # nchunks·nT·rng_band·n_f — prohibitive on production grids).
    Sₒ_buffer::Vector{Vector{OffsetVector{FT, Vector{FT}}}} = Vector{Vector{OffsetVector{FT, Vector{FT}}}}()
    Sᵢ_buffer::Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}} = Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}}()

    # --- Device buffers (GPU batched path) ---
    Sₒ_dev::Any = nothing      # (n_i, nT) — small, kept device-resident
    imap_i_dev::Any = nothing  # (nw, n_k)  physical-band → outer state index
    imap_f_dev::Any = nothing  # (nw, n_kq) physical-band → inner state index
    e_i_dev::Any = nothing     # (n_i) outer energies
    e_f_dev::Any = nothing     # (n_f) inner energies
    wq_dev::Any = nothing      # (n_kq) inner k+q weights
    μ_dev::Any = nothing       # (nT)
    T_dev::Any = nothing       # (nT)
    η_dev::Any = nothing       # (nT)

    # --- Sᵢ tile (streamed to the host per outer-k batch) ---
    # Sᵢ is never held whole on the device: each outer-k batch fills a tile and streams it to the host
    # (setup/flush_calculator_outer_batch!), so only one tile is device-resident at a time.
    Sᵢ_tile_dev::Any = nothing            # (ni_cap, n_f, nT)
    Sᵢ_tile_host::Array{FT, 3} = zeros(FT, 0, 0, 0)
    tile_i0::Int = 0
    tile_ni::Int = 0
end

ElectronPhonon.allow_eph_outer_k(::BoltzmannCalculator) = true
ElectronPhonon.allow_eph_outer_k_batched(::BoltzmannCalculator) = true

function ElectronPhonon.setup_calculator!(calc::BoltzmannCalculator{FT}, kpts, qpts, el_states;
        el_states_kq, kqpts, nelec_below_window_k, nelec_below_window_kq, nchunks_threads, rng_band,
        nw, use_gpu = false, kwargs...) where {FT}
    mpi_isroot() && println("Setting up BoltzmannCalculator")
    calc.use_gpu = use_gpu
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
    calc.el_i = electron_states_to_BandStates(el_states, gkpts, nelec_below_window_k)
    calc.el_f = electron_states_to_BandStates(el_states_kq, gkqpts, nelec_below_window_kq)
    n_i = calc.el_i.n
    n_f = calc.el_f.n
    nT = length(calc.occ)

    calc.Sₒ = [zeros(FT, n_i) for _ in 1:nT]
    calc.Sᵢ = [zeros(FT, n_i, n_f) for _ in 1:nT]
    # CPU path only: allocate the per-chunk thread buffers now (this runs once, single-threaded, so
    # no lock is needed). The GPU path leaves them empty (its device buffers are allocated lazily in
    # the batched hooks). Reallocated on every setup, so a reused calculator on a different grid gets
    # correctly-sized buffers.
    if use_gpu
        calc.Sₒ_buffer = empty(calc.Sₒ_buffer)
        calc.Sᵢ_buffer = empty(calc.Sᵢ_buffer)
    else
        rb = calc.rng_band
        calc.Sₒ_buffer = [[OffsetArray(zeros(FT, length(rb)), rb) for _ in 1:nT] for _ in 1:calc.nchunks]
        calc.Sᵢ_buffer = [[OffsetArray(zeros(FT, length(rb), n_f), rb, :) for _ in 1:nT] for _ in 1:calc.nchunks]
    end
    calc
end

# --- CPU (non-batched) path -----------------------------------------------------------------

# CPU path: zero the per-chunk buffers for the new outer k. No-op on the GPU path.
function ElectronPhonon.setup_calculator_inner!(calc::BoltzmannCalculator; kwargs...)
    calc.use_gpu && return calc
    for c in eachindex(calc.Sₒ_buffer)
        for x in calc.Sₒ_buffer[c]; x .= 0; end
        for x in calc.Sᵢ_buffer[c]; x .= 0; end
    end
    calc
end

# CPU path: called per (ik, iq, ikq) by the host e-ph loop; accumulates into the per-chunk thread
# buffers, reduced into Sₒ/Sᵢ by postprocess_calculator_inner!.
function ElectronPhonon.run_calculator!(calc::BoltzmannCalculator{FT}, epdata, ik, iq, ikq;
        id_chunk, kwargs...) where {FT}
    calc.use_gpu && error("run_calculator! is a CPU-only hook (use_gpu=true)")
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

# CPU path: reduce the per-chunk buffers into the global Sₒ/Sᵢ. No-op on the GPU path.
function ElectronPhonon.postprocess_calculator_inner!(calc::BoltzmannCalculator; ik, kwargs...)
    calc.use_gpu && return calc
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

# --- GPU batched path -----------------------------------------------------------------------

# GPU (batched) path: called by the GPU e-ph loop once per outer-k batch, before its k iterations.
# `proto` is a device array (the e-ph matrix) used as the allocation prototype. On the first batch it
# also builds the run's one-time device buffers (they need a device array, which `setup_calculator!`
# does not have); every batch it (re)points/zeros the Sᵢ tile.
#
# TODO: have run_eph_over_k_and_kq hand setup_calculator! a GPU array prototype (or the backend
# array type) so the one-time device init below can move into setup_calculator! and out of this
# per-batch hook entirely.
function ElectronPhonon.setup_calculator_outer_batch!(calc::BoltzmannCalculator{FT};
        kstart, kend, proto, kwargs...) where {FT}
    calc.use_gpu || error("setup_calculator_outer_batch! is a GPU-only hook")
    n_i, n_f, nT = calc.el_i.n, calc.el_f.n, length(calc.occ)

    # One-time device copies, built on the first batch and reused across all k in the run. `nw` (the
    # full Wannier band count, from setup) sizes the physical-band index maps; see `_indmap_to_device`.
    if calc.imap_i_dev === nothing
        calc.imap_i_dev = _indmap_to_device(proto, calc.el_i, calc.nw)
        calc.imap_f_dev = _indmap_to_device(proto, calc.el_f, calc.nw)
        calc.e_i_dev = copyto!(similar(proto, FT, n_i), calc.el_i.es)
        calc.e_f_dev = copyto!(similar(proto, FT, n_f), calc.el_f.es)
        calc.wq_dev  = copyto!(similar(proto, FT, calc.el_f.kpts.n), calc.el_f.kpts.weights)
        calc.μ_dev = copyto!(similar(proto, FT, nT), collect(FT, calc.occ.μlist))
        calc.T_dev = copyto!(similar(proto, FT, nT), collect(FT, calc.occ.Tlist))
        calc.η_dev = copyto!(similar(proto, FT, nT), FT[s[2] for s in calc.smearing])
        calc.Sₒ_dev = fill!(similar(proto, FT, n_i, nT), zero(FT))   # small, device-resident
    end

    # Sᵢ tile for this batch (re-sized/zeroed every batch — Sᵢ is streamed per tile).
    rng = ind_range_for_k_range(calc.el_i, kstart, kend)   # contiguous outer-state block for this k-batch
    calc.tile_i0 = first(rng) - 1; calc.tile_ni = length(rng)
    ni = calc.tile_ni
    if calc.Sᵢ_tile_dev === nothing || size(calc.Sᵢ_tile_dev, 1) < ni
        calc.Sᵢ_tile_dev = similar(proto, FT, ni, n_f, nT)
        calc.Sᵢ_tile_host = Array{FT, 3}(undef, ni, n_f, nT)
    end
    fill!(view(calc.Sᵢ_tile_dev, 1:ni, :, :), zero(FT))
    calc
end

# GPU (batched) path: stream this batch's Sᵢ tile from device to the host output (called by the GPU
# e-ph loop once per outer-k batch, after its k iterations).
function ElectronPhonon.flush_calculator_outer_batch!(calc::BoltzmannCalculator; kwargs...)
    calc.use_gpu || error("flush_calculator_outer_batch! is a GPU-only hook")
    if calc.tile_ni > 0
        ni = calc.tile_ni
        i0 = calc.tile_i0
        copyto!(calc.Sᵢ_tile_host, calc.Sᵢ_tile_dev)
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
device-resident work of `run_calculator_outer_k_batched!` (its sole caller, so it lives here). For every
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

function ElectronPhonon.run_calculator_outer_k_batched!(calc::BoltzmannCalculator{FT},
        ep_kq, ωq, ik, ikqs; g2=nothing, ibandk_offset=0) where {FT}
    calc.use_gpu || error("run_calculator_outer_k_batched! is a GPU-only hook")
    nbandkq, nbandk, nmodes, nqc = size(ep_kq)
    nT = length(calc.occ)
    # Device buffers (imap/energies/Sₒ) were built once in setup_calculator_outer_batch!.

    # The loop always folds g2 = |ep|²/(2ω) in get_eph_kR_to_kq_batched! and passes it here, so g2 is
    # never nothing at the call site; require it rather than recomputing.
    g2 === nothing && error("run_calculator_outer_k_batched!(BoltzmannCalculator) requires g2 from the loop")
    g2vals = g2
    # k-side window projection: ep_kq's band-n axis covers nbandk bands starting at physical band
    # ibandk_offset+1, so shift into the physical-band imap_i by that offset (full-band runs have
    # ibandk_offset = 0, nbandk = nw). The k+q axis (m ∈ 1:nbandkq = nw) is not projected.
    imap_i_at_k = view(calc.imap_i_dev, ibandk_offset+1:ibandk_offset+nbandk, ik)

    # Scatter into this batch's Sᵢ tile (streamed to the host by flush_calculator_outer_batch!).
    ElectronPhonon.bte_window_accumulate!(calc.Sₒ_dev, calc.Sᵢ_tile_dev, g2vals, ωq,
        imap_i_at_k, calc.imap_f_dev, ikqs, calc.e_i_dev, calc.e_f_dev, calc.wq_dev,
        calc.μ_dev, calc.T_dev, calc.η_dev, calc.occupation_method, calc.omega_cutoff,
        nbandkq, nbandk, nmodes, nqc, nT; i0 = calc.tile_i0)
    calc
end

function ElectronPhonon.postprocess_calculator!(calc::BoltzmannCalculator{FT}; kwargs...) where {FT}
    calc.use_gpu || return calc   # CPU path: no device buffers to stream/free
    # GPU path: Sₒ is kept device-resident, so stream it to the host here. (Sᵢ was already streamed
    # one tile per outer-k batch in flush_calculator_outer_batch!.) The `!== nothing` guard covers
    # the no-batch corner: if this rank owns no outer k (empty MPI slice / empty window)
    # run_calculator_outer_k_batched! never ran, so Sₒ_dev is still unset.
    if calc.Sₒ_dev !== nothing
        Sₒ_host = Array(calc.Sₒ_dev)        # (n_i, nT)
        @inbounds for iT in 1:length(calc.occ)
            @views calc.Sₒ[iT] .= Sₒ_host[:, iT]
        end
        calc.Sₒ_dev = nothing
    end
    # Free device buffers so the calc can be reused.
    calc.Sᵢ_tile_dev = nothing
    calc.Sᵢ_tile_host = zeros(FT, 0, 0, 0)
    calc.imap_i_dev = nothing; calc.imap_f_dev = nothing
    calc.e_i_dev = nothing; calc.e_f_dev = nothing; calc.wq_dev = nothing
    calc.μ_dev = nothing; calc.T_dev = nothing; calc.η_dev = nothing
    calc
end
