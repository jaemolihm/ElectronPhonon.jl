# GPUBoltzmannCalculator: an AbstractCalculator that accumulates the BTE scattering-out (Sₒ) and
# scattering-in (Sᵢ) matrices during a single pass of `run_eph_over_k_and_kq`, on the CPU
# (`run_calculator!`) or the GPU (`run_calculator_batched!`). It is the transport analogue of
# MigdalEliashberg's `EliashbergCalculator`: same BandStates / imap addressing and the same
# full-resident vs block-device-resident memory strategy, but instead of copying g2/ωq it folds
# the temperature-dependent occupation physics into Sₒ/Sᵢ via the shared `bte_scattering_increments`
# (so CPU and GPU compute identical scattering — see src/boltzmann/bte_scattering_core.jl).
#
# Output layout is what the transport solver (`solve_electron_bte` / `solve_thermoelectric_bte`)
# consumes unchanged:
#   Sₒ :: Vector{Vector}  — Sₒ[iT][i]      (inverse SERTA lifetime γ_{nk})
#   Sᵢ :: Vector{Matrix}  — Sᵢ[iT][i, f]   (scattering-in kernel)
#
# Supported configuration (asserted at setup): FermiDirac occupation + Gaussian smearing — the
# configuration `bte_scattering_increments` implements and the one used for transport.

export GPUBoltzmannCalculator

Base.@kwdef mutable struct GPUBoltzmannCalculator{FT} <: AbstractCalculator
    # --- Parameters ---
    const occ::ElectronOccupationParams
    const smearing::Vector{Tuple{Symbol, Float64}}        # one (type, η) per temperature
    const occupation_method::Symbol = :Method5            # :Method1 … :Method6
    const scattering_method::Symbol = :BTE                # :SERTA or :BTE (affects only the solver)
    const omega_cutoff::FT = FT(omega_acoustic)           # skip modes below this (acoustic at Γ)

    # Integer form of `occupation_method` (1..6) for the device-safe core; set at setup.
    method_int::Int = 5
    # Number of CPU thread-chunks (for the lazily-allocated CPU buffers); set at setup.
    nchunks::Int = 1
    rng_band::UnitRange{Int} = 1:0

    # --- State (BandStates) ---
    el_i::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing
    imap_el_i::Union{Nothing, OffsetMatrix{Int64, Matrix{Int64}}} = nothing
    el_f::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing
    imap_el_f::Union{Nothing, OffsetMatrix{Int64, Matrix{Int64}}} = nothing

    # --- Host outputs (solver-facing) ---
    Sₒ::Vector{Vector{FT}} = Vector{Vector{FT}}()         # per iT, length n_i
    Sᵢ::Vector{Matrix{FT}} = Vector{Matrix{FT}}()         # per iT, (n_i, n_f)

    # --- CPU-path thread buffers (run_calculator!) ---
    # Allocated lazily on the first `run_calculator!` (CPU loop only), so the GPU path never pays
    # for them (Sᵢ_buffer would be nchunks·nT·rng_band·n_f — prohibitive on production grids).
    Sₒ_buffer::Vector{Vector{OffsetVector{FT, Vector{FT}}}} = Vector{Vector{OffsetVector{FT, Vector{FT}}}}()
    Sᵢ_buffer::Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}} = Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}}()
    alloc_lock::ReentrantLock = ReentrantLock()

    # --- Device buffers (GPU batched path) ---
    Sₒ_dev::Any = nothing      # (n_i, nT) — small, always full-resident
    Sᵢ_dev::Any = nothing      # (n_i, n_f, nT) — full-resident path
    imap_i_dev::Any = nothing  # (nw, n_k)  physical-band → outer state index
    imap_f_dev::Any = nothing  # (nw, n_kq) physical-band → inner state index
    e_i_dev::Any = nothing     # (n_i) outer energies
    e_f_dev::Any = nothing     # (n_f) inner energies
    wq_dev::Any = nothing      # (n_kq) inner k+q weights
    μ_dev::Any = nothing       # (nT)
    T_dev::Any = nothing       # (nT)
    η_dev::Any = nothing       # (nT)

    # --- Block-device-resident path (Sᵢ tiled over outer k) ---
    gpu_block::Bool = false
    residency_decided::Bool = false
    Sᵢ_tile_dev::Any = nothing            # (ni_cap, n_f, nT)
    Sᵢ_tile_host::Array{FT, 3} = zeros(FT, 0, 0, 0)
    tile_i0::Int = 0
    tile_ni::Int = 0
end

ElectronPhonon.allow_eph_outer_k(::GPUBoltzmannCalculator) = true
ElectronPhonon.allow_eph_batched(::GPUBoltzmannCalculator) = true

_method_to_int(m::Symbol) = begin
    s = String(m)
    startswith(s, "Method") || throw(ArgumentError("occupation_method must be :Method1…:Method6, got $m"))
    parse(Int, s[7:end])
end

function ElectronPhonon.setup_calculator!(calc::GPUBoltzmannCalculator{FT}, kpts, qpts, el_states;
        el_states_kq, kqpts, nelec_below_window_k, nchunks_threads, rng_band, kwargs...) where {FT}
    mpi_isroot() && println("Setting up GPUBoltzmannCalculator")
    calc.scattering_method === :MRTA &&
        throw(ArgumentError("scattering_method :MRTA not implemented"))
    calc.occ.occ_type === :FermiDirac ||
        throw(ArgumentError("GPUBoltzmannCalculator supports occ_type = :FermiDirac only (got $(calc.occ.occ_type))"))
    all(s -> s[1] === :Gaussian, calc.smearing) ||
        throw(ArgumentError("GPUBoltzmannCalculator supports :Gaussian smearing only"))
    FT === Float64 ||
        @warn "GPUBoltzmannCalculator: FT=$FT; FP64 (Float64) is required for transport accuracy."
    calc.method_int = _method_to_int(calc.occupation_method)
    calc.nchunks = nchunks_threads
    calc.rng_band = rng_band

    # Chemical potential: computed on a temporary BTStates exactly as the CPU reference, so μ is
    # identical (bte_compute_μ! reads BTStates fields directly).
    if !chemical_potential_is_computed(calc.occ)
        el, _ = electron_states_to_BTStates(el_states, kpts, nelec_below_window_k)
        bte_compute_μ!(calc.occ, el; do_print=true)
    end

    calc.el_i, calc.imap_el_i = electron_states_to_BandStates(el_states, kpts)
    calc.el_f, calc.imap_el_f = electron_states_to_BandStates(el_states_kq, kqpts)
    n_i = calc.el_i.n
    n_f = calc.el_f.n
    nT = length(calc.occ)

    calc.Sₒ = [zeros(FT, n_i) for _ in 1:nT]
    calc.Sᵢ = [zeros(FT, n_i, n_f) for _ in 1:nT]
    # Clear CPU thread buffers from any previous run: their size depends on rng_band / el_f.n, and
    # the lazy allocation in `_ensure_cpu_buffers!` keys on emptiness, so a reused calculator on a
    # different grid must start empty here to re-allocate correctly.
    calc.Sₒ_buffer = empty(calc.Sₒ_buffer)
    calc.Sᵢ_buffer = empty(calc.Sᵢ_buffer)
    calc
end

# --- CPU path -------------------------------------------------------------------------------
# Lazily allocate per-chunk thread buffers (CPU loop only). Single allocation guarded by a lock;
# concurrent first calls (the @threads inner loop) double-check under the lock.
function _ensure_cpu_buffers!(calc::GPUBoltzmannCalculator{FT}) where {FT}
    # Gate on Sᵢ_buffer (the LAST field published below): any thread that sees it non-empty is
    # guaranteed Sₒ_buffer was published first, so the unlocked fast path never observes a
    # half-built state (Sₒ set but Sᵢ still empty). Each buffer is built fully into a local and
    # published with a single field assignment (no resized-but-unfilled `#undef` intermediate).
    isempty(calc.Sᵢ_buffer) || return calc
    lock(calc.alloc_lock) do
        isempty(calc.Sᵢ_buffer) || return
        nT = length(calc.occ); rb = calc.rng_band; n_f = calc.el_f.n
        Sₒ = [[OffsetArray(zeros(FT, length(rb)), rb) for _ in 1:nT] for _ in 1:calc.nchunks]
        Sᵢ = [[OffsetArray(zeros(FT, length(rb), n_f), rb, :) for _ in 1:nT] for _ in 1:calc.nchunks]
        calc.Sₒ_buffer = Sₒ   # publish first
        calc.Sᵢ_buffer = Sᵢ   # publish last → unblocks the fast path (release barrier on unlock)
    end
    calc
end

function ElectronPhonon.setup_calculator_inner!(calc::GPUBoltzmannCalculator; kwargs...)
    # Zero the CPU buffers for the new outer k (no-op for the GPU path, where they are unallocated).
    for c in eachindex(calc.Sₒ_buffer)
        for x in calc.Sₒ_buffer[c]; x .= 0; end
        for x in calc.Sᵢ_buffer[c]; x .= 0; end
    end
    calc
end

function ElectronPhonon.run_calculator!(calc::GPUBoltzmannCalculator{FT}, epdata, ik, iq, ikq;
        id_chunk, kwargs...) where {FT}
    _ensure_cpu_buffers!(calc)
    Sₒ = calc.Sₒ_buffer[id_chunk]
    Sᵢ = calc.Sᵢ_buffer[id_chunk]
    (; el_k, el_kq, ph, wtq) = epdata
    method = calc.method_int
    @inbounds for n in el_k.rng
        ek = el_k.e[n]
        for m in el_kq.rng
            ind_el_f = calc.imap_el_f[m, ikq]
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

function ElectronPhonon.postprocess_calculator_inner!(calc::GPUBoltzmannCalculator; ik, kwargs...)
    # Reduce the per-chunk CPU buffers into the global Sₒ/Sᵢ (no-op for the GPU path).
    isempty(calc.Sₒ_buffer) && return calc
    @inbounds @views for n in axes(calc.imap_el_i, 1)
        ind_el_i = calc.imap_el_i[n, ik]
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
# (`_imap_to_device_bte` / `_tile_i_range_bte` mirror EliashbergCalculator's; defined here to keep
# this calculator self-contained within ElectronPhonon.)
function _imap_to_device_bte(proto, imap, nphys::Integer)
    host = zeros(Int, nphys, size(imap, 2))
    ilo, ihi = first(axes(imap, 1)), last(axes(imap, 1))
    @views host[ilo:ihi, :] .= parent(imap)
    copyto!(similar(proto, Int, nphys, size(imap, 2)), host)
end

# Global-i range [i0+1, i0+ni] of the outer states in the k-tile `kstart:kend`. Outer states are
# enumerated in k order, so a contiguous k-tile maps to a contiguous i-block (asserted).
function _tile_i_range_bte(calc::GPUBoltzmannCalculator, kstart::Integer, kend::Integer)
    imap = calc.imap_el_i
    imin = typemax(Int); imax = 0; count = 0
    @inbounds for ik in kstart:kend, b in axes(imap, 1)
        v = imap[b, ik]
        v > 0 || continue
        imin = min(imin, v); imax = max(imax, v); count += 1
    end
    count == 0 && return (0, 0)
    ni = imax - imin + 1
    ni == count || error("GPUBoltzmannCalculator block path: outer-k tile $kstart:$kend maps to a " *
                         "non-contiguous i-range ($count states span $ni rows).")
    (imin - 1, ni)
end

function ElectronPhonon.setup_calculator_outer_batch!(calc::GPUBoltzmannCalculator{FT};
        kstart, kend, proto, kwargs...) where {FT}
    n_i, n_f, nT = calc.el_i.n, calc.el_f.n, length(calc.occ)
    if !calc.residency_decided && calc.Sᵢ_dev === nothing
        # Full device-resident Sᵢ = n_i·n_f·nT·sizeof(FT). With headroom for the per-chunk g2
        # transient + the D2H host copy, fall back to the block path if it would not fit.
        Sᵢ_bytes = sizeof(FT) * n_i * n_f * nT
        if 1.5 * Sᵢ_bytes > ElectronPhonon.device_free_bytes(proto)
            calc.gpu_block = true
            @info "GPUBoltzmannCalculator: full device-resident Sᵢ ($(round(Sᵢ_bytes/1e9, digits=2)) GB) " *
                  "would not fit; using block-device-resident (per-tile D2H)."
        end
        calc.residency_decided = true
    end
    calc.gpu_block || return calc
    i0, ni = _tile_i_range_bte(calc, kstart, kend)
    calc.tile_i0 = i0; calc.tile_ni = ni
    if calc.Sᵢ_tile_dev === nothing || size(calc.Sᵢ_tile_dev, 1) < ni
        calc.Sᵢ_tile_dev = similar(proto, FT, ni, n_f, nT)
        calc.Sᵢ_tile_host = Array{FT, 3}(undef, ni, n_f, nT)
    end
    fill!(view(calc.Sᵢ_tile_dev, 1:ni, :, :), zero(FT))
    calc
end

function ElectronPhonon.flush_calculator_outer_batch!(calc::GPUBoltzmannCalculator; kwargs...)
    calc.gpu_block || return calc
    ni = calc.tile_ni
    ni == 0 && return calc
    i0 = calc.tile_i0
    copyto!(calc.Sᵢ_tile_host, calc.Sᵢ_tile_dev)
    @inbounds for iT in 1:length(calc.occ)
        @views calc.Sᵢ[iT][i0+1:i0+ni, :] .= calc.Sᵢ_tile_host[1:ni, :, iT]
    end
    calc
end

"""
    bte_window_scatter!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_col, imap_f, ikqs, e_i, e_f, wq,
                        μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nm, nqc, nT; i0=0)

Device-resident BTE scatter — the transport analogue of `eph_window_scatter!`, BTE-specific so it
lives with its sole caller (`run_calculator_batched!` below). For every `(m, n, j)` of the chunk
look up the outer/inner states `i = imap_i_col[n]`, `f = imap_f[m, ikqs[j]]` (skip if either is
out-of-window, `== 0`), then for each temperature `iocc` sum the shared per-mode physics
(`bte_scattering_increments`) over the `nm` phonon modes (`ωqmat[ν,j] ≥ ω_cutoff`) and:

  * `Sₒ_out[i, iocc] += Σ_ν sₒ`      — scattering-out, **accumulated** over `(m, ν, j)` (many `(m,j)`
    map to the same outer `i`), so the device method uses an atomic add here. `Sₒ` is small
    (`n_i × nT`) and always full-resident, so it is indexed by the GLOBAL outer state `i`;
  * `Sᵢ_out[i−i0, f, iocc] = Σ_ν sᵢ` — scattering-in; each `(i, f)` pair is produced by a unique
    `(n, m, j)` across the whole run (distinct k → distinct i, distinct k+q → distinct f), so this
    is a collision-free plain write (matching the no-atomics insight of `eph_window_scatter!`).
    `Sᵢ` is the large object that may be block-tiled, hence the `i − i0` (tile-local) row.

`i0` is the global-i offset of the block-tiled `Sᵢ` buffer (`i0 = 0` for the full-resident buffer).
Generic (CPU/fallback) method; the CUDA extension provides the `CuArray` kernel. The physics lives
entirely in `bte_scattering_increments` so the two paths agree.
"""
function bte_window_scatter!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_col, imap_f, ikqs,
        e_i, e_f, wq, μs, Ts, ηs, method::Int, ω_cutoff,
        nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, nT::Int; i0::Int = 0)
    @inbounds for j in 1:nqc, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        i > 0 || continue
        ikq = ikqs[j]
        f = imap_f[m, ikq]
        f > 0 || continue
        ek = e_i[i]; ekq = e_f[f]; wtq = wq[ikq]
        il = i - i0
        for iocc in 1:nT
            μ = μs[iocc]; T = Ts[iocc]; η = ηs[iocc]
            sₒ = zero(eltype(Sₒ_out)); sᵢ = sₒ
            for ν in 1:nm
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

function ElectronPhonon.run_calculator_batched!(calc::GPUBoltzmannCalculator{FT},
        ep_kq, ωq, ik, ikqs; g2=nothing, ibandk_offset=0) where {FT}
    nbandkq, nbandk, nm, nqc = size(ep_kq)
    n_i, n_f, nT = calc.el_i.n, calc.el_f.n, length(calc.occ)

    if calc.imap_i_dev === nothing
        # imap_i spans ALL physical bands (nphys = nw = nbandkq, the un-projected k+q band count):
        # under the loop's k-side window projection the ep_kq band axis covers only nbandk bands
        # starting at physical band ibandk_offset+1, addressed below by a shifted view (ibandk_offset+nbandk ≤ nw
        # by construction). Full-band runs have ibandk_offset = 0, nbandk = nw — identical to before.
        calc.imap_i_dev = _imap_to_device_bte(ep_kq, calc.imap_el_i, nbandkq)
        calc.imap_f_dev = _imap_to_device_bte(ep_kq, calc.imap_el_f, nbandkq)
        calc.e_i_dev = copyto!(similar(ep_kq, FT, n_i), calc.el_i.e)
        calc.e_f_dev = copyto!(similar(ep_kq, FT, n_f), calc.el_f.e)
        calc.wq_dev  = copyto!(similar(ep_kq, FT, calc.el_f.kpts.n), calc.el_f.kpts.weights)
        calc.μ_dev = copyto!(similar(ep_kq, FT, nT), collect(FT, calc.occ.μlist))
        calc.T_dev = copyto!(similar(ep_kq, FT, nT), collect(FT, calc.occ.Tlist))
        calc.η_dev = copyto!(similar(ep_kq, FT, nT), FT[s[2] for s in calc.smearing])
        calc.Sₒ_dev = fill!(similar(ep_kq, FT, n_i, nT), zero(FT))   # small, always resident
    end

    g2vals = g2 === nothing ? abs2.(ep_kq) ./ (2 .* reshape(ωq, 1, 1, nm, nqc)) : g2
    # Shifted by the k-side projection offset: ep_kq band n ↔ physical band ibandk_offset + n.
    imap_i_col = view(calc.imap_i_dev, ibandk_offset+1:ibandk_offset+nbandk, ik)

    if calc.gpu_block
        ElectronPhonon.bte_window_scatter!(calc.Sₒ_dev, calc.Sᵢ_tile_dev, g2vals, ωq,
            imap_i_col, calc.imap_f_dev, ikqs, calc.e_i_dev, calc.e_f_dev, calc.wq_dev,
            calc.μ_dev, calc.T_dev, calc.η_dev, calc.method_int, calc.omega_cutoff,
            nbandkq, nbandk, nm, nqc, nT; i0 = calc.tile_i0)
        return calc
    end

    # Full device-resident path
    if calc.Sᵢ_dev === nothing
        calc.Sᵢ_dev = fill!(similar(ep_kq, FT, n_i, n_f, nT), zero(FT))
    end
    ElectronPhonon.bte_window_scatter!(calc.Sₒ_dev, calc.Sᵢ_dev, g2vals, ωq,
        imap_i_col, calc.imap_f_dev, ikqs, calc.e_i_dev, calc.e_f_dev, calc.wq_dev,
        calc.μ_dev, calc.T_dev, calc.η_dev, calc.method_int, calc.omega_cutoff,
        nbandkq, nbandk, nm, nqc, nT; i0 = 0)
    calc
end

function ElectronPhonon.postprocess_calculator!(calc::GPUBoltzmannCalculator{FT}; kwargs...) where {FT}
    # GPU path: D2H Sₒ (always resident) and Sᵢ (full-resident; the block path already D2H'd Sᵢ
    # per tile in flush_calculator_outer_batch!). CPU path: nothing device-side to do.
    if calc.Sₒ_dev !== nothing
        Sₒ_host = Array(calc.Sₒ_dev)        # (n_i, nT)
        @inbounds for iT in 1:length(calc.occ)
            @views calc.Sₒ[iT] .= Sₒ_host[:, iT]
        end
        calc.Sₒ_dev = nothing
    end
    if calc.Sᵢ_dev !== nothing
        Sᵢ_host = Array(calc.Sᵢ_dev)        # (n_i, n_f, nT)
        @inbounds for iT in 1:length(calc.occ)
            @views calc.Sᵢ[iT] .= Sᵢ_host[:, :, iT]
        end
        calc.Sᵢ_dev = nothing
    end
    # Free device buffers / reset residency decision so the calc can be reused.
    calc.Sᵢ_tile_dev = nothing
    calc.Sᵢ_tile_host = zeros(FT, 0, 0, 0)
    calc.imap_i_dev = nothing; calc.imap_f_dev = nothing
    calc.e_i_dev = nothing; calc.e_f_dev = nothing; calc.wq_dev = nothing
    calc.μ_dev = nothing; calc.T_dev = nothing; calc.η_dev = nothing
    calc.gpu_block = false
    calc.residency_decided = false
    calc
end
