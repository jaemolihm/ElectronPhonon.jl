# BoltzmannCalculator: an AbstractCalculator that accumulates the BTE scattering-out (Sₒ) and
# scattering-in (Sᵢ) matrices during a single pass of `run_eph_over_k_and_kq`. It uses BandStates /
# imap addressing and a full-resident vs block-device-resident memory strategy; rather than copying
# g2/ωq it folds the temperature-dependent occupation physics into Sₒ/Sᵢ via the shared
# `bte_scattering_increments` (see src/boltzmann/bte_scattering_core.jl).
#
# ONE calculator, both backends: the SAME calculator instance is used on CPU or GPU — the backend is
# chosen by `run_eph_over_k_and_kq`'s `use_gpu`, which dispatches to `run_calculator!` (host loop,
# per (ik,iq,ikq)) or `run_calculator_batched!` (device, per k-batch). Both fold the identical
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

export BoltzmannCalculator

Base.@kwdef mutable struct BoltzmannCalculator{FT} <: AbstractCalculator
    # --- Parameters ---
    const occ::ElectronOccupationParams
    const smearing::Vector{Tuple{Symbol, Float64}}        # one (type, η) per temperature
    # Occupation-factor convention, an integer 1..6 passed to the device-safe core; the six
    # conventions are defined in `bte_scattering_increments` (src/boltzmann/bte_scattering_core.jl).
    const occupation_method::Int = 5
    const scattering_method::Symbol = :BTE                # :SERTA or :BTE (affects only the solver)
    const omega_cutoff::FT = FT(omega_acoustic)           # skip modes below this (acoustic at Γ)

    # Number of CPU thread-chunks (for the CPU-path buffers); set at setup.
    nchunks::Int = 1
    rng_band::UnitRange{Int} = 1:0
    # Backend selected by run_eph_over_k_and_kq (set at setup): false = CPU (run_calculator!),
    # true = GPU (run_calculator_batched!). Each per-path hook asserts the expected value.
    use_gpu::Bool = false

    # --- State (BandStates) ---
    el_i::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing
    imap_el_i::Union{Nothing, OffsetMatrix{Int64, Matrix{Int64}}} = nothing
    el_f::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing
    imap_el_f::Union{Nothing, OffsetMatrix{Int64, Matrix{Int64}}} = nothing

    # --- Host outputs (solver-facing) ---
    Sₒ::Vector{Vector{FT}} = Vector{Vector{FT}}()         # per iT, length n_i
    Sᵢ::Vector{Matrix{FT}} = Vector{Matrix{FT}}()         # per iT, (n_i, n_f)

    # --- CPU-path thread buffers (run_calculator!) ---
    # Allocated at setup when use_gpu = false; the GPU path never allocates them (Sᵢ_buffer would be
    # nchunks·nT·rng_band·n_f — prohibitive on production grids).
    Sₒ_buffer::Vector{Vector{OffsetVector{FT, Vector{FT}}}} = Vector{Vector{OffsetVector{FT, Vector{FT}}}}()
    Sᵢ_buffer::Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}} = Vector{Vector{OffsetMatrix{FT, Matrix{FT}}}}()

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

ElectronPhonon.allow_eph_outer_k(::BoltzmannCalculator) = true
ElectronPhonon.allow_eph_batched(::BoltzmannCalculator) = true

function ElectronPhonon.setup_calculator!(calc::BoltzmannCalculator{FT}, kpts, qpts, el_states;
        el_states_kq, kqpts, nelec_below_window_k, nelec_below_window_kq, nchunks_threads, rng_band,
        use_gpu = false, kwargs...) where {FT}
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

    # Chemical potential: computed on a temporary BTStates exactly as the CPU reference, so μ is
    # identical (bte_compute_μ! reads BTStates fields directly).
    if !chemical_potential_is_computed(calc.occ)
        el, _ = electron_states_to_BTStates(el_states, kpts, nelec_below_window_k)
        bte_compute_μ!(calc.occ, el; do_print=true)
    end

    # electron_states_to_BandStates requires GridKpoints (its k-vector hash is needed by the loop);
    # the non-symmetry path hands us a plain Kpoints, so promote here.
    gkpts   = kpts isa GridKpoints   ? kpts   : GridKpoints(kpts)
    gkqpts  = kqpts isa GridKpoints  ? kqpts  : GridKpoints(kqpts)
    calc.el_i, calc.imap_el_i = electron_states_to_BandStates(el_states, gkpts, nelec_below_window_k)
    calc.el_f, calc.imap_el_f = electron_states_to_BandStates(el_states_kq, gkqpts, nelec_below_window_kq)
    n_i = calc.el_i.n
    n_f = calc.el_f.n
    nT = length(calc.occ)

    calc.Sₒ = [zeros(FT, n_i) for _ in 1:nT]
    calc.Sᵢ = [zeros(FT, n_i, n_f) for _ in 1:nT]
    # CPU path only: allocate the per-chunk thread buffers now (this runs once, single-threaded, so
    # no lock is needed). The GPU path leaves them empty. Reallocated on every setup, so a reused
    # calculator on a different grid gets correctly-sized buffers.
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
    @assert !calc.use_gpu "run_calculator! is the CPU path but the calculator was set up with use_gpu=true"
    Sₒ = calc.Sₒ_buffer[id_chunk]
    Sᵢ = calc.Sᵢ_buffer[id_chunk]
    (; el_k, el_kq, ph, wtq) = epdata
    method = calc.occupation_method
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

# CPU path: reduce the per-chunk buffers into the global Sₒ/Sᵢ. No-op on the GPU path.
function ElectronPhonon.postprocess_calculator_inner!(calc::BoltzmannCalculator; ik, kwargs...)
    calc.use_gpu && return calc
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

# Build a device copy of a window imap for the device scatter: `imap` is the OffsetMatrix
# `(band, k) → state` (band axis offset to the in-window bands); write it densely into rows
# `1:nphys` of a zero-filled `(nphys, n_k)` host array (out-of-window bands stay 0), then copy to a
# device array of the same element/backend type as `proto` (a device array used only as a
# `similar` prototype). `nphys` is the physical band count the loop indexes by (the un-projected
# k+q band count `nw`), so the calculator can address the device imap by physical band.
function _imap_to_device_bte(proto, imap, nphys::Integer)
    host = zeros(Int, nphys, size(imap, 2))
    ilo, ihi = first(axes(imap, 1)), last(axes(imap, 1))
    @views host[ilo:ihi, :] .= parent(imap)
    copyto!(similar(proto, Int, nphys, size(imap, 2)), host)
end

# GPU (batched) path: (re)point/zero the batch-sized device Sᵢ buffer at the start of each outer-k
# batch (called by the GPU e-ph loop once per k-batch, before its k iterations). Also makes the
# one-time full-vs-block residency decision — deferred to here (not setup_calculator!) because it
# needs the device-array prototype `proto` to query free memory and to `similar(...)` the tile.
function ElectronPhonon.setup_calculator_outer_batch!(calc::BoltzmannCalculator{FT};
        kstart, kend, proto, kwargs...) where {FT}
    @assert calc.use_gpu "setup_calculator_outer_batch! is the GPU path but use_gpu=false"
    n_i, n_f, nT = calc.el_i.n, calc.el_f.n, length(calc.occ)
    if !calc.residency_decided && calc.Sᵢ_dev === nothing
        # Full device-resident Sᵢ = n_i·n_f·nT·sizeof(FT). With headroom for the per-chunk g2
        # transient + the D2H host copy, fall back to the block path if it would not fit.
        Sᵢ_bytes = sizeof(FT) * n_i * n_f * nT
        if 1.5 * Sᵢ_bytes > ElectronPhonon.device_free_bytes(proto)
            calc.gpu_block = true
            @info "BoltzmannCalculator: full device-resident Sᵢ ($(round(Sᵢ_bytes/1e9, digits=2)) GB) " *
                  "would not fit; using block-device-resident (per-tile D2H)."
        end
        calc.residency_decided = true
    end
    calc.gpu_block || return calc
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
# e-ph loop once per outer-k batch, after its k iterations). Only the block-resident path has a
# per-batch tile; the full-resident path keeps Sᵢ on the device and streams it once in
# postprocess_calculator!.
function ElectronPhonon.flush_calculator_outer_batch!(calc::BoltzmannCalculator; kwargs...)
    @assert calc.use_gpu "flush_calculator_outer_batch! is the GPU path but use_gpu=false"
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
    bte_window_accumulate!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_col, imap_f, ikqs, e_i, e_f, wq,
                           μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nm, nqc, nT; i0=0)

Accumulate the BTE scattering-out (Sₒ) and scattering-in (Sᵢ) contributions of one e-ph chunk into
the (in-energy-window) device buffers — the transport analogue of `eph_window_scatter!`, and the
device-resident work of `run_calculator_batched!` (its sole caller, so it lives here). For every
`(m, n, j)` of the chunk look up the outer/inner states `i = imap_i_col[n]`,
`f = imap_f[m, ikqs[j]]` (skip if either is out-of-window, `== 0`), then for each temperature
`iocc` sum the shared per-mode physics (`bte_scattering_increments`) over the `nm` phonon modes
(`ωqmat[ν,j] ≥ ω_cutoff`) and:

  * `Sₒ_out[i, iocc] += Σ_ν sₒ`      — scattering-out, added over `(m, ν, j)` (many `(m,j)` map to
    the same outer `i`), so the device method uses an atomic add here. `Sₒ` is small (`n_i × nT`)
    and always full-resident, so it is indexed by the GLOBAL outer state `i`;
  * `Sᵢ_out[i−i0, f, iocc] = Σ_ν sᵢ` — scattering-in; each `(i, f)` pair is produced by a unique
    `(n, m, j)` across the whole run (distinct k → distinct i, distinct k+q → distinct f), so this
    is a collision-free plain write (no atomics needed). `Sᵢ` is the large object that may be
    block-tiled, hence the `i − i0` (tile-local) row.

`i0` is the global-i offset of the block-tiled `Sᵢ` buffer (`i0 = 0` for the full-resident buffer).
Generic (CPU/fallback) method; the CUDA extension provides the `CuArray` kernel. The physics lives
entirely in `bte_scattering_increments` so the two paths agree.
"""
function bte_window_accumulate!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_col, imap_f, ikqs,
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

function ElectronPhonon.run_calculator_batched!(calc::BoltzmannCalculator{FT},
        ep_kq, ωq, ik, ikqs; g2=nothing, ibandk_offset=0) where {FT}
    @assert calc.use_gpu "run_calculator_batched! is the GPU path but use_gpu=false"
    nbandkq, nbandk, nm, nqc = size(ep_kq)
    n_i, n_f, nT = calc.el_i.n, calc.el_f.n, length(calc.occ)

    if calc.imap_i_dev === nothing
        # imap_i spans ALL physical bands (nphys = nw = nbandkq, the un-projected k+q band count):
        # under the loop's k-side window projection the ep_kq band axis covers only nbandk bands
        # starting at physical band ibandk_offset+1, addressed below by a shifted view (ibandk_offset+nbandk ≤ nw
        # by construction). Full-band runs have ibandk_offset = 0, nbandk = nw — identical to before.
        calc.imap_i_dev = _imap_to_device_bte(ep_kq, calc.imap_el_i, nbandkq)
        calc.imap_f_dev = _imap_to_device_bte(ep_kq, calc.imap_el_f, nbandkq)
        calc.e_i_dev = copyto!(similar(ep_kq, FT, n_i), calc.el_i.es)
        calc.e_f_dev = copyto!(similar(ep_kq, FT, n_f), calc.el_f.es)
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
        ElectronPhonon.bte_window_accumulate!(calc.Sₒ_dev, calc.Sᵢ_tile_dev, g2vals, ωq,
            imap_i_col, calc.imap_f_dev, ikqs, calc.e_i_dev, calc.e_f_dev, calc.wq_dev,
            calc.μ_dev, calc.T_dev, calc.η_dev, calc.occupation_method, calc.omega_cutoff,
            nbandkq, nbandk, nm, nqc, nT; i0 = calc.tile_i0)
        return calc
    end

    # Full device-resident path
    if calc.Sᵢ_dev === nothing
        calc.Sᵢ_dev = fill!(similar(ep_kq, FT, n_i, n_f, nT), zero(FT))
    end
    ElectronPhonon.bte_window_accumulate!(calc.Sₒ_dev, calc.Sᵢ_dev, g2vals, ωq,
        imap_i_col, calc.imap_f_dev, ikqs, calc.e_i_dev, calc.e_f_dev, calc.wq_dev,
        calc.μ_dev, calc.T_dev, calc.η_dev, calc.occupation_method, calc.omega_cutoff,
        nbandkq, nbandk, nm, nqc, nT; i0 = 0)
    calc
end

function ElectronPhonon.postprocess_calculator!(calc::BoltzmannCalculator{FT}; kwargs...) where {FT}
    # GPU path final device→host copy. CPU path: all device buffers are `nothing`, so both blocks
    # below are skipped (nothing device-side to do).
    #
    # Sₒ is always full-resident on the device (both residency modes), so it is streamed here.
    if calc.Sₒ_dev !== nothing
        Sₒ_host = Array(calc.Sₒ_dev)        # (n_i, nT)
        @inbounds for iT in 1:length(calc.occ)
            @views calc.Sₒ[iT] .= Sₒ_host[:, iT]
        end
        calc.Sₒ_dev = nothing
    end
    # Sᵢ: this runs ONLY for the full-resident path (its whole Sᵢ lives in `Sᵢ_dev`, streamed once
    # here). The block path never allocates `Sᵢ_dev` (it uses `Sᵢ_tile_dev`, already streamed to the
    # host per outer-k batch by flush_calculator_outer_batch!), so `Sᵢ_dev === nothing` and this is
    # skipped — no double copy.
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
