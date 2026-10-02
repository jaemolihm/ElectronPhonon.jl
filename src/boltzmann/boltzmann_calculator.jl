# BoltzmannCalculator: an AbstractCalculator that accumulates the BTE scattering-out (Sₒ) and
# scattering-in (Sᵢ) matrices during a single pass of `run_eph_over_k_and_kq`. It uses BandStates /
# imap addressing; rather than copying g2/ωq it folds the temperature-dependent occupation physics
# into Sₒ/Sᵢ via the shared `bte_scattering_increments` (see src/boltzmann/bte_scattering_core.jl).
#
# One method per block, `run_calculator!(::EPBlock{OuterKLoop})`, on every backend (`backend` selects
# where its arrays live): it forms g2 = |ep|²/(2ω) in a per-tile scratch and folds it through
# `bte_scattering_increments`. CPU thread chunks write their own Sₒ partial and g2 scratch.
#
# Output layout is what the transport solver (`solve_electron_bte` / `solve_thermoelectric_bte`)
# consumes unchanged:
#   Sₒ :: Vector{Vector}  — Sₒ[iT][i]      (inverse SERTA lifetime γ_{nk})
#   Sᵢ :: Vector{Matrix}  — Sᵢ[iT][i, f]   (scattering-in kernel)
#
# Device memory (Sᵢ): the scattering-in matrix Sᵢ (n_i·n_f·nT) is the large object. On the GPU it
# is never held whole on the device — it is tiled over outer k, each tile filled by one k-batch and
# streamed to the host (calculator_begin!/end! brackets), so only one tile (≈ one k-batch of rows)
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

# Device buffers for the batched path, built once in `setup_calculator!` (batched runs only) from
# `alloc(backend, …)` / `to_device(backend, …)`. Held behind `dev::Union{Nothing, …}` on the
# calculator; touched only at hook granularity (one kernel launch per call), so the function boundary
# keeps the hot code type-stable. The tiled Sᵢ output lives in `calc.tiled` (a `TiledDeviceOutput`).
# The energies/weights/index maps are intrinsic to the state sets, so they are uploaded at setup.
struct BoltzmannDeviceBuffers{MT, MI, VT, ST, GT}
    Sₒ       :: MT      # (n_i, nT, nchunks) per-chunk partials — small, device-resident
    imap_i   :: MI      # (nband_max_k, n_k)   box band → outer state index (0: none)
    imap_f   :: MI      # (nband_max_kq, n_kq) box band → inner state index (0: none)
    e_i      :: VT      # (n_i,) outer energies
    e_f      :: VT      # (n_f,) inner energies
    wf       :: VT      # (n_f,) per-final-state BZ weight; indexed by the inner state f
    μ        :: VT      # (nT,)
    T        :: VT      # (nT,)
    smearing :: ST      # (nT,)
    g2       :: GT      # (nband_max_kq, nband_max_k, nmodes, n_inner_tile, nchunks) per-tile scratch
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

    # Number of CPU thread chunks of the loop; set at setup (0 = not set yet).
    nchunks::Int = 0

    # --- State (BandStates) --- the (iband, ik) → state reverse map is `el_*.indmap` (via state_index)
    # The per-final-state BZ weight is `state_weights(el_f)` (el_f carries a materialized per-state
    # `weights`), indexed by the inner state f; it replaces the per-k+q-point weight in the scatter.
    el_i::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing
    el_f::Union{Nothing, BandStates{FT, GridKpoints{FT}}} = nothing

    # --- Host outputs (solver-facing) ---
    Sₒ::Vector{Vector{FT}} = Vector{Vector{FT}}()         # per iT, length n_i
    Sᵢ::Vector{Matrix{FT}} = Vector{Matrix{FT}}()         # per iT, (n_i, n_f)

    # --- Device buffers ---
    # Built once in `setup_calculator!`. See `BoltzmannDeviceBuffers`.
    dev::Union{Nothing, BoltzmannDeviceBuffers} = nothing

    # --- Tiled Sᵢ device output (batched path) ---
    # Sᵢ is never held whole on the device: `TiledDeviceOutput` (always block mode) keeps one outer-k
    # tile (i-extent = the largest k-batch) resident and streams it to `calc.Sᵢ` per batch. Built at
    # setup; device buffers allocated lazily on the first batch.
    tiled::Union{Nothing, TiledDeviceOutput{FT}} = nothing

    # Set by `postprocess_calculator!`; `setup_calculator!` errors if already `true`. A calculator
    # instance is single-use — reconstruct it rather than re-running it on a new grid.
    done::Bool = false
end

supports(::BoltzmannCalculator, ::Type{OuterKLoop}) = true
# The loop always provides `e`, `u` and the e-ph matrix elements; this lists the extra quantities:
# the band velocities of both sides (`BandStates`).
required_el_quantities(::BoltzmannCalculator) = [:vdiag]

# What `setup_calculator!` allocates on the backend: the whole-run Sₒ, index maps and per-state
# arrays, the Sᵢ tile (rows of the outer states of one k, all inner states and temperatures, per
# outer point), and the per-tile g2 scratch. The state counts are bounded by the containers' boxes;
# without containers (`estimate_device_memory`) only the scratch is counted.
function eph_batched_bytes_per_point(calc::BoltzmannCalculator{FT}, ::Type{<:EPBlock{OuterKLoop}};
        nmodes, nband_max_k, nband_max_kq, els_k = nothing, els_kq = nothing, kwargs...) where {FT}
    per_pair = sizeof(FT) * nband_max_kq * nband_max_k * nmodes
    (els_k === nothing || els_kq === nothing) && return (; persistent = 0, per_outer = 0, per_pair)
    n_i, n_f, nT = sum(els_k.nband), sum(els_kq.nband), length(calc.occ)
    persistent = sizeof(FT) * (n_i * nT + n_i + 2n_f + 3nT) +                      # Sₒ, e_i, e_f, wf, μ, T, smearing
        sizeof(Int) * (els_k.nband_max * els_k.nk + els_kq.nband_max * els_kq.nk)     # imap_i, imap_f
    (; persistent, per_outer = sizeof(FT) * nband_max_k * n_f * nT, per_pair)
end

function setup_calculator!(calc::BoltzmannCalculator{FT}, backend::AbstractBackend, els_k, els_kq, phs;
        sel_k, sel_kq, nmodes, nchunks_threads, n_outer_batch, n_inner_tile, kwargs...) where {FT}
    mpi_isroot() && println("Setting up BoltzmannCalculator")
    calc.done &&
        throw(ArgumentError("this BoltzmannCalculator has already been run; reconstruct the " *
                            "calculator, reuse is not supported"))
    (sel_k isa FilteredBandStates && sel_kq isa FilteredBandStates) ||
        throw(ArgumentError("BoltzmannCalculator requires a FilteredBandStates for both k and k+q " *
                            "(run it through run_eph_over_k_and_kq)."))
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

    # The selections' states with their energies and velocities, so el_i/el_f carry the
    # per-(k,band) weights and `nstates_base` of the selection (the multigrid double-grid partition;
    # on a uniform grid the weights derive to the per-k weight).
    calc.el_i = BandStates(els_k, sel_k)
    calc.el_f = BandStates(els_kq, sel_kq)

    # Chemical potential: solved directly on el_i, whose per-state energies/weights and `nstates_base`
    # give the correct auto-μ carrier count on a windowed selection (the below-window count rides on
    # the selection) so the bracket succeeds. `bte_compute_μ!` reads el_i through the shared accessors
    # (es / state_weights / ibands / nstates_base); the μ formula is unchanged.
    if !chemical_potential_is_computed(calc.occ)
        bte_compute_μ!(calc.occ, calc.el_i; do_print=true)
    end

    n_i = calc.el_i.n
    n_f = calc.el_f.n
    nT = length(calc.occ)

    calc.Sₒ = [zeros(FT, n_i) for _ in 1:nT]
    calc.Sᵢ = [zeros(FT, n_i, n_f) for _ in 1:nT]
    # Tiled Sᵢ device output: shape (n_i, n_f, nT), tiled over the outer-k state axis (axis 1), always
    # block mode (there is deliberately no full-device-resident Sᵢ path). Metadata only at setup; the
    # device/host tile buffers are lazy in `tile_begin!` (first batch).
    calc.tiled = TiledDeviceOutput{FT}((n_i, n_f, nT), 1, calc.el_i, n_outer_batch; narr = 1,
                                       force_block = true)

    # The whole-run device buffers: the band energies/weights/index maps are intrinsic to the state
    # sets and temperatures, so they are set up once here. `alloc`/`to_device` are backend-generic, so
    # on a `CPUBackend` these are host arrays.
    calc.dev = BoltzmannDeviceBuffers(
        alloc_zeros(backend, FT, n_i, nT, nchunks_threads),                 # Sₒ
        _indmap_to_device(backend, calc.el_i),                              # imap_i
        _indmap_to_device(backend, calc.el_f),                              # imap_f
        to_device(backend, calc.el_i.es),                                   # e_i  (per outer state)
        to_device(backend, calc.el_f.es),                                   # e_f  (per inner state)
        to_device(backend, collect(FT, state_weights(calc.el_f))),          # wf   (per inner state f)
        to_device(backend, collect(FT, calc.occ.μlist)),                    # μ
        to_device(backend, collect(FT, calc.occ.Tlist)),                    # T
        to_device(backend, calc.smearing_list),                             # smearing (one per T)
        alloc(backend, FT, el_kq.nband_max, el_k.nband_max, nmodes, n_inner_tile, nchunks_threads),   # g2
    )
    calc
end

# --- Blocks (EPBlock) ----------------------------------------------------------------

# Once per outer-k batch, before its blocks: record this batch's Sᵢ tile range and zero the tile's
# active region (via `calc.tiled`).
function calculator_begin!(calc::BoltzmannCalculator{FT}, ctx::LoopContext) where {FT}
    # Sᵢ tile for this batch (block mode: zeroed and its range recorded by the helper).
    tile_begin!(calc.tiled, ctx)
    calc
end

# Once per outer-k batch, after its blocks: stream the batch's Sᵢ tile from device to the host output.
function calculator_end!(calc::BoltzmannCalculator, ctx::LoopContext)
    t = calc.tiled
    ni = tile_length(t)
    if ni > 0
        i0 = tile_offset(t)
        tile_download!(t)             # contiguous device→host copy into the tile's host mirror
        host = host_array(t, 1)
        @inbounds for iT in 1:length(calc.occ)
            @views calc.Sᵢ[iT][i0+1:i0+ni, :] .= host[:, :, iT]
        end
    end
    calc
end

"""
    bte_window_accumulate!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs, e_i, e_f, wf,
                           μs, Ts, ηs, method, ω_cutoff, i0)

Accumulate the BTE scattering-out (Sₒ) and scattering-in (Sᵢ) contributions of one k+q batch into
the (in-energy-window) device buffers — the transport analogue of `eph_window_scatter!`, and the
device-resident work of `run_calculator!(::BoltzmannCalculator, ::EPBlock{OuterKLoop}, ctx)` (its sole caller, so it lives here). For every
`(m, n, j)` of the batch look up the outer/inner states `i = imap_i_at_k[n]`,
`f = imap_f[m, ikqs[j]]` (skip if either is out-of-window, `== 0`), the per-final-state weight
`wtq = wf[f]`, then for each temperature
`iT` sum the shared per-mode physics (`bte_scattering_increments`) over the `nmodes` phonon modes
(`ωqmat[ν,j] ≥ ω_cutoff`) and:

  * `Sₒ_out[i, iT] += Σ_ν sₒ`      — scattering-out, added over `(m, ν, j)` (many `(m,j)` map to
    the same outer `i`), so the device method uses an atomic add here. `Sₒ` is small (`n_i × nT`)
    and device-resident, so it is indexed by the GLOBAL outer state `i`;
  * `Sᵢ_out[i-i0, f, iT] = Σ_ν sᵢ` — scattering-in; each `(i, f)` pair is produced by a unique
    `(n, m, j)` across the whole run (distinct k → distinct i, distinct k+q → distinct f), so this
    is a collision-free plain write (no atomics needed). `Sᵢ` is streamed to the host one tile per
    outer-k batch, so `Sᵢ_out` is the current tile and the row is the tile-local `i - i0`.

`imap_i_at_k` is `imap_i[:, ik]` — the outer-state indices of the box bands at the batch's fixed
outer k `ik` — and `imap_f` is in the k+q container's box coordinates; both are 0 on the box padding,
so its `g2vals` entries are never read. `i0` is the global-i offset of the current `Sᵢ` tile. The
extents are those of `g2vals` `(nbandkq, nbandk, nmodes, nq_batch)`; `ωqmat`, `ikqs` and the index
maps must agree with them, or a `DimensionMismatch` is thrown.

The generic method below serves a run on a `CPUBackend`, where the whole loop runs on host arrays;
the CUDA extension provides a `CuArray` kernel. Dispatch is on `Sₒ_out::CuArray, Sᵢ_out::CuArray`
(and `CUDA.allowscalar(false)` turns any accidental host/device mixing into a hard error). The physics
lives entirely in `bte_scattering_increments`, so both methods and the per-(k,q) host loop above
compute the same scattering (validated in `test/boltzmann/test_gpu_boltzmann_calculator.jl`).
`eph_window_scatter!` (src/calculator/calculator_utils.jl) ships the same generic + `CuArray` pair.

The generic method adds into `Sₒ_out` with a plain `+=` where the device kernel needs an atomic:
each CPU thread chunk passes its own `Sₒ` partial, so there is no host counterpart to the kernel's
concurrent writes.
"""
function bte_window_accumulate!(Sₒ_out, Sᵢ_out, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs,
        e_i, e_f, wf, μs, Ts, ηs, method::Int, ω_cutoff, i0::Int)
    nbandkq, nbandk, nmodes, nq_batch = _scatter_extents(g2vals, ωqmat, ikqs, imap_i_at_k, imap_f)
    nT = length(μs)
    @inbounds for iq_batch in 1:nq_batch, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_at_k[n]         # outer (k) state index; 0 = out of window → skip
        i > 0 || continue
        f = imap_f[m, ikqs[iq_batch]]   # inner (k+q) state index; 0 = out of window → skip
        f > 0 || continue
        ek = e_i[i]; ekq = e_f[f]; wtq = wf[f]   # per-final-state weight
        il = i - i0                # tile-local outer row (i0 = the current Sᵢ tile's global offset)
        for iT in 1:nT             # one entry per temperature
            μ = μs[iT]; T = Ts[iT]; η = ηs[iT]
            sₒ = zero(eltype(Sₒ_out)); sᵢ = sₒ
            for ν in 1:nmodes
                ωq = ωqmat[ν, iq_batch]
                ωq < ω_cutoff && continue
                sₒ_ν, sᵢ_ν = bte_scattering_increments(method, ek, ekq, ωq,
                    g2vals[m, n, ν, iq_batch], wtq, μ, T, η)
                sₒ += sₒ_ν; sᵢ += sᵢ_ν
            end
            Sₒ_out[i, iT] += sₒ
            Sᵢ_out[il, f, iT] = sᵢ
        end
    end
    nothing
end

# One block: one outer k with a tile of k+q points. Forms g2 = |ep|²/(2ω) in the per-tile scratch and
# scatters into `dev.Sₒ` and the current Sᵢ tile via `bte_window_accumulate!` (same
# `bte_scattering_increments`), with this chunk's g2 scratch and Sₒ partial.
function run_calculator!(calc::BoltzmannCalculator{FT}, block::EPBlock{OuterKLoop}, ctx) where {FT}
    (; ep, phs, ik, ikq) = block
    dev = calc.dev
    nmodes, nq_batch = size(ep, 3), size(ep, 4)
    g2 = view(dev.g2, :, :, :, 1:nq_batch, ctx.chunk)
    g2 .= abs2.(ep) .* inv.(2 .* reshape(ph.e, 1, 1, nmodes, nq_batch))   # as `epstate_set_g2!`
    t = calc.tiled
    bte_window_accumulate!(view(dev.Sₒ, :, :, ctx.chunk), device_array(t, 1), g2, ph.e,
        view(dev.imap_i, :, ik), dev.imap_f, ikq, dev.e_i, dev.e_f, dev.wf,
        dev.μ, dev.T, dev.smearing, calc.occupation_method, calc.omega_cutoff, tile_offset(t))
    calc
end

function postprocess_calculator!(calc::BoltzmannCalculator{FT}; kwargs...) where {FT}
    calc.done = true    # single-use: `setup_calculator!` errors on a re-run
    # Sₒ is kept in `dev`, so copy it to the host output here (Sᵢ was already streamed one tile per
    # outer-k batch in the end bracket). With no batch on this rank (empty MPI slice or window)
    # `dev.Sₒ` is still the setup zeros.
    Sₒ_host = Array(calc.dev.Sₒ)        # (n_i, nT, nchunks)
    @views for iT in 1:length(calc.occ)
        calc.Sₒ[iT] .= Sₒ_host[:, iT, 1]
        for chunk in 2:size(Sₒ_host, 3)
            calc.Sₒ[iT] .+= Sₒ_host[:, iT, chunk]
        end
    end
    # Free device buffers (the calc is single-use; `done` forbids a re-run in `setup_calculator!`).
    calc.dev = nothing
    calc.tiled === nothing || tile_free!(calc.tiled)
    calc
end
