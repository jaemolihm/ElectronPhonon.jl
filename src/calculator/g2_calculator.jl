# G2Calculator: an AbstractCalculator that extracts the mode-resolved electron-phonon coupling
# g2 = |ep|²/(2ω) and the phonon frequency ωq of every in-window state pair, during a single pass of
# `run_eph_over_k_and_kq`. The coupling is temperature-independent, so a consumer that needs it at
# many temperatures or iterations runs the e-ph loop once and reads these arrays.
#
# Electronic states are flattened into `BandStates` and addressed through `state_index(el, ik, iband)`.
# The `BandStates` also carry `nstates_base`, the below-window carrier count of the selection
# they were built from.
#
# States i = (k, n) (outer, "el_i") and f = (k', m) (inner, "el_f"); q = k' - k.
#
# Two supported modes (selected by what is passed to `run_eph_over_k_and_kq`):
#   * Full BZ (`symmetry = nothing`, kpts == kqpts): el_i and el_f index the same physical states.
#   * Irreducible BZ (`symmetry = model.symmetry`): the outer k-grid (el_i) is reduced to the IBZ
#     while the inner k+q-grid (el_f) is the full BZ — exactly EPW's `mp_mesh_k`. el_i and el_f then
#     differ; `find_unfolding_indices` maps the full-BZ inner states to their IBZ representatives.
# This calculator stores g2[ν, i, f] rectangularly either way. `|ep|²` drops the phase of the
# element; `EPElementCalculator` keeps it.
#
# `EPElementCalculator` (src/calculator/ep_element_calculator.jl) runs the same protocol for the raw
# complex matrix element. The two are written out separately rather than over a shared supertype,
# so that each reads top to bottom on its own.

Base.@kwdef mutable struct G2Calculator{FT} <: AbstractCalculator
    # --- Parameters ---
    const nmodes::Int

    # --- Data fields (filled during the loop) ---
    # Mode-resolved coupling |g_{mn,ν}(k,k')|² = |M|²/(2ω), stored as g2[ν, i, f].
    # Mode-fastest layout (ν is the contiguous first axis): for a fixed state pair (i, f)
    # the nmodes values are adjacent, which makes the per-pair write in `run_calculator!` and a
    # consumer's sum over modes cache-friendly.
    g2::Array{FT,3} = zeros(FT, 0, 0, 0)

    # Phonon frequency ω_{qν} for the q = k'(f) - k(i) connecting each state pair; g2[ν, i, f].
    ωq::Array{FT,3} = zeros(FT, 0, 0, 0)

    # GPU loop output residency, forwarded to the `TiledDeviceOutput` helper. The batched loop
    # scatters each chunk on the device into either the FULL g2/ωq (downloaded once at
    # `postprocess_calculator!`) or ONE outer-k batch's tile of it (downloaded after every batch,
    # device memory bounded by the batch). `nothing` = choose from free device memory; `true` =
    # force streaming per batch; `false` = force the full arrays.
    force_stream_per_batch::Union{Nothing,Bool} = nothing

    # --- State fields ---
    el_i::Union{Nothing,BandStates{FT,GridKpoints{FT}}} = nothing
    el_f::Union{Nothing,BandStates{FT,GridKpoints{FT}}} = nothing

    # --- Device buffers for the batched loop (`run_eph_over_k_and_kq`, `backend = gpu_backend()`) ---
    # The (band, k) state-index maps are kept device-resident (small, reused across all chunks);
    # built at setup on the batched path via `_indmap_to_device`. Typed `Any` to avoid a CUDA
    # dependency: only device indexing and views are used.
    imap_i_dev::Any = nothing   # (nband_max_k, n_k) state index for (box band, k); device Int matrix
    imap_f_dev::Any = nothing   # (nband_max_kq, n_kq) state index for (box band, k+q); device Int matrix
    g2_tile::Any = nothing      # (nband_max_kq, nband_max_k, nmodes, n_inner_tile, nchunks) per-tile scratch

    # Device-resident g2/ωq output (full or streamed per batch, chosen from free device memory).
    # Owned by the shared `TiledDeviceOutput` helper: it holds both arrays (g2 = array 1, ωq =
    # array 2, tiled over the outer-k state axis = axis 2), the residency decision, tile ranges,
    # per-batch zeroing, and the per-tile / final device→host copy. Built at setup; device buffers
    # alloc'd lazily on the first batch.
    tile_dev::Union{Nothing,TiledDeviceOutput{FT}} = nothing

    # Set by `postprocess_calculator!`; `setup_calculator!` errors if already `true`. A calculator
    # instance is single-use — reconstruct it rather than re-running it on a new grid.
    done::Bool = false
end

supports(::G2Calculator, ::Type{OuterKLoop}) = true

# The loop always provides `e`, `u` and the e-ph matrix elements; this lists the extra quantities:
# the band velocities of both sides (`BandStates`).
required_el_quantities(::G2Calculator) = [:vdiag]

# What `setup_calculator!` allocates on the backend: the two index maps, bounded by the containers'
# boxes (nothing without containers, in `estimate_device_memory`), and the per-tile g2 scratch. The
# tiled output is not counted: on the first batch it chooses full or streamed residency from the
# memory then free.
function calculator_bytes(::G2Calculator{FT}, ::Type{<:EPBlock{OuterKLoop}}; nmodes,
        nband_max_k, nband_max_kq, els_k = nothing, els_kq = nothing, kwargs...) where {FT}
    per_pair = sizeof(FT) * nband_max_kq * nband_max_k * nmodes                     # g2_tile
    persistent = els_k === nothing || els_kq === nothing ? 0 :
        sizeof(Int) * (els_k.nband_max * els_k.nk + els_kq.nband_max * els_kq.nk)  # imap_i, imap_f
    (; persistent, per_outer = 0, per_pair)
end

function setup_calculator!(calc::G2Calculator{FT}, backend, els_k, els_kq, phs;
        sel_k, sel_kq, nchunks_threads, n_outer_batch, n_inner_tile, kwargs...) where {FT}
    calc.done && throw(ArgumentError("this $(nameof(typeof(calc))) has already been run; " *
                                     "reconstruct the calculator, reuse is not supported"))
    # `sel.nstates_base` is the below-window carrier count of the selection (Σ_ik (band_min[ik]-1)·w_ik
    # over the input grid); `el_i` carries it so that a consumer can count the electrons below the
    # window.
    calc.el_i = BandStates(els_k, sel_k)
    calc.el_f = BandStates(els_kq, sel_kq)

    n_i = calc.el_i.n
    n_f = calc.el_f.n
    calc.g2 = zeros(FT, calc.nmodes, n_i, n_f)
    calc.ωq = zeros(FT, calc.nmodes, n_i, n_f)
    # Device-resident output: `g2` and `ωq`, each of shape (nmodes, n_i, n_f), tiled over the outer-k
    # state axis (axis 2). `force_stream_per_batch` overrides the full-vs-streamed residency
    # decision. Device buffers alloc'd lazily on the first batch.
    calc.tile_dev = TiledDeviceOutput{FT}((calc.nmodes, n_i, n_f), 2, calc.el_i,
        n_outer_batch; narr = 2, force_stream_per_batch = calc.force_stream_per_batch)
    # The (band, k) → state-index maps on `backend`, in the containers' box coordinates (0 for a band
    # that is not a state, and on the box padding).
    calc.imap_i_dev = _indmap_to_device(backend, calc.el_i)
    calc.imap_f_dev = _indmap_to_device(backend, calc.el_f)
    # The per-tile g2 = |ep|²/(2ω), formed from each block before the scatter.
    calc.g2_tile = alloc(backend, FT, els_kq.nband_max, els_k.nband_max, calc.nmodes, n_inner_tile,
                         nchunks_threads)
    calc
end

# One bracket pair per outer-k batch, delegated to the `TiledDeviceOutput` helper: begin decides
# residency on the first batch, allocates the device buffers, and (streamed mode) records/zeros this
# batch's tile; end D2H's the tile into the host arrays.
function calculator_begin_batch!(calc::G2Calculator, ctx::OuterKContext)
    tile_begin!(calc.tile_dev, ctx)
    calc
end

function calculator_end_batch!(calc::G2Calculator, ctx::OuterKContext)
    tile_dev = calc.tile_dev
    # Full mode: the output stays on the device and is downloaded once in `postprocess_calculator!`.
    streamed_per_batch(tile_dev) || return calc
    ni = tile_length(tile_dev)
    ni == 0 && return calc
    i0 = tile_offset(tile_dev)
    inds_i = i0+1:i0+ni   # outer-state indices i of this tile in the full output
    # Copy each device tile buffer into its contiguous host mirror (one bulk D2H per array). A direct
    # device→strided-host-SubArray copyto! would fall back to scalar indexing.
    tile_download!(tile_dev)
    @views calc.g2[:, inds_i, :] .= host_array(tile_dev, 1)
    @views calc.ωq[:, inds_i, :] .= host_array(tile_dev, 2)
    calc
end

function postprocess_calculator!(calc::G2Calculator; kwargs...)
    # `TiledDeviceOutput`: full-resident buffers are copied back to the host arrays (one D2H);
    # streamed tiles were already D2H'd per batch. Then free the device buffers so they do not
    # occupy memory after the loop.
    tile_dev = calc.tile_dev
    if tile_dev !== nothing && is_allocated(tile_dev)
        if !streamed_per_batch(tile_dev)
            copyto!(vec(calc.g2), device_array(tile_dev, 1))
            copyto!(vec(calc.ωq), device_array(tile_dev, 2))
        end
        tile_free!(tile_dev)
    end
    calc.imap_i_dev = nothing
    calc.imap_f_dev = nothing
    calc.g2_tile = nothing
    calc.done = true
    calc
end

"""
One block: one outer k-point `ik` with a tile of k+q points `ikq[j]`. Forms `g2 = |ep|²/(2ω)` in the
per-tile scratch of the block's thread chunk and writes it with `ωq` into the device-resident output
at the `[ν, i, f]` slots, with `i = imap_i[n, ik]` and `f = imap_f[m, ikq[j]]` in the containers' box
coordinates. The `TiledDeviceOutput` helper's `tile_stride`/`tile_offset`
select the linear-index layout, so the same call serves the full and the streamed residency
(see `force_stream_per_batch`).

A band that is not a state, and the box padding, have `imap == 0` and are dropped. The target
`(ν, i, f)` slots are unique across the whole run (distinct k → distinct i, distinct k+q →
distinct f), so the scatter is overwrite-free.
"""
function run_calculator!(calc::G2Calculator, block::EPBlock{OuterKLoop}, ctx)
    (; ep, phs, ik, ikq) = block
    nm, npairs = size(ep, 3), size(ep, 4)
    g2 = view(calc.g2_tile, :, :, :, 1:npairs, ctx.chunk)
    g2 .= abs2.(ep) .* inv.(2 .* reshape(phs.e, 1, 1, nm, npairs))   # as `epstate_set_g2!`
    tile_dev = calc.tile_dev
    eph_window_scatter!(
        device_array(tile_dev, 1), device_array(tile_dev, 2),
        g2, view(calc.imap_i_dev, :, ik), calc.imap_f_dev, ikq, phs.e,
        tile_stride(tile_dev), tile_offset(tile_dev))
    calc
end
