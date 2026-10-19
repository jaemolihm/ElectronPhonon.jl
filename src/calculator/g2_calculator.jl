# G2Calculator: an AbstractCalculator that extracts the mode-resolved electron-phonon coupling
# g2 = |ep|²/(2ω) of every in-window state pair during a single pass of `run_eph_over_k_and_kq`,
# with the run's phonon table `ωph[ν, iq]` and the q index `iq_kk[ik, ikq]` of every point pair. The
# coupling is temperature-independent, so a consumer that needs it at many temperatures or
# iterations runs the e-ph loop once and reads these arrays. The pair frequency is a gather
# (`gather_pair_table!`), never stored densely.
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

    # The run's phonon frequencies ω_{qν} as ωph[ν, iq], over the loop's q set (`phs.qpts`, the
    # window-connecting q points in its sorted order), and the index iq_kk[ik, ikq] into it of
    # q = k+q - k for outer k point ik (`el_i.kpts`) and inner point ikq (`el_f.kpts`). The
    # frequency of the pair (i, f) is ωph[ν, iq_kk[el_i.iks[i], el_f.iks[f]]], bitwise the ω in
    # its g2. Host arrays.
    ωph::Matrix{FT} = zeros(FT, 0, 0)
    iq_kk::Matrix{Int32} = zeros(Int32, 0, 0)

    # GPU loop output residency, forwarded to the `TiledDeviceOutput` helper. The batched loop
    # scatters each chunk on the device into either the FULL g2 (downloaded once at
    # `postprocess_calculator!`) or ONE outer-k batch's tile of it (downloaded after every batch,
    # device memory bounded by the batch). `nothing` = choose from free device memory; `true` =
    # force streaming per batch; `false` = force the full arrays.
    force_stream_per_batch::Union{Nothing,Bool} = nothing

    # --- State fields ---
    el_i::Union{Nothing,BandStates{FT,GridKpoints{FT}}} = nothing
    el_f::Union{Nothing,BandStates{FT,GridKpoints{FT}}} = nothing

    # Device-resident g2 output (full or streamed per batch, chosen from free device memory).
    # Owned by the shared `TiledDeviceOutput` helper: it holds g2 (array 1, tiled over the outer-k
    # state axis = axis 2), the residency decision, tile ranges, per-batch zeroing, and the
    # per-tile / final device→host copy. Built at setup; device buffers alloc'd lazily on the first
    # batch.
    tile_dev::Union{Nothing,TiledDeviceOutput{FT}} = nothing

    # Set by `postprocess_calculator!`; `setup_calculator!` errors if already `true`. A calculator
    # instance is single-use — reconstruct it rather than re-running it on a new grid.
    done::Bool = false
end

supports(::G2Calculator, ::Type{OuterKLoop}) = true

# The loop always provides `e`, `u` and the e-ph matrix elements; this lists the extra quantities:
# the band velocities of both sides (`BandStates`).
required_el_quantities(::G2Calculator) = [:vdiag]

# `setup_calculator!` allocates nothing on the backend: the state index maps are the engine's, and g2 is
# formed inside the scatter kernel. The tiled output is not counted: on the first batch it chooses full or
# streamed residency from the memory then free. `ωph` and `iq_kk` are host arrays.
calculator_bytes(::G2Calculator, ::Type{<:EPBlock{OuterKLoop}}; kwargs...) =
    (; persistent = 0, per_outer = 0, per_pair = 0)

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
    # The loop's phonon table and the q index of every (k, k+q) pair, by the loop's own lookup.
    calc.ωph = Array(phs.e)
    calc.iq_kk = q_index_table(sel_k.kpts, sel_kq.kpts, phs.qpts)
    # Device-resident output: `g2` of shape (nmodes, n_i, n_f), tiled over the outer-k state axis
    # (axis 2). `force_stream_per_batch` overrides the full-vs-streamed residency decision. Device
    # buffers alloc'd lazily on the first batch.
    calc.tile_dev = TiledDeviceOutput{FT}((calc.nmodes, n_i, n_f), 2, calc.el_i,
        n_outer_batch; narr = 1, force_stream_per_batch = calc.force_stream_per_batch)
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
    # Copy the device tile buffer into its contiguous host mirror (one bulk D2H). A direct
    # device→strided-host-SubArray copyto! would fall back to scalar indexing.
    tile_download!(tile_dev)
    @views calc.g2[:, inds_i, :] .= host_array(tile_dev, 1)
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
        end
        tile_free!(tile_dev)
    end
    calc.done = true
    calc
end

"""
One tile: one outer k-point `ik` with its (k, k+q) pairs in band-class order. One scatter over the tile's items
forms `g2 = |ep|²/(2ω)` and writes it into the device-resident output at the `[ν, i, f]` slots, with
`i = state_k(tile, n, j)` and `f = state_kq(tile, m, j)`. The `TiledDeviceOutput` helper's
`tile_stride`/`tile_offset` select the linear-index layout, so the same call serves the full and the
streamed residency (see `force_stream_per_batch`).

A band that is not a state has index 0 and is dropped. The target `(ν, i, f)` slots are unique across the
whole run (distinct k → distinct i, distinct k+q → distinct f), so the scatter is overwrite-free.
"""
function run_calculator!(calc::G2Calculator, tile::EPBlock{OuterKLoop}, ctx)
    tile_dev = calc.tile_dev
    eph_window_scatter!(device_array(tile_dev, 1), tile, tile_stride(tile_dev), tile_offset(tile_dev))
    calc
end
