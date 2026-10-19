# EPElementCalculator: an AbstractCalculator that stores one e-ph run's RAW complex
# matrix element ep_{mnν}(k, q) as a complex array `ep` on the `[ν, i, f]` state-pair slots, plus
# the run's phonon table `ωph` and q index `iq_kk`.
#
# States i = (k, n) (outer, "el_i") and f = (k', m) (inner, "el_f"); q = k' - k. The run protocol
# — setup, the tiled device output, the D2H brackets, the device index maps — is
# `G2Calculator`'s (src/calculator/g2_calculator.jl), written out here rather than shared through a
# supertype, so that each reads top to bottom on its own.

"""
    EPElementCalculator{FT}(; nmodes, force_stream_per_batch)

One electron-phonon run's RAW matrix element `ep_{mnν}(k, q)` on the `[ν, i, f]` slots
[`G2Calculator`](@ref) uses, as the complex array `ep`, plus `G2Calculator`'s phonon table `ωph`
and q index `iq_kk` (the pair frequency is a [`gather_pair_table!`](@ref)). Unlike `g2` it
keeps the phase, which a vertex without time-reversal symmetry needs: two such runs, the second over
both k-lists negated, give the elements at (k, q) and at (-k, -q). No `1/(2ω)` is applied here.
`force_stream_per_batch` is `G2Calculator`'s GPU output residency option.

A separate type rather than a mode of `G2Calculator`, so that a field always means one
thing: `g2` is always `|ep|²/(2ω)`, `ep` always the raw element of this run.
"""
Base.@kwdef mutable struct EPElementCalculator{FT} <: AbstractCalculator
    const nmodes::Int

    # The raw element ep_{mnν}(k, q), stored as [ν, i, f].
    ep::Array{Complex{FT},3} = zeros(Complex{FT}, 0, 0, 0)
    # ωph[ν, iq] over the loop's q set and iq_kk[ik, ikq] into it; the pair (i, f) has frequency
    # ωph[ν, iq_kk[el_i.iks[i], el_f.iks[f]]]. As `G2Calculator`'s; host arrays.
    ωph::Matrix{FT} = zeros(FT, 0, 0)
    iq_kk::Matrix{Int32} = zeros(Int32, 0, 0)

    force_stream_per_batch::Union{Nothing,Bool} = nothing

    el_i::Union{Nothing,BandStates{FT,GridKpoints{FT}}} = nothing
    el_f::Union{Nothing,BandStates{FT,GridKpoints{FT}}} = nothing

    # (band, k) → state index in the containers' box coordinates, on the loop backend; read by the
    # scatter kernel. Typed `Any` to avoid a CUDA dependency.
    # Tiled device output, real arrays: Re(ep) = array 1, Im(ep) = array 2.
    tile_dev::Union{Nothing,TiledDeviceOutput{FT}} = nothing
    done::Bool = false
end

supports(::EPElementCalculator, ::Type{OuterKLoop}) = true

# The loop always provides `e`, `u` and the e-ph matrix elements; this lists the extra quantities:
# the band velocities of both sides (`BandStates`).
required_el_quantities(::EPElementCalculator) = [:vdiag]

# `setup_calculator!` allocates nothing on the backend: the state index maps are the engine's, and
# `run_calculator!` scatters the tile's `ep` directly, so there is no per-pair scratch. The tiled output is
# not counted: on the first batch it chooses full or streamed residency from the memory then free. `ωph`
# and `iq_kk` are host arrays.
calculator_bytes(::EPElementCalculator, ::Type{<:EPBlock{OuterKLoop}}; kwargs...) =
    (; persistent = 0, per_outer = 0, per_pair = 0)

function setup_calculator!(calc::EPElementCalculator{FT}, backend, els_k, els_kq, phs;
        sel_k, sel_kq, n_outer_batch, n_inner_tile, kwargs...) where {FT}
    calc.done && throw(ArgumentError("this $(nameof(typeof(calc))) has already been run; " *
                                     "reconstruct the calculator, reuse is not supported"))
    # `sel.nstates_base` is the below-window carrier count of the selection (Σ_ik (band_min[ik]-1)·w_ik
    # over the input grid); `el_i` carries it so that a consumer can count the electrons below the
    # window.
    calc.el_i = BandStates(els_k, sel_k)
    calc.el_f = BandStates(els_kq, sel_kq)

    n_i = calc.el_i.n
    n_f = calc.el_f.n
    calc.ep = zeros(Complex{FT}, calc.nmodes, n_i, n_f)
    # The loop's phonon table and the q index of every (k, k+q) pair, by the loop's own lookup.
    calc.ωph = Array(phs.e)
    calc.iq_kk = q_index_table(sel_k.kpts, sel_kq.kpts, phs.qpts)
    # Device-resident output: Re(ep) and Im(ep), each a real array of shape (nmodes, n_i, n_f),
    # tiled over the outer-k state axis (axis 2). `force_stream_per_batch` overrides the
    # full-vs-streamed residency decision. Device buffers alloc'd lazily on the first batch.
    calc.tile_dev = TiledDeviceOutput{FT}((calc.nmodes, n_i, n_f), 2, calc.el_i,
        n_outer_batch; narr = 2, force_stream_per_batch = calc.force_stream_per_batch)
    calc
end

# One bracket pair per outer-k batch, delegated to the `TiledDeviceOutput` helper: begin decides
# residency on the first batch, allocates the device buffers, and (streamed mode) records/zeros this
# batch's tile; end D2H's the tile into the host arrays.
function calculator_begin_batch!(calc::EPElementCalculator, ctx::OuterKContext)
    tile_begin!(calc.tile_dev, ctx)
    calc
end

function calculator_end_batch!(calc::EPElementCalculator, ctx::OuterKContext)
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
    @views calc.ep[:, inds_i, :] .= complex.(host_array(tile_dev, 1),
                                            host_array(tile_dev, 2))
    calc
end

function postprocess_calculator!(calc::EPElementCalculator; kwargs...)
    # `TiledDeviceOutput`: full-resident buffers are copied back to the host arrays (one D2H);
    # streamed tiles were already D2H'd per batch. Then free the device buffers so they do not
    # occupy memory after the loop.
    tile_dev = calc.tile_dev
    if tile_dev !== nothing && is_allocated(tile_dev)
        if !streamed_per_batch(tile_dev)
            vec(calc.ep) .= complex.(vec(Array(device_array(tile_dev, 1))),
                                     vec(Array(device_array(tile_dev, 2))))
        end
        tile_free!(tile_dev)
    end
    calc.done = true
    calc
end

"""
One tile: one outer k-point `ik` with its (k, k+q) pairs in band-class order. Writes `real(ep)` and
`imag(ep)` of the RAW matrix element of every item into the device-resident output at the `[ν, i, f]` slots,
with `i = state_k(tile, n, j)` and `f = state_kq(tile, m, j)`; a band that is not a state has index 0 and
is dropped. One `eph_window_scatter_reim!` kernel serves the full and the streamed residency (see
`G2Calculator`'s `force_stream_per_batch`).
"""
function run_calculator!(calc::EPElementCalculator, tile::EPBlock{OuterKLoop}, ctx)
    tile_dev = calc.tile_dev
    eph_window_scatter_reim!(device_array(tile_dev, 1), device_array(tile_dev, 2), tile,
        tile_stride(tile_dev), tile_offset(tile_dev))
    calc
end
