"""
    AbstractCalculator

A calculator computes properties of the system during a single pass of one of the e-ph drivers
(`run_eph_over_k_and_kq`, `run_eph_over_k_and_q`, `run_eph_over_q_and_k`). The driver hands each calculator the e-ph matrix
of one block, an outer point with a tile of inner points, as an [`EPBlock`](@ref) together with a
context of the loop order, [`OuterKContext`](@ref) or [`OuterQContext`](@ref).

Users subtype `AbstractCalculator` and implement:
* `supports(calc, ::Type{<:LoopTag})` — the loop orders (`OuterKLoop`, `OuterQLoop`) it handles.
* `required_el_quantities(calc)`, `required_ph_quantities(calc)` — the electron and phonon
  quantities it reads beyond `e` and `u`, as `Symbol` field names of `BatchedElectronState` /
  `BatchedPhononState`.
* `setup_calculator!(calc, backend, els_k, els_kq, phs; order, sel_k, sel_kq, nchunks_threads,
  n_outer_batch, n_inner_tile, verbosity)` — run once, before the loop. `order` is the loop order
  of the run, `OuterKLoop()` or `OuterQLoop()`, for buffers only one order needs. `els_k`, `els_kq`, `phs` are
  the run's state containers (`els_k.nw` Wannier functions, `phs.nmodes` modes) and `sel_k`, `sel_kq` the `FilteredBandStates` they were built from
  (the selected states and their weights: `BandStates(els_k, sel_k)`); `els_kq` and `sel_kq` are
  `nothing` when k+q is solved per tile. `n_outer_batch` and `n_inner_tile` are the widths the
  loop chose, which per-batch and per-tile buffers are sized to.
* `run_calculator!(calc, tile::EPBlock{Loop}, ctx)` — one method per supported loop order `Loop`, once per
  tile.
* `calculator_begin_batch!(calc, ctx)` / `calculator_end_batch!(calc, ctx)` — around every outer batch
  (`ctx.iks_batch` or `ctx.iqs_batch`, never empty). There is no default; a calculator with nothing
  to do defines `= nothing`.
* `postprocess_calculator!(calc; kwargs...)` — run once, after the loop.

The contract:
* **Any batch width.** Every batch length ≥ 1 must work; a per-outer-point reduction loops over
  `ctx.iks_batch` (or `ctx.iqs_batch`) in the brackets.
* **Writes.** On the CPU the tiles of different thread chunks run concurrently. Writes indexed by
  an inner-tile point are disjoint across tiles; every other write, and every per-tile scratch,
  is per `ctx.chunk` (`nchunks_threads` at setup), with partials reduced in `calculator_end_batch!`
  (`ctx.chunk` is 1 on the device).
* **Extents.** The tile is a flat list of items `(m, n, ν, j)` holding only real bands: no padding is
  an item. Read it through `item_indices` / `entry_indices` (device) or `eachitem` / `eachentry` /
  `eachsegment` (host), and the states through `state_k`, `state_kq`, `e_k`, `e_kq`, `u_k`, `u_kq`;
  the raw `els_k` / `els_kq` hold class segments at different strides.
* **No allocation in `run_calculator!`**: size buffers at setup from `n_outer_batch` /
  `n_inner_tile`, and declare their bytes in `calculator_bytes`.

Optionally:
* `calculator_bytes(calc, ::Type{<:EPBlock{Loop}}; kwargs...)` — device bytes the
  calculator holds, `(; persistent, per_outer, per_pair)`, for the loops' memory planning.
* `allowed_eph_phonon_basis(calc)` — phonon bases the calculator accepts.
* `allows_phonon_mode_truncation(calc)` — whether a run may drop the modes no pair can use.

See `docs/writing_a_calculator.md` for a worked example. The public (unexported) API of the
calculators, the drivers and the engines is the one `public` declaration at the bottom of this file.
For manual execution, construct an `OuterKEngine` / `OuterQEngine`, call `setup_calculator!` on
its states, use `stage1!`, `stage2!` and `OuterKContext(eng)` / `OuterQContext(eng)`, then invoke
the same hooks.
"""
abstract type AbstractCalculator end


# =============================================================================
#  Loop-order tags: the order a calculator supports (`supports`) and the type parameter of
#  `EPBlock`.
abstract type LoopTag end
struct OuterKLoop <: LoopTag end    # run_eph_over_k_and_kq / _k_and_q (outer k, inner k+q / q)
struct OuterQLoop <: LoopTag end    # run_eph_over_q_and_k (outer q, inner k)

"""
    OuterKContext{BT <: AbstractBackend}(backend, iks_batch, chunk)

Loop-level state passed to every calculator hook of an outer-k run (`run_eph_over_k_and_kq`,
`run_eph_over_k_and_q`); [`OuterQContext`](@ref) is its outer-q twin. The type is the loop order, so
a calculator dispatches on it.

Fields:
- `backend` :: `CPUBackend()` or `GPUBackend(proto)`.
- `iks_batch` :: the outer k indices of the current outer batch, never empty.
- `chunk` :: the CPU thread slot of this `run_calculator!` call; 1 in the brackets and on a device.
"""
struct OuterKContext{BT <: AbstractBackend}
    backend :: BT
    iks_batch :: UnitRange{Int}
    chunk :: Int
end

"""
    OuterQContext{BT <: AbstractBackend}(backend, iqs_batch, chunk)

The context of an outer-q run (`run_eph_over_q_and_k`), as [`OuterKContext`](@ref) with
`iqs_batch`, the outer q indices of the current outer batch.
"""
struct OuterQContext{BT <: AbstractBackend}
    backend :: BT
    iqs_batch :: UnitRange{Int}
    chunk :: Int
end

"""
    EPBlock{Loop <: LoopTag, ...}

The e-ph matrix of one tile as a flat list of items `(m, n, ν, j)`: band `m` at k+q, band `n` at k, mode `ν`,
pair `j` (a (k, k+q) combination of the tile, in class order), on the run's backend. Pairs are split by band
class `(b_kq, b_k)`; a class's items are stored back to back (`m` fastest, then `n`, `ν`, pair), so under
no padding is an item. Decode an item with
[`item_indices`](@ref), an entry (a band pair with the mode sum inside) with [`entry_indices`](@ref), on the
host with [`eachitem`](@ref) / [`eachentry`](@ref) / [`eachsegment`](@ref); [`device_tile`](@ref) gives the
accessors' arrays to a kernel. The arrays borrow the chunk's storage: consume the tile in `run_calculator!`.

Fields (`n_items` items, `n_pairs` pairs, `nmodes` modes):
- `ep` :: `(n_items,)` eigenbasis e-ph matrix, before `1/(2ω)`, in the run's phonon basis, polar term included.
- `dg` :: `(n_items, 3)` covariant derivative of `ep` (`OuterKLoop` with `covariant_derivative_of_g`), else `nothing`.
- `phs` :: `BatchedPhononState` of the pairs (`phs.e[ν, j]`, polar coefficients if requested), `phs.nmodes`
  modes (the kept modes if the run truncates them). Under `OuterQLoop` no eigenvectors `u` are gathered.
- `wtk`, `wtq` :: weights, per pair on the backend, except `wtk::Float64` under `OuterKLoop`.
- `xk`, `xq` :: momenta, per pair in host vectors, except `xk::Vec3` under `OuterKLoop`; `xkmat` is `xk` as a
  `(3, n_pairs)` matrix on the backend under `OuterQLoop`, else `nothing`.
- `ik`, `ikq`, `iq` :: point indices, per pair on the backend (a k or a q can repeat under `OuterQLoop`), except
  the outer point's `ik::Int` under `OuterKLoop`; `ikq` is `nothing` when k+q is solved per tile.
- `seg`, `seg_host` :: segment table, one column per class (at most `nw²`), on the backend and the host. Rows:
  item offset, pair offset, number of pairs, `b_kq`, `b_k`, point and band-slot offsets of the class's states in
  `els_kq` and `els_k`, and 1 if the class shares one k point.
- `els_k`, `els_kq` :: the gathered states in class segments; read them with `e_k`, `e_kq`, `u_k`, `u_kq`.
- `imap_k`, `imap_kq` :: `[band, point]` state indices in the run's selections (0: not a state), or `nothing`
  without a selection; read them with `state_k`, `state_kq`.
"""
struct EPBlock{Loop <: LoopTag, AT, DGT, EK <: BatchedElectronState, EKQ <: BatchedElectronState,
               PH <: BatchedPhononState, WK, WQ, XK, XKM, XQ, IK, IKQ, IQ, SG, SGH, MK, MKQ}
    ep    :: AT
    dg    :: DGT
    els_k  :: EK
    els_kq :: EKQ
    phs    :: PH
    wtk   :: WK
    wtq   :: WQ
    xk    :: XK
    xkmat :: XKM
    xq    :: XQ
    ik    :: IK
    ikq   :: IKQ
    iq    :: IQ
    seg   :: SG
    seg_host :: SGH
    imap_k  :: MK
    imap_kq :: MKQ
end

function EPBlock{Loop}(; ep::AT, dg::DGT, els_k::EK, els_kq::EKQ, phs::PH, wtk::WK, wtq::WQ, xk::XK,
        xkmat::XKM = nothing, xq::XQ, ik::IK, ikq::IKQ, iq::IQ, seg::SG, seg_host::SGH,
        imap_k::MK = nothing, imap_kq::MKQ = nothing) where {Loop <: LoopTag, AT, DGT, EK, EKQ, PH, WK, WQ,
        XK, XKM, XQ, IK, IKQ, IQ, SG, SGH, MK, MKQ}
    EPBlock{Loop, AT, DGT, EK, EKQ, PH, WK, WQ, XK, XKM, XQ, IK, IKQ, IQ, SG, SGH, MK, MKQ}(ep, dg, els_k, els_kq,
        phs, wtk, wtq, xk, xkmat, xq, ik, ikq, iq, seg, seg_host, imap_k, imap_kq)
end


"""
    supports(calc, ::Type{T}) -> Bool

Declare that `calc` handles loop order `T` (`OuterKLoop` / `OuterQLoop`). Default `false`. The
drivers check this up front and fail loudly on a calculator that does not support their order.

The second argument must be a *type* (e.g. `supports(calc, OuterKLoop)`), not an instance: a non-Type
argument throws, so a typo like `supports(calc, OuterKLoop())` fails loudly instead of silently
returning `false`.
"""
supports(::AbstractCalculator, ::Type{<:LoopTag}) = false
supports(::AbstractCalculator, x) = error(
    "supports(calc, x) expects a loop-tag type, OuterKLoop or OuterQLoop; got $x (::$(typeof(x))).")


# =============================================================================
#  Lifecycle

"""
    allowed_eph_phonon_basis(calc::AbstractCalculator) -> Vector{Symbol}

Return the list of phonon bases the calculator supports for e-ph matrix elements.
- `:eigenmode`: e-ph coupling in phonon eigenmode basis (default)
- `:cartesian`: e-ph coupling in Cartesian displacement basis
"""
allowed_eph_phonon_basis(::AbstractCalculator) = [:eigenmode]

"""
    allows_phonon_mode_truncation(calc::AbstractCalculator) -> Bool

Whether the calculator accepts a run that keeps only the lowest phonon modes: with a finite
`energy_conservation_tol`, `run_eph_over_k_and_kq` on a GPU may drop the modes with `|ω| > Ω` at
every q, `Ω` the energy range of the resident states plus the tolerance, when every calculator
allows it. Such a mode has no process inside the tolerance for any pair. `phs.nmodes` is then below `model.nmodes`. When the phonons of all q do not fit the device
budget, the run builds them per outer batch, for the q of that batch's kept pairs: the `phs` handed
to `setup_calculator!` then holds no q, and the tiles carry the pairs' phonons as always. Both
choices depend on the device budget, so a run that may truncate returns `phs = nothing`.
Default `false`.
"""
allows_phonon_mode_truncation(::AbstractCalculator) = false

"""
    required_el_quantities(calc) -> Vector{Symbol}
    required_ph_quantities(calc) -> Vector{Symbol}

The electron (k and k+q side alike) and phonon quantities the calculator reads beyond the ones the
loop always provides (the energies `e`, the eigenvectors `u` and the e-ph matrix elements), named
as the fields of `BatchedElectronState` (`:vdiag`, `:v`, `:rbar`) and `BatchedPhononState`
(`:vdiag`, `:eph_dipole_coeff`, ...). The loop builds the union. Default: none.
"""
required_el_quantities(::AbstractCalculator) = Symbol[]
required_ph_quantities(::AbstractCalculator) = Symbol[]

# Mandatory hook, no working default. The state arguments are positional and the catch-all leaves
# them unannotated, so it is never ambiguous with a calculator method that leaves them untyped.
function setup_calculator!(::AbstractCalculator, backend, els_k, els_kq, phs; kwargs...)
    error("setup_calculator! has to be implemented")
end

function postprocess_calculator!(::AbstractCalculator; kwargs...)
    error("postprocess_calculator! has to be implemented")
end

# The one bracket, around every outer batch. There is NO no-op default: a missing method is a loud
# error, never a silent skip; a calculator that does nothing there defines `= nothing`. `ctx` is
# unannotated so the fallback is never ambiguous with a calculator method that leaves it untyped.
function calculator_begin_batch!(calc::AbstractCalculator, ctx)
    error("calculator_begin_batch!($(typeof(calc)), ::$(typeof(ctx))) is not defined. Every calculator " *
          "defines the begin/end brackets, even as an explicit no-op (`= nothing`).")
end
function calculator_end_batch!(calc::AbstractCalculator, ctx)
    error("calculator_end_batch!($(typeof(calc)), ::$(typeof(ctx))) is not defined. Every calculator " *
          "defines the begin/end brackets, even as an explicit no-op (`= nothing`).")
end


# =============================================================================
#  Execution hook, dispatched on the block's loop order. There is deliberately no catch-all
#  default: the drivers reject unsupported calculators up front via `supports`.
function run_calculator! end

"""
    calculator_bytes(calc, ::Type{<:EPBlock{Loop}}; nw, nmodes, nband_max_k, nband_max_kq,
                     els_k, els_kq, phs, nchunks_threads) -> (; persistent, per_outer, per_pair)

Device bytes the calculator allocates for a run of loop order `Loop`: whole-run buffers
(`persistent`), per outer point of a batch (`per_outer`) and per inner pair of a tile (`per_pair`).
The loops add them to their own counts to size the inner tile against `free_bytes`, counting
`per_pair` once per thread chunk; a per-outer buffer held per chunk multiplies by `nchunks_threads`
itself. `els_k`, `els_kq`, `phs` are `nothing` in `estimate_device_memory`. Default zeros.
"""
calculator_bytes(::AbstractCalculator, ::Type{<:EPBlock}; kwargs...) =
    (; persistent = 0, per_outer = 0, per_pair = 0)


# =============================================================================
#  Public (but unexported) API. `public` (Julia ≥ 1.11) marks these names as the supported
#  interface without exporting them (users still reach them as `ElectronPhonon.<name>`). The
#  drivers (run_eph.jl), the engines (eph_engine.jl), `eph_window_scatter!` (calculator_utils.jl)
#  and the backend primitives (gpu_utils.jl) are marked here too: `public`, like `export`, permits
#  forward references to names defined later in the module.
public run_eph_over_k_and_kq, run_eph_over_k_and_q, run_eph_over_q_and_k,
    OuterKEngine, OuterQEngine, stage1!, stage2!,
    AbstractCalculator, supports, setup_calculator!, run_calculator!, postprocess_calculator!,
    calculator_begin_batch!, calculator_end_batch!, OuterKLoop, OuterQLoop, EPBlock,
    item_indices, entry_indices, state_k, state_kq, e_k, e_kq, u_k, u_kq, device_tile, eachsegment, eachitem, eachentry, OuterKContext, OuterQContext,
    AbstractBackend, CPUBackend, GPUBackend,
    gpu_backend, alloc, free_bytes, synchronize, batched_gemm!, eph_window_scatter!,
    bte_window_accumulate!, calculator_bytes, allowed_eph_phonon_basis, allows_phonon_mode_truncation,
    required_el_quantities, required_ph_quantities, _indmap_to_device,
    TiledDeviceOutput, tile_begin!, tile_download!, tile_free!, device_array, host_array,
    tile_offset, tile_length, tile_stride, streamed_per_batch, is_allocated, should_stream_per_batch,
    G2Calculator, EPElementCalculator, gather_pair_table!, dedup_phonon_frequencies, phonon_index,
    to_device,
    plan_batch, estimate_device_memory
