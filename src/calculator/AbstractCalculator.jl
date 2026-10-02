"""
    AbstractCalculator

A calculator computes properties of the system during a single pass of one of the e-ph drivers
(`run_eph_over_k_and_kq`, `run_eph_over_q_and_k`). The driver hands each calculator the e-ph matrix
of one block, an outer point with a tile of inner points, as an [`EPBlock`](@ref) together with a
[`LoopContext`](@ref).

Users subtype `AbstractCalculator` and implement:
* `supports(calc, ::Type{<:LoopTag})` — the loop orders (`OuterKLoop`, `OuterQLoop`) it handles.
* `required_el_quantities(calc)`, `required_ph_quantities(calc)` — the electron and phonon
  quantities it reads beyond `e` and `u`, as `Symbol` field names of `BatchedElectronState` /
  `BatchedPhononState`.
* `setup_calculator!(calc, backend, el_k, el_kq, ph; sel_k, sel_kq, nw, nmodes, nchunks_threads,
  n_outer_batch, n_inner_tile, verbosity)` — run once, before the loop. `el_k`, `el_kq`, `ph` are
  the run's state containers and `sel_k`, `sel_kq` the `FilteredBandStates` they were built from
  (the selected states and their weights: `BandStates(el_k, sel_k)`); `el_kq` and `sel_kq` are
  `nothing` when k+q is solved per tile. `n_outer_batch` and `n_inner_tile` are the widths the
  loop chose, which per-batch and per-tile buffers are sized to.
* `run_calculator!(calc, block::EPBlock{O}, ctx)` — one method per supported order `O`.
* `calculator_begin!(calc, ctx)` / `calculator_end!(calc, ctx)` — around every outer batch
  (`ctx.batch`, never empty). There is no default; a calculator with nothing to do defines `= nothing`.
* `postprocess_calculator!(calc; kwargs...)` — run once, after the loop.

The contract:
* **Any batch width.** Every `length(ctx.batch) ≥ 1` must work; a per-outer-point reduction loops
  over `ctx.batch` in the brackets.
* **Writes.** Writes indexed by an inner-tile point are disjoint across blocks; every other write
  needs per-`ctx.chunk` partials reduced in `calculator_end!` (`ctx.chunk` is 1 on the device).
* **Extents.** Every array of a block holds exactly the block's points. Band `n` of a block's
  electron side is physical band `iband_offset[j] + n` only for `n ≤ nband[j]`; entries past it
  (including box columns past band `nw`) are undefined. Loop over `1:nband[j]` or through an index
  map that is 0 past it (`_indmap_to_device`); a reduction over the whole box selects with `ifelse`
  on `n ≤ nband[j]`, never by multiplying with a mask.
* **No allocation in `run_calculator!`**: size buffers at setup from `n_outer_batch` /
  `n_inner_tile`, and declare their bytes in `eph_batched_bytes_per_point`.

Optionally:
* `eph_batched_bytes_per_point(calc, ::Type{<:EPBlock{O}}; kwargs...)` — device bytes the
  calculator holds, `(; persistent, per_outer, per_pair)`, for the loops' memory planning.
* `allowed_eph_phonon_basis(calc)` — phonon bases the calculator accepts.

See `docs/writing_a_calculator.md` for a worked example. The public (unexported) calculator API is
the `public` declaration at the bottom of this file.
"""
abstract type AbstractCalculator end


# =============================================================================
#  Loop-order tags: the order a calculator supports, and the type parameter of `EPBlock` and
#  `LoopContext`.
abstract type LoopTag end
struct OuterKLoop <: LoopTag end    # run_eph_over_k_and_kq (outer k, inner k+q)
struct OuterQLoop <: LoopTag end    # run_eph_over_q_and_k (outer q, inner k)

"""
    LoopContext{BT <: AbstractBackend, OT <: LoopTag}

Loop-level state passed to every calculator hook.

Fields:
- `backend` :: `CPUBackend()` or `GPUBackend(proto)`.
- `order` :: `OuterKLoop()` or `OuterQLoop()`.
- `batch` :: the outer indices of the current outer batch, never empty.
- `chunk` :: the CPU thread slot of this `run_calculator!` call; 1 in the brackets and on a device.
"""
struct LoopContext{BT <: AbstractBackend, OT <: LoopTag}
    backend :: BT
    order :: OT
    batch :: UnitRange{Int}
    chunk :: Int
end

"""
    EPBlock{Order <: LoopTag, ...} <: AbstractElPhPayload

The e-ph matrix of one block: one outer point with a tile of inner points, on the run's backend.
Order-agnostic code broadcasts over the pair axis (the last axis of `ep` and of the tile-shaped
fields); the side shared by the whole block has extent 1 along it and a scalar index.

Fields (pair axis `j`):
- `ep` :: `(nband_max_kq, nband_max_k, nmodes, nb)` eigenbasis e-ph matrix, before `1/(2ω)`. Defined
  on each pair's windows only: entry `[m, n, ν, j]` is meaningful for `m ≤ el_kq.nband[j]` and
  `n ≤ el_k.nband[j]` (the shared side's index is 1).
- `dg` :: `nothing` (the covariant derivative is not produced by the batched loops).
- `el_k`, `el_kq` :: `BatchedElectronState` views at block extent; `el_k` has extent 1 under
  `OuterKLoop`.
- `ph` :: `BatchedPhononState` view; extent 1 under `OuterQLoop`.
- `wtk`, `wtq` :: the weights; the shared side's is a scalar, the pair side's a device vector.
- `xk`, `xq` :: the momenta, `Vec3` on the shared side and a host vector on the pair side.
- `ik`, `ikq`, `iq` :: indices into the run's point sets: under `OuterKLoop` `ik::Int`, `ikq` a
  `UnitRange` into the k+q container and `iq` a device vector into the q set; under `OuterQLoop`
  `iq::Int`, `ik` a `UnitRange` and `ikq === nothing` (k+q solved per tile).
"""
struct EPBlock{Order <: LoopTag, AT, DGT, EK, EKQ, PH, WK, WQ, XK, XQ, IK, IKQ, IQ} <: AbstractElPhPayload
    ep    :: AT
    dg    :: DGT
    el_k  :: EK
    el_kq :: EKQ
    ph    :: PH
    wtk   :: WK
    wtq   :: WQ
    xk    :: XK
    xq    :: XQ
    ik    :: IK
    ikq   :: IKQ
    iq    :: IQ
end
function EPBlock{O}(; ep::AT, dg::DGT, el_k::EK, el_kq::EKQ, ph::PH, wtk::WK, wtq::WQ, xk::XK, xq::XQ,
        ik::IK, ikq::IKQ, iq::IQ) where {O <: LoopTag, AT, DGT, EK, EKQ, PH, WK, WQ, XK, XQ, IK, IKQ, IQ}
    EPBlock{O, AT, DGT, EK, EKQ, PH, WK, WQ, XK, XQ, IK, IKQ, IQ}(ep, dg, el_k, el_kq, ph, wtk, wtq, xk, xq, ik, ikq, iq)
end


"""
    supports(calc, ::Type{T}) -> Bool

Declare that `calc` handles loop order `T` (`OuterKLoop` / `OuterQLoop`). The per-point loops also
ask about their payload type (`EPData`). Default `false`. The drivers check this up front and fail
loudly on a calculator that does not support their order.

The second argument must be a *type* (e.g. `supports(calc, OuterKLoop)`), not an instance: a non-Type
argument throws, so a typo like `supports(calc, OuterKLoop())` fails loudly instead of silently
returning `false`.
"""
supports(::AbstractCalculator, ::Type{<:LoopTag}) = false
supports(::AbstractCalculator, ::Type{<:AbstractElPhPayload}) = false
supports(::AbstractCalculator, x) = error(
    "supports(calc, x) expects a loop-tag TYPE (e.g. supports(calc, OuterKLoop)); got " *
    "x::$(typeof(x)). Pass the Type, not an instance.")


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
    required_el_quantities(calc) -> Vector{Symbol}
    required_ph_quantities(calc) -> Vector{Symbol}

The electron (k and k+q side alike) and phonon quantities the calculator reads beyond the ones the
loop always provides (the energies `e`, the eigenvectors `u` and the e-ph matrix elements), named
as the fields of `BatchedElectronState` (`:vdiag`, `:v`, `:rbar`) and `BatchedPhononState`
(`:vdiag`, `:eph_dipole_coeff`, ...). The loop builds the union. Default: none.
"""
required_el_quantities(::AbstractCalculator) = Symbol[]
required_ph_quantities(::AbstractCalculator) = Symbol[]

"""
    required_el_k_quantities(calc::AbstractCalculator) -> Vector{String}

The per-point outer-q loop's k-side electron quantities, as `compute_electron_states` strings.
Default is the full list.
"""
required_el_k_quantities(::AbstractCalculator) = ["eigenvalue", "eigenvector", "velocity", "position"]

# Mandatory hook, no working default. The state arguments are positional and the catch-all leaves
# them unannotated, so it is never ambiguous with a calculator method that leaves them untyped.
function setup_calculator!(::AbstractCalculator, backend, el_k, el_kq, ph; kwargs...)
    error("setup_calculator! has to be implemented")
end

function postprocess_calculator!(::AbstractCalculator; kwargs...)
    error("postprocess_calculator! has to be implemented")
end

# The one bracket, around every outer batch. There is NO no-op default: a missing method is a loud
# error, never a silent skip; a calculator that does nothing there defines `= nothing`. `ctx` is
# unannotated so the fallback is never ambiguous with a calculator method that leaves it untyped.
function calculator_begin!(calc::AbstractCalculator, ctx)
    error("calculator_begin!($(typeof(calc)), ::$(typeof(ctx))) is not defined. Every calculator " *
          "defines the begin/end brackets, even as an explicit no-op (`= nothing`).")
end
function calculator_end!(calc::AbstractCalculator, ctx)
    error("calculator_end!($(typeof(calc)), ::$(typeof(ctx))) is not defined. Every calculator " *
          "defines the begin/end brackets, even as an explicit no-op (`= nothing`).")
end

# The per-point loops' scope bracket (with their `EPData` payload).
struct OuterIteration end
function calculator_begin!(calc::AbstractCalculator, scope, ctx)
    error("calculator_begin!($(typeof(calc)), ::$(typeof(scope)), ::$(typeof(ctx))) is not defined.")
end
function calculator_end!(calc::AbstractCalculator, scope, ctx)
    error("calculator_end!($(typeof(calc)), ::$(typeof(scope)), ::$(typeof(ctx))) is not defined.")
end


# =============================================================================
#  Execution hook, dispatched on the block's loop order. There is deliberately no catch-all
#  default: the drivers reject unsupported calculators up front via `supports`.
function run_calculator! end

"""
    eph_batched_bytes_per_point(calc, ::Type{<:EPBlock{O}}; nw, nmodes, nband_max_k, nband_max_kq,
                                el_k, el_kq, ph) -> (; persistent, per_outer, per_pair)

Device bytes the calculator allocates for a batched run of order `O`: whole-run buffers
(`persistent`), per outer point of a batch (`per_outer`) and per inner pair of a tile (`per_pair`).
The loops add them to their own counts to size the inner tile against `free_bytes`. Default zeros.
"""
eph_batched_bytes_per_point(::AbstractCalculator, ::Type{<:EPBlock}; kwargs...) =
    (; persistent = 0, per_outer = 0, per_pair = 0)


# =============================================================================
#  Public (but unexported) calculator API. `public` (Julia ≥ 1.11) marks these names as the
#  supported interface without exporting them (users still reach them as `ElectronPhonon.<name>`).
#  `eph_window_scatter!` (calculator_utils.jl) and the backend primitives (gpu_utils.jl) are marked
#  here too — `public`, like `export`, permits forward references to names defined later in the module.
public AbstractCalculator, supports, setup_calculator!, run_calculator!, postprocess_calculator!,
    calculator_begin!, calculator_end!, OuterKLoop, OuterQLoop, OuterIteration,
    AbstractElPhPayload, EPData, EPBlock, LoopContext, AbstractBackend, CPUBackend, GPUBackend,
    gpu_backend, alloc, free_bytes, synchronize, batched_gemm!, eph_window_scatter!,
    bte_window_accumulate!, eph_batched_bytes_per_point, allowed_eph_phonon_basis,
    required_el_quantities, required_ph_quantities, required_el_k_quantities, _indmap_to_device,
    TiledDeviceOutput, tile_begin!, tile_download!, tile_free!, device_array, host_array,
    tile_offset, tile_length, tile_stride, is_block, is_allocated, residency_use_block, to_device,
    plan_batch, estimate_device_memory

