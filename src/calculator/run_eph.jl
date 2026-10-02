using ChunkSplitters
using Base.Threads: nthreads, @threads
using Dates: now

"""
    run_eph_over_k_and_kq(model, kpts, kqpts; calculators, backend, kwargs...)

Sweep the outer k points and, for each, the inner k+q points (on a grid commensurate with the k
grid), handing each calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterKLoop}`: one outer k
with a tile of k+q points. `kpts` and `kqpts` are grids: a grid size, a k-point set on a grid or a
prebuilt `FilteredBandStates` of one (the phonons are built on the q grid the two span). Returns `(; kpts, qpts, els_k, els_kq, phs)`, the run's state containers
(`BatchedElectronState`, `BatchedPhononState`).

Keywords:
* `calculators` — at least one; each must `supports(calc, OuterKLoop)`.
* `backend = CPUBackend()` — `ElectronPhonon.gpu_backend()` for a GPU run.
* `window_k`, `window_kq` — energy windows of the two sides (ignored for a `FilteredBandStates`).
* `symmetry = model.symmetry` — reduces the outer k to the irreducible wedge and builds the k+q
  set by unfolding its selection; `nothing` for full grids.
* `energy_conservation = (:None, 0.0)` — `(:Fixed, tol)` or `(:Linear, curvature)` drops the pairs
  with no energy-conserving process before the e-ph matrix is computed (`CPUBackend` only).
* `covariant_derivative_of_g = false` — also compute the covariant derivative `block.dg`.
* `eph_phonon_basis = :eigenmode` — or `:cartesian` (identity phonon rotation).
* `fourier_mode = "gridopt"` — or `"normal"`: the interpolation of the setup-time state solves on a
  `CPUBackend` (any other value is an `ArgumentError`; a GPU backend ignores it). The e-ph matrix
  always uses the batched interpolator.
* `n_outer_batch = 256` — outer k points per stage-1 batch and per calculator bracket.
* `n_inner_tile` — k+q points per block: as many as fit the free device memory on a GPU, at most
  1024 per thread chunk on the CPU.
* `nchunks_threads = nthreads()` — CPU thread chunks over the k+q points (1 on a GPU).
* `el_k_eigenpairs`, `el_kq_eigenpairs`, `ph_eigenpairs` — caches from
  [`electron_eigenpairs`](@ref) / [`phonon_eigenpairs`](@ref) giving runs over overlapping point
  sets a shared eigenvector gauge inside degenerate multiplets. Each must cover every point the run
  visits on its side (the q points are `combine_kpoint_grids(kpts, kqpts)`) and be resident on
  `backend`.
* `mpi_comm_k` — splits the outer k points across ranks.
"""
run_eph_over_k_and_kq(model::Model, kpts_input, kqpts_input; kwargs...) =
    _run_eph(OuterKLoop(), model, kpts_input, kqpts_input; kwargs...)

"""
    run_eph_over_k_and_q(model, kpts, qpts; calculators, backend, kwargs...)

Sweep the outer k points and, for each, the inner q points (any q list, e.g. a path, or a q grid),
handing each calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterKLoop}`: one outer k with a
tile of q points, whose index is `iq` (`ikq === nothing`). The k+q states are solved per tile, with
`e` and `u` only, and the phonons are built once on `qpts`. Returns
`(; kpts, qpts, els_k, els_kq = nothing, phs)`.

Keywords as in [`run_eph_over_k_and_kq`](@ref), except: `kpts` is any k list (e.g. a band path,
not necessarily on a grid), a grid size or a prebuilt `FilteredBandStates`; `el_k_eigenpairs` only
for k points on a grid (the cache is looked up on one); `symmetry = nothing`, the only value
accepted (the outer k points are not reduced); `n_inner_tile` q points per block; no `vdiag` (for a
calculator or `:Linear` energy conservation) and no `el_kq_eigenpairs`, which need the k+q points
on a grid. Calculators that read the k+q selection (`sel_kq`) are not supported.
"""
run_eph_over_k_and_q(model::Model, kpts_input, qpts_input; symmetry = nothing, kwargs...) =
    _run_eph(OuterKLoop(), model, kpts_input, qpts_input, true; symmetry, kwargs...)

"""
    run_eph_over_q_and_k(model, kpts, qpts; calculators, backend, kwargs...)

Sweep the outer q points (any q list, e.g. a path) and, for each, the inner k points, handing each
calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterQLoop}`: one q with a tile of k points.
The k+q states are solved per tile, with `e` and `u` only, unless `precompute_el_kq = true` (a grid
q set), which builds them once on the k+q grid; a pair whose k+q has no state in `window_kq` is
then dropped. Returns `(; kpts, qpts, els_k, els_kq, phs)`, `els_kq = nothing` when solved per tile.

Keywords as in [`run_eph_over_k_and_kq`](@ref), except: `use_symmetry = true` reduces the k points
with `model.symmetry`; `keep_all_qpts = false` (outer q only) drops the q points with no k+q state
in `window_kq`; `precompute_el_kq` needs a q grid that is a multiple of the k grid; `mpi_comm_k` is
refused;
`n_outer_batch` q points per stage-1 batch and per bracket, 16 on a GPU and 1 on the CPU (stage 1
gains nothing from a wider batch there, while a calculator's per-q buffers are held per thread
chunk); `n_inner_tile` k points per block, at most `2^15` on a GPU; no `covariant_derivative_of_g`;
`el_kq_eigenpairs` only with `precompute_el_kq`.
"""
run_eph_over_q_and_k(model::Model, kpts_input, qpts_input; use_symmetry::Bool = true, kwargs...) =
    _run_eph(OuterQLoop(), model, kpts_input, qpts_input;
             symmetry = use_symmetry ? model.symmetry : nothing, kwargs...)


# The quantities the loop provides itself, on both electron sides and on the phonons: always the
# energies and eigenvectors, the dipole coefficients of a polar model, and for the `:Linear` energy
# conservation the velocities of the k+q side and the phonons.
loop_el_quantities((mode, _)) = [:e; :u; mode === :Linear ? [:vdiag] : Symbol[]]
function loop_ph_quantities(model, (mode, _))
    [:e; :u; model.polar_eph.use ? [:eph_dipole_coeff] : Symbol[];
     mode === :Linear ? [:vdiag] : Symbol[]]
end

# `kq_per_tile` (`OuterKLoop` only): `second_input` is the inner q set and the k+q states are solved
# per tile (`run_eph_over_k_and_q`); otherwise it is the k+q grid.
function _run_eph(order::LoopTag, model::Model{FT}, kpts_input, second_input, kq_per_tile = false;
        calculators = [],
        backend::AbstractBackend = CPUBackend(),
        window_k = (-Inf, Inf),
        window_kq = (-Inf, Inf),
        symmetry = model.symmetry,
        precompute_el_kq = false,
        keep_all_qpts = false,
        energy_conservation = (:None, 0.0),
        covariant_derivative_of_g = false,
        eph_phonon_basis::Symbol = :eigenmode,
        fourier_mode = "gridopt",
        screening_params = nothing,
        mpi_comm_k = nothing,
        n_outer_batch = nothing,
        n_inner_tile = nothing,
        nchunks_threads = nthreads(),
        el_k_eigenpairs::Union{Nothing, Eigenpairs} = nothing,
        el_kq_eigenpairs::Union{Nothing, Eigenpairs} = nothing,
        ph_eigenpairs::Union{Nothing, Eigenpairs} = nothing,
        progress_print_step = 20,
        verbosity::Int = 1,
    ) where {FT}

    el_qty = union(loop_el_quantities(energy_conservation), required_el_quantities.(calculators)...)
    ph_qty = union(loop_ph_quantities(model, energy_conservation),
                   required_ph_quantities.(calculators)...)
    _check_run(order, model, backend, calculators, kpts_input, second_input, el_qty;
        energy_conservation, covariant_derivative_of_g, eph_phonon_basis, fourier_mode,
        precompute_el_kq, screening_params, mpi_comm_k, el_kq_eigenpairs, symmetry, kq_per_tile,
        el_k_eigenpairs)

    (; els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq) = _setup_states(order, model, kpts_input,
        second_input, el_qty, ph_qty; kq_per_tile, backend, window_k, window_kq, symmetry, precompute_el_kq,
        keep_all_qpts, eph_phonon_basis, fourier_mode, mpi_comm_k, el_k_eigenpairs,
        el_kq_eigenpairs, ph_eigenpairs, verbosity)
    # The containers' types depend on the runtime quantity lists, and inference of the loop on the
    # abstract types does not terminate (> 30 min for an outer-q run), so it is not let through.
    Base.inferencebarrier(_run_eph_loop)(order, model, els_k, els_kq, phs, kpts, kqpts, qpts, sel_k,
        sel_kq, el_qty, ph_qty; calculators, backend, symmetry, precompute_el_kq, energy_conservation,
        covariant_derivative_of_g, eph_phonon_basis, n_outer_batch, n_inner_tile, nchunks_threads,
        window_kq, progress_print_step, verbosity)
end

# The loop of `_run_eph` on the built states, compiled for their concrete types (`els_kq` and `kqpts`
# are `nothing` for the per-tile k+q solve).
function _run_eph_loop(order, model, els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq, el_qty,
        ph_qty; calculators, backend, symmetry, precompute_el_kq, energy_conservation,
        covariant_derivative_of_g, eph_phonon_basis, n_outer_batch, n_inner_tile, nchunks_threads,
        window_kq, progress_print_step, verbosity)
    (; nw, nmodes) = model

    nchunks = backend isa CPUBackend ? nchunks_threads : 1
    drop_pairs = energy_conservation[1] !== :None || (order isa OuterQLoop && precompute_el_kq)
    # The inner set of the outer-k loop: the k+q grid, or the q points with k+q solved per tile.
    kq_per_tile = order isa OuterKLoop && els_kq === nothing
    inner_pts = order isa OuterKLoop ? something(kqpts, qpts) : kpts
    n_outer = order isa OuterKLoop ? kpts.n : qpts.n
    n_inner = inner_pts.n
    (; nb_outer, nb_inner, committed, bytes) = _plan_widths(order, model, backend, calculators;
        n_outer, n_inner, nk = kpts.n, nkq = order isa OuterKLoop ? inner_pts.n : 0, nchunks,
        kq_per_tile,
        n_outer_batch, n_inner_tile, nband_max_k = els_k.nband_max,
        nband_max_kq = els_kq === nothing ? nw : els_kq.nband_max, els_k, els_kq, phs, el_qty, ph_qty,
        drop_pairs, precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis)
    if verbosity > 0 && mpi_isroot()
        @info "e-ph loop: committed = $(round(committed / 1e9, digits = 2)) GB, " *
              "$(round(bytes.per_pair / 1e3, digits = 1)) kB per pair; outer batch = $nb_outer, " *
              "inner tile = $nb_inner, $nchunks chunk(s)"
    end

    eng = order isa OuterKLoop ?
        OuterKEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, kqpts, qpts,
            n_outer_batch = nb_outer, n_inner_tile = nb_inner, nchunks, drop_pairs,
            covariant_derivative_of_g, eph_phonon_basis) :
        OuterQEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, qpts,
            n_outer_batch = nb_outer, n_inner_tile = nb_inner, nchunks, drop_pairs, eph_phonon_basis)
    foreach(c -> setup_calculator!(c, backend, els_k, els_kq, phs; sel_k, sel_kq, nw, nmodes,
        nchunks_threads = nchunks, n_outer_batch = nb_outer, n_inner_tile = nb_inner, verbosity),
        calculators)

    ngrid = order isa OuterKLoop ? inner_pts.ngrid : qpts.ngrid   # the energy-conservation box
    for batch in Iterators.partition(1:n_outer, nb_outer)
        if mpi_isroot() &&
                div(last(batch), progress_print_step) > div(first(batch) - 1, progress_print_step)
            @info "$(now()) $(order isa OuterKLoop ? "ik" : "iq") = $batch / $n_outer"
            flush(stdout); flush(stderr)
        end
        ctx = LoopContext(backend, order, batch, 1)
        foreach(c -> calculator_begin!(c, ctx), calculators)
        if kq_per_tile
            _loop_outer_k_over_q!(eng, batch, els_k, phs, kpts, qpts, calculators, model,
                                  energy_conservation, ngrid, window_kq)
        elseif order isa OuterKLoop
            _loop_outer_k!(eng, batch, els_k, els_kq, phs, kpts, qpts, calculators, model,
                           energy_conservation, ngrid)
        else
            _loop_outer_q!(eng, batch, els_k, els_kq, phs, kpts, kqpts, qpts, calculators, model,
                           energy_conservation, ngrid, eph_phonon_basis, window_kq)
        end
        foreach(c -> calculator_end!(c, ctx), calculators)
        # Bound the host look-ahead to one batch, so its device scratch does not pile up in the pool.
        synchronize(backend)
    end

    # The outer-q loop hands the model's symmetry to postprocess whatever `use_symmetry` was.
    symmetry_post = order isa OuterQLoop ? model.symmetry : symmetry
    foreach(c -> postprocess_calculator!(c; qpts, symmetry = symmetry_post), calculators)
    (; kpts, qpts, els_k, els_kq, phs)
end


# The widths of a run and the device bytes behind them, for `_run_eph` and `estimate_device_memory`
# alike. The outer batch is a fixed default (256 outer k; 16 q on a device and 1 on the CPU). The
# inner tile fills the free device memory left after the persistent and per-batch buffers
# (`plan_batch` on `engine_bytes` plus each calculator's `eph_batched_bytes_per_point`, `per_pair`
# once per chunk), capped at all inner points on a device (`2^15` k under `OuterQLoop`) and at a
# cache-sized 1024 per chunk on the CPU.
function _plan_widths(order, model, backend, calculators; n_outer, n_inner, nk, nkq, nchunks,
        n_outer_batch, n_inner_tile, nband_max_k, nband_max_kq, els_k, els_kq, phs, el_qty, ph_qty,
        drop_pairs, precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis, kq_per_tile = false)
    (; nw, nmodes) = model
    outer_default = order isa OuterKLoop ? 256 : backend isa CPUBackend ? 1 : 16
    nb_outer = max(1, min(something(n_outer_batch, outer_default), n_outer))
    inner_default = backend isa CPUBackend ? 1024 : order isa OuterKLoop ? n_inner : 2^15
    inner_cap = max(1, min(cld(n_inner, nchunks), something(n_inner_tile, inner_default)))
    bytes = order isa OuterKLoop ?
        engine_bytes(OuterKEngine, model; nband_max_k, nband_max_kq, nk, nkq, el_qty, ph_qty,
            drop_pairs, covariant_derivative_of_g, eph_phonon_basis, kq_per_tile) :
        engine_bytes(OuterQEngine, model; nband_max_k, nband_max_kq, nk, n_outer_batch = nb_outer,
            el_qty, ph_qty, drop_pairs, precompute_el_kq, eph_phonon_basis)
    for c in calculators
        b = eph_batched_bytes_per_point(c, EPBlock{typeof(order)}; nw, nmodes, nband_max_k,
                                        nband_max_kq, els_k, els_kq, phs, nchunks_threads = nchunks)
        bytes = (; persistent = bytes.persistent + b.persistent,
                   per_outer = bytes.per_outer + b.per_outer, per_pair = bytes.per_pair + b.per_pair)
    end
    committed = bytes.persistent + bytes.per_outer * nb_outer
    nb_inner = plan_batch(backend, bytes.per_pair * nchunks, committed, inner_cap;
                          what = order isa OuterKLoop ? "outer-k" : "outer-q")
    (; nb_outer, nb_inner, committed, bytes)
end

"""
    plan_batch(backend, per_point, committed, cap; headroom_num = 7, headroom_den = 10, what = "",
               warn = true) -> nbatch

Size a batched e-ph loop's batch to free device memory:
`nbatch = min(cap, (free - committed) · headroom ÷ per_point)`, clamped to at least 1, where
`per_point` / `committed` are the per-(batched-inner index) and whole-run device-byte counts. On a
CPU backend `free_bytes` is `typemax(Int)`, so `nbatch = cap`. Errors if the whole-run commitments
alone exceed free device memory (a clear early failure instead of an OOM mid-loop); `what` names the
loop in that message. `headroom_num / headroom_den` is the usable fraction of free memory (default
`7/10`, i.e. a 30% headroom for the batched drivers' recycled temporaries), applied as
`x ÷ den * num` to match the integer arithmetic of the formulas this replaces. Pass `warn = false`
for a counterfactual query ("how wide would the batch be at a different `per_point`?"), which must
not tell the user their batch was reduced.
"""
function plan_batch(backend::AbstractBackend, per_point::Integer, committed::Integer, cap::Integer;
        headroom_num::Integer = 7, headroom_den::Integer = 10, what::AbstractString = "",
        warn::Bool = true)
    free = free_bytes(backend)
    if free != typemax(Int) && committed > free
        error("batched $(what): committed device memory ($(round(committed / 1e9, digits = 2)) GB, " *
              "whole-run stacks) exceeds free device memory ($(round(free / 1e9, digits = 2)) GB). " *
              "Reduce the batch cap or the grid size.")
    end
    nb_mem = free == typemax(Int) ? Int(cap) :
        max(1, ((free - committed) ÷ headroom_den * headroom_num) ÷ per_point)
    nbatch = min(Int(cap), nb_mem)
    if warn && free != typemax(Int) && nb_mem < cap
        @warn "WARNING : batch width reduced from the requested cap $(Int(cap)) to $nbatch to fit " *
            "free device memory in plan_batch$(isempty(what) ? "" : " ($what)"). Performance may " *
            "degrade compared to smaller calculations."
    end
    nbatch
end


# Every refusal of the loop, before any state is built. `el_qty` is the union of the electron
# quantities of the loop and the calculators.
function _check_run(order, model, backend, calculators, kpts_input, second_input, el_qty;
        energy_conservation, covariant_derivative_of_g, eph_phonon_basis, fourier_mode,
        precompute_el_kq, screening_params, mpi_comm_k, el_kq_eigenpairs, symmetry = nothing,
        kq_per_tile = false, el_k_eigenpairs = nothing)
    Order = typeof(order)
    isempty(calculators) && throw(ArgumentError("the e-ph loop requires at least one calculator."))
    for calc in calculators
        supports(calc, Order) || throw(ArgumentError("$calc does not support the " *
            (order isa OuterKLoop ? "outer-k loop. Use run_eph_over_q_and_k instead." :
                                    "outer-q loop. Use run_eph_over_k_and_kq instead.")))
        eph_phonon_basis ∈ allowed_eph_phonon_basis(calc) || throw(ArgumentError(
            "Calculator $calc does not support eph_phonon_basis = :$eph_phonon_basis. " *
            "Allowed: $(allowed_eph_phonon_basis(calc))"))
    end
    eph_phonon_basis ∈ (:eigenmode, :cartesian) ||
        throw(ArgumentError("eph_phonon_basis must be :eigenmode or :cartesian, got :$eph_phonon_basis"))
    (backend isa CPUBackend && fourier_mode ∉ ("gridopt", "normal")) && throw(ArgumentError(
        "fourier_mode = \"$fourier_mode\" is not supported on a CPU backend. Use \"gridopt\" " *
        "(the default) or \"normal\"."))
    model.epmat isa WannierObject || throw(ArgumentError(
        "a disk-backed epmat ($(typeof(model.epmat))) is not supported by the e-ph loop; load the " *
        "model into memory"))
    screening_params === nothing || error(
        "screening_params is not supported: dielectric screening is currently disabled (ϵ ≡ 1). " *
        "Pass screening_params = nothing.")
    mode = energy_conservation[1]
    mode ∈ (:None, :Fixed, :Linear) ||
        throw(ArgumentError("energy_conservation mode must be :None, :Fixed or :Linear, got :$mode"))
    (mode === :None || backend isa CPUBackend) || throw(ArgumentError(
        "energy_conservation is a CPUBackend feature: on a GPU computing every pair and letting " *
        "the calculators' delta functions discard is cheaper. Pass energy_conservation = (:None, 0.0)."))
    if order isa OuterKLoop
        precompute_el_kq && throw(ArgumentError("precompute_el_kq is an outer-q option"))
        covariant_derivative_of_g && model.epmat_outer_momentum != "el" && throw(ArgumentError(
            "covariant_derivative_of_g needs a model loaded with epmat_outer_momentum = \"el\""))
        if kq_per_tile
            symmetry === nothing || throw(ArgumentError(
                "run_eph_over_k_and_q does not reduce the outer k points: pass symmetry = nothing"))
            (el_k_eigenpairs === nothing || all(_input_ngrid(kpts_input) .> 0)) || throw(ArgumentError(
                "el_k_eigenpairs needs the outer k points on a grid: the cache is looked up on one"))
            mode === :Linear && throw(ArgumentError(
                "run_eph_over_k_and_q solves the k+q states per tile, which adaptive energy " *
                "conservation (:Linear) cannot use: it needs the k+q velocities on a grid"))
            issubset(el_qty, (:e, :u)) || throw(ArgumentError(
                "run_eph_over_k_and_q solves the k+q states per tile, with `e` and `u` only; the " *
                "loop and the calculators request $(setdiff(el_qty, (:e, :u)))"))
            el_kq_eigenpairs === nothing || throw(ArgumentError(
                "el_kq_eigenpairs is not supported by run_eph_over_k_and_q: the k+q states are " *
                "solved per tile"))
        else
            ng_k, ng_kq = _input_ngrid(kpts_input), _input_ngrid(second_input)
            (all(ng_k .> 0) && all(ng_kq .> 0) &&
             (all(mod.(ng_kq, ng_k) .== 0) || all(mod.(ng_k, ng_kq) .== 0))) || throw(ArgumentError(
                "run_eph_over_k_and_kq needs commensurate k and k+q grids (got $ng_k and $ng_kq): " *
                "the phonon states are built on the q grid they span"))
        end
    else
        covariant_derivative_of_g && throw(ArgumentError(
            "covariant_derivative_of_g is not supported by run_eph_over_q_and_k"))
        all(_input_ngrid(kpts_input) .> 0) || throw(ArgumentError(
            "run_eph_over_q_and_k needs the k points on a grid: a grid size, a k-point set on a " *
            "grid or a FilteredBandStates of one"))
        mpi_comm_k === nothing || throw(ArgumentError("mpi_comm_k is not implemented for run_eph_over_q_and_k"))
        q_on_grid = second_input isa NTuple{3, Int} || (second_input isa AbstractKpoints && all(second_input.ngrid .> 0))
        (precompute_el_kq || mode === :Linear) && !q_on_grid && throw(ArgumentError(
            "precompute_el_kq and adaptive energy conservation need a q grid, not a q list"))
        # The precomputed k+q states live on the q grid, which holds every k + q only when it is a
        # multiple of the k grid.
        if precompute_el_kq
            ng_k, ng_q = _input_ngrid(kpts_input), _input_ngrid(second_input)
            all(mod.(ng_q, ng_k) .== 0) || throw(ArgumentError(
                "precompute_el_kq needs a q grid that is a multiple of the k grid (got k $ng_k, " *
                "q $ng_q): k + q is off the q grid otherwise"))
        end
        if !precompute_el_kq
            issubset(el_qty, (:e, :u)) || throw(ArgumentError(
                "run_eph_over_q_and_k solves the k+q states per tile, with `e` and `u` only; the " *
                "loop and the calculators request $(setdiff(el_qty, (:e, :u))). Pass precompute_el_kq = true."))
            el_kq_eigenpairs === nothing || throw(ArgumentError(
                "el_kq_eigenpairs needs precompute_el_kq = true: the k+q states are solved per tile"))
        end
    end
    nothing
end

_input_ngrid(x::NTuple{3, Int}) = x
_input_ngrid(x::FilteredBandStates) = x.kpts.ngrid
_input_ngrid(x::AbstractKpoints) = x.ngrid


# The state containers of a run: the k side from its selection, the k+q side (`nothing` when solved
# per tile: under `OuterQLoop`, or under `OuterKLoop` with `kq_per_tile`), the q set and its
# phonons, all on `backend`.
function _setup_states(order, model::Model{FT}, kpts_input, second_input, el_qty, ph_qty;
        kq_per_tile = false, backend,
        window_k, window_kq, symmetry, precompute_el_kq, keep_all_qpts, eph_phonon_basis,
        fourier_mode, mpi_comm_k, el_k_eigenpairs, el_kq_eigenpairs, ph_eigenpairs,
        verbosity) where {FT}
    (; nw, nmodes) = model
    # The host solves of a GPU run (q filter, polar phonons) keep the default interpolation.
    backend isa CPUBackend || (fourier_mode = "gridopt")
    sel_k = kpts_input isa FilteredBandStates ? kpts_input : maybe_time(verbosity) do
        filter_electron_states(kpts_input, nw, model.el_ham, window_k; symmetry, fourier_mode, backend,
                               mpi_comm = mpi_comm_k)
    end
    kpts = sel_k.kpts
    els_k = maybe_time(verbosity) do
        compute_electron_states_batched(model, sel_k, el_qty; fourier_mode, backend,
            eigenpairs = el_k_eigenpairs)
    end

    if order isa OuterKLoop && kq_per_tile
        # The inner set is the q set itself; k + q is solved per tile.
        qpts = second_input isa NTuple{3, Int} ? kpoints_grid(second_input) : second_input
        sel_kq = kqpts = els_kq = nothing
    elseif order isa OuterKLoop
        if second_input isa FilteredBandStates
            # A prebuilt full-BZ selection, consumed as it is.
            sel_kq = second_input
            kqpts = sel_kq.kpts
            els_kq = maybe_time(verbosity) do
                compute_electron_states_batched(model, sel_kq, el_qty; fourier_mode, backend,
                    eigenpairs = el_kq_eigenpairs)
            end
        else
            # A grid, filtered to the window; under symmetry IBZ-filtered, then unfolded.
            sel_kqf = maybe_time(verbosity) do
                filter_electron_states(second_input, nw, model.el_ham, window_kq; symmetry, fourier_mode, backend)
            end
            kqpts = symmetry === nothing ? sel_kqf.kpts : unfold_kpoints(sel_kqf.kpts, symmetry)[1]
            els_kq = maybe_time(verbosity) do
                compute_electron_states_batched(model, kqpts, el_qty, window_kq; fourier_mode, backend,
                    eigenpairs = el_kq_eigenpairs)
            end
            sel_kq = electron_states_to_FilteredBandStates(kqpts, els_kq, sel_kqf.nstates_base; nw)
        end
        # The q grid spanned by the commensurate k and k+q grids (`_check_run`).
        ngrid_q = all(mod.(kqpts.ngrid, kpts.ngrid) .== 0) ? kqpts.ngrid : kpts.ngrid
        qpts = maybe_time(verbosity) do
            combine_kpoint_grids(kqpts, kpts, -, ngrid_q)
        end
    else
        qpts_all = second_input isa NTuple{3, Int} ? kpoints_grid(second_input) : second_input
        qpts = keep_all_qpts ? qpts_all : maybe_time(verbosity) do
            filter_qpoints(qpts_all, kpts, nw, model.el_ham, window_kq; fourier_mode)
        end
        if precompute_el_kq
            sel_kq = maybe_time(verbosity) do
                filter_electron_states(qpts.ngrid, nw, model.el_ham, window_kq;
                    shift = kpts.shift + qpts.shift, fourier_mode, backend)
            end
            kqpts = GridKpoints(sel_kq.kpts)
            els_kq = maybe_time(verbosity) do
                compute_electron_states_batched(model, sel_kq, el_qty; fourier_mode, backend,
                    eigenpairs = el_kq_eigenpairs)
            end
        else
            sel_kq = kqpts = els_kq = nothing
        end
    end

    # The phonons on `backend`. The device builder fills `e` and `u` of a non-polar model; the
    # other quantities are built on the host and copied over.
    phs = maybe_time(verbosity) do
        if backend isa CPUBackend || (issubset(ph_qty, (:e, :u)) && !model.polar_phonon.use)
            compute_phonon_states_batched(model, qpts, ph_qty; fourier_mode, eph_phonon_basis, backend,
                                          eigenpairs = ph_eigenpairs)
        else
            ph_host = compute_phonon_states_batched(model, qpts, ph_qty; fourier_mode, eph_phonon_basis,
                                                    eigenpairs = ph_eigenpairs)
            copy_batched_phonon_states!(BatchedPhononState(backend, nmodes, qpts.n, ph_qty; qpts, FT),
                                        ph_host, 1:qpts.n)
        end
    end

    if verbosity > 0 && mpi_isroot()
        @info "Number of k points = $(kpts.n)"
        kqpts === nothing || @info "Number of k+q points = $(kqpts.n)"
        @info "Number of q points = $(qpts.n)"
    end
    (; els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq)
end


# Fill `iqs[1:nq]` with the index into `qpts` of `x_{k+q} - x_k` for outer k `ik` and every k+q of
# the tile `qstart .+ (0:nq-1)`, by integer grid hash on the coordinates the engine reduced once.
function _fill_iqs!(iqs, qpts, xkqs_int, xks_int, ik, qstart, nq)
    ng1, ng2, ng3 = qpts.ngrid
    k1, k2, k3 = xks_int[1, ik], xks_int[2, ik], xks_int[3, ik]
    for j in 1:nq
        ikq = qstart + j - 1
        # Both operands are pre-reduced into 0:ng-1, so the fold is a compare-and-add.
        h1 = _wrap_reduced(xkqs_int[1, ikq] - k1, ng1)
        h2 = _wrap_reduced(xkqs_int[2, ikq] - k2, ng2)
        h3 = _wrap_reduced(xkqs_int[3, ikq] - k3, ng3)
        hash = (h1 * ng2 + h2) * ng3 + h3
        iq = _ik_from_hash(qpts, hash)
        # 0 = miss on either index.
        (iq < 1 || iq > qpts.n) && throw(ArgumentError("kq - k = q point not found in precomputed qpts"))
        iqs[j] = iq
    end
    iqs
end

# `f(chunk, inds)` for each CPU thread chunk of the inner points `1:n`, threaded only when there is
# more than one chunk: a device run stays on the calling task, whose stream the brackets use.
function _foreach_chunk(f, nchunks, n)
    n == 0 && return nothing
    nchunks == 1 && return f(1, 1:n)
    @threads for (chunk, inds) in collect(enumerate(chunks(1:n; n = min(nchunks, n))))
        f(chunk, inds)
    end
end


# ---- OuterKLoop --------------------------------------------------------------------------------

# One outer-k batch: stage 1 for the batch, then per thread chunk its k+q tiles, each tile's phase
# built once and reused by every k of the batch (tile-major).
function _loop_outer_k!(eng::OuterKEngine, batch, els_k, els_kq, phs, kpts, qpts, calculators, model,
        energy_conservation, ngrid)
    stage1!(eng, els_k, kpts, batch)
    _foreach_chunk(length(eng.tiles), els_kq.nk) do chunk, ikqs
        tile_bufs = eng.tiles[chunk]
        ctx = LoopContext(eng.backend, OuterKLoop(), batch, chunk)
        for tile in Iterators.partition(ikqs, eng.n_inner_tile)
            n = length(tile)
            # The k+q side is a contiguous slice of the resident container; its phase is shared by
            # every k of the batch.
            els_kq_t = view(els_kq, tile)
            phase = view(tile_bufs.P_kq, :, 1:n)
            @views build_fourier_phase!(phase, eng.irvecp_mat, eng.xkq[:, tile])
            for (iouter, ik) in enumerate(batch)
                # This (k, tile)'s q indices, checked on the host and copied once into the device buffer.
                _fill_iqs!(tile_bufs.iq, qpts, eng.xkqs_int, eng.xks_int, ik, first(tile), n)
                copyto!(tile_bufs.iq_dev, 1, tile_bufs.iq, 1, n)
                iq = view(tile_bufs.iq_dev, 1:n)
                copy_batched_phonon_states!(tile_bufs.phs, phs, iq)
                pairs = (; n, iouter,
                    els_k = view(eng.els_k_batch, iouter:iouter),
                    els_kq = els_kq_t, phs = view(tile_bufs.phs, 1:n), phase,
                    ik, ikq = tile, iq, wtk = kpts.weights[ik], wtq = view(eng.wtkq, tile),
                    xk = kpts.vectors[ik], xq = view(qpts.vectors, view(tile_bufs.iq, 1:n)))
                _block!(eng, tile_bufs, pairs, ctx, calculators, model, energy_conservation, ngrid)
            end
        end
    end
end

# One outer-k batch of `run_eph_over_k_and_q`: stage 1 for the batch, then per thread chunk its q
# tiles. The q tile's phonons are a view of the run's; the k+q states and the stage-2 phase at
# x_k + x_q are built per (k, tile).
function _loop_outer_k_over_q!(eng::OuterKEngine, batch, els_k, phs, kpts, qpts, calculators, model,
        energy_conservation, ngrid, window_kq)
    stage1!(eng, els_k, kpts, batch)
    _foreach_chunk(length(eng.tiles), qpts.n) do chunk, iqs
        tile_bufs = eng.tiles[chunk]
        ctx = LoopContext(eng.backend, OuterKLoop(), batch, chunk)
        for tile in Iterators.partition(iqs, eng.n_inner_tile)
            n = length(tile)
            phs_t = view(phs, tile)
            for (iouter, ik) in enumerate(batch)
                xk = kpts.vectors[ik]
                for (j, iq) in enumerate(tile)
                    tile_bufs.kqs[j] = xk + qpts.vectors[iq]
                end
                els_kq_t = compute_electron_states_batched!(tile_bufs.els_kq, tile_bufs.itp_el_ham,
                    tile_bufs.hk, model, view(tile_bufs.kqs, 1:n), window_kq)
                # x_k + x_q on the backend, as the host sum above: stage 1 folded exp(-2πi R_p · x_k)
                # into g(k, R_p), so the phase is the one at x_{k+q}.
                xkq = view(tile_bufs.xkq, :, 1:n)
                @views xkq .= eng.xkq[:, tile] .+ eng.xk[:, iouter]
                phase = view(tile_bufs.P_kq, :, 1:n)
                build_fourier_phase!(phase, eng.irvecp_mat, xkq)
                pairs = (; n, iouter,
                    els_k = view(eng.els_k_batch, iouter:iouter),
                    els_kq = els_kq_t, phs = phs_t, phase,
                    ik, ikq = nothing, iq = tile, wtk = kpts.weights[ik], wtq = view(eng.wtkq, tile),
                    xk, xq = view(qpts.vectors, tile))
                _block!(eng, tile_bufs, pairs, ctx, calculators, model, energy_conservation, ngrid)
            end
        end
    end
end


# ---- OuterQLoop --------------------------------------------------------------------------------

# One outer-q batch: stage 1 for the batch, then per q one threaded region over the k tiles.
function _loop_outer_q!(eng::OuterQEngine, batch, els_k, els_kq, phs, kpts, kqpts, qpts, calculators,
        model, energy_conservation, ngrid, eph_phonon_basis, window_kq)
    stage1!(eng, phs, qpts, batch, eph_phonon_basis)
    for (iouter, iq) in enumerate(batch)
        copyto!(eng.eRpq.op_r, view(eng.ep_Rq, :, :, iouter))
        _foreach_chunk(length(eng.tiles), els_k.nk) do chunk, iks
            tile_bufs = eng.tiles[chunk]
            ctx = LoopContext(eng.backend, OuterQLoop(), batch, chunk)
            xq = qpts.vectors[iq]
            phs_q = view(phs, iq:iq)
            for tile in Iterators.partition(iks, eng.n_inner_tile)
                n = length(tile)
                copy_batched_electron_states!(tile_bufs.els_k, els_k, tile)
                for (j, ik) in enumerate(tile)
                    tile_bufs.kqs[j] = kpts.vectors[ik] + xq
                end
                if els_kq === nothing
                    # k+q solved into the tile, at the box of its largest window.
                    els_kq_t = compute_electron_states_batched!(tile_bufs.els_kq, tile_bufs.itp_el_ham,
                        tile_bufs.hk, model, view(tile_bufs.kqs, 1:n), window_kq)
                    ikq = nothing
                else
                    # Precomputed k+q by grid lookup; 0 (dropped by `filter_pairs!`) where it is absent,
                    # which the copy reads as point 1.
                    for j in 1:n
                        tile_bufs.ikq[j] = something(xk_to_ik_unsafe(tile_bufs.kqs[j], kqpts), 0)
                        tile_bufs.ikq_copy[j] = max(tile_bufs.ikq[j], 1)
                    end
                    copy_batched_electron_states!(tile_bufs.els_kq, els_kq, view(tile_bufs.ikq_copy, 1:n))
                    els_kq_t = view(tile_bufs.els_kq, 1:n)
                    ikq = view(tile_bufs.ikq, 1:n)
                end
                pairs = (; n, iouter = 0, els_k = view(tile_bufs.els_k, 1:n),
                    els_kq = els_kq_t, phs = phs_q, phase = nothing, ik = tile, ikq, iq,
                    wtk = view(eng.wtk, tile), wtq = qpts.weights[iq], xk = view(kpts.vectors, tile), xq)
                _block!(eng, tile_bufs, pairs, ctx, calculators, model, energy_conservation, ngrid)
            end
        end
    end
end


# ---- Both orders -------------------------------------------------------------------------------

# One block: drop the pairs the loop does not compute, contract the e-ph matrix, add its terms and
# hand it to the calculators.
function _block!(eng, tile_bufs, pairs, ctx, calculators, model, energy_conservation, ngrid)
    pairs = filter_pairs!(tile_bufs, pairs, ctx.order, model, energy_conservation, ngrid)
    pairs.n == 0 && return nothing
    ep, dg = stage2!(eng, tile_bufs, pairs)
    # Every pair field but the engine inputs is an `EPBlock` field.
    block = EPBlock{typeof(ctx.order)}(; ep, dg,
        Base.structdiff(pairs, NamedTuple{(:n, :iouter, :phase)})...)
    finish_ep!(block, tile_bufs, model)
    foreach(c -> run_calculator!(c, block, ctx), calculators)
    nothing
end

"""
    filter_pairs!(tile_bufs, pairs, order, model, energy_conservation, ngrid) -> pairs

The pairs of a block that the loop computes: `pairs` itself when none is dropped, otherwise the kept
ones copied into the tile buffers' `tile_bufs.kept` (`pairs` is never modified, since under `OuterKLoop`
one tile serves every k of a batch). A pair is dropped when its k+q is absent from the precomputed
states, or when `energy_conservation` (`CPUBackend`) finds no energy-conserving process in it
(`check_energy_conservation` over every mode, band pair and phonon sign, with `ngrid` the grid of
the box). The side with a scalar index is shared by the block and carried over as it is.
"""
function filter_pairs!(tile_bufs, pairs, order, model, energy_conservation, ngrid)
    kept_bufs = tile_bufs.kept
    kept_bufs === nothing && return pairs
    mode, tol = energy_conservation
    # Whether pair `j` has an energy-conserving process, by `check_energy_conservation` on its host
    # arrays (local box bands; the shared side has extent 1).
    vec3(v, i) = v === nothing ? nothing : reinterpret(reshape, Vec3{eltype(v)}, view(v, :, :, i))
    function conserves(j)
        jk, jq = min(j, pairs.els_k.nk), min(j, pairs.phs.nq)
        states_k = (; e = view(pairs.els_k.e, :, jk))
        states_kq = (; e = view(pairs.els_kq.e, :, j), vdiag = vec3(pairs.els_kq.vdiag, j))
        states_ph = (; e = view(pairs.phs.e, :, jq), vdiag = vec3(pairs.phs.vdiag, jq))
        any(check_energy_conservation(states_k, states_kq, states_ph, ib, jb, imode, sign_ph, ngrid,
                                      model.recip_lattice, mode, tol)
            for imode in 1:pairs.phs.nmodes, jb in 1:pairs.els_kq.nband[j], ib in 1:pairs.els_k.nband[jk],
                sign_ph in (-1, 1))
    end
    nkeep = 0
    for j in 1:pairs.n
        pairs.ikq === nothing || pairs.ikq[j] != 0 || continue
        mode === :None || conserves(j) || continue
        kept_bufs.keep[nkeep += 1] = j
    end
    nkeep == pairs.n && return pairs
    nkeep == 0 && return merge(pairs, (; n = 0))
    copyto!(kept_bufs.keep_dev, 1, kept_bufs.keep, 1, nkeep)
    _copy_kept_pairs(order, kept_bufs, pairs, nkeep)
end

# Copy the kept pairs of a block into the kept buffers, as views of the kept extent: under `OuterKLoop`
# the k+q side, the phonons and the shared phase; under `OuterQLoop` the k and k+q sides. The side the
# block shares is carried over as it is.
function _copy_kept_pairs(::OuterKLoop, kept_bufs, pairs, n)
    keep, keep_dev, kept = view(kept_bufs.keep, 1:n), view(kept_bufs.keep_dev, 1:n), 1:n
    els_kq = view(copy_batched_electron_states!(reshape_view_batched_electron_states(
        kept_bufs.els_kq, pairs.els_kq.nband_max, kept_bufs.els_kq.nk), pairs.els_kq, keep_dev), kept)
    phs = view(copy_batched_phonon_states!(kept_bufs.phs, pairs.phs, keep_dev), kept)
    phase = view(_copy_last_axis!(kept_bufs.P_kq, pairs.phase, keep_dev), :, kept)
    iq = view(_copy_last_axis!(kept_bufs.iq_dev, pairs.iq, keep_dev), kept)
    wtq = view(_copy_last_axis!(kept_bufs.wtq, pairs.wtq, keep_dev), kept)
    for (i, j) in enumerate(keep)
        pairs.ikq === nothing || (kept_bufs.ikq[i] = pairs.ikq[j])
        kept_bufs.xq[i] = pairs.xq[j]
    end
    ikq = pairs.ikq === nothing ? nothing : view(kept_bufs.ikq, kept)
    merge(pairs, (; n, els_kq, phs, phase, iq, wtq, ikq, xq = view(kept_bufs.xq, kept)))
end

function _copy_kept_pairs(::OuterQLoop, kept_bufs, pairs, n)
    keep, keep_dev, kept = view(kept_bufs.keep, 1:n), view(kept_bufs.keep_dev, 1:n), 1:n
    copy_els(els_dst, els_src) = view(copy_batched_electron_states!(reshape_view_batched_electron_states(
        els_dst, els_src.nband_max, els_dst.nk), els_src, keep_dev), kept)
    els_k, els_kq = copy_els(kept_bufs.els_k, pairs.els_k), copy_els(kept_bufs.els_kq, pairs.els_kq)
    wtk = view(_copy_last_axis!(kept_bufs.wtk, pairs.wtk, keep_dev), kept)
    for (i, j) in enumerate(keep)
        kept_bufs.ik[i] = pairs.ik[j]
        kept_bufs.xk[i] = pairs.xk[j]
        pairs.ikq === nothing || (kept_bufs.ikq[i] = pairs.ikq[j])
    end
    ikq = pairs.ikq === nothing ? nothing : view(kept_bufs.ikq, kept)
    merge(pairs, (; n, els_k, els_kq, wtk, ikq, ik = view(kept_bufs.ik, kept), xk = view(kept_bufs.xk, kept)))
end

"""
    estimate_device_memory(model; nk, nkq, n_outer_batch = nothing, n_inner_tile = nothing,
                           calculators = [], backend = CPUBackend(), nchunks_threads = nthreads())
        -> NamedTuple

Estimate the device memory of an e-ph run without running it, from the byte counts the loop plans
with (`engine_bytes` and the calculators' `eph_batched_bytes_per_point`) at box widths `nw`, so a
windowed run uses less. The order follows `model.epmat_outer_momentum` (`el` → outer-k, `ph` →
outer-q); the state containers are not counted. Returns `(; loop, committed, per_pair, batch,
free)`, `batch` the inner tile the run would pick on `backend`, with the run's defaults.

Actual device usage starts ~100-150 MB higher: the CUDA library context and workspace (cuBLAS
etc.) are allocated lazily on the first kernel launch and are not a per-run buffer.
"""
function estimate_device_memory(model::Model{FT}; nk::Integer, nkq::Integer, n_outer_batch = nothing,
        n_inner_tile = nothing, calculators = [], backend::AbstractBackend = CPUBackend(),
        nchunks_threads = nthreads()) where {FT}
    outer_k = model.epmat_outer_momentum == "el"
    order = outer_k ? OuterKLoop() : OuterQLoop()
    el_qty = union(loop_el_quantities((:None, 0.0)), required_el_quantities.(calculators)...)
    ph_qty = union(loop_ph_quantities(model, (:None, 0.0)), required_ph_quantities.(calculators)...)
    (; nb_inner, committed, bytes) = _plan_widths(order, model, backend, calculators;
        n_outer = outer_k ? Int(nk) : Int(nkq), n_inner = outer_k ? Int(nkq) : Int(nk), nk, nkq,
        nchunks = backend isa CPUBackend ? nchunks_threads : 1, n_outer_batch, n_inner_tile,
        nband_max_k = model.nw, nband_max_kq = model.nw, els_k = nothing, els_kq = nothing, phs = nothing,
        el_qty, ph_qty, drop_pairs = false, precompute_el_kq = false, covariant_derivative_of_g = false,
        eph_phonon_basis = :eigenmode)
    (; loop = outer_k ? :outer_k : :outer_q, committed, bytes.per_pair, batch = nb_inner,
       free = free_bytes(backend))
end


# =============================================================================
# Deprecated driver name — forwarder, removed after one release. Explicit @warn (maxlog=1) because
# Base.@deprecate depwarns are invisible in ordinary script runs (Julia ≥ 1.5). The driver was renamed
# to the run_eph_over_<outer>_and_<inner> scheme. Delete this block when the old name goes.
function run_eph_outer_q(args...; kwargs...)
    @warn "run_eph_outer_q is deprecated; use run_eph_over_q_and_k (identical arguments)." maxlog=1
    run_eph_over_q_and_k(args...; kwargs...)
end
