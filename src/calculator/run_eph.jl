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

Requires a model loaded with `epmat_outer_momentum = "el"` so stage 1 contracts R_e.

Keywords:
* `calculators` — at least one; each must `supports(calc, OuterKLoop)`.
* `backend = CPUBackend()` — `ElectronPhonon.gpu_backend()` for a GPU run.
* `window_k`, `window_kq` — energy windows of the two sides (ignored for a `FilteredBandStates`).
* `symmetry = model.symmetry` — reduces the outer k to the irreducible wedge and builds the k+q
  set by unfolding its selection; `nothing` for full grids.
* `energy_conservation = (:None, 0.0)` — `(:Fixed, tol)` or `(:Linear, curvature)` drops the point pairs
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
The model must use `epmat_outer_momentum = "el"`.
"""
function run_eph_over_k_and_q(model::Model, kpts_input, qpts_input; symmetry = nothing, kwargs...)
    # The inner points are q, not k+q; solve the k+q electron states within each tile.
    _run_eph(OuterKLoop(), model, kpts_input, qpts_input; inner_loop_kq = false, symmetry, kwargs...)
end

"""
    run_eph_over_q_and_k(model, kpts, qpts; calculators, backend, kwargs...)

Sweep the outer q points (any q list, e.g. a path) and, for each, the inner k points, handing each
calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterQLoop}`: one q with a tile of k points.
The k+q states are solved per tile, with `e` and `u` only, unless `precompute_el_kq = true` (a grid
q set), which builds them once on the k+q grid; a pair whose k+q has no state in `window_kq` is
then dropped. Returns `(; kpts, qpts, els_k, els_kq, phs)`, `els_kq = nothing` when solved per tile.
Requires a model loaded with `epmat_outer_momentum = "ph"` so stage 1 contracts R_p.

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
loop_el_quantities((energy_conservation_mode, _)) =
    [:e; :u; energy_conservation_mode === :Linear ? [:vdiag] : Symbol[]]
function loop_ph_quantities(model, (energy_conservation_mode, _))
    [:e; :u; model.polar_eph.use ? [:eph_dipole_coeff] : Symbol[];
     energy_conservation_mode === :Linear ? [:vdiag] : Symbol[]]
end

# `inner_loop_kq` (`OuterKLoop` only): true for run_eph_over_k_and_kq (inner k+q grid),
# false for run_eph_over_k_and_q (inner q points, with k+q solved per tile).
function _run_eph(order::LoopTag, model::Model{FT}, kpts_input, second_input;
        inner_loop_kq = true,
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
        precompute_el_kq, screening_params, mpi_comm_k, el_kq_eigenpairs, symmetry, inner_loop_kq,
        el_k_eigenpairs)

    # Prepare resident electron/phonon states and the point sets needed by the chosen driver.
    (; els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq) = _setup_states(
        order,
        model,
        kpts_input,
        second_input,
        el_qty,
        ph_qty;
        inner_loop_kq,
        backend,
        window_k,
        window_kq,
        symmetry,
        precompute_el_kq,
        keep_all_qpts,
        eph_phonon_basis,
        fourier_mode,
        mpi_comm_k,
        el_k_eigenpairs,
        el_kq_eigenpairs,
        ph_eigenpairs,
        verbosity,
    )

    # One setup-time inference barrier: runtime quantity lists determine the concrete state types.
    Base.inferencebarrier(_run_eph_loop)(order, model;
        els_k,
        els_kq,
        phs,
        kpts,
        kqpts,
        qpts,
        sel_k,
        sel_kq,
        el_qty,
        ph_qty,
        calculators,
        backend,
        symmetry,
        precompute_el_kq,
        energy_conservation,
        covariant_derivative_of_g,
        eph_phonon_basis,
        n_outer_batch,
        n_inner_tile,
        nchunks_threads,
        window_kq,
        progress_print_step,
        verbosity,
    )
end

# The loop of `_run_eph` on the built states, compiled for their concrete types (`els_kq` and `kqpts`
# are `nothing` for the per-tile k+q solve).
function _run_eph_loop(order, model; els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq, el_qty,
        ph_qty, calculators, backend, symmetry, precompute_el_kq, energy_conservation,
        covariant_derivative_of_g, eph_phonon_basis, n_outer_batch, n_inner_tile, nchunks_threads,
        window_kq, progress_print_step, verbosity)
    (; nw, nmodes) = model

    nchunks = backend isa CPUBackend ? nchunks_threads : 1
    # Allocate compaction buffers only when a point pair can be dropped: energy conservation,
    # or an absent k+q point in the outer-q driver's precomputed state container.
    drop_pairs = energy_conservation[1] !== :None || (order isa OuterQLoop && precompute_el_kq)

    if order isa OuterKLoop
        # Outer k: the inner set is either resident k+q points or q points with a per-tile solve.
        inner_loop_kq = els_kq !== nothing
        inner_pts = inner_loop_kq ? kqpts : qpts
        n_outer = kpts.n
    else
        # Outer q: the inner set is the resident k points.
        inner_loop_kq = false
        inner_pts = kpts
        n_outer = qpts.n
    end
    n_inner = inner_pts.n

    (; nbatch_outer, nbatch_inner, committed, bytes) = _plan_widths(order, model, backend, calculators;
        n_outer, n_inner, nk = kpts.n, nkq = order isa OuterKLoop ? inner_pts.n : 0, nchunks,
        inner_loop_kq,
        n_outer_batch, n_inner_tile, nband_max_k = els_k.nband_max,
        nband_max_kq = els_kq === nothing ? nw : els_kq.nband_max, els_k, els_kq, phs, el_qty, ph_qty,
        drop_pairs, precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis)
    if verbosity > 0 && mpi_isroot()
        @info "e-ph loop: committed = $(round(committed / 1e9, digits = 2)) GB, " *
              "$(round(bytes.per_pair / 1e3, digits = 1)) kB per pair; outer batch = $nbatch_outer, " *
              "inner tile = $nbatch_inner, $nchunks chunk(s)"
    end

    # Allocate the engine for the requested outer momentum.
    if order isa OuterKLoop
        eng = OuterKEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, kqpts, qpts,
            n_outer_batch = nbatch_outer, n_inner_tile = nbatch_inner, nchunks, drop_pairs,
            covariant_derivative_of_g, eph_phonon_basis)
    else
        eng = OuterQEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, qpts,
            n_outer_batch = nbatch_outer, n_inner_tile = nbatch_inner, nchunks, drop_pairs, eph_phonon_basis)
    end
    for calculator in calculators
        setup_calculator!(calculator, backend, els_k, els_kq, phs; sel_k, sel_kq, nw, nmodes,
            nchunks_threads = nchunks, n_outer_batch = nbatch_outer, n_inner_tile = nbatch_inner, verbosity)
    end

    ngrid_econv = order isa OuterKLoop ? inner_pts.ngrid : qpts.ngrid   # the energy-conservation box
    for batch in Iterators.partition(1:n_outer, nbatch_outer)
        if mpi_isroot() &&
                div(last(batch), progress_print_step) > div(first(batch) - 1, progress_print_step)
            @info "$(now()) $(order isa OuterKLoop ? "ik" : "iq") = $batch / $n_outer"
            flush(stdout); flush(stderr)
        end
        ctx = LoopContext(backend, order, batch, 1)
        for calculator in calculators
            calculator_begin!(calculator, ctx)
        end

        if order isa OuterKLoop && !inner_loop_kq
            # Outer k, inner q: solve k+q states per tile.
            _loop_outer_k_over_q!(eng, batch, els_k, phs, kpts, qpts, calculators, model,
                                  energy_conservation, ngrid_econv, window_kq)
        elseif order isa OuterKLoop
            # Outer k, inner k+q: gather the corresponding resident phonons.
            _loop_outer_k!(eng, batch, els_k, els_kq, phs, kpts, qpts, calculators, model,
                           energy_conservation, ngrid_econv)
        else
            # Outer q, inner k: solve or gather k+q states per tile.
            _loop_outer_q!(eng, batch, els_k, els_kq, phs, kpts, kqpts, qpts, calculators, model,
                           energy_conservation, ngrid_econv, eph_phonon_basis, window_kq)
        end
        for calculator in calculators
            calculator_end!(calculator, ctx)
        end
        # Bound the host look-ahead to one batch, so its device scratch does not pile up in the pool.
        synchronize(backend)
    end

    # Preserve the pre-PR outer-q postprocessing contract; use_symmetry controls k-point reduction.
    symmetry_post = order isa OuterQLoop ? model.symmetry : symmetry
    for calculator in calculators
        postprocess_calculator!(calculator; qpts, symmetry = symmetry_post)
    end
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
        drop_pairs, precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq = true)
    (; nw, nmodes) = model
    outer_default = order isa OuterKLoop ? 256 : backend isa CPUBackend ? 1 : 16
    nbatch_outer = max(1, min(something(n_outer_batch, outer_default), n_outer))
    inner_default = backend isa CPUBackend ? 1024 : order isa OuterKLoop ? n_inner : 2^15
    inner_cap = max(1, min(cld(n_inner, nchunks), something(n_inner_tile, inner_default)))
    bytes = order isa OuterKLoop ?
        engine_bytes(OuterKEngine, model; nband_max_k, nband_max_kq, nk, nkq, el_qty, ph_qty,
            drop_pairs, covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq) :
        engine_bytes(OuterQEngine, model; nband_max_k, nband_max_kq, nk, n_outer_batch = nbatch_outer,
            el_qty, ph_qty, drop_pairs, precompute_el_kq, eph_phonon_basis)
    for c in calculators
        b = eph_batched_bytes_per_point(c, EPBlock{typeof(order)}; nw, nmodes, nband_max_k,
                                        nband_max_kq, els_k, els_kq, phs, nchunks_threads = nchunks)
        bytes = (; persistent = bytes.persistent + b.persistent,
                   per_outer = bytes.per_outer + b.per_outer, per_pair = bytes.per_pair + b.per_pair)
    end
    committed = bytes.persistent + bytes.per_outer * nbatch_outer
    nbatch_inner = plan_batch(backend, bytes.per_pair * nchunks, committed, inner_cap;
                          what = order isa OuterKLoop ? "outer-k" : "outer-q")
    (; nbatch_outer, nbatch_inner, committed, bytes)
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
        inner_loop_kq = true, el_k_eigenpairs = nothing)
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
    _require_epmat_layout(order, model)
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
        if !inner_loop_kq
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
# per tile: under `OuterQLoop`, or under `OuterKLoop` with `inner_loop_kq = false`), the q set and its
# phonons, all on `backend`.
function _setup_states(order, model::Model{FT}, kpts_input, second_input, el_qty, ph_qty;
        inner_loop_kq = true, backend,
        window_k, window_kq, symmetry, precompute_el_kq, keep_all_qpts, eph_phonon_basis,
        fourier_mode, mpi_comm_k, el_k_eigenpairs, el_kq_eigenpairs, ph_eigenpairs,
        verbosity) where {FT}
    (; nw, nmodes) = model
    # The host solves of a GPU run (q filter, polar phonons) keep the default interpolation.
    backend isa CPUBackend || (fourier_mode = "gridopt")
    # Reuse a supplied k selection, or select its bands within the requested energy window.
    if kpts_input isa FilteredBandStates
        sel_k = kpts_input
    else
        sel_k = maybe_time(verbosity) do
            filter_electron_states(kpts_input, nw, model.el_ham, window_k; symmetry, fourier_mode, backend,
                                   mpi_comm = mpi_comm_k)
        end
    end

    # Compute the resident electron states at the selected k points.
    kpts = sel_k.kpts
    els_k = maybe_time(verbosity) do
        compute_electron_states_batched(model, sel_k, el_qty; fourier_mode, backend,
            eigenpairs = el_k_eigenpairs)
    end

    # Prepare electron states at k+q, or leave them for the inner-tile eigensolve.
    if order isa OuterKLoop && !inner_loop_kq
        # The inner set is the q set itself; k + q is solved per tile.
        qpts = second_input isa NTuple{3, Int} ? kpoints_grid(second_input) : second_input
        sel_kq = kqpts = els_kq = nothing
    elseif order isa OuterKLoop
        # Inner k+q grid: build all resident k+q electron states before the e-ph loop.
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
        # Outer q: filter the q set, then optionally precompute the k+q grid's states.
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


# ---- OuterKLoop --------------------------------------------------------------------------------

# Compute stage 1 once, then explicitly divide the inner k+q points into thread chunks.
function _loop_outer_k!(eng::OuterKEngine, batch, els_k, els_kq, phs, kpts, qpts, calculators, model,
        energy_conservation, ngrid_econv)
    stage1!(eng, els_k, kpts, batch)
    els_kq.nk == 0 && return nothing
    eng_fields = _workspace_fields(eng)

    # A single chunk (including every GPU run) stays on the caller's task and CUDA stream.
    if length(eng.tiles) == 1
        _loop_outer_k_chunk!(eng_fields, _workspace_fields(eng.tiles[1]), els_kq, phs, kpts, qpts;
            batch, chunk = 1, ikqs = 1:els_kq.nk, calculators, model, energy_conservation, ngrid_econv)
    else
        # Each CPU chunk owns its scratch; all chunks finish before the next outer batch.
        inner_chunks = collect(enumerate(chunks(1:els_kq.nk; n = min(length(eng.tiles), els_kq.nk))))
        @threads for (chunk, ikqs) in inner_chunks
            _loop_outer_k_chunk!(eng_fields, _workspace_fields(eng.tiles[chunk]), els_kq, phs, kpts, qpts;
                batch, chunk, ikqs, calculators, model, energy_conservation, ngrid_econv)
        end
    end
    nothing
end

function _loop_outer_k_chunk!(eng, tile_workspace, els_kq, phs, kpts, qpts;
        batch, chunk, ikqs, calculators, model, energy_conservation, ngrid_econv)
    # function barrier for one CPU/GPU chunk's concrete engine and tile buffers.
    ctx = LoopContext(eng.backend, OuterKLoop(), batch, chunk)
    # Iterate over this chunk's inner-point tiles.
    for tile in Iterators.partition(ikqs, eng.n_inner_tile)
        n = length(tile)
        # The k+q side is a contiguous slice of the resident container; its phase is shared by
        # every k of the batch.
        els_kq_t = view(els_kq, tile)
        phase = view(tile_workspace.P_kq, :, 1:n)
        @views build_fourier_phase!(phase, eng.irvecp_mat, eng.xkq[:, tile])

        # Work on one outer k point with the current inner tile.
        for (iouter, ik) in enumerate(batch)
            # This (k, tile)'s q indices, checked on the host and copied once into the device buffer.
            _fill_iqs!(tile_workspace.iq, qpts, eng.xkqs_int, eng.xks_int, ik, first(tile), n)
            copyto!(tile_workspace.iq_dev, 1, tile_workspace.iq, 1, n)
            iq = view(tile_workspace.iq_dev, 1:n)
            copy_batched_phonon_states!(tile_workspace.phs, phs, iq)
            eph_inputs = (; n, iouter,
                els_k = view(eng.els_k_batch, iouter:iouter),
                els_kq = els_kq_t, phs = view(tile_workspace.phs, 1:n), phase,
                ik, ikq = tile, iq, wtk = kpts.weights[ik], wtq = view(eng.wtkq, tile),
                xk = kpts.vectors[ik], xq = view(qpts.vectors, view(tile_workspace.iq, 1:n)))

            # Filter point pairs before computing the expensive e-ph matrix.
            eph_inputs = filter_pairs!(tile_workspace, eph_inputs, ctx.order, model, energy_conservation, ngrid_econv)
            eph_inputs.n == 0 && continue

            # Compute stage 2, add polar corrections, and pass the result to each calculator.
            ep, dg = _stage2!(ctx.order, eng, tile_workspace, eph_inputs)
            block = EPBlock{OuterKLoop}(; ep, dg,
                Base.structdiff(eph_inputs, NamedTuple{(:n, :iouter, :phase)})...)
            finish_ep!(block, tile_workspace, model)
            for calculator in calculators
                run_calculator!(calculator, block, ctx)
            end
        end
    end
    nothing
end

# Compute stage 1 once, then explicitly divide the inner q points into thread chunks.
function _loop_outer_k_over_q!(eng::OuterKEngine, batch, els_k, phs, kpts, qpts, calculators, model,
        energy_conservation, ngrid_econv, window_kq)
    stage1!(eng, els_k, kpts, batch)
    qpts.n == 0 && return nothing
    eng_fields = _workspace_fields(eng)

    # A single chunk stays on the caller's task; k+q states are solved within each q tile.
    if length(eng.tiles) == 1
        _loop_outer_k_over_q_chunk!(eng_fields, _workspace_fields(eng.tiles[1]), phs, kpts, qpts;
            batch, chunk = 1, iqs = 1:qpts.n, calculators, model, energy_conservation, ngrid_econv, window_kq)
    else
        # Each CPU chunk independently solves k+q and computes the e-ph matrix for its q tiles.
        inner_chunks = collect(enumerate(chunks(1:qpts.n; n = min(length(eng.tiles), qpts.n))))
        @threads for (chunk, iqs) in inner_chunks
            _loop_outer_k_over_q_chunk!(eng_fields, _workspace_fields(eng.tiles[chunk]), phs, kpts, qpts;
                batch, chunk, iqs, calculators, model, energy_conservation, ngrid_econv, window_kq)
        end
    end
    nothing
end

function _loop_outer_k_over_q_chunk!(eng, tile_workspace, phs, kpts, qpts;
        batch, chunk, iqs, calculators, model, energy_conservation, ngrid_econv, window_kq)
    # function barrier for one chunk's concrete buffers and per-tile k+q eigensolve.
    ctx = LoopContext(eng.backend, OuterKLoop(), batch, chunk)
    # Iterate over this chunk's inner-point tiles.
    for tile in Iterators.partition(iqs, eng.n_inner_tile)
        n = length(tile)
        phs_t = view(phs, tile)

        # Work on one outer k point with the current inner tile.
        for (iouter, ik) in enumerate(batch)
            xk = kpts.vectors[ik]
            for (j, iq) in enumerate(tile)
                tile_workspace.kqs[j] = xk + qpts.vectors[iq]
            end
            els_kq_t = compute_electron_states_batched!(tile_workspace.els_kq, tile_workspace.itp_el_ham,
                tile_workspace.hk, model, view(tile_workspace.kqs, 1:n), window_kq)
            # x_k + x_q on the backend, as the host sum above: stage 1 folded exp(-2πi R_p · x_k)
            # into g(k, R_p), so the phase is the one at x_{k+q}.
            xkq = view(tile_workspace.xkq, :, 1:n)
            @views xkq .= eng.xkq[:, tile] .+ eng.xk[:, iouter]
            phase = view(tile_workspace.P_kq, :, 1:n)
            build_fourier_phase!(phase, eng.irvecp_mat, xkq)
            eph_inputs = (; n, iouter,
                els_k = view(eng.els_k_batch, iouter:iouter),
                els_kq = els_kq_t, phs = phs_t, phase,
                ik, ikq = nothing, iq = tile, wtk = kpts.weights[ik], wtq = view(eng.wtkq, tile),
                xk, xq = view(qpts.vectors, tile))

            # Filter point pairs before computing the expensive e-ph matrix.
            eph_inputs = filter_pairs!(tile_workspace, eph_inputs, ctx.order, model, energy_conservation, ngrid_econv)
            eph_inputs.n == 0 && continue

            # Compute stage 2, add polar corrections, and pass the result to each calculator.
            ep, dg = _stage2!(ctx.order, eng, tile_workspace, eph_inputs)
            block = EPBlock{OuterKLoop}(; ep, dg,
                Base.structdiff(eph_inputs, NamedTuple{(:n, :iouter, :phase)})...)
            finish_ep!(block, tile_workspace, model)
            for calculator in calculators
                run_calculator!(calculator, block, ctx)
            end
        end
    end
    nothing
end


# ---- OuterQLoop --------------------------------------------------------------------------------

# Compute stage 1 once, then visit each outer q and explicitly thread its inner k chunks.
function _loop_outer_q!(eng::OuterQEngine, batch, els_k, els_kq, phs, kpts, kqpts, qpts, calculators,
        model, energy_conservation, ngrid_econv, eph_phonon_basis, window_kq)
    stage1!(eng, phs, qpts, batch, eph_phonon_basis)
    els_k.nk == 0 && return nothing
    eng_fields = _workspace_fields(eng)

    for (iouter, iq) in enumerate(batch)
        # Stage the current q's Fourier output; all k chunks share this read-only parent.
        copyto!(eng.eRpq.op_r, view(eng.ep_Rq, :, :, iouter))
        if length(eng.tiles) == 1
            _loop_outer_q_chunk!(eng_fields, _workspace_fields(eng.tiles[1]), els_k, els_kq, phs, kpts, kqpts, qpts;
                batch, chunk = 1, iks = 1:els_k.nk, iq, calculators, model, energy_conservation, ngrid_econv, window_kq)
        else
            # Finish every CPU k chunk before overwriting the shared parent for the next q.
            inner_chunks = collect(enumerate(chunks(1:els_k.nk; n = min(length(eng.tiles), els_k.nk))))
            @threads for (chunk, iks) in inner_chunks
                _loop_outer_q_chunk!(eng_fields, _workspace_fields(eng.tiles[chunk]), els_k, els_kq, phs, kpts, kqpts, qpts;
                    batch, chunk, iks, iq, calculators, model, energy_conservation, ngrid_econv, window_kq)
            end
        end
    end
    nothing
end

function _loop_outer_q_chunk!(eng, tile_workspace, els_k, els_kq, phs, kpts, kqpts, qpts;
        batch, chunk, iks, iq, calculators, model, energy_conservation, ngrid_econv, window_kq)
    # function barrier for one chunk's concrete k-tile states, interpolators, and scratch.
    ctx = LoopContext(eng.backend, OuterQLoop(), batch, chunk)
    xq = qpts.vectors[iq]
    phs_q = view(phs, iq:iq)
    # Iterate over this chunk's inner-point tiles.
    for tile in Iterators.partition(iks, eng.n_inner_tile)
        n = length(tile)
        copy_batched_electron_states!(tile_workspace.els_k, els_k, tile)
        for (j, ik) in enumerate(tile)
            tile_workspace.kqs[j] = kpts.vectors[ik] + xq
        end
        if els_kq === nothing
            # k+q solved into the tile, at the box of its largest window.
            els_kq_t = compute_electron_states_batched!(tile_workspace.els_kq, tile_workspace.itp_el_ham,
                tile_workspace.hk, model, view(tile_workspace.kqs, 1:n), window_kq)
            ikq = nothing
        else
            # Precomputed k+q by grid lookup; 0 (dropped by `filter_pairs!`) where it is absent,
            # which the copy reads as point 1.
            for j in 1:n
                tile_workspace.ikq[j] = something(xk_to_ik_unsafe(tile_workspace.kqs[j], kqpts), 0)
                tile_workspace.ikq_copy[j] = max(tile_workspace.ikq[j], 1)
            end
            copy_batched_electron_states!(tile_workspace.els_kq, els_kq, view(tile_workspace.ikq_copy, 1:n))
            els_kq_t = view(tile_workspace.els_kq, 1:n)
            ikq = view(tile_workspace.ikq, 1:n)
        end
        eph_inputs = (; n, iouter = 0, els_k = view(tile_workspace.els_k, 1:n),
            els_kq = els_kq_t, phs = phs_q, phase = nothing, ik = tile, ikq, iq,
            wtk = view(eng.wtk, tile), wtq = qpts.weights[iq], xk = view(kpts.vectors, tile), xq)

        # Filter point pairs before computing the expensive e-ph matrix.
        eph_inputs = filter_pairs!(tile_workspace, eph_inputs, ctx.order, model, energy_conservation, ngrid_econv)
        eph_inputs.n == 0 && continue

        # Compute stage 2, add polar corrections, and pass the result to each calculator.
        ep, dg = _stage2!(ctx.order, eng, tile_workspace, eph_inputs)
        block = EPBlock{OuterQLoop}(; ep, dg,
            Base.structdiff(eph_inputs, NamedTuple{(:n, :iouter, :phase)})...)
        finish_ep!(block, tile_workspace, model)
        for calculator in calculators
            run_calculator!(calculator, block, ctx)
        end
    end
    nothing
end


# ---- Both orders -------------------------------------------------------------------------------

"""
    filter_pairs!(tile_workspace, eph_inputs, order, model, energy_conservation, ngrid_econv) -> eph_inputs

The pairs of a block that the loop computes: `eph_inputs` itself when none is dropped, otherwise the kept
ones copied into the tile buffers' `tile_workspace.kept` (`eph_inputs` is never modified, since under `OuterKLoop`
one tile serves every k of a batch). A pair is dropped when its k+q is absent from the precomputed
states, or when `energy_conservation` (`CPUBackend`) finds no energy-conserving process in it
(`check_energy_conservation` over every mode, band pair and phonon sign, with `ngrid_econv` the grid of
the box). The side with a scalar index is shared by the block and carried over as it is.
"""
function filter_pairs!(tile_workspace, eph_inputs, order, model, energy_conservation, ngrid_econv)
    kept_bufs = tile_workspace.kept
    kept_bufs === nothing && return eph_inputs
    mode, tol = energy_conservation
    # Whether pair `j` has an energy-conserving process, by `check_energy_conservation` on its host
    # arrays (local box bands; the shared side has extent 1).
    vec3(v, i) = v === nothing ? nothing : reinterpret(reshape, Vec3{eltype(v)}, view(v, :, :, i))
    function conserves(j)
        jk, jq = min(j, eph_inputs.els_k.nk), min(j, eph_inputs.phs.nq)
        states_k = (; e = view(eph_inputs.els_k.e, :, jk))
        states_kq = (; e = view(eph_inputs.els_kq.e, :, j), vdiag = vec3(eph_inputs.els_kq.vdiag, j))
        states_ph = (; e = view(eph_inputs.phs.e, :, jq), vdiag = vec3(eph_inputs.phs.vdiag, jq))
        any(check_energy_conservation(states_k, states_kq, states_ph, ib, jb, imode, sign_ph, ngrid_econv,
                                      model.recip_lattice, mode, tol)
            for imode in 1:eph_inputs.phs.nmodes, jb in 1:eph_inputs.els_kq.nband[j], ib in 1:eph_inputs.els_k.nband[jk],
                sign_ph in (-1, 1))
    end
    nkeep = 0
    for j in 1:eph_inputs.n
        eph_inputs.ikq === nothing || eph_inputs.ikq[j] != 0 || continue
        mode === :None || conserves(j) || continue
        kept_bufs.keep[nkeep += 1] = j
    end
    nkeep == eph_inputs.n && return eph_inputs
    nkeep == 0 && return merge(eph_inputs, (; n = 0))
    copyto!(kept_bufs.keep_dev, 1, kept_bufs.keep, 1, nkeep)
    _copy_kept_pairs(order, kept_bufs, eph_inputs, nkeep)
end

# Copy the kept pairs of a block into the kept buffers, as views of the kept extent: under `OuterKLoop`
# the k+q side, the phonons and the shared phase; under `OuterQLoop` the k and k+q sides. The side the
# block shares is carried over as it is.
function _copy_kept_pairs(::OuterKLoop, kept_bufs, eph_inputs, n)
    keep, keep_dev, kept = view(kept_bufs.keep, 1:n), view(kept_bufs.keep_dev, 1:n), 1:n
    els_kq = view(copy_batched_electron_states!(reshape_view_batched_electron_states(
        kept_bufs.els_kq, eph_inputs.els_kq.nband_max, kept_bufs.els_kq.nk), eph_inputs.els_kq, keep_dev), kept)
    phs = view(copy_batched_phonon_states!(kept_bufs.phs, eph_inputs.phs, keep_dev), kept)
    phase = view(_copy_last_axis!(kept_bufs.P_kq, eph_inputs.phase, keep_dev), :, kept)
    iq = view(_copy_last_axis!(kept_bufs.iq_dev, eph_inputs.iq, keep_dev), kept)
    wtq = view(_copy_last_axis!(kept_bufs.wtq, eph_inputs.wtq, keep_dev), kept)
    for (i, j) in enumerate(keep)
        eph_inputs.ikq === nothing || (kept_bufs.ikq[i] = eph_inputs.ikq[j])
        kept_bufs.xq[i] = eph_inputs.xq[j]
    end
    ikq = eph_inputs.ikq === nothing ? nothing : view(kept_bufs.ikq, kept)
    merge(eph_inputs, (; n, els_kq, phs, phase, iq, wtq, ikq, xq = view(kept_bufs.xq, kept)))
end

function _copy_kept_pairs(::OuterQLoop, kept_bufs, eph_inputs, n)
    keep, keep_dev, kept = view(kept_bufs.keep, 1:n), view(kept_bufs.keep_dev, 1:n), 1:n
    copy_els(els_dst, els_src) = view(copy_batched_electron_states!(reshape_view_batched_electron_states(
        els_dst, els_src.nband_max, els_dst.nk), els_src, keep_dev), kept)
    els_k, els_kq = copy_els(kept_bufs.els_k, eph_inputs.els_k), copy_els(kept_bufs.els_kq, eph_inputs.els_kq)
    wtk = view(_copy_last_axis!(kept_bufs.wtk, eph_inputs.wtk, keep_dev), kept)
    for (i, j) in enumerate(keep)
        kept_bufs.ik[i] = eph_inputs.ik[j]
        kept_bufs.xk[i] = eph_inputs.xk[j]
        eph_inputs.ikq === nothing || (kept_bufs.ikq[i] = eph_inputs.ikq[j])
    end
    ikq = eph_inputs.ikq === nothing ? nothing : view(kept_bufs.ikq, kept)
    merge(eph_inputs, (; n, els_k, els_kq, wtk, ikq, ik = view(kept_bufs.ik, kept), xk = view(kept_bufs.xk, kept)))
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
    (; nbatch_inner, committed, bytes) = _plan_widths(order, model, backend, calculators;
        n_outer = outer_k ? Int(nk) : Int(nkq), n_inner = outer_k ? Int(nkq) : Int(nk), nk, nkq,
        nchunks = backend isa CPUBackend ? nchunks_threads : 1, n_outer_batch, n_inner_tile,
        nband_max_k = model.nw, nband_max_kq = model.nw, els_k = nothing, els_kq = nothing, phs = nothing,
        el_qty, ph_qty, drop_pairs = false, precompute_el_kq = false, covariant_derivative_of_g = false,
        eph_phonon_basis = :eigenmode)
    (; loop = outer_k ? :outer_k : :outer_q, committed, bytes.per_pair, batch = nbatch_inner,
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
