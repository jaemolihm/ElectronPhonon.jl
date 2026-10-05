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
* `symmetry = model.symmetry` — reduces the outer k to the irreducible wedge; `nothing` for the
  full grid. Symmetry reduces only a point set given as a grid size: a k-point set
  (`AbstractKpoints`, `FilteredBandStates`) is never reduced. A k+q grid size gives the full-BZ
  selection whatever `symmetry` is, and a k+q set is used as given.
* `energy_conservation_tol = Inf` — a finite tolerance drops the point pairs with no process inside
  it (`|e_k - e_{k+q} ± ω_q| <= energy_conservation_tol` for some bands and mode) before the e-ph
  matrix is computed (`CPUBackend` only).
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
    _run_eph(OuterKLoop(), model, kpts_input, kqpts_input; inner_loop_kq = true, kwargs...)

"""
    run_eph_over_k_and_q(model, kpts, qpts; calculators, backend, kwargs...)

Sweep the outer k points and, for each, the inner q points (any q list, e.g. a path, or a q grid),
handing each calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterKLoop}`: one outer k with a
tile of q points, whose index is `iq` (`ikq === nothing`). The k+q states are solved per tile, with
`e` and `u` only, and the phonons are built once on `qpts`. A pair whose k+q has no state in
`window_kq` is skipped. Returns
`(; kpts, qpts, els_k, els_kq = nothing, phs)`.

Keywords as in [`run_eph_over_k_and_kq`](@ref), except: `kpts` is any k list (e.g. a band path,
not necessarily on a grid), a grid size or a prebuilt `FilteredBandStates`; `el_k_eigenpairs` only
for k points on a grid (the cache is looked up on one); `symmetry` reduces the outer k points
(given as a grid size) and never the q points; `n_inner_tile` q points per block; no `vdiag` (for a
calculator) and no `el_kq_eigenpairs`, which need the k+q points on a grid. Calculators that read the k+q selection (`sel_kq`) are not supported.
The model must use `epmat_outer_momentum = "el"`.
"""
function run_eph_over_k_and_q(model::Model, kpts_input, qpts_input; kwargs...)
    # The inner points are q, not k+q; solve the k+q electron states within each tile.
    _run_eph(OuterKLoop(), model, kpts_input, qpts_input; inner_loop_kq = false, kwargs...)
end

"""
    run_eph_over_q_and_k(model, kpts, qpts; calculators, backend, kwargs...)

Sweep the outer q points (any q list, e.g. a path, or a q grid) and, for each, the inner k points
(a grid), handing each
calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterQLoop}`: one q with a tile of k points.
The k+q states are solved per tile, with `e` and `u` only, unless `precompute_el_kq = true` (a grid
q set), which builds them once on the k+q grid. A pair whose k+q has no state in `window_kq` is
skipped. Returns `(; kpts, qpts, els_k, els_kq, phs)`, `els_kq = nothing` when solved per tile.
Requires a model loaded with `epmat_outer_momentum = "ph"` so stage 1 contracts R_p.

Keywords as in [`run_eph_over_k_and_kq`](@ref), except: `symmetry` reduces the outer q points
(given as a grid size) to the irreducible wedge, and the inner k points are never reduced; every q
point is kept, also one with no k+q state in `window_kq`; `precompute_el_kq` needs a q grid that is a multiple of the k grid; `mpi_comm_k` is
refused;
`n_outer_batch` q points per stage-1 batch and per bracket, 16 on a GPU and 1 on the CPU (stage 1
gains nothing from a wider batch there, while a calculator's per-q buffers are held per thread
chunk); `n_inner_tile` k points per block, at most `2^15` on a GPU; no `covariant_derivative_of_g`;
`el_kq_eigenpairs` only with `precompute_el_kq`.
"""
run_eph_over_q_and_k(model::Model, kpts_input, qpts_input; kwargs...) =
    _run_eph(OuterQLoop(), model, kpts_input, qpts_input; inner_loop_kq = false, kwargs...)


# The quantities the loop provides itself, on both electron sides and on the phonons: always the
# energies and eigenvectors, and the dipole coefficients of a polar model.
function loop_el_quantities()
    [:e, :u]
end
function loop_ph_quantities(model)
    model.polar_eph.use ? [:e, :u, :eph_dipole_coeff] : [:e, :u]
end

# The options of an e-ph run with their defaults, shared by the drivers and the engine constructors.
# `inner_loop_kq`: true for run_eph_over_k_and_kq (inner k+q grid), false for
# run_eph_over_k_and_q (inner q points, with k+q solved per tile) and under `OuterQLoop`.
# `el_qty` / `ph_qty` are the state quantities of the run: those of the loop, of the caller
# (`el_quantities` / `ph_quantities`) and of the calculators.
function _run_options(model::Model;
        inner_loop_kq,
        calculators = [],
        el_quantities = Symbol[],
        ph_quantities = Symbol[],
        backend::AbstractBackend = CPUBackend(),
        window_k = (-Inf, Inf),
        window_kq = (-Inf, Inf),
        symmetry = model.symmetry,
        precompute_el_kq = false,
        energy_conservation_tol = Inf,
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
        verbosity::Int = 1,
    )
    nchunks_threads > 0 || throw(ArgumentError("nchunks_threads must be positive"))
    (n_outer_batch === nothing || n_outer_batch > 0) || throw(ArgumentError("n_outer_batch must be positive"))
    (n_inner_tile === nothing || n_inner_tile > 0) || throw(ArgumentError("n_inner_tile must be positive"))
    el_qty = union(loop_el_quantities(), el_quantities, required_el_quantities.(calculators)...)
    ph_qty = union(loop_ph_quantities(model), ph_quantities, required_ph_quantities.(calculators)...)
    (; inner_loop_kq, calculators, el_qty, ph_qty, backend, window_k, window_kq, symmetry,
       precompute_el_kq, energy_conservation_tol, covariant_derivative_of_g,
       eph_phonon_basis, fourier_mode, screening_params, mpi_comm_k, n_outer_batch, n_inner_tile,
       nchunks_threads, el_k_eigenpairs, el_kq_eigenpairs, ph_eigenpairs, verbosity)
end

# Plan the buffer widths and allocate the engine on the resident `states` of `_setup_states`
# (`els_kq` is nothing for per-tile k+q solves).
function _allocate_engine(order, model, states, options)
    # function barrier for allocation from concrete resident state containers.
    (; els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq) = states
    (; el_qty, ph_qty, calculators, backend, inner_loop_kq, precompute_el_kq, energy_conservation_tol,
       covariant_derivative_of_g, eph_phonon_basis, nchunks_threads, window_kq, verbosity) = options
    (; nw) = model

    nchunks = backend isa CPUBackend ? nchunks_threads : 1

    if order isa OuterKLoop
        # Outer k: the inner set is either resident k+q points or q points with a per-tile solve.
        inner_pts = inner_loop_kq ? kqpts : qpts
        n_outer = kpts.n
    else
        # Outer q: the inner set is the resident k points.
        inner_pts = kpts
        n_outer = qpts.n
    end
    n_inner = inner_pts.n

    # The widths the run uses, from the requested ones (`nothing`: the defaults).
    (; n_outer_batch, n_inner_tile, committed, bytes) = _plan_widths(order, model, backend, calculators;
        n_outer, n_inner, nk = kpts.n, nkq = order isa OuterKLoop ? inner_pts.n : 0, nchunks,
        inner_loop_kq, options.n_outer_batch, options.n_inner_tile, nband_max_k = els_k.nband_max,
        nband_max_kq = els_kq === nothing ? nw : els_kq.nband_max, els_k, els_kq, phs, el_qty, ph_qty,
        precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis)
    if verbosity > 0 && mpi_isroot()
        @info "e-ph loop: committed = $(round(committed / 1e9, digits = 2)) GB, " *
              "$(round(bytes.per_pair / 1e3, digits = 1)) kB per pair; outer batch = $n_outer_batch, " *
              "inner tile = $n_inner_tile, $nchunks chunk(s)"
    end

    # Allocate the engine for the requested outer momentum.
    if order isa OuterKLoop
        eng = OuterKEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, kqpts, qpts,
            n_outer_batch, n_inner_tile, nchunks,
            covariant_derivative_of_g, eph_phonon_basis, sel_k, sel_kq, window_kq, energy_conservation_tol)
    else
        eng = OuterQEngine(model, backend, els_k, els_kq, phs, el_qty, ph_qty; kpts, qpts,
            n_outer_batch, n_inner_tile, nchunks, eph_phonon_basis,
            kqpts, sel_k, sel_kq, window_kq, energy_conservation_tol)
    end
    eng
end


"""
    OuterKEngine(model, kpts, second_pts; inner_loop_kq=false, calculators=[], kwargs...)

Prepare resident states and reusable e-ph buffers without running calculators, for a model loaded
with `epmat_outer_momentum = "el"`. The inner points default to q points (k+q is solved per tile);
`inner_loop_kq = true` instead takes a commensurate k+q grid. A single point can be passed as
`Kpoints(Vec3(...))`. The defaults are `symmetry = nothing` and one CPU workspace
(`nchunks_threads = 1`).

Window, cache, energy-conservation, phonon-basis and capacity keywords match the e-ph drivers.
`calculators` contributes required state quantities and memory budgets but is not set up or run;
`el_quantities` / `ph_quantities` may request additional state fields explicitly. The prepared
states and selections are available as `eng.els_k`, `eng.els_kq`, `eng.phs`, `eng.sel_k` and
`eng.sel_kq`.

Call `stage1!(eng, outer_indices)`, then `stage2!(eng, outer_index, inner_indices)` to obtain an
`EPBlock` for `run_calculator!`. Indices refer to the selected `eng.kpts` / `eng.kqpts` /
`eng.qpts`, which can differ from the input point sets after window and symmetry filtering.
"""
function OuterKEngine(model::Model, kpts, second_pts; inner_loop_kq = false,
        symmetry = nothing, nchunks_threads = 1, kwargs...)
    options = _run_options(model; inner_loop_kq, symmetry, nchunks_threads, kwargs...)
    _check_run(OuterKLoop(), model, kpts, second_pts, options)
    states = _setup_states(OuterKLoop(), model, kpts, second_pts, options)
    _allocate_engine(OuterKLoop(), model, states, options)
end

"""
    OuterQEngine(model, kpts, qpts; calculators=[], kwargs...)

Prepare an outer-q engine without running calculators, for a model loaded with
`epmat_outer_momentum = "ph"`, from k and q point sets. State, capacity and lifecycle options
follow [`OuterKEngine`](@ref); `stage2!(eng, iq, k_indices)` returns a complete block.
"""
function OuterQEngine(model::Model, kpts, qpts; symmetry = nothing, nchunks_threads = 1, kwargs...)
    options = _run_options(model; inner_loop_kq = false, symmetry, nchunks_threads, kwargs...)
    _check_run(OuterQLoop(), model, kpts, qpts, options)
    states = _setup_states(OuterQLoop(), model, kpts, qpts, options)
    _allocate_engine(OuterQLoop(), model, states, options)
end

function _run_eph(order::LoopTag, model::Model, kpts_input, second_input; calculators = [],
        progress_print_step = 20, kwargs...)
    isempty(calculators) && throw(ArgumentError("the e-ph loop requires at least one calculator."))
    progress_print_step > 0 || throw(ArgumentError("progress_print_step must be positive"))
    options = _run_options(model; calculators, kwargs...)

    # Validate the request, then build the resident electron/phonon states and the point sets.
    _check_run(order, model, kpts_input, second_input, options)
    states = _setup_states(order, model, kpts_input, second_input, options)

    # Allocate the engine on those states, and run the calculators over every block.
    eng = _allocate_engine(order, model, states, options)
    _run_eph_loop(eng, calculators; options.symmetry, progress_print_step, options.verbosity)
end

function _run_eph_loop(eng, calculators; symmetry, progress_print_step, verbosity)
    for calculator in calculators
        setup_calculator!(calculator, eng.backend, eng.els_k, eng.els_kq, eng.phs;
            eng.sel_k, eng.sel_kq, nchunks_threads = length(eng.tiles), eng.n_outer_batch, eng.n_inner_tile, verbosity)
    end

    # Explicitly bracket each outer batch; each chunk consumes its blocks before buffers are reused.
    # The outer points and their batches are k under OuterKEngine and q under OuterQEngine.
    outer_pts = eng isa OuterKEngine ? eng.kpts : eng.qpts
    for outer_batch in Iterators.partition(1:outer_pts.n, eng.n_outer_batch)
        if mpi_isroot() && div(last(outer_batch), progress_print_step) > div(first(outer_batch) - 1, progress_print_step)
            @info "$(now()) $(eng isa OuterKEngine ? "ik" : "iq") = $outer_batch / $(outer_pts.n)"
            flush(stdout); flush(stderr)
        end
        # The calculators' bracket opens before the batch's stage 1, so a decision made in
        # `calculator_begin_batch!` from `free_bytes` (`TiledDeviceOutput`) sees no stage-1 transients.
        ctx = eng isa OuterKEngine ? OuterKContext(eng; iks_batch = outer_batch) :
                                     OuterQContext(eng; iqs_batch = outer_batch)
        for calculator in calculators
            calculator_begin_batch!(calculator, ctx)
        end
        stage1!(eng, outer_batch)
        if eng isa OuterKEngine
            _loop_outer_k!(eng, calculators)
        else
            _loop_outer_q!(eng, calculators)
        end
        for calculator in calculators
            calculator_end_batch!(calculator, ctx)
        end
        synchronize(eng.backend)
    end

    for calculator in calculators
        postprocess_calculator!(calculator; qpts = eng.qpts, symmetry)
    end
    (; kpts = eng.kpts, qpts = eng.qpts, els_k = eng.els_k, els_kq = eng.els_kq, phs = eng.phs)
end



# The widths of a run and the device bytes behind them, for `_run_eph` and `estimate_device_memory`
# alike, from the requested `n_outer_batch` / `n_inner_tile` (`nothing`: the default). The outer
# batch is a fixed default (256 outer k; 16 q on a device and 1 on the CPU). The
# inner tile fills the free device memory left after the persistent and per-batch buffers
# (`plan_batch` on `engine_bytes` plus each calculator's `calculator_bytes`, `per_pair`
# once per chunk), capped at all inner points on a device (`2^15` k under `OuterQLoop`) and at a
# cache-sized 1024 per chunk on the CPU.
function _plan_widths(order, model, backend, calculators; n_outer, n_inner, nk, nkq, nchunks,
        n_outer_batch, n_inner_tile, nband_max_k, nband_max_kq, els_k, els_kq, phs, el_qty, ph_qty,
        precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq)
    (; nw, nmodes) = model
    outer_default = order isa OuterKLoop ? 256 : backend isa CPUBackend ? 1 : 16
    n_outer_batch = max(1, min(something(n_outer_batch, outer_default), n_outer))
    inner_default = backend isa CPUBackend ? 1024 : order isa OuterKLoop ? n_inner : 2^15
    inner_cap = max(1, min(cld(n_inner, nchunks), something(n_inner_tile, inner_default)))
    bytes = order isa OuterKLoop ?
        engine_bytes(OuterKEngine, model; nband_max_k, nband_max_kq, nk, nkq, el_qty, ph_qty,
            covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq) :
        engine_bytes(OuterQEngine, model; nband_max_k, nband_max_kq, nk, el_qty, ph_qty,
            precompute_el_kq, eph_phonon_basis)
    for c in calculators
        b = calculator_bytes(c, EPBlock{typeof(order)}; nw, nmodes, nband_max_k, nband_max_kq,
                             els_k, els_kq, phs, nchunks_threads = nchunks)
        bytes = (; persistent = bytes.persistent + b.persistent,
                   per_outer = bytes.per_outer + b.per_outer, per_pair = bytes.per_pair + b.per_pair)
    end
    committed = bytes.persistent + bytes.per_outer * n_outer_batch
    n_inner_tile = plan_batch(backend, bytes.per_pair * nchunks, committed, inner_cap;
                              what = order isa OuterKLoop ? "outer-k" : "outer-q")
    (; n_outer_batch, n_inner_tile, committed, bytes)
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
`7/10`, i.e. a 30% headroom for the batched drivers' recycled temporaries), applied in integer
arithmetic as `x ÷ den * num`. Pass `warn = false`
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


# Every refusal of the loop, before any state is built.
function _check_run(order, model, kpts_input, second_input, options)
    (; backend, calculators, el_qty, energy_conservation_tol, covariant_derivative_of_g,
       eph_phonon_basis, fourier_mode, precompute_el_kq, screening_params, mpi_comm_k,
       el_k_eigenpairs, el_kq_eigenpairs, symmetry, inner_loop_kq) = options
    Order = typeof(order)
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
    energy_conservation_tol >= 0 ||
        throw(ArgumentError("energy_conservation_tol must be nonnegative, got $energy_conservation_tol"))
    (isinf(energy_conservation_tol) || backend isa CPUBackend) || throw(ArgumentError(
        "energy_conservation_tol is a CPUBackend feature: on a GPU computing every pair and letting " *
        "the calculators' delta functions discard is cheaper. Pass energy_conservation_tol = Inf."))
    if order isa OuterKLoop
        # Outer k: no precomputed k+q option; the inner set decides the rest.
        precompute_el_kq && throw(ArgumentError("precompute_el_kq is an outer-q option"))
        if !inner_loop_kq
            # Inner q points: k+q is solved per tile, with `e` and `u` only and no cache.
            (el_k_eigenpairs === nothing || all(_input_ngrid(kpts_input) .> 0)) || throw(ArgumentError(
                "el_k_eigenpairs needs the outer k points on a grid: the cache is looked up on one"))
            issubset(el_qty, (:e, :u)) || throw(ArgumentError(
                "run_eph_over_k_and_q solves the k+q states per tile, with `e` and `u` only; the " *
                "loop and the calculators request $(setdiff(el_qty, (:e, :u)))"))
            el_kq_eigenpairs === nothing || throw(ArgumentError(
                "el_kq_eigenpairs is not supported by run_eph_over_k_and_q: the k+q states are " *
                "solved per tile"))
        else
            # Inner k+q points: both sets on commensurate grids.
            ng_k, ng_kq = _input_ngrid(kpts_input), _input_ngrid(second_input)
            (all(ng_k .> 0) && all(ng_kq .> 0) &&
             (all(mod.(ng_kq, ng_k) .== 0) || all(mod.(ng_k, ng_kq) .== 0))) || throw(ArgumentError(
                "run_eph_over_k_and_kq needs commensurate k and k+q grids (got $ng_k and $ng_kq): " *
                "the phonon states are built on the q grid they span"))
        end
    else
        # Outer q: the inner k points on a grid, no covariant derivative and no MPI split.
        covariant_derivative_of_g && throw(ArgumentError(
            "covariant_derivative_of_g is not supported by run_eph_over_q_and_k"))
        second_input isa FilteredBandStates && throw(ArgumentError(
            "run_eph_over_q_and_k takes q points, not a state selection"))
        ng_k, ng_q = _input_ngrid(kpts_input), _input_ngrid(second_input)
        all(ng_k .> 0) || throw(ArgumentError(
            "run_eph_over_q_and_k needs the k points on a grid: a grid size, a k-point set on a " *
            "grid or a FilteredBandStates of one"))
        mpi_comm_k === nothing || throw(ArgumentError("mpi_comm_k is not implemented for run_eph_over_q_and_k"))
        if precompute_el_kq
            # k+q precomputed: the states live on the q grid, which holds every k + q only when it
            # is a multiple of the k grid.
            all(ng_q .> 0) || throw(ArgumentError("precompute_el_kq needs a q grid, not a q list"))
            all(mod.(ng_q, ng_k) .== 0) || throw(ArgumentError(
                "precompute_el_kq needs a q grid that is a multiple of the k grid (got k $ng_k, " *
                "q $ng_q): k + q is off the q grid otherwise"))
        else
            # k+q solved per tile, with `e` and `u` only and no cache.
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
function _setup_states(order, model::Model, kpts_input, second_input, options)
    (; el_qty, ph_qty, inner_loop_kq, backend, window_k, window_kq, symmetry, precompute_el_kq,
       eph_phonon_basis, mpi_comm_k, el_k_eigenpairs, el_kq_eigenpairs, ph_eigenpairs,
       verbosity) = options
    (; nw) = model
    # The host solves of a GPU run (q filter, polar phonons) keep the default interpolation.
    fourier_mode = backend isa CPUBackend ? options.fourier_mode : "gridopt"
    # Reuse a supplied k selection, or select its bands within the requested energy window.
    # Symmetry reduces the outer points, given as a grid size: k under outer k, q under outer q.
    if kpts_input isa FilteredBandStates
        sel_k = kpts_input
    else
        symmetry_k = order isa OuterKLoop ? symmetry : nothing
        sel_k = maybe_time(verbosity) do
            filter_electron_states(kpts_input, nw, model.el_ham, window_k; symmetry = symmetry_k,
                                   fourier_mode, backend, mpi_comm = mpi_comm_k)
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
            # The k+q set is always the full-BZ selection. A grid size is filtered to the window on
            # the irreducible wedge and unfolded, which only makes the filter cheaper; a k-point
            # set is filtered and used as given.
            symmetry_kq = second_input isa NTuple{3, Int} ? symmetry : nothing
            sel_kqf = maybe_time(verbosity) do
                filter_electron_states(second_input, nw, model.el_ham, window_kq; symmetry = symmetry_kq,
                                       fourier_mode, backend)
            end
            kqpts = symmetry_kq === nothing ? sel_kqf.kpts : unfold_kpoints(sel_kqf.kpts, symmetry_kq)[1]
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
        # Outer q: every q point is kept, also one with no k+q state in the window. Optionally
        # precompute the k+q grid's states.
        qpts = second_input isa NTuple{3, Int} ? kpoints_grid(second_input; symmetry) : second_input
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

    phs = maybe_time(verbosity) do
        compute_phonon_states_batched(model, qpts, ph_qty; fourier_mode, eph_phonon_basis, backend,
                                      eigenpairs = ph_eigenpairs)
    end

    if verbosity > 0 && mpi_isroot()
        @info "Number of k points = $(kpts.n)"
        kqpts === nothing || @info "Number of k+q points = $(kqpts.n)"
        @info "Number of q points = $(qpts.n)"
    end
    (; els_k, els_kq, phs, kpts, kqpts, qpts, sel_k, sel_kq)
end


# ---- Explicit outer-k sweep -------------------------------------------------------------------

# The blocks of the engine's current stage-1 batch of outer k points.
function _loop_outer_k!(eng::OuterKEngine, calculators)
    # Partition the inner k+q or q points into independent chunks.
    inner_pts = eng.inner_loop_kq ? eng.kqpts : eng.qpts
    inner_pts.n == 0 && return nothing
    eng_fields = _workspace_fields(eng)

    if length(eng.tiles) == 1
        # One chunk: GPU and serial runs remain on the caller's task and CUDA stream.
        _loop_outer_k_chunk!(eng_fields, _workspace_fields(eng.tiles[1]), calculators;
            chunk = 1, inner_indices = 1:inner_pts.n)
    else
        # Several chunks: one task per chunk of inner points, each on its own tile workspace.
        inner_chunks = collect(enumerate(chunks(1:inner_pts.n; n = min(length(eng.tiles), inner_pts.n))))
        @threads for (chunk, inner_indices) in inner_chunks
            _loop_outer_k_chunk!(eng_fields, _workspace_fields(eng.tiles[chunk]), calculators; chunk, inner_indices)
        end
    end
    nothing
end

function _loop_outer_k_chunk!(eng_fields, tile_workspace, calculators; chunk, inner_indices)
    # function barrier for one CPU/GPU chunk's concrete states and reusable buffers.
    ctx = OuterKContext(eng_fields.backend, eng_fields.iks_batch, chunk)

    for inner_indices_tile in Iterators.partition(inner_indices, eng_fields.n_inner_tile)
        # A resident k+q tile shares its Fourier phase across all outer k points, unless the
        # energy-conservation tolerance keeps different pairs for each k.
        phase = nothing
        if eng_fields.inner_loop_kq && isinf(eng_fields.energy_conservation_tol)
            phase = view(tile_workspace.P_kq, :, 1:length(inner_indices_tile))
            @views build_fourier_phase!(phase, eng_fields.irvecp_mat, eng_fields.xkqs[:, inner_indices_tile])
        end

        for ik in eng_fields.iks_batch
            # The same stage-2 worker also serves standalone stage2!(eng, ik, inner_indices).
            block = _stage2!(OuterKLoop(), eng_fields, tile_workspace, ik, inner_indices_tile; phase)
            block === nothing && continue

            for calculator in calculators
                run_calculator!(calculator, block, ctx)
            end
        end
    end
    nothing
end

# ---- Explicit outer-q sweep -------------------------------------------------------------------

# The blocks of the engine's current stage-1 batch of outer q points. The chunks only read the
# stage-1 output and own all writable scratch.
function _loop_outer_q!(eng::OuterQEngine, calculators)
    eng.els_k.nk == 0 && return nothing
    eng_fields = _workspace_fields(eng)
    (; iqs_batch) = eng

    # Visit each q explicitly and complete its k chunks before calculator_end_batch! or the next batch.
    for iq in iqs_batch
        if length(eng.tiles) == 1
            # One chunk: GPU and serial runs remain on the caller's task and CUDA stream.
            _loop_outer_q_chunk!(eng_fields, _workspace_fields(eng.tiles[1]), calculators;
                chunk = 1, iks = 1:eng.els_k.nk, iq)
        else
            # Several chunks: one task per chunk of k points, each on its own tile workspace.
            inner_chunks = collect(enumerate(chunks(1:eng.els_k.nk; n = min(length(eng.tiles), eng.els_k.nk))))
            @threads for (chunk, iks) in inner_chunks
                _loop_outer_q_chunk!(eng_fields, _workspace_fields(eng.tiles[chunk]), calculators; chunk, iks, iq)
            end
        end
    end
    nothing
end

function _loop_outer_q_chunk!(eng_fields, tile_workspace, calculators; chunk, iks, iq)
    # function barrier for one chunk's concrete state containers and Fourier/rotation scratch.
    ctx = OuterQContext(eng_fields.backend, eng_fields.iqs_batch, chunk)

    for iks_tile in Iterators.partition(iks, eng_fields.n_inner_tile)
        # The same stage-2 worker gathers/solves states and returns a complete block for direct calls.
        block = _stage2!(OuterQLoop(), eng_fields, tile_workspace, iq, iks_tile)
        block === nothing && continue

        for calculator in calculators
            run_calculator!(calculator, block, ctx)
        end
    end
    nothing
end

"""
    estimate_device_memory(model; nk, nkq, n_outer_batch = nothing, n_inner_tile = nothing,
                           calculators = [], backend = CPUBackend(), nchunks_threads = nthreads())
        -> NamedTuple

Estimate the device memory of an e-ph run without running it, from the byte counts the loop plans
with (`engine_bytes` and the calculators' `calculator_bytes`) at box widths `nw`, so a
windowed run uses less. The order follows `model.epmat_outer_momentum` (`el` → outer-k, `ph` →
outer-q). `nk` is the number of k points and `nkq` the number of k+q points of an outer-k model,
or of outer q points of an outer-q model. The state containers are not counted. Returns `(; loop, committed, per_pair, batch,
free)`, `batch` the inner tile the run would pick on `backend`, with the run's defaults.

Actual device usage starts ~100-150 MB higher: the CUDA library context and workspace (cuBLAS
etc.) are allocated lazily on the first kernel launch and are not a per-run buffer.
"""
function estimate_device_memory(model::Model{FT}; nk::Integer, nkq::Integer, n_outer_batch = nothing,
        n_inner_tile = nothing, calculators = [], backend::AbstractBackend = CPUBackend(),
        nchunks_threads = nthreads()) where {FT}
    outer_k = model.epmat_outer_momentum == "el"
    order = outer_k ? OuterKLoop() : OuterQLoop()
    el_qty = union(loop_el_quantities(), required_el_quantities.(calculators)...)
    ph_qty = union(loop_ph_quantities(model), required_ph_quantities.(calculators)...)
    plan = _plan_widths(order, model, backend, calculators;
        n_outer = outer_k ? Int(nk) : Int(nkq), n_inner = outer_k ? Int(nkq) : Int(nk), nk, nkq,
        nchunks = backend isa CPUBackend ? nchunks_threads : 1, n_outer_batch, n_inner_tile,
        nband_max_k = model.nw, nband_max_kq = model.nw, els_k = nothing, els_kq = nothing, phs = nothing,
        el_qty, ph_qty, precompute_el_kq = false, covariant_derivative_of_g = false,
        eph_phonon_basis = :eigenmode, inner_loop_kq = outer_k)
    (; loop = outer_k ? :outer_k : :outer_q, plan.committed, plan.bytes.per_pair,
       batch = plan.n_inner_tile, free = free_bytes(backend))
end


# =============================================================================
# Deprecated name of `run_eph_over_q_and_k`: a forwarder, to be removed after one release. An
# explicit @warn (maxlog = 1) because Base.@deprecate depwarns are invisible in ordinary script
# runs (Julia ≥ 1.5).
function run_eph_outer_q(args...; kwargs...)
    @warn "run_eph_outer_q is deprecated; use run_eph_over_q_and_k." maxlog=1
    run_eph_over_q_and_k(args...; kwargs...)
end
