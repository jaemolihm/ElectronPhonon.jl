using ChunkSplitters
using Base.Threads: nthreads, @threads
using Dates: now

"""
    run_eph_over_k_and_kq(model, kpts, kqpts; calculators, backend, kwargs...)

Sweep the outer k points and, for each, the inner k+q points (on a grid commensurate with the k
grid), handing each calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterKLoop}`: one outer k
with a tile of k+q points. `kpts` and `kqpts` are a grid size, a k-point set or a prebuilt
`FilteredBandStates`. Returns `(; kpts, qpts, el_k, el_kq, ph)`, the run's state containers
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
* `fill_padding_nan = false` — fill the box entries outside each window with NaN (a test switch).
"""
run_eph_over_k_and_kq(model::Model, kpts_input, kqpts_input; kwargs...) =
    _run_eph(OuterKLoop(), model, kpts_input, kqpts_input; kwargs...)

"""
    run_eph_over_q_and_k(model, kpts, qpts; calculators, backend, kwargs...)

Sweep the outer q points (any q list, e.g. a path) and, for each, the inner k points, handing each
calculator the e-ph coupling as an [`EPBlock`](@ref)`{OuterQLoop}`: one q with a tile of k points.
The k+q states are solved per tile, with `e` and `u` only, unless `precompute_el_kq = true` (a grid
q set), which builds them once on the k+q grid; a pair whose k+q has no state in `window_kq` is
then dropped. Returns `(; kpts, qpts, el_k, el_kq, ph)`, `el_kq = nothing` when solved per tile.

Keywords as in [`run_eph_over_k_and_kq`](@ref), except: `use_symmetry = true` reduces the k points
with `model.symmetry`; `keep_all_qpts = false` drops the q points with no k+q state in `window_kq`;
`n_outer_batch` q points per stage-1 batch and per bracket, 16 on a GPU and 1 on the CPU (stage 1
gains nothing from a wider batch there, while a calculator's per-q buffers are held per thread
chunk); `n_inner_tile` k points per block, at most `2^15` on a GPU; no `covariant_derivative_of_g`;
`el_kq_eigenpairs` only with `precompute_el_kq`.
"""
run_eph_over_q_and_k(model::Model, kpts_input, qpts_input; use_symmetry::Bool = true, kwargs...) =
    _run_eph(OuterQLoop(), model, kpts_input, qpts_input;
             symmetry = use_symmetry ? model.symmetry : nothing, kwargs...)


# The quantities the loop reads itself, on both electron sides and on the phonons.
# Energy conservation reads the energies, and its `:Linear` mode the velocities of the k+q side and
# the phonons.
function loop_el_quantities((mode, _))
    [:u; mode === :None ? Symbol[] : [:e]; mode === :Linear ? [:vdiag] : Symbol[]]
end
function loop_ph_quantities(model, (mode, _))
    [:u; model.polar_eph.use ? [:eph_dipole_coeff] : Symbol[];
     mode === :None ? Symbol[] : [:e]; mode === :Linear ? [:vdiag] : Symbol[]]
end

function _run_eph(order::LoopTag, model::Model{FT}, kpts_input, second_input;
        calculators = [],
        backend::AbstractBackend = CPUBackend(),
        window_k = (-Inf, Inf),
        window_kq = (-Inf, Inf),
        symmetry = model.symmetry,
        el_kq_from_unfolding = false,
        precompute_el_kq = false,
        keep_all_qpts = false,
        energy_conservation = (:None, 0.0),
        covariant_derivative_of_g = false,
        eph_phonon_basis::Symbol = :eigenmode,
        skip_eph = false,
        screening_params = nothing,
        mpi_comm_k = nothing,
        mpi_comm_q = nothing,
        n_outer_batch = nothing,
        n_inner_tile = nothing,
        nchunks_threads = nthreads(),
        el_k_eigenpairs::Union{Nothing, Eigenpairs} = nothing,
        el_kq_eigenpairs::Union{Nothing, Eigenpairs} = nothing,
        ph_eigenpairs::Union{Nothing, Eigenpairs} = nothing,
        fill_padding_nan::Bool = false,
        progress_print_step = 20,
        verbosity::Int = 1,
    ) where {FT}

    el_qty = union(loop_el_quantities(energy_conservation), required_el_quantities.(calculators)...)
    ph_qty = union(loop_ph_quantities(model, energy_conservation),
                   required_ph_quantities.(calculators)...)
    _check_run(order, model, backend, calculators, kpts_input, second_input, el_qty;
        energy_conservation, covariant_derivative_of_g, eph_phonon_basis, el_kq_from_unfolding,
        precompute_el_kq, skip_eph, screening_params, mpi_comm_k, mpi_comm_q, el_kq_eigenpairs)

    (; el_k, el_kq, ph, kpts, kqpts, qpts, sel_k, sel_kq) = _setup_states(order, model, kpts_input,
        second_input, el_qty, ph_qty; backend, window_k, window_kq, symmetry, precompute_el_kq,
        keep_all_qpts, eph_phonon_basis, mpi_comm_k, el_k_eigenpairs, el_kq_eigenpairs,
        ph_eigenpairs, fill_padding_nan, verbosity)
    # The containers' types depend on the runtime quantity lists, and inference of the loop on the
    # abstract types does not terminate (> 30 min for an outer-q run), so it is not let through.
    Base.inferencebarrier(_run_eph_loop)(order, model, el_k, el_kq, ph, kpts, kqpts, qpts, sel_k,
        sel_kq, el_qty, ph_qty; calculators, backend, symmetry, precompute_el_kq, energy_conservation,
        covariant_derivative_of_g, eph_phonon_basis, n_outer_batch, n_inner_tile, nchunks_threads,
        window_kq, fill_padding_nan, progress_print_step, verbosity)
end

# The loop of `_run_eph` on the built states, compiled for their concrete types (`el_kq` and `kqpts`
# are `nothing` for the per-tile k+q solve).
function _run_eph_loop(order, model, el_k, el_kq, ph, kpts, kqpts, qpts, sel_k, sel_kq, el_qty,
        ph_qty; calculators, backend, symmetry, precompute_el_kq, energy_conservation,
        covariant_derivative_of_g, eph_phonon_basis, n_outer_batch, n_inner_tile, nchunks_threads,
        window_kq, fill_padding_nan, progress_print_step, verbosity)
    (; nw, nmodes) = model

    # The widths. The outer batch is a fixed default; the inner tile fills the free device memory
    # left after the persistent and per-batch buffers (`plan_batch`), on the CPU a cache-sized tile
    # per thread chunk.
    nchunks = backend isa CPUBackend ? nchunks_threads : 1
    drop_pairs = energy_conservation[1] !== :None || (order isa OuterQLoop && precompute_el_kq)
    n_outer = order isa OuterKLoop ? kpts.n : qpts.n
    n_inner = order isa OuterKLoop ? kqpts.n : kpts.n
    outer_default = order isa OuterKLoop ? 256 : backend isa CPUBackend ? 1 : 16
    nb_outer = max(1, min(something(n_outer_batch, outer_default), n_outer))
    inner_default = backend isa CPUBackend ? 1024 : order isa OuterKLoop ? n_inner : 2^15
    inner_cap = max(1, min(cld(n_inner, nchunks), something(n_inner_tile, inner_default)))
    nband_max_k = el_k.nband_max
    nband_max_kq = el_kq === nothing ? nw : el_kq.nband_max
    bytes = order isa OuterKLoop ?
        engine_bytes(OuterKEngine, model; nband_max_k, nband_max_kq, nk = kpts.n, nkq = kqpts.n,
            el_qty, ph_qty, drop_pairs, covariant_derivative_of_g, eph_phonon_basis) :
        engine_bytes(OuterQEngine, model; nband_max_k, nband_max_kq, nk = kpts.n,
            n_outer_batch = nb_outer, el_qty, ph_qty, drop_pairs, precompute_el_kq, eph_phonon_basis)
    for c in calculators
        b = eph_batched_bytes_per_point(c, EPBlock{typeof(order)}; nw, nmodes, nband_max_k,
                                        nband_max_kq, el_k, el_kq, ph, nchunks_threads = nchunks)
        bytes = (; persistent = bytes.persistent + b.persistent,
                   per_outer = bytes.per_outer + b.per_outer, per_pair = bytes.per_pair + b.per_pair)
    end
    committed = bytes.persistent + bytes.per_outer * nb_outer
    nb_inner = plan_batch(backend, bytes.per_pair * nchunks, committed, inner_cap;
                          what = order isa OuterKLoop ? "outer-k" : "outer-q")
    if verbosity > 0 && mpi_isroot()
        @info "e-ph loop: committed = $(round(committed / 1e9, digits = 2)) GB, " *
              "$(round(bytes.per_pair / 1e3, digits = 1)) kB per pair; outer batch = $nb_outer, " *
              "inner tile = $nb_inner, $nchunks chunk(s)"
    end

    eng = order isa OuterKLoop ?
        OuterKEngine(model, backend, el_k, el_kq, ph, el_qty, ph_qty; kpts, kqpts, qpts,
            n_outer_batch = nb_outer, n_inner_tile = nb_inner, nchunks, drop_pairs,
            covariant_derivative_of_g, eph_phonon_basis) :
        OuterQEngine(model, backend, el_k, el_kq, ph, el_qty, ph_qty; kpts, qpts,
            n_outer_batch = nb_outer, n_inner_tile = nb_inner, nchunks, drop_pairs, eph_phonon_basis)
    foreach(c -> setup_calculator!(c, backend, el_k, el_kq, ph; sel_k, sel_kq, nw, nmodes,
        nchunks_threads = nchunks, n_outer_batch = nb_outer, n_inner_tile = nb_inner, verbosity),
        calculators)

    ngrid = order isa OuterKLoop ? kqpts.ngrid : qpts.ngrid   # the energy-conservation box
    for batch in Iterators.partition(1:n_outer, nb_outer)
        if mpi_isroot() &&
                div(last(batch), progress_print_step) > div(first(batch) - 1, progress_print_step)
            @info "$(now()) $(order isa OuterKLoop ? "ik" : "iq") = $batch / $n_outer"
            flush(stdout); flush(stderr)
        end
        ctx = LoopContext(backend, order, batch, 1)
        foreach(c -> calculator_begin!(c, ctx), calculators)
        if order isa OuterKLoop
            _loop_outer_k!(eng, batch, el_k, el_kq, ph, kpts, qpts, calculators, model,
                           energy_conservation, ngrid)
        else
            _loop_outer_q!(eng, batch, el_k, el_kq, ph, kpts, kqpts, qpts, calculators, model,
                           energy_conservation, ngrid, eph_phonon_basis, window_kq, fill_padding_nan)
        end
        foreach(c -> calculator_end!(c, ctx), calculators)
        # Bound the host look-ahead to one batch, so its device scratch does not pile up in the pool.
        synchronize(backend)
    end

    # The outer-q loop hands the model's symmetry to postprocess whatever `use_symmetry` was.
    symmetry_post = order isa OuterQLoop ? model.symmetry : symmetry
    foreach(c -> postprocess_calculator!(c; qpts, symmetry = symmetry_post), calculators)
    (; kpts, qpts, el_k, el_kq, ph)
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
        energy_conservation, covariant_derivative_of_g, eph_phonon_basis, el_kq_from_unfolding,
        precompute_el_kq, skip_eph, screening_params, mpi_comm_k, mpi_comm_q, el_kq_eigenpairs)
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
    model.epmat isa WannierObject || throw(ArgumentError(
        "a disk-backed epmat ($(typeof(model.epmat))) is not supported by the e-ph loop; load the " *
        "model into memory"))
    skip_eph && throw(ArgumentError("the e-ph loop requires skip_eph = false"))
    screening_params === nothing || error(
        "screening_params is not supported: dielectric screening is currently disabled (ϵ ≡ 1). " *
        "Pass screening_params = nothing.")
    mpi_comm_q === nothing || throw(ArgumentError("mpi_comm_q is not implemented"))
    el_kq_from_unfolding && throw(ArgumentError(
        "el_kq_from_unfolding = true is not supported: the k+q states are computed directly on the " *
        "full k+q set"))
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
        ng_k, ng_kq = _input_ngrid(kpts_input), _input_ngrid(second_input)
        (all(ng_k .> 0) && all(ng_kq .> 0) &&
         (all(mod.(ng_kq, ng_k) .== 0) || all(mod.(ng_k, ng_kq) .== 0))) || throw(ArgumentError(
            "run_eph_over_k_and_kq needs commensurate k and k+q grids (got $ng_k and $ng_kq): the " *
            "phonon states are built on the q grid they span"))
    else
        covariant_derivative_of_g && throw(ArgumentError(
            "covariant_derivative_of_g is not supported by run_eph_over_q_and_k"))
        mpi_comm_k === nothing || throw(ArgumentError("mpi_comm_k is not implemented for run_eph_over_q_and_k"))
        q_on_grid = second_input isa NTuple{3, Int} || (second_input isa AbstractKpoints && all(second_input.ngrid .> 0))
        (precompute_el_kq || mode === :Linear) && !q_on_grid && throw(ArgumentError(
            "precompute_el_kq and adaptive energy conservation need a q grid, not a q list"))
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
# per tile under `OuterQLoop`), the q set and its phonons, all on `backend`.
function _setup_states(order, model::Model{FT}, kpts_input, second_input, el_qty, ph_qty; backend,
        window_k, window_kq, symmetry, precompute_el_kq, keep_all_qpts, eph_phonon_basis,
        mpi_comm_k, el_k_eigenpairs, el_kq_eigenpairs, ph_eigenpairs, fill_padding_nan,
        verbosity) where {FT}
    (; nw, nmodes) = model
    fourier_mode = "gridopt"
    sel_k = kpts_input isa FilteredBandStates ? kpts_input : maybe_time(verbosity) do
        filter_electron_states(kpts_input, nw, model.el_ham, window_k; symmetry, fourier_mode, backend,
                               mpi_comm = mpi_comm_k)
    end
    kpts = sel_k.kpts
    el_k = maybe_time(verbosity) do
        compute_electron_states_batched(model, sel_k, el_qty; fourier_mode, backend,
            eigenpairs = el_k_eigenpairs, fill_padding_nan)
    end

    if order isa OuterKLoop
        if second_input isa FilteredBandStates
            # A prebuilt full-BZ selection, consumed as it is.
            sel_kq = second_input
            kqpts = sel_kq.kpts
            el_kq = maybe_time(verbosity) do
                compute_electron_states_batched(model, sel_kq, el_qty; fourier_mode, backend,
                    eigenpairs = el_kq_eigenpairs, fill_padding_nan)
            end
        else
            # A grid, filtered to the window; under symmetry IBZ-filtered, then unfolded.
            sel_kqf = maybe_time(verbosity) do
                filter_electron_states(second_input, nw, model.el_ham, window_kq; symmetry, fourier_mode, backend)
            end
            kqpts = symmetry === nothing ? sel_kqf.kpts : unfold_kpoints(sel_kqf.kpts, symmetry)[1]
            el_kq = maybe_time(verbosity) do
                compute_electron_states_batched(model, kqpts, el_qty, window_kq; fourier_mode, backend,
                    eigenpairs = el_kq_eigenpairs, fill_padding_nan)
            end
            sel_kq = electron_states_to_FilteredBandStates(kqpts, el_kq, sel_kqf.nstates_base; nw)
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
            el_kq = maybe_time(verbosity) do
                compute_electron_states_batched(model, sel_kq, el_qty; fourier_mode, backend,
                    eigenpairs = el_kq_eigenpairs, fill_padding_nan)
            end
        else
            sel_kq = kqpts = el_kq = nothing
        end
    end

    # The phonons on `backend`. The device builder fills `e` and `u` of a non-polar model; the
    # other quantities are built on the host and copied over.
    ph = maybe_time(verbosity) do
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
    (; el_k, el_kq, ph, kpts, kqpts, qpts, sel_k, sel_kq)
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
    nchunks == 1 && return f(1, 1:n)
    @threads for (chunk, inds) in collect(enumerate(chunks(1:n; n = min(nchunks, n))))
        f(chunk, inds)
    end
end


# ---- OuterKLoop --------------------------------------------------------------------------------

# One outer-k batch: stage 1 for the batch, then per thread chunk its k+q tiles, each tile's phase
# built once and reused by every k of the batch (tile-major).
function _loop_outer_k!(eng::OuterKEngine, batch, el_k, el_kq, ph, kpts, qpts, calculators, model,
        energy_conservation, ngrid)
    stage1!(eng, el_k, kpts, batch)
    _foreach_chunk(length(eng.tiles), el_kq.nk) do chunk, ikqs
        tile_bufs = eng.tiles[chunk]
        ctx = LoopContext(eng.backend, OuterKLoop(), batch, chunk)
        for tile in Iterators.partition(ikqs, eng.n_inner_tile)
            n = length(tile)
            # The k+q side is a contiguous slice of the resident container; its phase is shared by
            # every k of the batch.
            el_kq_t = view_batched_electron_states(el_kq, tile)
            phase = view(tile_bufs.P_kq, :, 1:n)
            @views build_fourier_phase!(phase, eng.irvecp_mat, eng.xkq[:, tile])
            for (iouter, ik) in enumerate(batch)
                # This (k, tile)'s q indices, checked on the host and copied once into the device buffer.
                _fill_iqs!(tile_bufs.iq, qpts, eng.xkqs_int, eng.xks_int, ik, first(tile), n)
                copyto!(tile_bufs.iq_dev, 1, tile_bufs.iq, 1, n)
                iq = view(tile_bufs.iq_dev, 1:n)
                copy_batched_phonon_states!(tile_bufs.ph, ph, iq)
                pairs = (; n, iouter,
                    el_k = view_batched_electron_states(eng.el_k_batch, iouter:iouter),
                    el_kq = el_kq_t, ph = view_batched_phonon_states(tile_bufs.ph, 1:n), phase,
                    ik, ikq = tile, iq, wtk = kpts.weights[ik], wtq = view(eng.wtkq, tile),
                    xk = kpts.vectors[ik], xq = view(qpts.vectors, view(tile_bufs.iq, 1:n)))
                _block!(eng, tile_bufs, pairs, ctx, calculators, model, energy_conservation, ngrid)
            end
        end
    end
end


# ---- OuterQLoop --------------------------------------------------------------------------------

# One outer-q batch: stage 1 for the batch, then per q one threaded region over the k tiles.
function _loop_outer_q!(eng::OuterQEngine, batch, el_k, el_kq, ph, kpts, kqpts, qpts, calculators,
        model, energy_conservation, ngrid, eph_phonon_basis, window_kq, fill_padding_nan)
    stage1!(eng, ph, qpts, batch, eph_phonon_basis)
    for (iouter, iq) in enumerate(batch)
        copyto!(eng.eRpq.op_r, view(eng.ep_Rq, :, :, iouter))
        _foreach_chunk(length(eng.tiles), el_k.nk) do chunk, iks
            tile_bufs = eng.tiles[chunk]
            ctx = LoopContext(eng.backend, OuterQLoop(), batch, chunk)
            xq = qpts.vectors[iq]
            ph_q = view_batched_phonon_states(ph, iq:iq)
            for tile in Iterators.partition(iks, eng.n_inner_tile)
                n = length(tile)
                copy_batched_electron_states!(tile_bufs.el_k, el_k, tile)
                for (j, ik) in enumerate(tile)
                    tile_bufs.kqs[j] = kpts.vectors[ik] + xq
                end
                if el_kq === nothing
                    # k+q solved into the tile, at the box of its largest window.
                    el_kq_t = compute_electron_states_batched!(tile_bufs.el_kq, tile_bufs.itp_el_ham,
                        tile_bufs.hk, model, view(tile_bufs.kqs, 1:n), window_kq; fill_padding_nan)
                    ikq = nothing
                else
                    # Precomputed k+q by grid lookup; 0 (dropped by `filter_pairs!`) where it is absent,
                    # which the copy reads as point 1.
                    for j in 1:n
                        tile_bufs.ikq[j] = something(xk_to_ik_unsafe(tile_bufs.kqs[j], kqpts), 0)
                        tile_bufs.ikq_copy[j] = max(tile_bufs.ikq[j], 1)
                    end
                    copy_batched_electron_states!(tile_bufs.el_kq, el_kq, view(tile_bufs.ikq_copy, 1:n))
                    el_kq_t = view_batched_electron_states(tile_bufs.el_kq, 1:n)
                    ikq = view(tile_bufs.ikq, 1:n)
                end
                pairs = (; n, iouter = 0, el_k = view_batched_electron_states(tile_bufs.el_k, 1:n),
                    el_kq = el_kq_t, ph = ph_q, phase = nothing, ik = tile, ikq, iq,
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
    pairs = filter_pairs!(tile_bufs, pairs, model, energy_conservation, ngrid)
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
    filter_pairs!(tile_bufs, pairs, model, energy_conservation, ngrid) -> pairs

The pairs of a block that the loop computes: `pairs` itself when none is dropped, otherwise the kept
ones copied into the tile buffers' `tile_bufs.kept` (`pairs` is never modified, since under `OuterKLoop`
one tile serves every k of a batch). A pair is dropped when its k+q is absent from the precomputed
states, or when `energy_conservation` (`CPUBackend`) finds no energy-conserving process in it
(`check_energy_conservation` over every mode, band pair and phonon sign, with `ngrid` the grid of
the box). The side with a scalar index is shared by the block and carried over as it is.
"""
function filter_pairs!(tile_bufs, pairs, model, energy_conservation, ngrid)
    kept_bufs = tile_bufs.kept
    kept_bufs === nothing && return pairs
    mode, tol = energy_conservation
    # Whether pair `j` has an energy-conserving process, by `check_energy_conservation` on its host
    # arrays (local box bands; the shared side has extent 1).
    vec3(v, i) = v === nothing ? nothing : reinterpret(reshape, Vec3{eltype(v)}, view(v, :, :, i))
    function conserves(j)
        jk, jq = min(j, pairs.el_k.nk), min(j, pairs.ph.nq)
        states_k = (; e = view(pairs.el_k.e, :, jk))
        states_kq = (; e = view(pairs.el_kq.e, :, j), vdiag = vec3(pairs.el_kq.vdiag, j))
        states_ph = (; e = view(pairs.ph.e, :, jq), vdiag = vec3(pairs.ph.vdiag, jq))
        any(check_energy_conservation(states_k, states_kq, states_ph, ib, jb, imode, sign_ph, ngrid,
                                      model.recip_lattice, mode, tol)
            for imode in 1:pairs.ph.nmodes, jb in 1:pairs.el_kq.nband[j], ib in 1:pairs.el_k.nband[jk],
                sign_ph in (-1, 1))
    end
    nkeep = 0
    for j in 1:pairs.n
        pairs.ikq === nothing || pairs.ikq[j] != 0 || continue
        mode === :None || conserves(j) || continue
        kept_bufs.keep[nkeep += 1] = j
    end
    nkeep == pairs.n && return pairs
    keep = view(kept_bufs.keep, 1:nkeep)
    keep_dev = _copy_indices_on_backend(pairs.el_kq.nband, keep, pairs.n)
    kept = 1:nkeep
    # Copy the kept columns of a host vector, of a backend array (by `keep_dev`), or of a state
    # container into the kept buffers, as views of the kept extent.
    copy_kept_host!(dst, src) = (for (i, j) in enumerate(keep); dst[i] = src[j]; end; view(dst, kept))
    copy_kept!(dst, src) =
        view(_copy_last_axis!(dst, src, keep_dev), ntuple(_ -> Colon(), ndims(dst) - 1)..., kept)
    copy_kept_electron_states!(buf, src) = view_batched_electron_states(copy_batched_electron_states!(
        prefix_batched_electron_states(buf, src.nband_max, buf.nk), src, keep_dev), kept)
    el_kq = copy_kept_electron_states!(kept_bufs.el_kq, pairs.el_kq)
    el_k = pairs.ik isa Integer ? pairs.el_k : copy_kept_electron_states!(kept_bufs.el_k, pairs.el_k)
    ph = pairs.iq isa Integer ? pairs.ph :
        view_batched_phonon_states(copy_batched_phonon_states!(kept_bufs.ph, pairs.ph, keep_dev), kept)
    phase = pairs.phase === nothing ? nothing : copy_kept!(kept_bufs.P_kq, pairs.phase)
    ikq = pairs.ikq === nothing ? nothing : copy_kept_host!(kept_bufs.ikq, pairs.ikq)
    ik = pairs.ik isa Integer ? pairs.ik : copy_kept_host!(kept_bufs.ik, pairs.ik)
    iq = pairs.iq isa Integer ? pairs.iq : copy_kept!(kept_bufs.iq_dev, pairs.iq)
    wtk = pairs.wtk isa Number ? pairs.wtk : copy_kept!(kept_bufs.wtk, pairs.wtk)
    wtq = pairs.wtq isa Number ? pairs.wtq : copy_kept!(kept_bufs.wtq, pairs.wtq)
    xk = pairs.xk isa Vec3 ? pairs.xk : copy_kept_host!(kept_bufs.xk, pairs.xk)
    xq = pairs.xq isa Vec3 ? pairs.xq : copy_kept_host!(kept_bufs.xq, pairs.xq)
    merge(pairs, (; n = nkeep, el_k, el_kq, ph, phase, ik, ikq, iq, wtk, wtq, xk, xq))
end

"""
    estimate_device_memory(model; nk, nkq, n_outer_batch = nothing, n_inner_tile = nothing,
                           calculators = [], backend = CPUBackend()) -> NamedTuple

Estimate the device memory of an e-ph run without running it, from the byte counts the loop plans
with (`engine_bytes` and the calculators' `eph_batched_bytes_per_point`) at box widths `nw`, so a
windowed run uses less. The order follows `model.epmat_outer_momentum` (`el` → outer-k, `ph` →
outer-q); the state containers are not counted. Returns `(; loop, committed, per_pair, batch,
free)`, `batch` the inner tile `plan_batch` would pick on `backend` (the cap on a `CPUBackend`).

Actual device usage starts ~100-150 MB higher: the CUDA library context and workspace (cuBLAS
etc.) are allocated lazily on the first kernel launch and are not a per-run buffer.
"""
function estimate_device_memory(model::Model{FT}; nk::Integer, nkq::Integer, n_outer_batch = nothing,
        n_inner_tile = nothing, calculators = [], backend::AbstractBackend = CPUBackend()) where {FT}
    (; nw, nmodes) = model
    outer_k = model.epmat_outer_momentum == "el"
    el_qty = union([:u], required_el_quantities.(calculators)...)
    ph_qty = union(loop_ph_quantities(model, (:None, 0.0)), required_ph_quantities.(calculators)...)
    nb = min(something(n_outer_batch, outer_k ? 256 : 16), outer_k ? Int(nk) : Int(nkq))
    bytes = outer_k ?
        engine_bytes(OuterKEngine, model; nband_max_k = nw, nband_max_kq = nw, nk, nkq, el_qty, ph_qty,
            drop_pairs = false, covariant_derivative_of_g = false, eph_phonon_basis = :eigenmode) :
        engine_bytes(OuterQEngine, model; nband_max_k = nw, nband_max_kq = nw, nk, n_outer_batch = nb,
            el_qty, ph_qty, drop_pairs = false, precompute_el_kq = false, eph_phonon_basis = :eigenmode)
    for c in calculators
        b = eph_batched_bytes_per_point(c, EPBlock{outer_k ? OuterKLoop : OuterQLoop}; nw, nmodes,
                                        nband_max_k = nw, nband_max_kq = nw)
        bytes = (; persistent = bytes.persistent + b.persistent, per_outer = bytes.per_outer + b.per_outer,
                   per_pair = bytes.per_pair + b.per_pair)
    end
    committed = bytes.persistent + bytes.per_outer * nb
    cap = something(n_inner_tile, outer_k ? Int(nkq) : min(2^15, Int(nk)))
    batch = plan_batch(backend, bytes.per_pair, committed, cap; what = outer_k ? "outer_k" : "outer_q")
    (; loop = outer_k ? :outer_k : :outer_q, committed, bytes.per_pair, batch, free = free_bytes(backend))
end


# =============================================================================
# Deprecated driver name — forwarder, removed after one release. Explicit @warn (maxlog=1) because
# Base.@deprecate depwarns are invisible in ordinary script runs (Julia ≥ 1.5). The driver was renamed
# to the run_eph_over_<outer>_and_<inner> scheme. Delete this block when the old name goes.
function run_eph_outer_q(args...; kwargs...)
    @warn "run_eph_outer_q is deprecated; use run_eph_over_q_and_k (identical arguments)." maxlog=1
    run_eph_over_q_and_k(args...; kwargs...)
end
