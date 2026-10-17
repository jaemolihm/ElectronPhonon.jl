# Which phonon modes a run builds, for which q, and where they live.
#
# An outer-k run over a k+q grid on a device with a finite `energy_conservation_tol`, whose
# calculators all `allows_phonon_mode_truncation`, may keep only the modes that can conserve energy:
# a mode with |ω| > Ω = (E_max - E_min) + tol, E over the resident states of both sides, has no
# process inside the tolerance for any pair. The ω pass that finds them (a value-only solve over all
# q) runs when the phonons of all q would take more than 1/PHONON_MODE_PASS_FREE_DIVISOR of the free
# device memory. The kept modes then live resident, or per outer batch (`_phonon_storage_plan`),
# for the q of that batch's kept pairs (`_select_qs_per_outer_batch`).
#
# Both choices depend on the free device memory, so a run that may truncate returns no phonons
# (`return_phs = false`): their modes and storage are not a property of the inputs.

# The free device memory a loop with `committed` bytes and an inner tile of `n_tile` points of
# `per_pair` bytes needs, at `plan_batch`'s default 7/10 headroom.
_needed_bytes_at_tile(committed, per_pair, n_tile) = committed + cld(per_pair * n_tile * 10, 7)

# Whether a run may keep only the lowest phonon modes (`_setup_phonons`): an outer-k run over a k+q
# grid on a device with a finite `energy_conservation_tol`, eigenmode phonons with `e` and `u` only
# from a non-polar model and no eigenpair cache, states on both sides, and every calculator
# `allows_phonon_mode_truncation`.
function _may_truncate_phonon_modes(order, model, els_k, els_kq, options)
    (; inner_loop_kq, energy_conservation_tol, backend, eph_phonon_basis, ph_qty, ph_eigenpairs,
       calculators) = options
    order isa OuterKLoop && _uses_device_pair_selection(backend, inner_loop_kq, energy_conservation_tol) &&
        eph_phonon_basis == :eigenmode && issubset(ph_qty, (:e, :u)) &&
        !model.polar_phonon.use && ph_eigenpairs === nothing && els_k.nk > 0 && els_kq.nk > 0 &&
        all(allows_phonon_mode_truncation, calculators)
end

# The ω pass of `_setup_phonons` (a value-only phonon solve over all q) runs only when the phonons of
# all q would take more than 1/PHONON_MODE_PASS_FREE_DIVISOR of the free device memory. A
# performance preference, not a measured threshold: the pass costs setup time (+3.2 s for SrVO3,
# 10 K, nk = 240, A100-SXM4-80GB at 225 W, commit 8985613) that a run whose phonons fit easily need
# not pay.
const PHONON_MODE_PASS_FREE_DIVISOR = 2

# The smallest inner tile a run over a k+q grid keeps before it builds the phonons per outer batch
# (`_phonon_storage_plan`). A performance preference, not a measured optimum: per-batch phonons
# rebuild the q of every batch (SrVO3, 10 K, nk = 400, A100-SXM4-80GB at 225 W, commit 7d5f68a: the
# whole BTE pass took 558.6 s with per-batch phonons vs 231.9 s resident), so a narrower tile with
# resident phonons is preferred down to here.
const GPU_PHONONS_RESIDENT_MIN_TILE = 2^16

# The number of phonon modes a run keeps from the frequencies `ω` `(nmodes, nq)`: those with
# `ω ≤ Ω` at some q. The eigenvalues ascend, so when no ω is below `-Ω` they are the lowest
# `nmodes_kept` of each q; otherwise every mode is kept.
_kept_phonon_modes(ω, Ω) =
    minimum(ω) >= -Ω ? max(1, Int(maximum(sum(ω .<= Ω; dims = 1)))) : size(ω, 1)

"""
    _phonon_storage_plan(; table_bytes, committed, per_pair, free, nkq, n_inner_tile, force_per_batch)
        -> (; storage, min_tile, needed)

Where the phonons of a run that may build them per outer batch live: `:resident` when the kept modes
of all q (`table_bytes`), the loop's committed buffers and an inner tile of
`min_tile = min(nkq, something(n_inner_tile, GPU_PHONONS_RESIDENT_MIN_TILE))` points of `per_pair`
bytes, at `plan_batch`'s 7/10 headroom, need no more than `free` (`needed`); `:per_batch` otherwise,
or when `force_per_batch`.
"""
function _phonon_storage_plan(; table_bytes, committed, per_pair, free, nkq, n_inner_tile, force_per_batch)
    min_tile = min(nkq, something(n_inner_tile, GPU_PHONONS_RESIDENT_MIN_TILE))
    needed = _needed_bytes_at_tile(table_bytes + committed, per_pair, min_tile)
    (; storage = force_per_batch || needed > free ? :per_batch : :resident, min_tile, needed)
end

"""
    _select_qs_per_outer_batch(backend, kpts, kqpts, qpts, els_k, els_kq, ω_all, n_outer_batch, tol)
        -> Vector{Vector{Int32}}

The q of the pairs `_select_kqpairs_on_device!` keeps in each outer batch
`Iterators.partition(1:kpts.n, n_outer_batch)`, with the frequencies `ω_all`, in ascending order:
the q whose phonons an engine with `phonons_per_batch` builds for that batch. Runs on its own
selection buffers, one tile of every k+q point, dropped on return.
"""
function _select_qs_per_outer_batch(backend, kpts, kqpts, qpts, els_k, els_kq, ω_all, n_outer_batch, tol)
    ck, ckq = _pair_grid_coords(kpts, kqpts, qpts)
    selection = DevicePairSelection(backend, ckq, qpts, kqpts.n; compact = false)
    q_marks = alloc(backend, Bool, qpts.n)
    map(Iterators.partition(1:kpts.n, n_outer_batch)) do iks_batch
        fill!(q_marks, false)
        for ik in iks_batch
            k_int = (ck[1, ik], ck[2, ik], ck[3, ik])
            _mark_conserving_qs!(q_marks, selection, k_int, qpts.ngrid, view(els_k, ik:ik), els_kq,
                                 1, kqpts.n, ω_all, tol)
        end
        Vector{Int32}(findall(Array(q_marks)))
    end
end

# The phonons of a run, from its resident electron states `els_k`, `els_kq` and q set `qpts`. Returns
# `(; phs, phonon_batches, return_phs)`: `phs` the resident table (or an empty container with the
# run's modes when they are built per outer batch), `phonon_batches` `nothing` or the per-batch
# frequencies `ω` of the kept modes, the q of each batch `iqs_per_batch` and `nq`, and `return_phs`
# whether the driver may hand `phs` back.
function _setup_phonons(order, model::Model{FT}, kpts, kqpts, qpts, els_k, els_kq, options) where {FT}
    (; el_qty, ph_qty, backend, eph_phonon_basis, ph_eigenpairs, energy_conservation_tol, calculators,
       inner_loop_kq, precompute_el_kq, covariant_derivative_of_g, _force_phonons_per_batch,
       verbosity) = options
    # The host solves of a GPU run (polar phonons) keep the default interpolation.
    fourier_mode = backend isa CPUBackend ? options.fourier_mode : "gridopt"
    truncation_allowed = _may_truncate_phonon_modes(order, model, els_k, els_kq, options)
    (!_force_phonons_per_batch || truncation_allowed) || throw(ArgumentError(
        "_force_phonons_per_batch needs a run that may truncate the phonon modes"))
    nmodes_kept, storage, ω_kept = model.nmodes, :resident, nothing
    # `free_bytes` reads the driver, which does not see the pool's cached memory.
    truncation_allowed && reclaim_device_memory(backend)
    if truncation_allowed && (_force_phonons_per_batch || qpts.n * _phonon_state_bytes(FT, model.nmodes,
            ph_qty) > free_bytes(backend) ÷ PHONON_MODE_PASS_FREE_DIVISOR)
        # The kept modes, from a value-only solve over all q.
        (emin_k, emax_k), (emin_kq, emax_kq) = _window_energy_range(els_k), _window_energy_range(els_kq)
        Ω = max(emax_k, emax_kq) - min(emin_k, emin_kq) + energy_conservation_tol
        ω = compute_phonon_states_batched(model, qpts, [:e]; backend).e
        nmodes_kept = _kept_phonon_modes(ω, Ω)
        if verbosity > 0 && mpi_isroot()
            Ω_meV = round(Ω / unit_to_aru(:meV), digits = 2)
            @info "Phonon modes kept = $nmodes_kept of $(model.nmodes) (|ω| ≤ $Ω_meV meV)"
        end
        ω_kept = ω[1:nmodes_kept, :]
        ω = nothing
        reclaim_device_memory(backend)

        # Where they live: all q resident, or per outer batch, for the q of its kept pairs, selected
        # with the value-only frequencies `ω_kept`.
        loop = _plan_widths(order, model, backend, calculators; n_outer = kpts.n, n_inner = kqpts.n,
            nk = kpts.n, nkq = kqpts.n, nchunks = 1, nq_grid = prod(qpts.ngrid), energy_conservation_tol,
            options.n_outer_batch, n_inner_tile = 1, nband_max_k = els_k.nband_max,
            nband_max_kq = els_kq.nband_max, els_k, els_kq, phs = (; nmodes = nmodes_kept), el_qty, ph_qty,
            precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq)
        table_bytes = qpts.n * _phonon_state_bytes(FT, nmodes_kept, ph_qty; ndisp = model.nmodes)
        free = free_bytes(backend)
        plan = _phonon_storage_plan(; table_bytes, loop.committed, loop.bytes.per_pair, free,
            nkq = kqpts.n, options.n_inner_tile, force_per_batch = _force_phonons_per_batch)
        storage = plan.storage
        if storage == :per_batch && verbosity > 0 && mpi_isroot()
            GB(x) = round(x / 1e9, digits = 1)
            @info "Phonons built per outer batch: " * (plan.needed > free ?
                "the kept modes of all q ($(GB(table_bytes)) GB) and the loop's buffers need " *
                "$(GB(plan.needed)) GB at an inner tile of $(plan.min_tile) points, but " *
                "$(GB(free)) GB are free" : "_force_phonons_per_batch")
        end
    end
    if storage == :resident
        phs = maybe_time(verbosity) do
            compute_phonon_states_batched(model, qpts, ph_qty; fourier_mode, eph_phonon_basis, backend,
                                          eigenpairs = ph_eigenpairs, nmodes_kept)
        end
        return (; phs, phonon_batches = nothing, return_phs = !truncation_allowed)
    end
    # An empty container with the run's modes: the engine fills one per outer batch.
    phs = BatchedPhononState(backend, nmodes_kept, 0, ph_qty; FT, ndisp = model.nmodes)
    # The q of each batch's kept pairs, found before the widths are planned so that the plan holds the
    # largest batch's phonons and their solve.
    n_outer_batch = _outer_batch_width(order, backend, options.n_outer_batch, kpts.n)
    iqs_per_batch = _select_qs_per_outer_batch(backend, kpts, kqpts, qpts,
        els_k, els_kq, ω_kept, n_outer_batch, energy_conservation_tol)::Vector{Vector{Int32}}
    reclaim_device_memory(backend)
    (; phs, phonon_batches = (; ω = ω_kept, iqs_per_batch, nq = qpts.n), return_phs = false)
end

# The device bytes the phonons per outer batch add to the outer-k engine (`engine_bytes`): the
# storage of the largest batch's phonons, the q index and upload, and the transients of one batch's
# phonon solve (`compute_phonon_states_batched` into the storage) in `persistent`; the pair's
# `iqs_ph_dev` entry in `per_pair`.
function _phonon_batches_bytes(model::Model{FT}, backend, phonon_batches, nmodes_kept) where {FT}
    phonon_batches === nothing && return (; persistent = 0, per_pair = 0)
    (; nmodes) = model
    cx, rl = sizeof(Complex{FT}), sizeof(FT)
    nq_max = maximum(length, phonon_batches.iqs_per_batch; init = 0)
    # The solve's transients: the dynamical matrices with their R vectors, the interpolator's Fourier
    # phase and output, the masses, and per Fourier block the D(q) stack, its eigenvalues and the
    # staged q.
    ph_solve = if nq_max == 0
        0
    else
        nr_ph, ndata_ph = length(model.ph_dyn.irvec), model.ph_dyn.ndata
        nblock = _default_batch_size(backend, nr_ph, ndata_ph; nk_hint = nq_max)
        cx * length(model.ph_dyn.op_r) + rl * 3nr_ph + cx * nr_ph * nblock + cx * ndata_ph + rl * nmodes +
            (cx * ndata_ph + rl * nmodes + rl * 3) * nblock
    end
    persistent = _phonon_state_bytes(FT, nmodes_kept, (:e, :u); ndisp = nmodes) * nq_max +
        sizeof(Int32) * (phonon_batches.nq + nq_max) + ph_solve
    (; persistent, per_pair = sizeof(Int))
end

# The (minimum, maximum) energy of the states `els` within their bands.
_window_energy_range(els) = (in_window = axes(els.e, 1) .<= reshape(els.nband, 1, :);
    (minimum(ifelse.(in_window, els.e, Inf)), maximum(ifelse.(in_window, els.e, -Inf))))
