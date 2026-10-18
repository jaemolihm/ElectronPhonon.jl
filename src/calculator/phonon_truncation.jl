# Which phonon modes a run builds, for which q, and where they live.
#
# An outer-k run over a k+q grid on a device with a finite `energy_conservation_tol`, whose
# calculators all `allows_phonon_mode_truncation`, may keep only the modes that can conserve energy:
# a mode with |ω| > Ω = (E_max - E_min) + tol, E over the resident states of both sides, has no
# process inside the tolerance for any pair. The ω pass that finds them (a value-only solve over all
# q) runs when the phonons of all modes and q would take more than 1/PHONON_MODE_PASS_BUDGET_DIVISOR
# of the device budget (`device_budget`) left after the loop's buffers. The kept modes then live
# resident, or per outer batch (`_phonon_storage_plan`), for the q of that batch's kept pairs
# (`_select_qs_per_outer_batch`).
#
# Both choices depend on the device budget, so a run that may truncate returns no phonons
# (`return_phs = false`): their modes and storage are not a property of the inputs.

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
# all modes and q would take more than 1/PHONON_MODE_PASS_BUDGET_DIVISOR of the device budget
# (`device_budget`) left after the loop's committed buffers. The factor 2 is a guess, not tuned: the
# pass costs setup time (+3.2 s for SrVO3, 10 K, nk = 240, A100-SXM4-80GB at 225 W, commit 8985613)
# that a run whose phonons fit easily need not pay.
const PHONON_MODE_PASS_BUDGET_DIVISOR = 2

# The smallest inner tile a run over a k+q grid keeps before it builds the phonons per outer batch
# (`_phonon_storage_plan`). Per-batch phonons rebuild the q of every batch (SrVO3, 10 K, nk = 400,
# A100-SXM4-80GB at 225 W, commit 7d5f68a: the whole BTE pass took 558.6 s with per-batch phonons
# vs 231.9 s resident), so resident phonons with a narrower tile are preferred down to this tile.
# The value 2^16 is a guess, not tuned.
const GPU_PHONONS_RESIDENT_MIN_TILE = 2^16

# The number of phonon modes a run keeps from the frequencies `ω` `(nmodes, nq)`: those with
# `ω ≤ Ω` at some q. The eigenvalues ascend, so when no ω is below `-Ω` they are the lowest
# `nmodes_kept` of each q; otherwise every mode is kept.
_kept_phonon_modes(ω, Ω) =
    minimum(ω) >= -Ω ? max(1, Int(maximum(sum(ω .<= Ω; dims = 1)))) : size(ω, 1)

"""
    _phonon_storage_plan(; table_bytes, committed, per_pair, budget, nkq, n_inner_tile)
        -> (; per_batch, min_tile, needed)

Whether the phonons of a run that may build them per outer batch are built per batch: the kept modes
of all q (`table_bytes`), the loop's committed buffers and an inner tile of
`min_tile = min(nkq, something(n_inner_tile, GPU_PHONONS_RESIDENT_MIN_TILE))` points of `per_pair`
bytes, at `plan_batch`'s default 7/10 headroom, need `needed` bytes. `per_batch` when `needed > budget`
(the device budget).
"""
function _phonon_storage_plan(; table_bytes, committed, per_pair, budget, nkq, n_inner_tile)
    min_tile = min(nkq, something(n_inner_tile, GPU_PHONONS_RESIDENT_MIN_TILE))
    needed = table_bytes + committed + cld(per_pair * min_tile * 10, 7)
    (; per_batch = needed > budget, min_tile, needed)
end

"""
    _select_qs_per_outer_batch(backend, kpts, kqpts, qpts, els_k, els_kq, ω_kept, n_outer_batch, tol)
        -> Vector{Vector{Int32}}

The q of the pairs `_select_kqpairs_on_device!` keeps in each outer batch
`Iterators.partition(1:kpts.n, n_outer_batch)`, with the frequencies `ω_kept`, in ascending order:
the q whose phonons an engine with `phonons_per_batch` builds for that batch. Runs on its own
selection buffers, one tile of every k+q point, dropped on return.
"""
function _select_qs_per_outer_batch(backend, kpts, kqpts, qpts, els_k, els_kq, ω_kept, n_outer_batch, tol)
    ck, ckq = _pair_grid_coords(kpts, kqpts, qpts)
    selection = DevicePairSelection(backend, ckq, qpts, kqpts.n; compact = false)
    q_marks = alloc(backend, Bool, qpts.n)
    map(Iterators.partition(1:kpts.n, n_outer_batch)) do iks_batch
        fill!(q_marks, false)
        for ik in iks_batch
            k_int = (ck[1, ik], ck[2, ik], ck[3, ik])
            _mark_conserving_qs!(q_marks, selection, k_int, qpts.ngrid, view(els_k, ik:ik), els_kq,
                                 1, kqpts.n, ω_kept, tol)
        end
        Vector{Int32}(findall(Array(q_marks)))
    end
end

# The loop's widths and committed bytes at an inner tile of 1, with the phonon container `phs` (of
# no q) fixing the modes. Valid before the engine exists because a device run has one chunk.
function _plan_loop_without_phonon_table(order, model, kpts, kqpts, qpts, els_k, els_kq, options, phs)
    (; el_qty, ph_qty, backend, eph_phonon_basis, energy_conservation_tol, calculators, inner_loop_kq,
       precompute_el_kq, covariant_derivative_of_g, max_device_bytes) = options
    _plan_widths(order, model, backend, calculators; n_outer = kpts.n, n_inner = kqpts.n,
        nk = kpts.n, nkq = kqpts.n, nchunks = 1, nq_grid = prod(qpts.ngrid), energy_conservation_tol,
        options.n_outer_batch, n_inner_tile = 1, nband_max_k = els_k.nband_max,
        nband_max_kq = els_kq.nband_max, els_k, els_kq, phs, el_qty, ph_qty,
        precompute_el_kq, covariant_derivative_of_g, eph_phonon_basis, inner_loop_kq,
        max_device_bytes)
end

# The phonons of a run, from its resident electron states `els_k`, `els_kq` and q set `qpts`. Returns
# `(; phs, phonon_batches, return_phs)`: `phs` the resident table (or an empty container with the
# run's modes when they are built per outer batch), `phonon_batches` `nothing` or the per-batch
# frequencies `ω_kept` of the kept modes and the q of each batch `iqs_per_batch`, and `return_phs`
# whether the driver may hand `phs` back.
function _setup_phonons(order, model::Model{FT}, kpts, kqpts, qpts, els_k, els_kq, options,
                        fourier_mode) where {FT}
    (; ph_qty, backend, eph_phonon_basis, ph_eigenpairs, energy_conservation_tol, max_device_bytes,
       verbosity) = options
    truncation_allowed = _may_truncate_phonon_modes(order, model, els_k, els_kq, options)
    resident_phonons(nmodes_kept) = maybe_time(verbosity) do
        compute_phonon_states_batched(model, qpts, ph_qty; fourier_mode, eph_phonon_basis, backend,
                                      eigenpairs = ph_eigenpairs, nmodes_kept)
    end
    if !truncation_allowed
        return (; phs = resident_phonons(model.nmodes), phonon_batches = nothing, return_phs = true)
    end
    # `free_bytes` reads the driver, which does not see the pool's cached memory.
    reclaim_device_memory(backend)
    # The loop's buffers with the phonons of all modes and none of the q, which bound what is left for
    # the phonon table.
    phs_empty_all = BatchedPhononState(backend, model.nmodes, 0, ph_qty; FT, ndisp = model.nmodes)
    loop_all = _plan_loop_without_phonon_table(order, model, kpts, kqpts, qpts, els_k, els_kq, options,
                                               phs_empty_all)
    left = device_budget(backend, max_device_bytes) - loop_all.committed
    if qpts.n * _phonon_state_bytes(FT, model.nmodes, ph_qty) <= left ÷ PHONON_MODE_PASS_BUDGET_DIVISOR
        # All modes of all q resident, with no ω pass.
        return (; phs = resident_phonons(model.nmodes), phonon_batches = nothing, return_phs = false)
    end
    # The kept modes, from a value-only solve over all q.
    (emin_k, emax_k), (emin_kq, emax_kq) = _window_energy_range(els_k), _window_energy_range(els_kq)
    # Ω is the window energy range plus the tolerance. Modes are dropped only when it is below the
    # top phonon frequency: a narrow window at low T with a small tolerance, where often only the
    # acoustic modes remain.
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

    # Where they live: all q resident, or per outer batch, for the q of its kept pairs, selected with
    # the value-only frequencies `ω_kept`. An empty container with the run's modes sizes the plan.
    phs_empty = nmodes_kept == model.nmodes ? phs_empty_all :
        BatchedPhononState(backend, nmodes_kept, 0, ph_qty; FT, ndisp = model.nmodes)
    loop = nmodes_kept == model.nmodes ? loop_all :
        _plan_loop_without_phonon_table(order, model, kpts, kqpts, qpts, els_k, els_kq, options, phs_empty)
    table_bytes = qpts.n * _phonon_state_bytes(FT, nmodes_kept, ph_qty; ndisp = model.nmodes)
    budget = device_budget(backend, max_device_bytes)
    plan = _phonon_storage_plan(; table_bytes, loop.committed, loop.bytes.per_pair, budget,
        nkq = kqpts.n, options.n_inner_tile)
    if !plan.per_batch
        return (; phs = resident_phonons(nmodes_kept), phonon_batches = nothing, return_phs = false)
    end
    if verbosity > 0 && mpi_isroot()
        GB(x) = round(x / 1e9, digits = 1)
        @info "Phonons built per outer batch: the kept modes of all q ($(GB(table_bytes)) GB) and " *
            "the loop's buffers need $(GB(plan.needed)) GB at an inner tile of $(plan.min_tile) " *
            "points, but $(GB(budget)) GB are available"
    end
    # The q of each batch's kept pairs, found before the widths are planned so that the plan holds the
    # largest batch's phonons and their solve. Without the inference barrier, compiling the GPU BTE
    # path in the full Pkg.test (test_gpu.jl) ran >1 h on an H100; test_gpu_boltzmann_calculator.jl
    # alone does not reproduce it. Cause unknown.
    n_outer_batch = _outer_batch_width(order, backend, options.n_outer_batch, kpts.n)
    iqs_per_batch = Base.inferencebarrier(_select_qs_per_outer_batch)(backend, kpts, kqpts, qpts,
        els_k, els_kq, ω_kept, n_outer_batch, energy_conservation_tol)::Vector{Vector{Int32}}
    reclaim_device_memory(backend)
    (; phs = phs_empty, phonon_batches = (; ω_kept, iqs_per_batch), return_phs = false)
end

# The (minimum, maximum) energy of the states `els` within their bands.
function _window_energy_range(els)
    in_window = axes(els.e, 1) .<= reshape(els.nband, 1, :)
    (minimum(ifelse.(in_window, els.e, Inf)), maximum(ifelse.(in_window, els.e, -Inf)))
end
