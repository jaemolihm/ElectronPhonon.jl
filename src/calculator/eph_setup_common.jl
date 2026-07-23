# Shared setup building blocks for the three e-ph drivers (run_eph_over_k_and_q,
# run_eph_over_q_and_k, run_eph_over_k_and_kq). Their `_setup_*` bodies are ~70% identical, so the
# common k-side filtering/state computation, the k+q electron-state branch, and the screening
# evaluation live here, each exactly once. (The per-thread EPState channel `get_epstates_channel`
# lives next to `EPState` in src/EPState.jl. The setup_calculator! fan-out is a plain `for` loop
# inlined in each driver — the shared kwarg contract is written at each call site regardless.)

# k-side setup shared by all three drivers: filter the outer k-points to the energy window and
# compute the electron states there. `verbosity` selects timing (@time when > 0, via `maybe_time`);
# `el_k_quantities` lets the outer-q GPU-batched path skip velocity/position.
function _setup_electron_k(
        model :: Model, kpts_input;
        window_k, mpi_comm_k, symmetry, fourier_mode, use_gpu = false, verbosity = 1,
        el_k_quantities = ["eigenvalue", "eigenvector", "velocity", "position"],
    )
    (; nw) = model
    kpts, iband_min, iband_max, nelec_below_window_k = maybe_time(verbosity) do
        filter_kpoints(kpts_input, nw, model.el_ham, window_k, mpi_comm_k; symmetry, fourier_mode, use_gpu)
    end
    el_k_save = maybe_time(verbosity) do
        compute_electron_states(model, kpts, el_k_quantities, window_k; fourier_mode, use_gpu)
    end
    (; kpts, iband_min, iband_max, nelec_below_window_k, el_k_save)
end


# Electron states at k+q, shared by all three drivers. Owns the full-`kqpts` → states step including
# the symmetry handling: with `el_kq_from_unfolding`, it folds the full `kqpts` to the irreducible BZ
# (`kqpts_irr`), computes the states there, and unfolds them back to `kqpts` (carrying the eigenvector
# gauge) for gauge consistency between symmetry-equivalent k points; otherwise it computes the states
# directly on `kqpts`. Callers pass only the full `kqpts` and need not pre-derive the irreducible grid.
# (`run_eph_over_k_and_kq` filters in the irreducible BZ and unfolds to build `kqpts`, so the fold here
# re-folds an already-unfolded grid — a cheap redundant fold accepted to keep one code path. It runs
# only on the CPU symmetry path, which `run_eph_over_k_and_kq` gates `el_kq_from_unfolding` on.)
function _setup_electron_kq(model, kqpts, symmetry, el_kq_from_unfolding, window_kq;
        fourier_mode, use_gpu = false)
    quantities = ["eigenvalue", "eigenvector", "velocity", "position"]
    if el_kq_from_unfolding
        symmetry !== nothing || throw(ArgumentError("el_kq_from_unfolding = true requires symmetry"))
        kqpts_irr, ik_to_ikirr_isym_kq = fold_kpoints(kqpts, symmetry)
        el_kq_save_irr = compute_electron_states(model, kqpts_irr, quantities, window_kq; fourier_mode, use_gpu)
        el_kq_save = unfold_ElectronStates(model, el_kq_save_irr, kqpts_irr, kqpts, ik_to_ikirr_isym_kq, symmetry; fourier_mode)
        # el_kq_save_irr is not used anymore.
        el_kq_save_irr !== el_kq_save && empty!(el_kq_save_irr)
    else
        el_kq_save = compute_electron_states(model, kqpts, quantities, window_kq; fourier_mode, use_gpu)
    end
    el_kq_save
end


# Dielectric screening is currently disabled: ϵ ≡ 1 always. Passing a nontrivial `screening_params`
# throws (the drivers reject it at entry; this is the defensive guard). The Lindhard evaluation is
# kept commented for reference — re-enabling it needs a self-contained (T, μ) source, not the
# `calculators[1].occ[1]` read it used to do (a layering violation that silently used the first
# calculator's first occupation set even in multi-T runs). See
# plans/calculator_gpu_extensibility.md [DECISION-3].
function _apply_screening!(ϵs, calculators, model, xq, epstate, screening_params)
    screening_params === nothing || error(
        "screening_params is not supported: dielectric screening is currently disabled (ϵ ≡ 1). " *
        "Pass screening_params = nothing.")
    ϵs .= 1
    # if screening_params !== nothing
    #     (; T, μ) = calculators[1].occ[1]
    #     xq_ = normalize_kpoint_coordinate(xq .+ 0.5) .- 0.5
    #     ϵs .= epsilon_lindhard.(Ref(model.recip_lattice * xq_), epstate.ph.e, T, μ, Ref(screening_params))
    #     ϵs .= real.(ϵs)
    # else
    #     ϵs .= 1
    # end
    ϵs
end
