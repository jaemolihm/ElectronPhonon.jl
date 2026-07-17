# Shared setup building blocks for the three e-ph drivers (run_eph_over_k_and_q,
# run_eph_over_q_and_k, run_eph_over_k_and_kq). Their `_setup_*` bodies are ~70% identical, so the
# common k-side filtering/state computation, the k+q electron-state branch, the per-thread ElPhData
# channel, the setup_calculator! fan-out, and the screening evaluation live here, each exactly once.

using Base.Threads: nthreads


# k-side setup shared by all three drivers: filter the outer k-points to the energy window and
# compute the electron states there. `verbosity` selects timing (@time when > 0, via `maybe_time`);
# `el_k_quantities` lets the outer-q GPU-batched path skip velocity/position.
function _setup_eph_common(
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


# Electron states at k+q, shared by all three drivers. With `el_kq_from_unfolding`, the states are
# computed only in the irreducible BZ (`kqpts_irr`) and unfolded to `kqpts` (carrying the eigenvector
# gauge) to keep gauge consistency between symmetry-equivalent k points; otherwise they are computed
# directly on `kqpts`. `kqpts_irr` / `ik_to_ikirr_isym_kq` are only read on the unfolding path.
function _compute_el_kq_states(
        model, kqpts, kqpts_irr, ik_to_ikirr_isym_kq, symmetry, el_kq_from_unfolding, window_kq;
        fourier_mode, use_gpu = false,
    )
    quantities = ["eigenvalue", "eigenvector", "velocity", "position"]
    if el_kq_from_unfolding
        symmetry !== nothing || throw(ArgumentError("el_kq_from_unfolding = true requires symmetry"))
        el_kq_save_irr = compute_electron_states(model, kqpts_irr, quantities, window_kq; fourier_mode, use_gpu)
        el_kq_save = unfold_ElectronStates(model, el_kq_save_irr, kqpts_irr, kqpts, ik_to_ikirr_isym_kq, symmetry; fourier_mode)
        # el_kq_save_irr is not used anymore.
        el_kq_save_irr !== el_kq_save && empty!(el_kq_save_irr)
    else
        el_kq_save = compute_electron_states(model, kqpts, quantities, window_kq; fourier_mode, use_gpu)
    end
    el_kq_save
end


# Per-thread ElPhData channel used by the CPU inner loops of the outer-k and over-k-and-kq drivers.
function _make_epdatas_channel(::Type{FT}, nw, nmodes, nband_max) where {FT}
    ch = Channel{ElPhData{FT}}(nthreads())
    foreach(1:nthreads()) do _
        put!(ch, ElPhData{FT}(nw, nmodes, nband_max))
    end
    ch
end


# setup_calculator! fan-out shared by all three drivers. The common keyword payload (band range,
# k+q states/grid, carrier counts, thread chunking) is passed explicitly; driver-specific extras
# (backend / verbosity) forward through `kwargs`.
function _setup_calculators!(
        calculators, kpts, qpts, el_k_save;
        nw, nmodes, rng_band, el_states_kq, kqpts,
        nelec_below_window_k, nelec_below_window_kq, nchunks_threads, kwargs...,
    )
    for calc in calculators
        setup_calculator!(calc, kpts, qpts, el_k_save;
            nw, nmodes, rng_band, el_states_kq, kqpts,
            nelec_below_window_k, nelec_below_window_kq, nchunks_threads, kwargs...)
    end
end


# Dielectric screening ε(q, ω=0) per phonon mode, evaluated once for all three inner loops. `ϵs` is
# filled in place (set to 1 when screening is off).
# FIXME: T, μ are read from calculators[1].occ[1] (a layering violation); moving the screening
#        parameters into calculator ownership lands with the stage-4 payload redesign.
function _apply_screening!(ϵs, calculators, model, xq, epdata, screening_params)
    if screening_params !== nothing
        (; T, μ) = calculators[1].occ[1]
        xq_ = normalize_kpoint_coordinate(xq .+ 0.5) .- 0.5
        ϵs .= epsilon_lindhard.(Ref(model.recip_lattice * xq_), epdata.ph.e, T, μ, Ref(screening_params))
        ϵs .= real.(ϵs)
    else
        ϵs .= 1
    end
    ϵs
end
