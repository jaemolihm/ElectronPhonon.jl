using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads

export compute_electron_states
export compute_phonon_states

"""
    compute_electron_states(model, kpts, quantities, window=(-Inf, Inf); fourier_mode="normal",
                            backend=CPUBackend(), eigenpairs=nothing)
Compute the quantities listed in `quantities` and return a vector of ElectronState.
`quantities` can containing the following: "eigenvalue", "eigenvector", "velocity_diagonal", "velocity"
`eigenpairs`: an [`Eigenpairs`](@ref) covering every k point of `kpts`. Its `e_full` and
`u_full` are copied in per k instead of diagonalizing H(k), so runs sharing one cache share the
eigenvector gauge; everything else, `window` included, is computed exactly as without it.
`fourier_mode` then no longer affects the eigenpairs -- the cache's own Fourier mode already did --
so a with-cache run reproduces a without-cache one bit for bit only when the cache was built with
the same `fourier_mode`. The one exception is `quantities == ["eigenvalue"]`: a cache holds the
eigenvalues of the full eigensolve, which agree with the value-only solve's only to round-off. A
cache is resident on the backend that built it, so it must be built with the `backend` the run uses.
"""
function compute_electron_states(model::Model{FT}, kpts, quantities, window=(-Inf, Inf);
        fourier_mode="normal", backend=CPUBackend(),
        eigenpairs::Union{Nothing, Eigenpairs}=nothing) where FT
    # TODO: MPI, threading
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "velocity", "position"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end
    (; nw) = model
    _check_eigenpairs(eigenpairs, nw, backend)

    states = [ElectronState{FT}(nw) for _ in 1:kpts.n]
    if quantities == []
        return states
    end

    if backend isa CPUBackend
        _compute_electron_states_cpu!(states, model, kpts, quantities, window, eigenpairs;
                                      fourier_mode)
    else
        _compute_electron_states_device!(states, model, kpts, quantities, window, backend,
                                        eigenpairs)
    end
    states
end

"""
    compute_electron_states(model, sel::FilteredBandStates, quantities; fourier_mode="normal",
                            backend=CPUBackend(), eigenpairs=nothing)

Compute electron states for exactly the per-k bands selected by `sel`: each state's band range is
`sel.band_extent[ik]` (from the selection) rather than a single energy window, so a multigrid (whose
per-k band extent is narrow at fine-only nodes and wide at coincident nodes) gets the right bands per
k. Returns a vector of `ElectronState` over `sel.kpts`. `eigenpairs` is as in the window method.
"""
function compute_electron_states(model::Model{FT}, sel::FilteredBandStates, quantities;
        fourier_mode="normal", backend=CPUBackend(),
        eigenpairs::Union{Nothing, Eigenpairs}=nothing) where FT
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "velocity", "position"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end
    _check_eigenpairs(eigenpairs, model.nw, backend)
    kpts = sel.kpts
    states = [ElectronState{FT}(model.nw) for _ in 1:kpts.n]
    isempty(quantities) && return states
    if backend isa CPUBackend
        _compute_electron_states_cpu!(states, model, kpts, quantities, sel.band_extent, eigenpairs;
                                      fourier_mode)
    else
        _compute_electron_states_device!(states, model, kpts, quantities, sel.band_extent, backend,
                                        eigenpairs)
    end
    states
end

# `compute_electron_states` accepts either a single energy-window `Tuple` (uniform; applied to every
# k — still used by the k+q / window-based callers) or a per-k `Vector` of band ranges (from a
# `FilteredBandStates`). `_window_for` selects the per-k argument that `set_window!` then applies: the
# same tuple for every k, or the k-th band range from the vector.
_window_for(window::Tuple, ik) = window
_window_for(window::AbstractVector, ik) = window[ik]

# Which quantities each backend needs to compute, derived from `quantities` (kept in one place so
# the caller passes only `quantities`). `need_velocity` gates the velocity-operator interpolation;
# `need_position` the position (Berry-connection) interpolation.
function _electron_state_needs(model, quantities)
    need_vfull    = "velocity" ∈ quantities
    need_vdiag    = "velocity_diagonal" ∈ quantities
    need_position = "position" ∈ quantities ||
        (need_vfull && model.el_velocity_mode === :BerryConnection)
    need_velocity = need_vfull || need_vdiag || "position" ∈ quantities
    (; need_vfull, need_vdiag, need_position, need_velocity)
end

function _compute_electron_states_cpu!(states, model::Model{FT}, kpts, quantities, window,
                                       eigenpairs; fourier_mode) where FT
    (; el_velocity_mode) = model
    (; need_vfull, need_vdiag, need_position, need_velocity) = _electron_state_needs(model, quantities)
    @threads for iks in chunks(kpts.vectors; n=2nthreads())
        # Setup thread-local WannierInterpolators. With supplied eigenpairs there is no H(k) to
        # interpolate, and nothing else uses `ham`.
        ham = if eigenpairs === nothing
            itp_ham = get_interpolator(model.el_ham; fourier_mode)
            register_kpoints!(itp_ham, view(kpts.vectors, iks))
            itp_ham
        end
        if need_velocity
            vel = if el_velocity_mode === :Direct
                get_interpolator(model.el_vel; fourier_mode)
            else
                get_interpolator(model.el_ham_R; fourier_mode)
            end
            register_kpoints!(vel, view(kpts.vectors, iks))
        end
        if need_position
            pos = get_interpolator(model.el_pos; fourier_mode)
            register_kpoints!(pos, view(kpts.vectors, iks))
        end

        for ik in iks
            xk = kpts.vectors[ik]
            el = states[ik]

            if quantities == ["eigenvalue"]
                _set_eigen_valueonly_from!(el, eigenpairs, ham, xk)
                set_window!(el, _window_for(window, ik))
            else
                _set_eigen_from!(el, eigenpairs, ham, xk)
                set_window!(el, _window_for(window, ik))
                if need_position
                    set_position!(el, pos, xk)
                end
                if need_vfull
                    set_velocity!(el, vel, xk, el_velocity_mode)
                    for i in el.rng
                        el.vdiag[i] = real.(el.v[i, i])
                    end
                elseif need_vdiag
                    set_velocity_diag!(el, vel, xk, el_velocity_mode)
                end
            end
        end
    end # ik
end

# Device: one batched eigensolve on the backend replaces the per-k solve, then a host loop copies the
# results into the states. The energy window is applied by set_window! after the full-band
# eigenpair is copied in (e_full/u_full); windowed quantities (rbar/velocity) then read the
# in-window block of the full-band device result.
# TODO: this solves all kpts.n k-points in one batch — the device H(k)/U stacks (nw²·nk) are not
# bounded, so a very large k-grid can OOM. Chunk over k like `_filter_kpoints` (which caps the
# per-chunk stack) if that becomes a problem.
# NOTE: the batched solver does NOT apply the EPW degeneracy gauge-fixing of the per-k
# get_el_eigen!, so for degenerate bands the eigenvectors (and per-band-pair e-ph matrix elements /
# g2) can differ from the CPU path by a unitary rotation within the degenerate subspace. Gauge-
# independent quantities (eigenvalues, ωq, BZ-summed observables) are unaffected.
function _compute_electron_states_device!(states, model::Model{FT}, kpts, quantities, window,
                                          backend, eigenpairs) where FT
    (; nw, el_velocity_mode) = model
    (; need_vfull, need_vdiag, need_position) = _electron_state_needs(model, quantities)

    # Supplied eigenpairs replace the batched eigensolve: the cache is already on this backend
    # (`_check_eigenpairs`), so its columns for this k list are gathered in the list's order and the
    # rbar/velocity interpolations below consume them exactly as they consume a solved `U_dev`. So
    # `itp_elham` is not needed at all.
    itp_elham = if eigenpairs === nothing
        get_interpolator(to_device(backend, model.el_ham); fourier_mode="batched", batch_size=kpts.n)
    end

    # The cache's columns for this k list, in the list's order. Resolved here, on the host: a
    # `nothing` carried into the gather below would be consumed inside the indexing kernel and
    # surface as a bare `KernelException` naming only the device.
    iks = eigenpairs === nothing ? nothing :
        map(xk -> _eigenpairs_ik(eigenpairs, xk), kpts.vectors)

    if quantities == ["eigenvalue"]
        E = if eigenpairs === nothing
            Array(get_el_eigen_valueonly_batched(itp_elham, kpts.vectors))
        else
            Array(eigenpairs.e_full[:, iks])
        end
        return _scatter_electron_states!(states, kpts.vectors, window, E, nothing, nothing, nothing,
                                        need_vfull, need_vdiag)
    end

    E_dev, U_dev = if eigenpairs === nothing
        get_el_eigen_batched(itp_elham, kpts.vectors)
    else
        (eigenpairs.e_full[:, iks], eigenpairs.u_full[:, :, iks])
    end
    rbar_dev = if need_position
        itp_pos = get_interpolator(to_device(backend, model.el_pos); fourier_mode="batched", batch_size=kpts.n)
        get_el_velocity_direct_batched(itp_pos, kpts.vectors, U_dev)
    else
        nothing
    end
    vel_dev = if need_vfull || need_vdiag
        Mop = if el_velocity_mode === :Direct
            model.el_vel
        elseif el_velocity_mode === :BerryConnection
            model.el_ham_R
        else
            throw(ArgumentError("unknown el_velocity_mode $el_velocity_mode"))
        end
        itp_vel = get_interpolator(to_device(backend, Mop); fourier_mode="batched", batch_size=kpts.n)
        v_dev = get_el_velocity_direct_batched(itp_vel, kpts.vectors, U_dev)
        if el_velocity_mode === :BerryConnection && need_vfull
            nk = kpts.n
            v_dev .+= im .* (reshape(E_dev, nw, 1, 1, nk) .- reshape(E_dev, 1, nw, 1, nk)) .* rbar_dev
        end
        v_dev
    else
        nothing
    end

    _scatter_electron_states!(states, kpts.vectors, window, Array(E_dev), Array(U_dev),
                             rbar_dev === nothing ? nothing : Array(rbar_dev),
                             vel_dev === nothing ? nothing : Array(vel_dev),
                             need_vfull, need_vdiag)
end

# Function barrier: `U`/`rbar`/`vel` are `nothing` or an `Array` depending on `quantities`, so they
# must arrive as typed arguments for the reads below to be static.
function _scatter_electron_states!(states::Vector{ElectronState{FT}}, xks, window, E, U, rbar, vel,
                                   need_vfull, need_vdiag) where FT
    @threads for iks in chunks(xks; n=2nthreads())
        for ik in iks
            el = states[ik]
            el.xk = xks[ik]
            @views el.e_full .= E[:, ik]
            U === nothing || (@views el.u_full .= U[:, :, ik])
            el.nband = 0; el.rng = 1:0
            set_window!(el, _window_for(window, ik))
            U === nothing && continue
            r = el.rng
            if rbar !== nothing
                rbar_w = reshape(reinterpret(Complex{FT}, no_offset_view(el.rbar)), 3, el.nband, el.nband)
                @views for idir in 1:3
                    rbar_w[idir, :, :] .= rbar[r, r, idir, ik]
                end
            end
            if need_vfull
                v_w = reshape(reinterpret(Complex{FT}, no_offset_view(el.v)), 3, el.nband, el.nband)
                @views for idir in 1:3
                    v_w[idir, :, :] .= vel[r, r, idir, ik]
                end
                for i in el.rng
                    el.vdiag[i] = real.(el.v[i, i])
                end
            elseif need_vdiag
                for i in el.rng
                    el.vdiag[i] = real.(Vec3(vel[i, i, 1, ik], vel[i, i, 2, ik], vel[i, i, 3, ik]))
                end
            end
        end  # ik
    end  # iks
end

"""
    compute_phonon_states(model, kpts, quantities; fourier_mode="normal", eigenpairs=nothing)
Compute the quantities listed in `quantities` and return a vector of PhononState.
`quantities` can containing the following: "eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"
`eigenpairs`: an [`Eigenpairs`](@ref) with `nbasis = nmodes` covering every q point of `kpts`. Holds
ω and the mass-scaled eigenmodes in `e_full`/`u_full`, so the diagonalization is skipped. Used to
fix the gauge of the phonon eigenvectors.
TODO: Implement quantities "velocity"
"""
function compute_phonon_states(model::Model{FT}, kpts, quantities; fourier_mode="normal",
        eph_phonon_basis::Symbol = :eigenmode, backend=CPUBackend(),
        eigenpairs::Union{Nothing, Eigenpairs}=nothing) where FT
    # TODO: MPI, threading
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end

    (; nmodes) = model
    _check_eigenpairs(eigenpairs, nmodes, backend)

    states = [PhononState(nmodes, FT) for ik=1:kpts.n]
    if quantities == []
        return states
    end

    if backend isa CPUBackend
        _compute_phonon_states_cpu!(states, model, kpts, quantities, eph_phonon_basis, eigenpairs;
                                    fourier_mode)
    else
        _compute_phonon_states_device!(states, model, kpts, quantities, eph_phonon_basis, backend,
                                       eigenpairs)
    end
    states
end

function _compute_phonon_states_cpu!(states, model::Model{FT}, kpts, quantities,
                                     eph_phonon_basis, eigenpairs; fourier_mode) where FT
    (; mass) = model
    need_velocity = "velocity_diagonal" ∈ quantities
    polar = model.polar_phonon
    @threads for iks in chunks(kpts.vectors; n = nthreads())
        # Setup thread-local WannierInterpolators. With supplied eigenpairs there is no dynamical
        # matrix to interpolate, and nothing else uses `dyn`.
        dyn = eigenpairs === nothing ? get_interpolator(model.ph_dyn; fourier_mode) : nothing
        if need_velocity
            dyn_R = get_interpolator(model.ph_dyn_R; fourier_mode)
        end

        for ik in iks
            xk = kpts.vectors[ik]
            ph = states[ik]

            if quantities == ["eigenvalue"]
                _set_eigen_valueonly_from!(ph, eigenpairs, dyn, mass, polar, xk)
            else
                _set_eigen_from!(ph, eigenpairs, dyn, mass, polar, xk)
                if "velocity" ∈ quantities
                    # not implemented
                    error("full velocity for phonons not implemented")
                elseif "velocity_diagonal" ∈ quantities
                    set_velocity_diag!(ph, xk, dyn_R)
                end
                if "eph_dipole_coeff" ∈ quantities
                    # Use ph.u for eigenmode basis, nothing for Cartesian basis
                    u_ph_for_dipole = (eph_phonon_basis == :eigenmode) ? ph.u : nothing
                    get_eph_dipole_coeffs!(ph.eph_dipole_coeff, ph.eph_r_coeff, xk, polar, u_ph_for_dipole)
                end
            end
        end  # ik
    end  # iks
end

# Device: batch the phonon eigensolve on the backend (same idea as _compute_electron_states_device!),
# then a host loop copies the results into the states. velocity_diagonal is rotated on the device
# too. polar is unsupported (the batched e-ph loop asserts no polar). Same degeneracy-gauge caveat as
# the electrons — small g2 differences for degenerate modes, most visible on COARSE q grids.
# Supplied eigenpairs replace the eigensolve outright, which also removes that caveat: the run
# inherits whichever basis built the cache.
function _compute_phonon_states_device!(states, model::Model{FT}, kpts, quantities,
                                        eph_phonon_basis, backend, eigenpairs) where FT
    (; nmodes, mass) = model
    need_velocity = "velocity_diagonal" ∈ quantities
    polar = model.polar_phonon
    polar.use && error("compute_phonon_states on a non-CPU backend does not support polar phonons")
    "velocity" ∈ quantities && error("full velocity for phonons not implemented")

    # ω and the mass-scaled eigenmodes, either solved for here or taken from the cache, which
    # already holds both in that form (it is built from an earlier run's `PhononState`s).
    E_dev, U_dev = if eigenpairs === nothing
        itp_dyn = get_interpolator(to_device(backend, model.ph_dyn);
                                   fourier_mode="batched", batch_size=kpts.n)
        D = _fourier_hk_batched(itp_dyn, kpts.vectors)  # (nmodes,nmodes,nq)
        msqrt_d = similar(D, FT, nmodes); copyto!(msqrt_d, sqrt.(mass))
        D ./= reshape(msqrt_d, nmodes, 1, 1)         # dynq[i,j] /= sqrt(mass[i] mass[j])
        D ./= reshape(msqrt_d, 1, nmodes, 1)
        Esq_dev, U_solved = eigen_batched(D)         # ω² (nmodes,nq), U (nmodes,nmodes,nq)
        U_solved ./= reshape(msqrt_d, nmodes, 1, 1)  # mass factor: u[i,:] /= sqrt(mass[i])
        (sign.(Esq_dev) .* sqrt.(abs.(Esq_dev)), U_solved)  # ω = sign(ω²)·√|ω²|
    else
        # The lookup runs on the host: a miss inside the view would surface as a bare
        # `KernelException` naming only the device.
        iqs = map(xq -> _eigenpairs_ik(eigenpairs, xq), kpts.vectors)
        # Views, not copies. Both consumers read `U_dev` and nothing writes it (`Array` copies it
        # out; `get_el_velocity_direct_batched` only broadcasts it into its own `urep`), so
        # aliasing the cache is safe, and a materialized gather would duplicate `u_full`'s
        # `16*nmodes^2*nq` device bytes. Measured at nmodes = 6, nq = 5e5: the gather costs 8.2 ms
        # and 288 MB against 1.0 ms and 4 MB for the view, while consuming the view instead of a
        # copy costs 1.1 ms on the host download and 0.03 ms on the velocity broadcast.
        (view(eigenpairs.e_full, :, iqs), view(eigenpairs.u_full, :, :, iqs))
    end
    E = Array(E_dev)
    U = quantities == ["eigenvalue"] ? nothing : Array(U_dev)
    vel = if need_velocity
        itp_phvel = get_interpolator(to_device(backend, model.ph_dyn_R); fourier_mode="batched", batch_size=kpts.n)
        Array(get_el_velocity_direct_batched(itp_phvel, kpts.vectors, U_dev))
    else
        nothing
    end

    _scatter_phonon_states!(states, kpts.vectors, E, U, vel,
                            "eph_dipole_coeff" ∈ quantities, eph_phonon_basis, polar)
end

# Function barrier: `U`/`vel` are `nothing` or an `Array` depending on `quantities`, so they must
# arrive as typed arguments for the reads below to be static.
#
# TODO: replace the `Vector{PhononState}` with a struct-of-arrays `BatchedPhononState{T, AT}` holding
# the whole q-grid in dense stacks — `e` (nmodes, nq), `u` (nmodes, nmodes, nq), `vdiag`,
# `eph_dipole_coeff` — with `AT` selecting host or device storage, so the same type serves both
# backends and the GPU path never leaves the device. A `Vector` of per-q mutable structs costs ~13
# allocations and ~1.5 kB per q point, which at the q-grids the outer-k driver builds (nq = 6.9 M for
# Cu at nk = 200) is ~90 M allocations and ~10 GiB of churn, most of the setup's GC time.
# `_loop_eph_over_k_and_kq_batched` shows how little of it is wanted: it reads only `.u` and `.e`, and
# gathers them straight back into dense stacks to re-upload — data `E_dev`/`U_dev` above already hold
# on the device — while `velocity_diagonal` and `eph_dipole_coeff`, which that driver requests, are
# never read. Measured on Cu at nq = 436 k: 1.62 s / 5.67 M allocations / 669 MiB for the per-q
# states, versus 0.60 s / ~3 k allocations for the device stacks alone.
# Blast radius: `PhononState` is also consumed per-q by the CPU drivers (`epstate.ph = ph_save[iq]`),
# `run_eph_over_q_and_k`, `wfpt.jl`, `run_coherence.jl` and `gamma_adaptive.jl`, so a per-q view into
# the batch has to keep the `set_*!`/`copyto!` interface those rely on.
function _scatter_phonon_states!(states, xqs, E, U, vel, need_dipole, eph_phonon_basis, polar)
    @threads for iqs in chunks(xqs; n = nthreads())
        for iq in iqs
            ph = states[iq]
            ph.xq = xqs[iq]
            @views ph.e .= E[:, iq]
            U === nothing && continue
            @views ph.u .= U[:, :, iq]
            if vel !== nothing
                for i in 1:ph.nmodes
                    ph.vdiag[i] = real.(Vec3(vel[i, i, 1, iq], vel[i, i, 2, iq], vel[i, i, 3, iq])) ./ (2 * ph.e[i])
                end
            end
            if need_dipole
                # Use ph.u for eigenmode basis, nothing for Cartesian basis
                u_ph_for_dipole = (eph_phonon_basis == :eigenmode) ? ph.u : nothing
                get_eph_dipole_coeffs!(ph.eph_dipole_coeff, ph.eph_r_coeff, ph.xq, polar, u_ph_for_dipole)
            end
        end  # iq
    end  # iqs
end
