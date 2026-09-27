using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads

export compute_electron_states
export compute_phonon_states
export compute_phonon_states_batched

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
        get_interpolator(to_device(backend, model.el_ham); fourier_mode="batched", backend, nk_hint=kpts.n)
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
        itp_pos = get_interpolator(to_device(backend, model.el_pos); fourier_mode="batched", backend, nk_hint=kpts.n)
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
        itp_vel = get_interpolator(to_device(backend, Mop); fourier_mode="batched", backend, nk_hint=kpts.n)
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
`backend` must be `CPUBackend()`: one mutable `PhononState` per q point is what the per-point CPU
drivers consume, and costs ~13 allocations and ~1.5 kB per q point. For dense stacks, on the host or
on a device, use [`compute_phonon_states_batched`](@ref) instead.
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

    backend isa CPUBackend || throw(ArgumentError(
        "compute_phonon_states runs on CPUBackend only. For the phonons on a device use " *
        "compute_phonon_states_batched(model, qpts, quantities; backend), the dense stacks the " *
        "batched e-ph loop reads."))
    (; nmodes) = model
    _check_eigenpairs(eigenpairs, nmodes, backend)

    states = [PhononState(nmodes, FT) for ik=1:kpts.n]
    if quantities == []
        return states
    end

    (; mass) = model
    (; valueonly, need_vdiag, need_dipole) = _phonon_state_needs(quantities)
    polar = model.polar_phonon
    @threads for iks in chunks(kpts.vectors; n = nthreads())
        # Setup thread-local WannierInterpolators. With supplied eigenpairs there is no dynamical
        # matrix to interpolate, and nothing else uses `dyn`.
        dyn = if eigenpairs === nothing
            itp_dyn = get_interpolator(model.ph_dyn; fourier_mode)
            register_kpoints!(itp_dyn, view(kpts.vectors, iks))
            itp_dyn
        end
        if need_vdiag
            dyn_R = get_interpolator(model.ph_dyn_R; fourier_mode)
            register_kpoints!(dyn_R, view(kpts.vectors, iks))
        end

        for ik in iks
            xk = kpts.vectors[ik]
            ph = states[ik]

            if valueonly
                _set_eigen_valueonly_from!(ph, eigenpairs, dyn, mass, polar, xk)
            else
                _set_eigen_from!(ph, eigenpairs, dyn, mass, polar, xk)
                if need_vdiag
                    set_velocity_diag!(ph, dyn_R, xk)
                end
                if need_dipole
                    # Use ph.u for eigenmode basis, nothing for Cartesian basis
                    u_ph_for_dipole = (eph_phonon_basis == :eigenmode) ? ph.u : nothing
                    get_eph_dipole_coeffs!(ph.eph_dipole_coeff, ph.eph_r_coeff, xk, polar, u_ph_for_dipole)
                end
            end
        end  # ik
    end  # iks
    states
end

# Which quantities a phonon builder computes, derived from `quantities` (kept in one place so both
# builders branch on the same flags). `valueonly` runs the value-only eigensolve, and leaves `u` zero.
function _phonon_state_needs(quantities)
    valueonly = quantities == ["eigenvalue"]
    need_vdiag = "velocity_diagonal" ∈ quantities
    need_dipole = "eph_dipole_coeff" ∈ quantities
    (; valueonly, need_vdiag, need_dipole)
end

# Why this is not `compute_phonon_states` stacked afterwards: that function returns one mutable
# `PhononState` per q point, ~13 heap objects per q, which is what the per-point loops need in their
# `EPState` and what the batched loop over millions of q points must not pay; and it is host-only. The
# per-q physics (eigensolve, cache copy, velocity, dipole) is shared: both builders call the same
# kernels on the same arrays, so the host stacks here are `compute_phonon_states` bit for bit.
"""
    compute_phonon_states_batched(model, qpts, quantities; fourier_mode = "gridopt",
        eph_phonon_basis = :eigenmode, backend = CPUBackend(), eigenpairs = nothing)
        -> BatchedPhononState

The phonons of `qpts` as a [`BatchedPhononState`](@ref) on `backend`: what
[`compute_phonon_states`](@ref) computes for the same arguments, stored as dense stacks instead of
one `PhononState` per q point. `quantities` and `eph_phonon_basis` are as there; a stack whose
quantity is not requested is zero-length.

`eigenpairs` is a gauge-fixing lookup table, as in `compute_phonon_states`: ω and `u` of every q are
copied from it instead of diagonalizing, so it must cover every q point of `qpts` and be resident on
`backend`.

On a GPU backend only `["eigenvalue"]` and `["eigenvalue", "eigenvector"]` are supported, polar
phonons are not, and `fourier_mode` is unused. The q set is solved in chunks of the batched
dynamical-matrix interpolator's block width, so the device `D(q)` transient is bounded whatever
`qpts.n`. As in [`electron_eigenpairs`](@ref), the batched eigensolve picks its own basis inside a
degenerate mode multiplet, so device and host `u` differ there.
"""
function compute_phonon_states_batched(model::Model{FT}, qpts, quantities; fourier_mode = "gridopt",
        eph_phonon_basis::Symbol = :eigenmode, backend = CPUBackend(),
        eigenpairs::Union{Nothing, Eigenpairs} = nothing) where FT
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end
    (; nmodes, mass) = model
    _check_eigenpairs(eigenpairs, nmodes, backend)
    (; valueonly, need_vdiag, need_dipole) = _phonon_state_needs(quantities)
    nq = qpts.n
    backend isa CPUBackend || valueonly || issetequal(quantities, ["eigenvalue", "eigenvector"]) ||
        throw(ArgumentError("compute_phonon_states_batched on $(nameof(typeof(backend))) " *
            "computes only [\"eigenvalue\"] or [\"eigenvalue\", \"eigenvector\"], got $quantities"))

    # Zero-filled as a `PhononState` is: `u` stays zero on the value-only path, and a 3D dipole
    # leaves `eph_r_coeff` unwritten.
    vdiag = alloc_zeros(backend, FT, 3, nmodes, need_vdiag ? nq : 0)
    eph_dipole_coeff = alloc_zeros(backend, Complex{FT}, nmodes, need_dipole ? nq : 0)
    eph_r_coeff = alloc_zeros(backend, Complex{FT}, nmodes, 3, need_dipole ? nq : 0)
    if backend isa CPUBackend
        e = alloc_zeros(backend, FT, nmodes, nq)
        u = alloc_zeros(backend, Complex{FT}, nmodes, nmodes, nq)
        isempty(quantities) ||
            _compute_phonon_states_batched_cpu!(e, u, vdiag, eph_dipole_coeff, eph_r_coeff, model,
                qpts, eigenpairs, valueonly, need_vdiag, need_dipole, eph_phonon_basis; fourier_mode)
    elseif eigenpairs === nothing
        model.polar_phonon.use && throw(ArgumentError(
            "compute_phonon_states_batched on a non-CPU backend does not support polar phonons"))
        itp_dyn = get_interpolator(to_device(backend, model.ph_dyn); fourier_mode = "batched",
                                   backend, nk_hint = nq)
        msqrt_d = alloc(backend, FT, nmodes); copyto!(msqrt_d, sqrt.(mass))
        e = alloc(backend, FT, nmodes, nq)
        u = valueonly ? alloc_zeros(backend, Complex{FT}, nmodes, nmodes, nq) :
                        alloc(backend, Complex{FT}, nmodes, nmodes, nq)
        # One chunk per Fourier block of `itp_dyn`, so the Fourier partition is that of one call
        # over the whole set; the eigensolve is per matrix, so the chunking does not change ω or u.
        # (`nk_hint = 0` gives a zero block width.)
        for c in Iterators.partition(1:nq, max(itp_dyn.batch_size, 1))
            D = _fourier_hk_batched(itp_dyn, view(qpts.vectors, c))  # (nmodes, nmodes, length(c))
            D ./= reshape(msqrt_d, nmodes, 1, 1)         # dynq[i,j] /= sqrt(mass[i] mass[j])
            D ./= reshape(msqrt_d, 1, nmodes, 1)
            Esq = if valueonly
                eigvals_batched(D)
            else
                Esq_c, U = eigen_batched(D)
                U ./= reshape(msqrt_d, nmodes, 1, 1)     # mass factor: u[i,:] /= sqrt(mass[i])
                u[:, :, c] .= U
                Esq_c
            end
            e[:, c] .= sign.(Esq) .* sqrt.(abs.(Esq))  # ω = sign(ω²)·√|ω²|
        end
    else
        # The cache's columns for this q list, resolved on the host: a miss inside the device
        # gather would surface as a bare `KernelException` naming only the device.
        iqs = Vector{Int}(undef, nq)
        @threads for iqs_chunk in chunks(1:nq; n = nthreads())
            for iq in iqs_chunk
                iqs[iq] = _eigenpairs_ik(eigenpairs, qpts.vectors[iq])
            end
        end
        e = eigenpairs.e_full[:, iqs]
        u = valueonly ? alloc_zeros(backend, Complex{FT}, nmodes, nmodes, nq) :
                        eigenpairs.u_full[:, :, iqs]
    end
    BatchedPhononState(nmodes, qpts, e, u, vdiag, eph_dipole_coeff, eph_r_coeff)
end

# The host fill: `compute_phonon_states`' loop, same chunks and same per-q kernels, on q slices of
# the stacks. A function barrier, so the `@threads` closure captures typed arguments.
function _compute_phonon_states_batched_cpu!(e, u, vdiag, eph_dipole_coeff, eph_r_coeff,
        model, qpts, eigenpairs, valueonly, need_vdiag, need_dipole, eph_phonon_basis;
        fourier_mode)
    (; mass) = model
    polar = model.polar_phonon
    @threads for iqs in chunks(qpts.vectors; n = nthreads())
        # Thread-local interpolators; with supplied eigenpairs there is no D(q) to interpolate.
        dyn = if eigenpairs === nothing
            itp_dyn = get_interpolator(model.ph_dyn; fourier_mode)
            register_kpoints!(itp_dyn, view(qpts.vectors, iqs))
            itp_dyn
        end
        if need_vdiag
            dyn_R = get_interpolator(model.ph_dyn_R; fourier_mode)
            register_kpoints!(dyn_R, view(qpts.vectors, iqs))
        end

        @views for iq in iqs
            xq = qpts.vectors[iq]
            if valueonly
                _set_eigen_valueonly_from!(e[:, iq], eigenpairs, dyn, mass, polar, xq)
            else
                _set_eigen_from!(e[:, iq], u[:, :, iq], eigenpairs, dyn, mass, polar, xq)
                if need_vdiag
                    _ph_velocity_diag!(vdiag[:, :, iq], dyn_R, xq, u[:, :, iq], e[:, iq])
                end
                if need_dipole
                    # u for the eigenmode basis, nothing for the Cartesian basis
                    u_ph = eph_phonon_basis == :eigenmode ? u[:, :, iq] : nothing
                    get_eph_dipole_coeffs!(eph_dipole_coeff[:, iq], eph_r_coeff[:, :, iq], xq,
                                           polar, u_ph)
                end
            end
        end
    end
    nothing
end
