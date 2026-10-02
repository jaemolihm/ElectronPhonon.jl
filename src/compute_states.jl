using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads

export compute_electron_states
export compute_electron_states_batched
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
        "compute_phonon_states_batched(model, qpts, quantities; backend)."))
    (; nmodes) = model
    _check_eigenpairs(eigenpairs, nmodes, backend)

    states = [PhononState(nmodes, FT) for ik=1:kpts.n]
    if quantities == []
        return states
    end

    (; mass) = model
    valueonly = quantities == ["eigenvalue"]
    need_vdiag = "velocity_diagonal" ∈ quantities
    need_dipole = "eph_dipole_coeff" ∈ quantities
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

# Why this is not `compute_phonon_states` stacked afterwards: that function returns one mutable
# `PhononState` per q point, ~13 heap objects per q, which is what the per-point loops need in their
# `EPState` and what the batched loop over millions of q points must not pay; and it is host-only.
# The host fill below runs the same per-q kernels on slices of the stacks, so they are
# `compute_phonon_states` bit for bit.
"""
    compute_phonon_states_batched(model, qpts, quantities; fourier_mode = "gridopt",
        eph_phonon_basis = :eigenmode, backend = CPUBackend(), eigenpairs = nothing)
        -> BatchedPhononState

The phonons of `qpts` as a [`BatchedPhononState`](@ref) on `backend`: what
[`compute_phonon_states`](@ref) computes for the same arguments, stored as dense stacks instead of
one `PhononState` per q point. The `fourier_mode` default is `"gridopt"` here and `"normal"` there.
`quantities` lists the fields to fill (`:e`, `:u`, `:vdiag`, `:eph_dipole_coeff`, `:eph_r_coeff`).
The eigenvalue-only solve runs when none of them needs the eigenmodes; `eph_phonon_basis` is as in
`compute_phonon_states`.

`eigenpairs` is a gauge-fixing lookup table, as in `compute_phonon_states`: ω and `u` of every q are
copied from it instead of diagonalizing, so it must cover every q point of `qpts` and be resident on
`backend`.

On a GPU backend only `:e` and `:u` are supported, polar phonons are refused and `fourier_mode` is
unused. The q set is solved in chunks of the batched dynamical-matrix interpolator's block width,
so the device `D(q)` transient is bounded whatever `qpts.n`. As in [`electron_eigenpairs`](@ref),
the batched eigensolve picks its own basis inside a degenerate mode multiplet, so device and host
`u` differ there. A GPU e-ph run therefore takes its phonon gauge from the device solve when it needs
`e` and `u` only, and from host LAPACK (built here, then copied over) for a polar model or other
quantities.
"""
function compute_phonon_states_batched(model::Model{FT}, qpts, quantities; fourier_mode = "gridopt",
        eph_phonon_basis::Symbol = :eigenmode, backend = CPUBackend(),
        eigenpairs::Union{Nothing, Eigenpairs} = nothing) where FT
    (; nmodes, mass) = model
    nq = qpts.n
    phs = BatchedPhononState(backend, nmodes, nq, quantities; qpts, FT)
    _check_eigenpairs(eigenpairs, nmodes, backend)
    need_dipole = :eph_dipole_coeff ∈ quantities || :eph_r_coeff ∈ quantities
    valueonly = !(:u ∈ quantities || :vdiag ∈ quantities || need_dipole)
    if !(backend isa CPUBackend)
        unsupported = setdiff(quantities, (:e, :u))
        isempty(unsupported) || throw(ArgumentError("quantities $unsupported are not supported " *
            "by compute_phonon_states_batched on $(nameof(typeof(backend)))"))
        model.polar_phonon.use && throw(ArgumentError(
            "compute_phonon_states_batched on a non-CPU backend does not support polar phonons"))
    end
    (nq == 0 || isempty(quantities)) && return phs

    if backend isa CPUBackend
        _compute_phonon_states_batched_cpu!(phs, model, eigenpairs, valueonly, need_dipole,
                                            eph_phonon_basis; fourier_mode)
        return phs
    end
    # Device: ω into `e` (or a temporary), eigenmodes into `u` when requested.
    e = phs.e === nothing ? alloc(backend, FT, nmodes, nq) : phs.e
    if eigenpairs === nothing
        itp_dyn = get_interpolator(to_device(backend, model.ph_dyn); fourier_mode = "batched", backend, nk_hint = nq)
        msqrt_d = alloc(backend, FT, nmodes); copyto!(msqrt_d, sqrt.(mass))
        # One chunk per Fourier block of `itp_dyn`, so the Fourier partition is that of one call
        # over the whole set; the eigensolve is per matrix, so the chunking does not change ω or u.
        for iq_chunk in Iterators.partition(1:nq, itp_dyn.batch_size)
            D = _fourier_hk_batched(itp_dyn, view(qpts.vectors, iq_chunk))  # (nmodes, nmodes, length(iq_chunk))
            D ./= reshape(msqrt_d, nmodes, 1, 1)         # dynq[i,j] /= sqrt(mass[i] mass[j])
            D ./= reshape(msqrt_d, 1, nmodes, 1)
            Esq = if valueonly
                eigvals_batched(D)
            else
                Esq_c, U = eigen_batched(D)
                U ./= reshape(msqrt_d, nmodes, 1, 1)     # mass factor: u[i,:] /= sqrt(mass[i])
                phs.u[:, :, iq_chunk] .= U
                Esq_c
            end
            e[:, iq_chunk] .= sign.(Esq) .* sqrt.(abs.(Esq))  # ω = sign(ω²)·√|ω²|
        end
    else
        # Gather from the cache. Its columns for this q list are resolved on the host: a miss
        # inside the device gather would surface as a bare `KernelException` naming only the device.
        iqs = Vector{Int}(undef, nq)
        @threads for iqs_chunk in chunks(1:nq; n = nthreads())
            for iq in iqs_chunk
                iqs[iq] = _eigenpairs_ik(eigenpairs, qpts.vectors[iq])
            end
        end
        e .= eigenpairs.e_full[:, iqs]
        valueonly || (phs.u .= eigenpairs.u_full[:, :, iqs])
    end
    phs
end

# The host fill: `compute_phonon_states`' loop, same chunks and same per-q kernels, on q slices of
# the stacks (or per-chunk scratch for what was not requested). A function barrier, so the
# `@threads` closure captures typed arguments.
function _compute_phonon_states_batched_cpu!(phs, model::Model{FT}, eigenpairs, valueonly,
        need_dipole, eph_phonon_basis; fourier_mode) where FT
    (; mass, nmodes) = model
    (; qpts) = phs
    polar = model.polar_phonon
    @threads for iqs in chunks(qpts.vectors; n = nthreads())
        # Thread-local interpolators; with supplied eigenpairs there is no D(q) to interpolate.
        dyn = if eigenpairs === nothing
            itp_dyn = get_interpolator(model.ph_dyn; fourier_mode)
            register_kpoints!(itp_dyn, view(qpts.vectors, iqs))
            itp_dyn
        end
        if phs.vdiag !== nothing
            dyn_R = get_interpolator(model.ph_dyn_R; fourier_mode)
            register_kpoints!(dyn_R, view(qpts.vectors, iqs))
        end
        e_s = zeros(FT, nmodes); u_s = zeros(Complex{FT}, nmodes, nmodes)
        d_s = zeros(Complex{FT}, nmodes); r_s = zeros(Complex{FT}, nmodes, 3)
        @views for iq in iqs
            xq = qpts.vectors[iq]
            e = phs.e === nothing ? e_s : phs.e[:, iq]
            u = phs.u === nothing ? u_s : phs.u[:, :, iq]
            if eigenpairs !== nothing
                jq = _eigenpairs_ik(eigenpairs, xq)
                e .= eigenpairs.e_full[:, jq]
                valueonly || (u .= eigenpairs.u_full[:, :, jq])
            elseif valueonly
                get_ph_eigen_valueonly!(e, dyn, mass, polar, xq)
            else
                get_ph_eigen!(e, u, dyn, mass, polar, xq)
            end
            valueonly && continue
            if phs.vdiag !== nothing
                # dω/dk = (dω²/dk) / (2ω), as `set_velocity_diag!(::PhononState, ...)`
                get_ph_velocity_diag!(phs.vdiag[:, :, iq], dyn_R, xq, u)
                for imode in 1:nmodes
                    phs.vdiag[:, imode, iq] ./= 2 .* e[imode]
                end
            end
            if need_dipole
                # u for the eigenmode basis, nothing for the Cartesian basis
                get_eph_dipole_coeffs!(
                    phs.eph_dipole_coeff === nothing ? d_s : phs.eph_dipole_coeff[:, iq],
                    phs.eph_r_coeff === nothing ? r_s : phs.eph_r_coeff[:, :, iq], xq, polar,
                    eph_phonon_basis == :eigenmode ? u : nothing)
            end
        end
    end
    nothing
end


# ---- Batched electron states --------------------------------------------------------------------

# Why this is not `compute_electron_states` stacked afterwards: that function returns one mutable
# `ElectronState` per k point, and its device arm copies every result back to the host to scatter it
# into them. The host fill below calls the same kernels on the same windowed eigenvectors, and the
# device fill runs the same batched solve and rotations without the copy back.
"""
    compute_electron_states_batched(model, sel::FilteredBandStates, quantities; kwargs...)
    compute_electron_states_batched(model, kpts, quantities, window = (-Inf, Inf); kwargs...)
        -> BatchedElectronState

The electron states of `sel.kpts` (each k restricted to `sel.band_extent[ik]`) or of `kpts`
(restricted to the energy `window`) as a [`BatchedElectronState`](@ref) in box storage on
`backend`: what [`compute_electron_states`](@ref) computes for the same arguments, as dense stacks.
`quantities` lists the fields to fill (`:e`, `:u`, `:vdiag`, `:v`, `:rbar`). The eigenvalue-only
solve runs when none of them needs the eigenvectors.

Keywords as in `compute_electron_states`: `fourier_mode = "normal"`, `backend = CPUBackend()`,
`eigenpairs = nothing` (a gauge-fixing cache covering every k, resident on `backend`).

On a GPU backend only `:e`, `:u` and `:vdiag` are supported and `fourier_mode` is unused. The
batched eigensolve picks its own basis inside a degenerate multiplet, as in `compute_electron_states`.
"""
function compute_electron_states_batched(model::Model, sel::FilteredBandStates, quantities;
        fourier_mode = "normal", backend = CPUBackend(), eigenpairs::Union{Nothing, Eigenpairs} = nothing)
    _compute_electron_states_batched(model, sel.kpts, quantities, sel.band_extent; fourier_mode, backend,
                                     eigenpairs)
end

function compute_electron_states_batched(model::Model, kpts::AbstractKpoints, quantities,
        window::Tuple = (-Inf, Inf); fourier_mode = "normal", backend = CPUBackend(),
        eigenpairs::Union{Nothing, Eigenpairs} = nothing)
    _compute_electron_states_batched(model, kpts, quantities, window; fourier_mode, backend, eigenpairs)
end

function _compute_electron_states_batched(model::Model{FT}, kpts, quantities, window;
        fourier_mode, backend, eigenpairs) where FT
    (; nw) = model
    nk = kpts.n
    backend isa CPUBackend || isempty(setdiff(quantities, (:e, :u, :vdiag))) || throw(ArgumentError(
        "quantities $(setdiff(quantities, (:e, :u, :vdiag))) are not supported by the batched " *
        "electron builder on $(nameof(typeof(backend)))"))
    unknown = setdiff(quantities, (:e, :u, :vdiag, :v, :rbar))
    isempty(unknown) || throw(ArgumentError("unknown electron quantities $unknown"))
    _check_eigenpairs(eigenpairs, nw, backend)
    need_u = any(∈(quantities), (:u, :vdiag, :v, :rbar))

    # Full-band eigenpairs first: the box width is the largest window, known only once every k is
    # solved.
    E, U = if backend isa CPUBackend
        _electron_eigenpairs_cpu(model, kpts, eigenpairs, need_u; fourier_mode)
    elseif eigenpairs === nothing
        itp = get_interpolator(to_device(backend, model.el_ham); fourier_mode = "batched", backend, nk_hint = nk)
        need_u ? get_el_eigen_batched(itp, kpts.vectors) : (get_el_eigen_valueonly_batched(itp, kpts.vectors), nothing)
    else
        # Resolved on the host: a miss inside the device gather would surface as a bare
        # `KernelException` naming only the device.
        iks = map(xk -> _eigenpairs_ik(eigenpairs, xk), kpts.vectors)
        (eigenpairs.e_full[:, iks], need_u ? eigenpairs.u_full[:, :, iks] : nothing)
    end
    E_host = Array(E)
    # Each k's window, as `set_window!` takes it: an energy window or an explicit band range.
    rngs = map(1:nk) do ik
        window_ik = _window_for(window, ik)
        rng = window_ik isa Tuple ? inside_window(view(E_host, :, ik), window_ik...) : intersect(window_ik, 1:nw)
        isempty(rng) ? (1:0) : rng
    end
    nband_h = length.(rngs)
    offset_h = [isempty(rng) ? 0 : first(rng) - 1 for rng in rngs]
    els = BatchedElectronState(backend, nw, maximum(nband_h; init = 0), nk, quantities; kpts, FT)
    copyto!(els.iband_offset, offset_h); copyto!(els.nband, nband_h)
    nk == 0 && return els
    if backend isa CPUBackend
        _fill_electron_states_batched_cpu!(els, model, E, U, rngs; fourier_mode)
    else
        _fill_electron_states_batched_device!(els, model, E, U, offset_h, backend)
    end
    els
end

# Full-band `E` (nw, nk) and, when `need_u`, `U` (nw, nw, nk) on the host, with
# `compute_electron_states`' chunks and per-k solve.
function _electron_eigenpairs_cpu(model::Model{FT}, kpts, eigenpairs, need_u; fourier_mode) where FT
    (; nw) = model
    E = zeros(FT, nw, kpts.n)
    U = need_u ? zeros(Complex{FT}, nw, nw, kpts.n) : nothing
    @threads for iks in chunks(kpts.vectors; n = 2nthreads())
        ham = if eigenpairs === nothing
            itp_ham = get_interpolator(model.el_ham; fourier_mode)
            register_kpoints!(itp_ham, view(kpts.vectors, iks))
            itp_ham
        end
        @views for ik in iks
            xk = kpts.vectors[ik]
            if eigenpairs !== nothing
                jk = _eigenpairs_ik(eigenpairs, xk)
                E[:, ik] .= eigenpairs.e_full[:, jk]
                U === nothing || (U[:, :, ik] .= eigenpairs.u_full[:, :, jk])
            elseif U === nothing
                get_el_eigen_valueonly!(E[:, ik], nw, ham, xk)
            else
                get_el_eigen!(E[:, ik], U[:, :, ik], nw, ham, xk)
            end
        end
    end
    E, U
end

# The host fill: the in-window block of each k into the box, then the windowed quantities with the
# per-point kernels on the in-window eigenvectors (`compute_electron_states`' chunks), through
# contiguous per-chunk scratch so the kernels see the arrays they see on the per-point path.
function _fill_electron_states_batched_cpu!(els, model::Model{FT}, E, U, rngs; fourier_mode) where FT
    (; nw, el_velocity_mode) = model
    (; kpts) = els
    need_position = els.rbar !== nothing || (els.v !== nothing && el_velocity_mode === :BerryConnection)
    need_velocity = els.vdiag !== nothing || els.v !== nothing
    @threads for iks in chunks(kpts.vectors; n = 2nthreads())
        if need_velocity
            vel = get_interpolator(el_velocity_mode === :Direct ? model.el_vel : model.el_ham_R; fourier_mode)
            register_kpoints!(vel, view(kpts.vectors, iks))
        end
        if need_position
            pos = get_interpolator(model.el_pos; fourier_mode)
            register_kpoints!(pos, view(kpts.vectors, iks))
        end
        v_s = zeros(Complex{FT}, 3 * nw * nw); r_s = zeros(Complex{FT}, 3 * nw * nw)
        @views for ik in iks
            rng = rngs[ik]
            nb = length(rng)
            xk = kpts.vectors[ik]
            els.e === nothing || (els.e[1:nb, ik] .= E[rng, ik])
            U === nothing && continue
            els.u === nothing || (els.u[:, 1:nb, ik] .= U[:, rng, ik])
            u_w = U[:, rng, ik]
            rbar_w = reshape(r_s[1:3nb*nb], 3, nb, nb)
            if need_position
                get_el_velocity_direct!(rbar_w, nw, pos, xk, u_w)
                els.rbar === nothing || (els.rbar[:, 1:nb, 1:nb, ik] .= rbar_w)
            end
            v_w = reshape(v_s[1:3nb*nb], 3, nb, nb)
            if els.v !== nothing
                if el_velocity_mode === :Direct
                    get_el_velocity_direct!(v_w, nw, vel, xk, u_w)
                else
                    get_el_velocity_berry_connection!(v_w, nw, vel, E[rng, ik], xk, u_w,
                        reinterpret(reshape, Vec3{Complex{FT}}, rbar_w))
                end
                els.v[:, 1:nb, 1:nb, ik] .= v_w
                if els.vdiag !== nothing
                    for i in 1:nb
                        els.vdiag[:, i, ik] .= real.(v_w[:, i, i])
                    end
                end
            elseif els.vdiag !== nothing
                # As `set_velocity_diag!(::ElectronState, ...)`: direct interpolation has no
                # diagonal-only form, and the Berry connection term is zero on the diagonal.
                if el_velocity_mode === :Direct
                    get_el_velocity_direct!(v_w, nw, vel, xk, u_w)
                    for i in 1:nb
                        els.vdiag[:, i, ik] .= real.(v_w[:, i, i])
                    end
                elseif el_velocity_mode === :BerryConnection
                    get_el_velocity_diag_berry_connection!(els.vdiag[:, 1:nb, ik], nw, vel, xk, u_w)
                else
                    throw(ArgumentError("mode must be :Direct or :BerryConnection, not $el_velocity_mode."))
                end
            end
        end
    end
    nothing
end

# The device fill: gather each k's in-window columns into the box, and the band-diagonal velocity
# from the full-band rotation, as `_compute_electron_states_device!` reads it. Box entries past
# `nband` gather a clamped in-range index, so they hold some other band's value (undefined).
function _fill_electron_states_batched_device!(els, model, E, U, offset_h, backend)
    (; nw) = model
    (; nk, nband_max, kpts) = els
    # band[n, k]: the physical band of box column n at k, clamped into 1:nw on the padding.
    band = [min(offset_h[ik] + n, nw) for n in 1:nband_max, ik in 1:nk]
    col = to_device_copy(backend, vec(band .+ nw .* (0:nk-1)'))   # column of U's (nw, nw*nk) view
    els.e === nothing || (vec(els.e) .= view(vec(E), col))
    U === nothing && return nothing
    els.u === nothing || (reshape(els.u, nw, :) .= view(reshape(U, nw, :), :, col))
    if els.vdiag !== nothing
        Mop = model.el_velocity_mode === :Direct ? model.el_vel :
              model.el_velocity_mode === :BerryConnection ? model.el_ham_R :
              throw(ArgumentError("unknown el_velocity_mode $(model.el_velocity_mode)"))
        itp_vel = get_interpolator(to_device(backend, Mop); fourier_mode = "batched", backend, nk_hint = nk)
        vel = get_el_velocity_direct_batched(itp_vel, kpts.vectors, U)   # (nw, nw, 3, nk)
        # vdiag[d, n, k] = real(vel[b, b, d, k]) with b = band[n, k]
        diag_lin = to_device_copy(backend, [band[n, ik] + nw * (band[n, ik] - 1) +
            nw^2 * (d - 1) + 3nw^2 * (ik - 1) for d in 1:3, n in 1:nband_max, ik in 1:nk])
        els.vdiag .= real.(view(vec(vel), diag_lin))
    end
    nothing
end

"""
    compute_electron_states_batched!(dst, itp_ham, hk, model, xks, window)
        -> BatchedElectronState

Solve the electron states at the k points `xks` (a host vector of at most `dst.nk` points) into the
buffers of `dst`, a [`BatchedElectronState`](@ref) with `nband_max = nw` and fields `e` and/or `u`:
one batched Fourier transform with `itp_ham` (the `BatchedWannierInterpolator` of `model.el_ham` on
`dst`'s backend, block width at least `length(xks)`) into the `(nw^2, ≥ length(xks))` scratch `hk`,
one batched eigensolve, then each point's bands inside the energy `window` moved to local bands
`1:nband`. Returns the `length(xks)` points as a container whose box is the largest `nband` of
them (at least 1), stored in the leading elements of `dst`'s arrays (`dense_prefix`), so the
blocks built on it shrink with the window. The batched eigensolve applies no degeneracy gauge fix,
as in `eigen_batched`.
"""
function compute_electron_states_batched!(els::BatchedElectronState, itp_ham, hk, model::Model, xks,
        window::Tuple)
    (; nw) = model
    els.vdiag === nothing && els.v === nothing && els.rbar === nothing || throw(ArgumentError(
        "the in-tile electron builder fills e and u only"))
    els.nw == nw && els.nband_max == nw ||
        throw(ArgumentError("a buffer solved in place needs nband_max = nw = $nw, got $(els.nband_max)"))
    nx = length(xks)
    nx <= dst.nk || throw(ArgumentError("$nx points do not fit a buffer of width $(dst.nk)"))
    nx == 0 && return prefix_batched_electron_states(dst, 1, 0)
    hk_x = view(hk, :, 1:nx)
    get_fourier_batched!(hk_x, itp_ham, xks)
    H = reshape(hk_x, nw, nw, nx)
    E, U = els.u === nothing ? (eigvals_batched(H), nothing) : eigen_batched(H)
    # The eigenvalues are sorted per point, so counts give `inside_window`'s range.
    wmin, wmax = window
    off = vec(sum(E .< wmin; dims = 1))
    nb = max.(vec(sum(E .<= wmax; dims = 1)) .- off, 0)
    view(dst.iband_offset, 1:nx) .= ifelse.(nb .> 0, off, 0)
    view(dst.nband, 1:nx) .= nb
    out = prefix_batched_electron_states(dst, max(maximum(nb), 1), nx)
    # col[n, j]: the column of point j's band offset + n in the (nw, nw * nx) view, clamped into
    # 1:nw on the padding.
    col = vec(min.(reshape(off, 1, nx) .+ (1:out.nband_max), nw) .+ nw .* reshape(0:nx-1, 1, nx))
    out.e === nothing || (vec(out.e) .= view(vec(E), col))
    out.u === nothing || (reshape(out.u, nw, :) .= view(reshape(U, nw, :), :, col))
    out
end

"""
    BandStates(els::BatchedElectronState, sel::FilteredBandStates) -> BandStates

The states of the selection `sel` with their energies and velocities from the container `els`
built from it: state `i` reads local band `sel.ibands[i] - els.iband_offset[k]` of its k point.
The states, weights and `nstates_base` are the selection's. `vs` is empty when `els` holds no
`vdiag`. Every selected band must be inside the box of its k point.
"""
function BandStates(els::BatchedElectronState{T}, sel::FilteredBandStates{T}) where {T}
    els.nk == sel.kpts.n || throw(ArgumentError("els holds $(els.nk) k points, sel $(sel.kpts.n)"))
    e = Array(els.e); off = Array(els.iband_offset); nband = Array(els.nband)
    vdiag = els.vdiag === nothing ? nothing : Array(els.vdiag)
    n = sel.n
    es = zeros(T, n)
    vs = zeros(Vec3{T}, vdiag === nothing ? 0 : n)
    for i in 1:n
        ik = sel.iks[i]
        nl = sel.ibands[i] - off[ik]
        1 <= nl <= nband[ik] || throw(ArgumentError("band $(sel.ibands[i]) at k point $ik is " *
            "outside the window of the states (bands $(off[ik] + 1):$(off[ik] + nband[ik]))"))
        es[i] = e[nl, ik]
        vdiag === nothing || (vs[i] = Vec3{T}(vdiag[1, nl, ik], vdiag[2, nl, ik], vdiag[3, nl, ik]))
    end
    BandStates{T, typeof(sel.kpts)}(n, sel.nband, sel.nband_ignore, sel.nw, sel.kpts,
        copy(sel.iks), copy(sel.ibands), es, vs, copy(sel.weights), sel.nstates_base,
        copy(sel.indmap), copy(sel.band_extent))
end

"""
    electron_states_to_FilteredBandStates(kpts, els::BatchedElectronState, nstates_base; nw)

The `BatchedElectronState` method of the `Vector{ElectronState}` one: the bands of each k point are
its box window, `iband_offset[k] .+ (1:nband[k])`.
"""
function electron_states_to_FilteredBandStates(kpts, els::BatchedElectronState, nstates_base; nw)
    gkpts = kpts isa GridKpoints ? kpts : GridKpoints(kpts)
    off = Array(els.iband_offset); nband = Array(els.nband)
    iks = Int[]; ibands = Int[]
    for ik in 1:els.nk, n in 1:nband[ik]
        push!(iks, ik); push!(ibands, off[ik] + n)
    end
    FilteredBandStates(gkpts, iks, ibands; nw, nstates_base)
end
