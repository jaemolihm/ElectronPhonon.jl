using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads

export compute_electron_states
export compute_phonon_states

"""
    compute_electron_states(model, kpts, quantities, window=(-Inf, Inf); fourier_mode="normal")
Compute the quantities listed in `quantities` and return a vector of ElectronState.
`quantities` can containing the following: "eigenvalue", "eigenvector", "velocity_diagonal", "velocity"
"""
function compute_electron_states(model::Model{FT}, kpts, quantities, window=(-Inf, Inf); fourier_mode="normal", use_gpu=false) where FT
    # TODO: MPI, threading
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "velocity", "position"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end
    (; nw) = model

    states = [ElectronState{FT}(nw) for _ in 1:kpts.n]
    if quantities == []
        return states
    end

    if use_gpu
        _compute_electron_states_gpu!(states, model, kpts, quantities, window)
    else
        _compute_electron_states_cpu!(states, model, kpts, quantities, window; fourier_mode)
    end
    states
end

"""
    compute_electron_states(model, sel::StateSelection, quantities; fourier_mode="normal", use_gpu=false)

Compute electron states for exactly the per-k bands selected by `sel` ([DECISION 7]): each state's
band range is `sel.band_extent[ik]` (from the selection) rather than a single energy window, so a
multigrid (whose per-k band extent is narrow at fine-only nodes and wide at coincident nodes) gets
the right bands per k. Returns a vector of `ElectronState` over `sel.kpts`.
"""
function compute_electron_states(model::Model{FT}, sel::StateSelection, quantities;
        fourier_mode="normal", use_gpu=false) where FT
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "velocity", "position"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end
    kpts = sel.kpts
    states = [ElectronState{FT}(model.nw) for _ in 1:kpts.n]
    isempty(quantities) && return states
    if use_gpu
        _compute_electron_states_gpu!(states, model, kpts, quantities, sel.band_extent)
    else
        _compute_electron_states_cpu!(states, model, kpts, quantities, sel.band_extent; fourier_mode)
    end
    states
end

# The per-k band window for `set_window!`: an energy-window tuple applies to every k; a per-k vector
# of band ranges (from a `StateSelection`) is indexed by k.
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

function _compute_electron_states_cpu!(states, model::Model{FT}, kpts, quantities, window;
                                       fourier_mode) where FT
    (; el_velocity_mode) = model
    (; need_vfull, need_vdiag, need_position, need_velocity) = _electron_state_needs(model, quantities)
    @threads for iks in chunks(kpts.vectors; n=2nthreads())
        # Setup thread-local WannierInterpolators
        ham = get_interpolator(model.el_ham; fourier_mode)
        register_kpoints!(ham, view(kpts.vectors, iks))
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
                set_eigen_valueonly!(el, ham, xk)
                set_window!(el, _window_for(window, ik))
            else
                set_eigen!(el, ham, xk)
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

# GPU: one batched eigensolve on the device replaces the per-k solve, then a host loop copies the
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
function _compute_electron_states_gpu!(states, model::Model{FT}, kpts, quantities, window) where FT
    (; nw, el_velocity_mode) = model
    (; need_vfull, need_vdiag, need_position) = _electron_state_needs(model, quantities)

    E = nothing; U = nothing; rbar = nothing; vel = nothing
    itp_elham = get_interpolator(to_device(model.el_ham); fourier_mode="batched", batch_size=kpts.n)
    if quantities == ["eigenvalue"]
        E = Array(get_el_eigen_valueonly_batched(itp_elham, kpts.vectors))
    else
        E_dev, U_dev = get_el_eigen_batched(itp_elham, kpts.vectors)
        E = Array(E_dev); U = Array(U_dev)
        rbar_dev = nothing
        if need_position
            itp_pos = get_interpolator(to_device(model.el_pos); fourier_mode="batched", batch_size=kpts.n)
            rbar_dev = get_el_velocity_direct_batched(itp_pos, kpts.vectors, U_dev)
            rbar = Array(rbar_dev)
        end
        if need_vfull || need_vdiag
            Mop = if el_velocity_mode === :Direct
                model.el_vel
            elseif el_velocity_mode === :BerryConnection
                model.el_ham_R
            else
                throw(ArgumentError("unknown el_velocity_mode $el_velocity_mode"))
            end
            itp_vel = get_interpolator(to_device(Mop); fourier_mode="batched", batch_size=kpts.n)
            vel_dev = get_el_velocity_direct_batched(itp_vel, kpts.vectors, U_dev)
            if el_velocity_mode === :BerryConnection && need_vfull
                nk = kpts.n
                vel_dev .+= im .* (reshape(E_dev, nw, 1, 1, nk) .- reshape(E_dev, 1, nw, 1, nk)) .* rbar_dev
            end
            vel = Array(vel_dev)
        end
    end

    # Copy the batched device results into the per-k states (on the host).
    @threads for iks in chunks(kpts.vectors; n=2nthreads())
        for ik in iks
            xk = kpts.vectors[ik]
            el = states[ik]

            if quantities == ["eigenvalue"]
                el.xk = xk; @views el.e_full .= E[:, ik]; el.nband = 0; el.rng = 1:0
                set_window!(el, _window_for(window, ik))
            else
                el.xk = xk
                @views el.e_full .= E[:, ik]
                @views el.u_full .= U[:, :, ik]
                el.nband = 0; el.rng = 1:0
                set_window!(el, _window_for(window, ik))
                r = el.rng
                if need_position
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
            end
        end
    end
end

"""
    compute_phonon_states(model, kpts, quantities; fourier_mode="normal")
Compute the quantities listed in `quantities` and return a vector of PhononState.
`quantities` can containing the following: "eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"
TODO: Implement quantities "velocity"
"""
function compute_phonon_states(model::Model{FT}, kpts, quantities; fourier_mode="normal", eph_phonon_basis::Symbol = :eigenmode, use_gpu=false) where FT
    # TODO: MPI, threading
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end

    (; nmodes) = model

    states = [PhononState(nmodes, FT) for ik=1:kpts.n]
    if quantities == []
        return states
    end

    if use_gpu
        _compute_phonon_states_gpu!(states, model, kpts, quantities, eph_phonon_basis)
    else
        _compute_phonon_states_cpu!(states, model, kpts, quantities, eph_phonon_basis; fourier_mode)
    end
    states
end

function _compute_phonon_states_cpu!(states, model::Model{FT}, kpts, quantities,
                                     eph_phonon_basis; fourier_mode) where FT
    (; mass) = model
    need_velocity = "velocity_diagonal" ∈ quantities
    polar = model.polar_phonon
    @threads for iks in chunks(kpts.vectors; n = nthreads())
        # Setup thread-local WannierInterpolators
        dyn = get_interpolator(model.ph_dyn; fourier_mode)
        if need_velocity
            dyn_R = get_interpolator(model.ph_dyn_R; fourier_mode)
        end

        for ik in iks
            xk = kpts.vectors[ik]
            ph = states[ik]

            if quantities == ["eigenvalue"]
                set_eigen_valueonly!(ph, xk, dyn, mass, polar)
            else
                set_eigen!(ph, xk, dyn, mass, polar)
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

# GPU: batch the phonon eigensolve on the device (same idea as _compute_electron_states_gpu!),
# then a host loop copies the results into the states. velocity_diagonal is rotated on the device
# too. polar is unsupported (the GPU e-ph loop asserts no polar). Same degeneracy-gauge caveat as
# the electrons — small g2 differences for degenerate modes, most visible on COARSE q grids.
function _compute_phonon_states_gpu!(states, model::Model{FT}, kpts, quantities,
                                     eph_phonon_basis) where FT
    (; nmodes, mass) = model
    need_velocity = "velocity_diagonal" ∈ quantities
    polar = model.polar_phonon
    polar.use && error("compute_phonon_states use_gpu does not support polar phonons")

    itp_dyn = get_interpolator(to_device(model.ph_dyn); fourier_mode="batched", batch_size=kpts.n)
    D = _fourier_hk_batched(itp_dyn, kpts.vectors)  # (nmodes,nmodes,nq)
    msqrt_d = similar(D, FT, nmodes); copyto!(msqrt_d, sqrt.(mass))
    D ./= reshape(msqrt_d, nmodes, 1, 1)         # dynq[i,j] /= sqrt(mass[i] mass[j])
    D ./= reshape(msqrt_d, 1, nmodes, 1)
    E_dev, U_dev = eigen_batched(D)              # E_dev = ω² (nmodes,nq), U_dev (nmodes,nmodes,nq)
    U_dev ./= reshape(msqrt_d, nmodes, 1, 1)     # eigenvector mass factor: u[i,:] /= sqrt(mass[i])
    E = Array(sign.(E_dev) .* sqrt.(abs.(E_dev)))  # ω = sign(ω²)·√|ω²|
    U = Array(U_dev)
    vel = nothing
    if need_velocity
        itp_phvel = get_interpolator(to_device(model.ph_dyn_R); fourier_mode="batched", batch_size=kpts.n)
        vel = Array(get_el_velocity_direct_batched(itp_phvel, kpts.vectors, U_dev))
    end

    # Copy the batched device results into the per-q states (on the host).
    @threads for iks in chunks(kpts.vectors; n = nthreads())
        for ik in iks
            xk = kpts.vectors[ik]
            ph = states[ik]

            if quantities == ["eigenvalue"]
                ph.xq = xk; @views ph.e .= E[:, ik]
            else
                ph.xq = xk; @views ph.e .= E[:, ik]; @views ph.u .= U[:, :, ik]
                if "velocity" ∈ quantities
                    # not implemented
                    error("full velocity for phonons not implemented")
                elseif "velocity_diagonal" ∈ quantities
                    for i in 1:nmodes
                        ph.vdiag[i] = real.(Vec3(vel[i, i, 1, ik], vel[i, i, 2, ik], vel[i, i, 3, ik])) ./ (2 * ph.e[i])
                    end
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
