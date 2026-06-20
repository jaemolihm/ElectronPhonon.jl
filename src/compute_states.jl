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
    (; nw, el_velocity_mode) = model

    states = [ElectronState{FT}(nw) for _ in 1:kpts.n]
    if quantities == []
        return states
    end

    need_velocity = ("position" ∈ quantities
        || "velocity" ∈ quantities || "velocity_diagonal" ∈ quantities)
    need_position = ("position" ∈ quantities
        || ("velocity" ∈ quantities && el_velocity_mode === :BerryConnection))

    # GPU: replace the per-k eigensolve with one batched solve on the device (eigenvalues, and
    # eigenvectors unless valueonly). Velocity/position stay on the CPU loop below (they reuse
    # these eigenvectors). NOTE: the batched solver does NOT apply the EPW degeneracy gauge-fixing
    # that the per-k `get_el_eigen!` does, so for degenerate bands the eigenvectors (and hence e-ph
    # matrix elements / g2 for those band pairs) can differ from the CPU path by a unitary rotation
    # within the degenerate subspace. This can produce small numerical differences vs the CPU path,
    # most visibly on COARSE k grids (which are more likely to sample exact high-symmetry
    # degeneracies); the difference shrinks as the grid is refined and BZ-summed results converge.
    gpu_W = nothing; gpu_V = nothing; gpu_rbar = nothing; gpu_v = nothing
    if use_gpu
        elham_dev = to_device(model.el_ham)
        if quantities == ["eigenvalue"]
            gpu_W = Array(get_el_eigen_valueonly_batched(elham_dev, kpts.vectors))
        else
            W, V = get_el_eigen_batched(elham_dev, kpts.vectors)
            gpu_W = Array(W); gpu_V = Array(V)
            rbar_dev = nothing
            if need_position
                # Position matrix rbar = uk' * A(k) * uk, batched on the device (full-band; the
                # in-window block is sliced per k in the loop below). `el_pos` has ndata = nw^2*3.
                rbar_dev = get_el_velocity_direct_batched(to_device(model.el_pos), kpts.vectors, V)
                gpu_rbar = Array(rbar_dev)
            end
            if "velocity" ∈ quantities || "velocity_diagonal" ∈ quantities
                # Full-band velocity rotation v = uk' * (dH/dk) * uk on the device. :Direct interpolates
                # dH/dk (el_vel) directly; :BerryConnection interpolates H(R) (el_ham_R) for the rotation
                # and adds the Berry term im*(e_i - e_j)*rbar_{ij}. The Berry term is added only for the
                # full "velocity" quantity (rbar is on device then; it is zero on the diagonal, so the
                # "velocity_diagonal" path takes real(diag(v)) without it).
                Mop = el_velocity_mode === :Direct ? model.el_vel : model.el_ham_R
                v_dev = get_el_velocity_direct_batched(to_device(Mop), kpts.vectors, V)
                if el_velocity_mode === :BerryConnection && "velocity" ∈ quantities
                    nk = kpts.n
                    v_dev .+= im .* (reshape(W, nw, 1, 1, nk) .- reshape(W, 1, nw, 1, nk)) .* rbar_dev
                end
                gpu_v = Array(v_dev)
            end
        end
    end

    # compute quantities
    @threads for iks in chunks(kpts.vectors; n=2nthreads())
        # Setup thread-local WannierInterpolators (ham only needed for the CPU eigensolve)
        ham = use_gpu ? nothing : get_interpolator(model.el_ham; fourier_mode)
        use_gpu || register_kpoints!(ham, view(kpts.vectors, iks))
        # GPU handles both "velocity" and "velocity_diagonal" on the device; the CPU interpolator is
        # only needed for the CPU path.
        if need_velocity && !use_gpu
            vel = if el_velocity_mode === :Direct
                get_interpolator(model.el_vel; fourier_mode)
            else
                get_interpolator(model.el_ham_R; fourier_mode)
            end
            register_kpoints!(vel, view(kpts.vectors, iks))
        end
        if need_position && !use_gpu
            pos = get_interpolator(model.el_pos; fourier_mode)
            register_kpoints!(pos, view(kpts.vectors, iks))
        end

        for ik in iks
            xk = kpts.vectors[ik]
            el = states[ik]

            if quantities == ["eigenvalue"]
                if use_gpu
                    el.xk = xk; @views el.e_full .= gpu_W[:, ik]; el.nband = 0; el.rng = 1:0
                else
                    set_eigen_valueonly!(el, ham, xk)
                end
                set_window!(el, window)
            else
                if use_gpu
                    el.xk = xk
                    @views el.e_full .= gpu_W[:, ik]
                    @views el.u_full .= gpu_V[:, :, ik]
                    el.nband = 0; el.rng = 1:0
                else
                    set_eigen!(el, ham, xk)
                end
                set_window!(el, window)
                if need_position
                    if use_gpu
                        # Inject the in-window block of the device-computed full-band rbar.
                        # (in-window block of the full-u rotation == windowed-u rotation.)
                        rbar_w = reshape(reinterpret(Complex{FT}, no_offset_view(el.rbar)), 3, el.nband, el.nband)
                        r = el.rng
                        @views for idir in 1:3
                            rbar_w[idir, :, :] .= gpu_rbar[r, r, idir, ik]
                        end
                    else
                        set_position!(el, pos, xk)
                    end
                end
                if "velocity" ∈ quantities
                    if use_gpu
                        # Inject the in-window block of the device-computed full-band velocity.
                        v_w = reshape(reinterpret(Complex{FT}, no_offset_view(el.v)), 3, el.nband, el.nband)
                        r = el.rng
                        @views for idir in 1:3
                            v_w[idir, :, :] .= gpu_v[r, r, idir, ik]
                        end
                    else
                        set_velocity!(el, vel, xk, el_velocity_mode)
                    end
                    for i in el.rng
                        el.vdiag[i] = real.(el.v[i, i])
                    end
                elseif "velocity_diagonal" ∈ quantities
                    if use_gpu
                        # vdiag = real(diag(v)); the BerryConnection term is zero on the diagonal.
                        for i in el.rng
                            el.vdiag[i] = real.(Vec3(gpu_v[i, i, 1, ik], gpu_v[i, i, 2, ik], gpu_v[i, i, 3, ik]))
                        end
                    else
                        set_velocity_diag!(el, vel, xk, el_velocity_mode)
                    end
                end
            end
        end
    end # ik
    states
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

    (; nmodes, mass) = model
    polar = model.polar_phonon

    states = [PhononState(nmodes, FT) for ik=1:kpts.n]
    if quantities == []
        return states
    end

    need_velocity = "velocity_diagonal" ∈ quantities

    # GPU: batch the phonon eigensolve on the device — Fourier(dyn) over all q, mass-scale, batched
    # Hermitian eigensolve, ω = sign(ω²)·√|ω²|, then the eigenvector mass factor — replacing the
    # per-q `get_ph_eigen!`. velocity_diagonal / eph_dipole_coeff stay on the CPU loop below.
    # polar is unsupported here (the GPU e-ph loop asserts no polar; `dynmat_dipole!` no-ops when
    # polar.use is false anyway). Same gauge caveat as the electrons: the batched solver omits the
    # per-q degeneracy gauge-fixing, so eigenvectors of degenerate modes can differ from the CPU
    # path by a unitary rotation — small numerical differences in g2, most visible on COARSE q grids.
    gpu_phe = nothing; gpu_phu = nothing; gpu_phvel = nothing
    if use_gpu
        polar.use && error("compute_phonon_states use_gpu does not support polar phonons")
        D = _fourier_hk_batched(to_device(model.ph_dyn), kpts.vectors; batch_size = kpts.n)  # (nmodes,nmodes,nq)
        msqrt_d = similar(D, FT, nmodes); copyto!(msqrt_d, sqrt.(mass))
        D ./= reshape(msqrt_d, nmodes, 1, 1)        # dynq[i,j] /= sqrt(mass[i] mass[j])
        D ./= reshape(msqrt_d, 1, nmodes, 1)
        W, V = eigen_batched(D)                     # W = ω² (nmodes,nq), V (nmodes,nmodes,nq)
        V ./= reshape(msqrt_d, nmodes, 1, 1)        # eigenvector mass factor: u[i,:] /= sqrt(mass[i])
        gpu_phe = Array(sign.(W) .* sqrt.(abs.(W))) # ω = sign(ω²)·√|ω²|
        gpu_phu = Array(V)
        if need_velocity
            # Phonon band velocity (diagonal): reuse the batched uk'*M*uk rotation on dyn_R (ndata =
            # nmodes^2*3) with the mass-factored eigenvectors V (mass factor already applied above,
            # as get_ph_velocity_diag! assumes). get_el_velocity_direct_batched is matrix-dim-generic
            # (nw -> nmodes), so it is reused here for phonons. The diagonal real part /(2ω) is taken
            # per q in the loop below (matches set_velocity_diag!). Polar unsupported (asserted above).
            gpu_phvel = Array(get_el_velocity_direct_batched(to_device(model.ph_dyn_R), kpts.vectors, V))
        end
    end

    # compute quantities
    @threads for iks in chunks(kpts.vectors; n = nthreads())

        # Setup thread-local WannierInterpolators (dyn only needed for the CPU eigensolve)
        dyn = use_gpu ? nothing : get_interpolator(model.ph_dyn; fourier_mode)
        if need_velocity && !use_gpu
            dyn_R = get_interpolator(model.ph_dyn_R; fourier_mode)
        end

        for ik in iks
            xk = kpts.vectors[ik]
            ph = states[ik]

            if quantities == ["eigenvalue"]
                if use_gpu
                    ph.xq = xk; @views ph.e .= gpu_phe[:, ik]
                else
                    set_eigen_valueonly!(ph, xk, dyn, mass, polar)
                end
            else
                if use_gpu
                    ph.xq = xk; @views ph.e .= gpu_phe[:, ik]; @views ph.u .= gpu_phu[:, :, ik]
                else
                    set_eigen!(ph, xk, dyn, mass, polar)
                end
                if "velocity" ∈ quantities
                    # not implemented
                    error("full velocity for phonons not implemented")
                elseif "velocity_diagonal" ∈ quantities
                    if use_gpu
                        # vdiag[idir,i] = real(u[:,i]' * dD/dk[:,:,idir] * u[:,i]) / (2 ω_i)
                        # (dw/dk = (d(w²)/dk)/(2w); same as set_velocity_diag!). ω≈0 acoustic modes
                        # at Γ divide by ~0 exactly as the CPU path does.
                        for i in 1:nmodes
                            ph.vdiag[i] = real.(Vec3(gpu_phvel[i, i, 1, ik], gpu_phvel[i, i, 2, ik], gpu_phvel[i, i, 3, ik])) ./ (2 * ph.e[i])
                        end
                    else
                        set_velocity_diag!(ph, xk, dyn_R)
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
    states
end
