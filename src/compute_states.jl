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
    gpu_W = nothing; gpu_V = nothing
    if use_gpu
        elham_dev = to_device(model.el_ham)
        if quantities == ["eigenvalue"]
            gpu_W = Array(get_el_eigen_valueonly_batched(elham_dev, kpts.vectors))
        else
            W, V = get_el_eigen_batched(elham_dev, kpts.vectors)
            gpu_W = Array(W); gpu_V = Array(V)
        end
    end

    # compute quantities
    @threads for iks in chunks(kpts.vectors; n=2nthreads())
        # Setup thread-local WannierInterpolators (ham only needed for the CPU eigensolve)
        ham = use_gpu ? nothing : get_interpolator(model.el_ham; fourier_mode)
        use_gpu || register_kpoints!(ham, view(kpts.vectors, iks))
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
                if "position" ∈ quantities || ("velocity" ∈ quantities && el_velocity_mode === :BerryConnection)
                    set_position!(el, pos, xk)
                end
                if "velocity" ∈ quantities
                    set_velocity!(el, vel, xk, el_velocity_mode)
                    for i in el.rng
                        el.vdiag[i] = real.(el.v[i, i])
                    end
                elseif "velocity_diagonal" ∈ quantities
                    set_velocity_diag!(el, vel, xk, el_velocity_mode)
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
    gpu_phe = nothing; gpu_phu = nothing
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
    end

    # compute quantities
    @threads for iks in chunks(kpts.vectors; n = nthreads())

        # Setup thread-local WannierInterpolators (dyn only needed for the CPU eigensolve)
        dyn = use_gpu ? nothing : get_interpolator(model.ph_dyn; fourier_mode)
        if need_velocity
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
    states
end
