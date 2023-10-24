export compute_electron_states
export compute_phonon_states

"""
    compute_electron_states(model, kpts, quantities, window=(-Inf, Inf); fourier_mode="normal")
Compute the quantities listed in `quantities` and return a vector of ElectronState.
`quantities` can containing the following: "eigenvalue", "eigenvector", "velocity_diagonal", "velocity"
"""
function compute_electron_states(model::ModelEPW{FT}, kpts, quantities, window=(-Inf, Inf); fourier_mode="normal") where FT
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

    # Setup WannierInterpolators
    ham_threads = [get_interpolator(model.el_ham; fourier_mode) for _ in 1:nthreads()]
    if need_velocity
        vel_threads = if el_velocity_mode === :Direct
            [get_interpolator(model.el_vel; fourier_mode) for _ in 1:nthreads()]
        else
            [get_interpolator(model.el_ham_R; fourier_mode) for _ in 1:nthreads()]
        end
    end
    if need_position
        pos_threads = [get_interpolator(model.el_pos; fourier_mode) for _ in 1:nthreads()]
    end

    # compute quantities
    Threads.@threads :static for ik in 1:kpts.n
        ham = ham_threads[threadid()]
        need_velocity && (vel = vel_threads[threadid()])
        need_position && (pos = pos_threads[threadid()])

        xk = kpts.vectors[ik]
        el = states[ik]

        if quantities == ["eigenvalue"]
            set_eigen_valueonly!(el, ham, xk)
            set_window!(el, window)
        else
            set_eigen!(el, ham, xk)
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
    end # ik
    states
end

"""
    compute_phonon_states(model, kpts, quantities; fourier_mode="normal")
Compute the quantities listed in `quantities` and return a vector of PhononState.
`quantities` can containing the following: "eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"
TODO: Implement quantities "velocity"
"""
function compute_phonon_states(model::ModelEPW{FT}, kpts, quantities; fourier_mode="normal") where FT
    # TODO: MPI, threading
    allowed_quantities = ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]
    for quantity in quantities
        quantity ∉ allowed_quantities && error("$quantity is not an allowed quantity.")
    end

    nmodes = model.nmodes

    states = [PhononState(nmodes, FT) for ik=1:kpts.n]
    if quantities == []
        return states
    end

    # compute quantities
    Threads.@threads for ik in 1:kpts.n
        xk = kpts.vectors[ik]
        ph = states[ik]

        if quantities == ["eigenvalue"]
            set_eigen_valueonly!(ph, model, xk; fourier_mode)
        else
            set_eigen!(ph, model, xk; fourier_mode)
            if "velocity" ∈ quantities
                # not implemented
                error("velocity not implemented")
            elseif "velocity_diagonal" ∈ quantities
                set_velocity_diag!(ph, model, xk; fourier_mode)
            end
            if "eph_dipole_coeff" ∈ quantities
                set_eph_dipole_coeff!(ph, model, xk)
            end
        end
    end # ik
    states
end
