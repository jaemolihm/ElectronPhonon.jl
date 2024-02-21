using ChunkSplitters
using Base.Threads: nthreads, threadid, @threads

export compute_electron_states
export compute_phonon_states

"""
    compute_electron_states(model, kpts, quantities, window=(-Inf, Inf); fourier_mode="normal")
Compute the quantities listed in `quantities` and return a vector of ElectronState.
`quantities` can containing the following: "eigenvalue", "eigenvector", "velocity_diagonal", "velocity"
"""
function compute_electron_states(model::Model{FT}, kpts, quantities, window=(-Inf, Inf); fourier_mode="normal") where FT
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


    # compute quantities
    @threads for iks in chunks(kpts.vectors; n=2nthreads())
        # Setup thread-local WannierInterpolators
        ham = get_interpolator(model.el_ham; fourier_mode)
        if need_velocity
            vel = if el_velocity_mode === :Direct
                get_interpolator(model.el_vel; fourier_mode)
            else
                get_interpolator(model.el_ham_R; fourier_mode)
            end
        end
        if need_position
            pos = get_interpolator(model.el_pos; fourier_mode)
        end

        for ik in iks
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
function compute_phonon_states(model::Model{FT}, kpts, quantities; fourier_mode="normal") where FT
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

    # Setup WannierInterpolators
    dyn_threads = [get_interpolator(model.ph_dyn; fourier_mode) for _ in 1:nthreads()]
    if need_velocity
        dyn_R_threads = [get_interpolator(model.ph_dyn_R; fourier_mode) for _ in 1:nthreads()]
    end

    # compute quantities
    Threads.@threads for ik in 1:kpts.n
        dyn = dyn_threads[threadid()]
        need_velocity && (dyn_R = dyn_R_threads[threadid()])

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
                set_eph_dipole_coeff!(ph, xk, polar)
            end
        end
    end # ik
    states
end
