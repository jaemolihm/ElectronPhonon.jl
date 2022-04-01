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
    nw = model.nw

    # if position (rbar) is calculated, it does not need to be recalculated in set_velocity!
    skip_rbar = "position" ∈ quantities

    states = [ElectronState{FT}(nw, 0) for ik=1:kpts.n]
    if quantities == []
        return states
    end

    # compute quantities
    for ik in 1:kpts.n
        xk = kpts.vectors[ik]
        el = states[ik]

        if quantities == ["eigenvalue"]
            set_eigen_valueonly!(el, model, xk, fourier_mode)
            set_window!(el, window)
        else
            set_eigen!(el, model, xk, fourier_mode)
            set_window!(el, window)
            if "position" ∈ quantities
                set_position!(el, model, xk, fourier_mode)
            end
            if "velocity" ∈ quantities
                set_velocity!(el, model, xk, fourier_mode; skip_rbar)
                for i in el.rng
                    el.vdiag[i] = real.(el.v[i, i])
                end
            elseif "velocity_diagonal" ∈ quantities
                set_velocity_diag!(el, model, xk, fourier_mode)
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
    for ik in 1:kpts.n
        xk = kpts.vectors[ik]
        ph = states[ik]

        if quantities == ["eigenvalue"]
            set_eigen_valueonly!(ph, model, xk, fourier_mode)
        else
            set_eigen!(ph, model, xk, fourier_mode)
            if "velocity" ∈ quantities
                # not implemented
                error("velocity not implemented")
            elseif "velocity_diagonal" ∈ quantities
                set_velocity_diag!(ph, model, xk, fourier_mode)
            end
            if "eph_dipole_coeff" ∈ quantities
                set_eph_dipole_coeff!(ph, model, xk)
            end
        end
    end # ik
    states
end