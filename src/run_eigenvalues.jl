"""
Routines for calculating eigenvalues
"""

export compute_eigenvalues_el
export compute_eigenvalues_ph

function compute_eigenvalues_el(model, kpts; fourier_mode="gridopt")
    el = ElectronState(Float64, model.nw)
    e = zeros(model.nw, kpts.n)
    for ik in 1:kpts.n
        set_eigen_valueonly!(el, model.el_ham, kpts.vectors[ik], fourier_mode)
        e[:, ik] .= el.e_full
    end # ik
    e
end

function compute_eigenvalues_ph(model, kpts; fourier_mode="gridopt")
    ph = PhononState(Float64, model.nmodes)
    e = zeros(model.nmodes, kpts.n)
    for ik in 1:kpts.n
        set_eigen!(ph, model, kpts.vectors[ik], fourier_mode)
        e[:, ik] .= ph.e
    end # ik
    e
end