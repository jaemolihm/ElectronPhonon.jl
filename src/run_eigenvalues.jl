"""
Routines for calculating eigenvalues
"""

export compute_eigenvalues_el
export compute_eigenvalues_ph

function compute_eigenvalues_el(model::ModelEPW{FT}, kpts; fourier_mode="gridopt") where FT
    el = ElectronState(model.nw, FT)
    e = zeros(model.nw, kpts.n)
    for ik in 1:kpts.n
        set_eigen_valueonly!(el, model, kpts.vectors[ik], fourier_mode)
        e[:, ik] .= el.e_full
    end # ik
    e
end

function compute_eigenvalues_ph(model::ModelEPW{FT}, kpts; fourier_mode="gridopt") where FT
    ph = PhononState(model.nmodes, FT)
    e = zeros(model.nmodes, kpts.n)
    for ik in 1:kpts.n
        set_eigen!(ph, model, kpts.vectors[ik], fourier_mode)
        e[:, ik] .= ph.e
    end # ik
    e
end