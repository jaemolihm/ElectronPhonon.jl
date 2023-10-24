"""
Routines for calculating eigenvalues
"""

export compute_eigenvalues_el
export compute_eigenvalues_ph

function compute_eigenvalues_el(model::ModelEPW{FT}, kpts; fourier_mode="gridopt") where FT
    ham = get_interpolator(model.el_ham; fourier_mode)
    e = zeros(FT, model.nw, kpts.n)
    el = ElectronState{FT}(model.nw)
    for ik in 1:kpts.n
        set_eigen_valueonly!(el, ham, kpts.vectors[ik])
        e[:, ik] .= el.e_full
    end # ik
    e
end


function compute_eigenvalues_ph(model::ModelEPW{FT}, kpts; fourier_mode="gridopt") where FT
    dyn = get_interpolator(model.ph_dyn; fourier_mode)
    ph = PhononState(model.nmodes, FT)
    e = zeros(model.nmodes, kpts.n)
    for ik in 1:kpts.n
        set_eigen_valueonly!(ph, kpts.vectors[ik], dyn, model.mass, model.polar_phonon)
        e[:, ik] .= ph.e
    end # ik
    e
end
