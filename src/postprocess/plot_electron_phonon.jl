import PyPlot

export plot_electron_phonon_deformation_potential

"""
    plot_electron_phonon_deformation_potential(model, xk=Vec3(0., 0., 0.); kline_density=40,
        band_rng=1:model.nw, include_polar=true, close_fig=true)
Calculate and plot the total electrin-phonon deformation potential ``D(q, ν)`` along a high-symmetry q-point path.
Reference: J. Sjakste et al., Phys. Rev. B 92, 054307 (2015), Eqs. (3-4)

``D(q, ν) = \\sqrt{2 M_{\\rm uc} ω_{q,ν} / ħ^2 * ∑_{m, n ∈ band_rng} |g_{m,n,ν}(k,q)|^2}``

Here, ``k`` is a single point (`xk`) is used while ``q`` are multiple points on a high-symmetry path.
- `xk`: k point to calculate the deformation potential. Default: `Vec3(0., 0., 0.)`
- `kline_density`: number of ``k``-points per inverse bohrs (i.e. overall in units of length).
- `band_rng`: range of bands to include in the deformation potential. Default: `1:model.nw`
- `include_polar`: if true, include the polar e-ph interaction if present. Default: `true`
"""
function plot_electron_phonon_deformation_potential(model, xk=Vec3(0., 0., 0.);
        kline_density=40, band_rng=1:model.nw, include_polar=true, close_fig=true)
    model.epmat_outer_momentum != "el" && error("model.epmat_outer_momentum must be el")
    nw = model.nw
    nmodes = model.nmodes
    fourier_mode = "normal" # Since we use a band path, gridopt is not useful.

    # Setup k, q, and kq points
    kpts = Kpoints(xk)
    qpts, plot_xdata = high_symmetry_kpath(model; kline_density)
    xkq = qpts.vectors .+ Ref(xk)
    kqpts = Kpoints(xkq)
    nq = qpts.n

    deformation_potential = zeros(nmodes, nq)
    e_ph = zeros(nmodes, nq)

    el_k_save = compute_electron_states(model, kpts, ["eigenvector"]; fourier_mode)
    el_kq_save = compute_electron_states(model, kqpts, ["eigenvector"]; fourier_mode)
    ph_save = compute_phonon_states(model, qpts, ["eigenvector", "eph_dipole_coeff"]; fourier_mode)

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epdata = ElPhData(nw, nmodes)
    epobj_ekpR = WannierObject(model.epmat.irvec_next, zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    ik = 1
    xk = kpts.vectors[ik]
    el_k = el_k_save[ik]
    epdata.el_k = el_k
    get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, get_u(el_k), fourier_mode)

    # Calculate electron-phonon coupling matrix elements
    for iq in 1:nq
        xkq = kqpts.vectors[iq]
        xq = qpts.vectors[iq]

        # Set electron and phonon states in epdata
        epdata.el_kq = el_kq_save[iq]
        epdata.ph = ph_save[iq]

        # Compute electron-phonon coupling
        get_eph_kR_to_kq!(epdata, epobj_ekpR, xq, fourier_mode)
        if include_polar && any(abs.(xq) .> 1.0e-8) && model.use_polar_dipole
            epdata_set_mmat!(epdata)
            model.polar_eph.use && epdata_compute_eph_dipole!(epdata)
        end
        epdata_set_g2!(epdata)

        @views for imode in 1:nmodes
            # Here, |g|^2 = |epdata.ep|^2 / 2ω, so 2ω|g|^2 = |epdata.ep|^2.
            deformation_potential[imode, iq] = norm(epdata.ep[band_rng, band_rng, imode])
            e_ph[imode, iq] = epdata.ph.e[imode]
        end
    end # iq

    unit_cell_mass = sum(model.mass[1:3:end])
    deformation_potential .*= sqrt(unit_cell_mass)

    # Plot deformation potential and phonon band structure
    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    deformation_title = "Deformation potential, bands $(band_rng)"
    if model.use_polar_dipole
        if include_polar
            deformation_title *= "\n(Long-range part included)"
        else
            deformation_title *= "\n(Long-range part excluded)"
        end
    else
        deformation_title *= "\n(No long-range part in model)"
    end
    plot_band_data(plotaxes[1], deformation_potential ./ (unit_to_aru(:eV) / unit_to_aru(:Å)),
                    plot_xdata, ylabel="D(q) (eV/Å)", title=deformation_title)
    plot_band_data(plotaxes[2], e_ph ./ unit_to_aru(:meV), plot_xdata,
                    ylabel="energy (meV)", title="Phonon dispersion")
    plotaxes[1].axhline(0, c="k", lw=1)
    plotaxes[2].axhline(0, c="k", lw=1)
    display(fig)
    close_fig && close(fig)

    (;fig, e_ph, deformation_potential, qpts, plot_xdata)
end