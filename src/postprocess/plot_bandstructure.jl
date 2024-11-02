import PyPlot

export plot_bandstructure

# TODO: Fermi level
# TODO: Time reversal symmetry
# TODO: Magnetic moments

function plot_bandstructure(model; kline_density=40, close_fig=true, εF=nothing, is_2d = false)
    kpts, plot_xdata = high_symmetry_kpath(model; kline_density, is_2d)

    # Calculate eigenvalues
    # Since we use a band path, gridopt is not useful.
    e_el = compute_eigenvalues_el(model, kpts; fourier_mode="normal")
    e_ph = compute_eigenvalues_ph(model, kpts; fourier_mode="normal")

    # Plot band structure
    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    plot_band_data(plotaxes[1], e_el ./ unit_to_aru(:eV),  plot_xdata, ylabel="energy (eV)", title="Electron", fmt="k")
    plot_band_data(plotaxes[2], e_ph ./ unit_to_aru(:meV), plot_xdata, ylabel="energy (meV)", title="Phonon", fmt="k")
    plotaxes[2].axhline(0, c="k", lw=1)

    # Plot Fermi level if given
    if εF !== nothing
        plotaxes[1].axhline(εF / unit_to_aru(:eV), c="b", lw=1)
        plotaxes[1].set_ylim(εF / unit_to_aru(:eV) .+ [-2.0, 2.0])
    end

    display(fig)
    close_fig && close(fig)
    (; fig, plotaxes, kpts, e_el, e_ph, plot_xdata)
end
