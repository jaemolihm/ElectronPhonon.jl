module ElectronPhononPyPlotExt

# Rendering of the band structure, deformation potential, DOS and Wannier-decay plots.
# The numerics live in the base package (`compute_deformation_potential`, `compute_dos`,
# `high_symmetry_kpath`, `compute_eigenvalues_el/ph`); only the matplotlib calls are here, so a
# multithreaded run of the base package never holds a `PyObject`.

using ElectronPhonon
using ElectronPhonon: Model, AbstractWannierObject
using PyPlot
using LinearAlgebra: norm

# TODO: Fermi level
# TODO: Time reversal symmetry
# TODO: Magnetic moments

function ElectronPhonon.plot_bandstructure(model; kline_density=40, close_fig=true, εF=nothing, is_2d = false)
    kpts, plot_xdata = high_symmetry_kpath(model; kline_density, is_2d)

    # Calculate eigenvalues
    # Since we use a band path, gridopt is not useful.
    e_el = compute_eigenvalues_el(model, kpts; fourier_mode="normal")
    e_ph = compute_eigenvalues_ph(model, kpts; fourier_mode="normal")

    # Plot band structure
    # fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    # plot_band_data(plotaxes[1], e_el ./ unit_to_aru(:eV),  plot_xdata, ylabel="energy (eV)", title="Electron", fmt="k")
    # plot_band_data(plotaxes[2], e_ph ./ unit_to_aru(:meV), plot_xdata, ylabel="energy (meV)", title="Phonon", fmt="k")
    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3) .* (0.9, 1.0))
    plot_band_data(plotaxes[1], e_el ./ unit_to_aru(:eV),  plot_xdata, title="Electron dispersion (eV)", fmt="k")
    plot_band_data(plotaxes[2], e_ph ./ unit_to_aru(:meV), plot_xdata, title="Phonon dispersion (meV)", fmt="k")
    plotaxes[2].axhline(0, c="gray", lw=1)

    # Plot Fermi level if given
    if εF !== nothing
        plotaxes[1].axhline(εF / unit_to_aru(:eV), c="b", lw=1)
        plotaxes[1].set_ylim(εF / unit_to_aru(:eV) .+ [-2.0, 2.0])
    end

    display(fig)
    close_fig && close(fig)
    (; fig, plotaxes, kpts, e_el, e_ph, plot_xdata)
end

function ElectronPhonon.plot_deformation_potential(model, xk=Vec3(0., 0., 0.);
        kline_density=40, band_rng=1:model.nw, include_polar=true, close_fig=true, is_2d = false)
    nmodes = model.nmodes
    (; e_ph, deformation_potential, qpts, plot_xdata) = compute_deformation_potential(model, xk;
        kline_density, band_rng, include_polar, is_2d)

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

    # Compute mode averaged deformation potential
    deformation_potential_avg = sqrt.(sum(deformation_potential.^2, dims=1)[1, :] ./ nmodes)

    plot_band_data(plotaxes[1], deformation_potential ./ (unit_to_aru(:eV) / unit_to_aru(:Å)),
                    plot_xdata, ylabel="D(q) (eV/Å)", title=deformation_title)
    plot_band_data(plotaxes[1], deformation_potential_avg ./ (unit_to_aru(:eV) / unit_to_aru(:Å)), plot_xdata, fmt = "k--")
    plot_band_data(plotaxes[2], e_ph ./ unit_to_aru(:meV), plot_xdata,
                    ylabel="energy (meV)", title="Phonon dispersion")
    plotaxes[1].axhline(0, c="k", lw=1)
    plotaxes[2].axhline(0, c="k", lw=1)
    display(fig)
    close_fig && close(fig)

    (; fig, e_ph, deformation_potential, qpts, plot_xdata)
end

function ElectronPhonon.plot_dos(model, nks; pdos_inds = 1:model.nw, η = 100.0 * unit_to_aru(:meV), elist = nothing, close_fig=true)
    elist, dos, pdos = compute_dos(model, nks; η, elist)

    # Plot band structure
    fig, ax = PyPlot.subplots(1, 1)
    ax.plot(elist./ unit_to_aru(:eV), dos .* unit_to_aru(:eV); c = "k", label = "DOS")

    for i in pdos_inds
        ax.plot(elist ./ unit_to_aru(:eV), pdos[:, i] .* unit_to_aru(:eV); label = "PDOS $i")
    end

    ax.legend()
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (1 / eV)")
    display(fig)
    close_fig && close(fig)
    (; fig, ax, elist, dos, pdos)
end

function ElectronPhonon.plot_decay(obj::AbstractWannierObject, lattice, ax=PyPlot.gca(); logscale=true, display_fig=true, close_fig=true)
    absR = norm.(Ref(lattice) .* obj.irvec)
    norms = real.(sqrt.(vec(sum(x -> abs2(x), obj.op_r; dims=1))))
    ax.plot(absR, norms, "o")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("norm of op(R)")
    if logscale
        # drop very small values
        if any(norms .> 1e-10)
            ax.set_ylim([minimum(norms[norms .> 1e-10]) / 2, maximum(norms) * 2])
        end
        ax.set_yscale("log")
    end
    fig = PyPlot.gcf()
    display_fig && display(fig)
    close_fig && close(fig)
    fig
end

function ElectronPhonon.plot_decay(model::Model)
    operators = [:el_ham, :el_ham_R, :el_pos, :el_vel, :ph_dyn, :ph_dyn_R]
    fig, plotaxes = PyPlot.subplots(2, 3, figsize=(8, 5))
    for (ax, key) in zip(vec(plotaxes), operators)
        ax.set_title(String(key))
        getfield(model, key) === nothing && continue
        plot_decay(getfield(model, key), model.lattice, ax, display_fig=false, close_fig=false)
    end
    fig.tight_layout(pad=0.0)
    display(fig)
    close(fig)
    fig
end

function ElectronPhonon.plot_decay_eph(model::Model)
    # epmat with outer momentum changed
    nR1 = length(model.epmat.irvec_next)
    nR2 = length(model.epmat.irvec)
    n = div(model.epmat.ndata, nR1)
    data = Base.ReshapedArray(model.epmat.op_r, (n, nR1, nR2), ())
    tmp = PermutedDimsArray(data, (1, 3, 2))
    data_new = Base.ReshapedArray(tmp, (n * nR2, nR1), ())
    epmat_new = ElectronPhonon.WannierObject(model.epmat.irvec_next, data_new)

    if model.epmat_outer_momentum == "el"
        labels = ["epmat(R_e)", "epmat(R_p)"]
    else
        labels = ["epmat(R_p)", "epmat(R_e)"]
    end

    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    for (ax, obj, key) in zip(vec(plotaxes), [model.epmat, epmat_new], labels)
        ax.set_title(String(key))
        plot_decay(obj, model.lattice, ax, display_fig=false, close_fig=false)
    end
    fig.tight_layout(pad=0.0)
    display(fig)
    close(fig)
    fig
end

function plot_band_data(axis, data, plot_xdata; add_style = true,
                        ylabel=nothing, title=nothing, fmt=nothing, kwargs...)
    if ndims(data) == 1
        if fmt === nothing
            axis.plot(plot_xdata.x, data; kwargs...)
        else
            axis.plot(plot_xdata.x, data, fmt; kwargs...)
        end
    elseif ndims(data) == 2
        get_fmt(i) = fmt === nothing ? "C$(mod(i-1, 10))" : fmt
        for iband in 1:size(data, 1)
            axis.plot(plot_xdata.x, data[iband, :], get_fmt(iband); kwargs...)
        end
    else
        @warn "data should be a vector or matrix to be plotted"
    end

    if add_style
        plot_band_data_style(axis, plot_xdata; ylabel, title)
    end
    nothing
end

function plot_band_data_style(axis, plot_xdata; ylabel=nothing, title=nothing)
    axis.axvline.(plot_xdata.xticks; c="gray", lw=1, ls="--")
    axis.set_xticks(plot_xdata.xticks)
    axis.set_xticklabels(plot_xdata.xlabels)
    axis.set_xlim(extrema(plot_xdata.x))
    ylabel !== nothing && axis.set_ylabel(ylabel)
    title !== nothing && axis.set_title(title)
    nothing
end

end # module ElectronPhononPyPlotExt
