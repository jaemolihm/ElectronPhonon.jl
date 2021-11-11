import Brillouin
import PyPlot

# TODO: Add test

function plot_bandstructure(model; kline_density=40)
    # - Brillouin.jl expects the input direct lattice to be in the conventional lattice
    #   in the convention of the International Table of Crystallography Vol A (ITA).
    # - spglib uses this convention for the returned conventional lattice,
    #   so it can be directly used as input to Brillouin.jl
    # - The output k-Points and reciprocal lattices will be in the CDML convention.
    conv_latt = EPW.get_spglib_lattice(model, to_primitive=false)
    sgnum     = EPW.spglib_spacegroup_number(model)  # Get ITA space-group number

    # Calculate high-symmetry k path. The k points are in crystal coordinates of primitive_latt.
    kp     = Brillouin.irrfbz_path(sgnum, Vec3(eachcol(conv_latt)))
    kinter = Brillouin.interpolate(kp, density=kline_density)

    # Convert kp and kinter to crystal coordiantes in model.lattice.
    kp_cart = Brillouin.cartesianize(kp)
    kinter_cart = Brillouin.cartesianize(kinter)
    recip_basis = Vec3(eachcol(model.recip_lattice))
    for (lab, kv) in Brillouin.points(kp_cart)
        Brillouin.points(kp)[lab] = Brillouin.latticize(kv, recip_basis)
    end
    for ik in 1:length(kinter_cart)
        kinter[ik] = Brillouin.latticize(kinter_cart[ik], recip_basis)
    end

    plot_xdata = get_band_plot_xdata(kinter)
    kpts = Kpoints(kinter)

    # Calculate eigenvalues
    e_el = compute_eigenvalues_el(model, kpts)
    e_ph = compute_eigenvalues_ph(model, kpts)

    # Plot band structure
    # TODO: Cleanup
    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    for iband in 1:size(e_el, 1)
        plotaxes[1].plot(plot_xdata.x, e_el[iband, :] ./ unit_to_aru(:eV), c="k")
    end
    for iband in 1:size(e_ph, 1)
        plotaxes[2].plot(plot_xdata.x, e_ph[iband, :] ./ unit_to_aru(:meV), c="k")
    end
    for ax in plotaxes
        for x in plot_xdata.xticks
            ax.axvline(x, c="k", lw=1, ls="--")
        end
        ax.set_xticks(plot_xdata.xticks)
        ax.set_xticklabels(plot_xdata.xlabels)
        ax.set_xlim(extrema(plot_xdata.x))
    end
    plotaxes[1].set_ylabel("energy (eV)")
    plotaxes[2].set_ylabel("energy (meV)")
    plotaxes[1].set_title("Electron")
    plotaxes[2].set_title("Phonon")
    plotaxes[2].axhline(0, c="k", lw=1)
    display(fig)
    (;fig, kpts, e_el, e_ph, plot_xdata)
end

"""
    get_band_plot_xdata(kinter)
`kinter`` is in different branches (discontinuous k paths). Merge these into a single line,
and set appropriate the xticks and xlabels.
"""
function get_band_plot_xdata(kinter)
    kinter_cart = Brillouin.cartesianize(kinter)
    xticks = Float64[]
    xlabels = String[]
    xs = Brillouin.cumdists.(kinter_cart.kpaths)
    xshift = zero(xs[1][1])
    for (ibranch, labels) in enumerate(kinter_cart.labels)
        xs[ibranch] .+= xshift
        for (ilab, x_idx) in enumerate(sort(collect(keys(labels))))
            label = labels[x_idx]
            if ibranch > 1 && ilab == 1
                # This branch is not continuous from the previous
                xlabels[end] *= "|" * String(label)
            else
                # This branch is continuous from the previous
                push!(xticks, xs[ibranch][x_idx])
                push!(xlabels, String(label))
            end
        end
        xshift += xs[ibranch][end]
    end
    x = vcat(xs...)
    (;x, xticks, xlabels)
end