import Brillouin
import PyPlot

# TODO: Add test
# TODO: Fermi level
# TODO: Time reversal symmetry
# TODO: Magnetic moments

function plot_bandstructure(model; kline_density=40)
    kpts, plot_xdata = high_symmetry_kpath(model, kline_density)

    # Calculate eigenvalues
    e_el = compute_eigenvalues_el(model, kpts)
    e_ph = compute_eigenvalues_ph(model, kpts)

    # Plot band structure
    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    plot_band_data(plotaxes[1], e_el ./ unit_to_aru(:eV),  plot_xdata, ylabel="energy (eV)", title="Electron")
    plot_band_data(plotaxes[2], e_ph ./ unit_to_aru(:meV), plot_xdata, ylabel="energy (meV)", title="Phonon")
    plotaxes[2].axhline(0, c="k", lw=1)
    display(fig)
    (;fig, kpts, e_el, e_ph, plot_xdata)
end

function plot_band_data(axis, data, plot_xdata; ylabel=nothing, title=nothing)
    for iband in 1:size(data, 1)
        axis.plot(plot_xdata.x, data[iband, :], c="k")
    end
    for x in plot_xdata.xticks
        axis.axvline(x, c="k", lw=1, ls="--")
    end
    axis.set_xticks(plot_xdata.xticks)
    axis.set_xticklabels(plot_xdata.xlabels)
    axis.set_xlim(extrema(plot_xdata.x))
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    nothing
end

"""
Extract the high-symmetry ``k``-point path corresponding to the passed model
using `Brillouin.jl`. Uses the conventions described in the reference work by
Cracknell, Davies, Miller, and Love (CDML). Of note, this has minor differences to
the ``k``-path reference
([Y. Himuma et. al. Comput. Mater. Sci. **128**, 140 (2017)](https://doi.org/10.1016/j.commatsci.2016.10.015))
underlying the path-choices of `Brillouin.jl`, specifically for oA and mC Bravais types.
The `kline_density` is given in number of ``k``-points per inverse bohrs (i.e.
overall in units of length).
(Adapted from DFTK.jl)
"""
function high_symmetry_kpath(model, kline_density=40)
    # - Brillouin.jl expects the input direct lattice to be in the conventional lattice
    #   in the convention of the International Table of Crystallography Vol A (ITA).
    # - spglib uses this convention for the returned conventional lattice,
    #   so it can be directly used as input to Brillouin.jl
    # - The output k-Points and reciprocal lattices will be in the CDML convention.
    conv_latt = EPW.get_spglib_lattice(model, to_primitive=false)
    sgnum     = EPW.spglib_spacegroup_number(model)  # Get ITA space-group number

    # Calculate high-symmetry k path. The k points are in crystal coordinates of the
    # primitive lattice vector in CDML convention.
    kp     = Brillouin.irrfbz_path(sgnum, Vec3(eachcol(conv_latt)))
    kinter = Brillouin.interpolate(kp, density=kline_density)

    # Now, kinter is in crystal coordinates of the primitive lattice vector in CDML convention,
    # not model.lattuce. So, we convert kinter to crystal coordiantes in model.lattice by
    # converting as follows:
    # crystal (standard primitive) -> Cartesian -> crystal (model.lattice)
    kinter_cart = Brillouin.cartesianize(kinter)
    recip_basis = Vec3(eachcol(model.recip_lattice))
    for ik in 1:length(kinter_cart)
        kinter[ik] = Brillouin.latticize(kinter_cart[ik], recip_basis)
    end

    plot_xdata = get_band_plot_xdata(kinter)
    kpts = Kpoints(kinter)

    kpts, plot_xdata
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