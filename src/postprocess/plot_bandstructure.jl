import Brillouin
import PyPlot
import Bravais

export plot_bandstructure

# TODO: Fermi level
# TODO: Time reversal symmetry
# TODO: Magnetic moments

function plot_bandstructure(model; kline_density=40, close_fig=true)
    kpts, plot_xdata = high_symmetry_kpath(model; kline_density)

    # Calculate eigenvalues
    e_el = compute_eigenvalues_el(model, kpts, fourier_mode="normal")
    e_ph = compute_eigenvalues_ph(model, kpts, fourier_mode="normal")

    # Plot band structure
    fig, plotaxes = PyPlot.subplots(1, 2, figsize=(8, 3))
    plot_band_data(plotaxes[1], e_el ./ unit_to_aru(:eV),  plot_xdata, ylabel="energy (eV)", title="Electron", c="k")
    plot_band_data(plotaxes[2], e_ph ./ unit_to_aru(:meV), plot_xdata, ylabel="energy (meV)", title="Phonon", c="k")
    plotaxes[2].axhline(0, c="k", lw=1)
    display(fig)
    close_fig && close(fig)
    (;fig, kpts, e_el, e_ph, plot_xdata)
end

function plot_band_data(axis, data, plot_xdata; ylabel=nothing, title=nothing, c=nothing)
    get_color(i) = c === nothing ? "C$(mod(i, 10))" : c
    for iband in 1:size(data, 1)
        axis.plot(plot_xdata.x, data[iband, :], c=c)
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

# Generalizes Brillouin.irrfbz_path to work with arbitrary cell.
# TODO: Add tests
# TODO: Move to a general package
function irrfbz_path_for_cell(cell)
    # standardize cell
    dset = Spglib.get_dataset(cell)
    sgnum = dset.spacegroup_number
    std_lattice = Bravais.DirectBasis(collect(eachcol(dset.std_lattice)))
    # TODO: For triclinic, one may need additional reduction (niggli) because seek-path
    #       uses a different convention from spglib.

    # If the input cell is a supercell (without any distortion), then the irrfbz algorithm cannot work
    if round(Int, det(cell.lattice) / det(dset.primitive_lattice)) != 1
        @warn "input cell is a supercell. irrfbz Does not give a correct k path."
    end

    # Calculate kpath for standard primitive cell
    kp = Brillouin.irrfbz_path(sgnum, std_lattice)

    # Convert to original lattice
    # cell.lattice = rotation * primitive_lattice * transformation
    rotation = dset.std_rotation_matrix
    transformation = inv(Bravais.primitivebasismatrix(Bravais.centering(sgnum, 3))) * dset.transformation_matrix'

    # Rotate k points in Cartesian space by `rotation`
    recip_basis = Bravais.reciprocalbasis(Bravais.DirectBasis(collect(eachcol(Matrix(cell.lattice)))))
    kp_cart = Brillouin.cartesianize(kp)
    for (lab, kv) in Brillouin.points(kp_cart)
        Brillouin.points(kp_cart)[lab] = rotation * kv
    end
    kp_new = Brillouin.latticize(kp_cart, recip_basis)
    kp_new
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
function high_symmetry_kpath(model; kline_density=40)
    spg_positions, spg_numbers, _ = spglib_atoms(atom_pos_crys(model), model.atom_labels)
    structure = Spglib.Cell(model.lattice, spg_positions, spg_numbers)

    kp = irrfbz_path_for_cell(structure)
    kinter = Brillouin.interpolate(kp, density=kline_density)

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
        xshift = xs[ibranch][end]
    end
    x = vcat(xs...)
    (;x, xticks, xlabels)
end