import Brillouin
import PyPlot
import Bravais

export high_symmetry_kpath

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
function high_symmetry_kpath(model; kline_density=40, is_2d = false)
    spg_positions, spg_numbers, _ = spglib_atoms(atom_pos_crys(model), model.atom_labels)
    cell = Spglib.Cell(model.lattice, spg_positions, spg_numbers)

    kp = Brillouin.irrfbz_path(cell)
    kinter = Brillouin.interpolate(kp, density=kline_density)

    plot_xdata = get_band_plot_xdata(kinter)
    kpts = Kpoints(kinter)

    if is_2d
        kpts, plot_xdata = kpath_truncate_to_2d(kpts, plot_xdata)
    end

    kpts, plot_xdata
end

function kpath_truncate_to_2d(kpts, plot_xdata)
    nk_2d = findfirst(xk -> xk[3] != 0, kpts.vectors) - 1
    nlabels = findlast(plot_xdata.xticks .<= plot_xdata.x[nk_2d])

    kpts = Kpoints(nk_2d, kpts.vectors[1:nk_2d], kpts.weights[1:nk_2d], kpts.ngrid)
    plot_xdata = (; x = plot_xdata.x[1:nk_2d], xlabels = plot_xdata.xlabels[1:nlabels], xticks = plot_xdata.xticks[1:nlabels])

    (; kpts, plot_xdata)
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
    axis.axvline.(plot_xdata.xticks; c="k", lw=1, ls="--")
    axis.set_xticks(plot_xdata.xticks)
    axis.set_xticklabels(plot_xdata.xlabels)
    axis.set_xlim(extrema(plot_xdata.x))
    ylabel !== nothing && axis.set_ylabel(ylabel)
    title !== nothing && axis.set_title(title)
    nothing
end
