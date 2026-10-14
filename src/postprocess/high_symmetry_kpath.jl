import Brillouin
import Bravais
using Printf

export high_symmetry_kpath
export write_kpath

"""
    high_symmetry_kpath(model; kline_density=40, is_2d = false) => (kpts, plot_xdata)
Extract the high-symmetry ``k``-point path corresponding to the passed model (or a
[`Structure`](@ref), e.g. from [`load_structure_from_qe`](@ref), before any DFT run)
using `Brillouin.jl`. Uses the conventions described in the reference work by
Cracknell, Davies, Miller, and Love (CDML). Of note, this has minor differences to
the ``k``-path reference
([Y. Himuma et. al. Comput. Mater. Sci. **128**, 140 (2017)](https://doi.org/10.1016/j.commatsci.2016.10.015))
underlying the path-choices of `Brillouin.jl`, specifically for oA and mC Bravais types.
The `kline_density` is given in number of ``k``-points per inverse bohrs (i.e.
overall in units of length).
`plot_xdata` also contains `kpts_by_label`, a `Dict` mapping each high-symmetry label
(`String`) to its ``k`` point in crystal (fractional reciprocal) coordinates (`Vec3`).
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
    xlabels = plot_xdata.xlabels[1:nlabels]
    kept = Set(String.(reduce(vcat, split.(xlabels, "|"))))
    kpts_by_label = filter(p -> first(p) in kept, plot_xdata.kpts_by_label)
    plot_xdata = (; x = plot_xdata.x[1:nk_2d], xlabels, xticks = plot_xdata.xticks[1:nlabels], kpts_by_label)

    (; kpts, plot_xdata)
end

"""
    get_band_plot_xdata(kinter)
`kinter`` is in different branches (discontinuous k paths). Merge these into a single line,
and set appropriate the xticks and xlabels.
`kpts_by_label` maps each high-symmetry label (as a `String`) to its `k` point in crystal
(fractional reciprocal) coordinates.
"""
function get_band_plot_xdata(kinter)
    kinter_cart = Brillouin.cartesianize(kinter)
    xticks = Float64[]
    xlabels = String[]
    kpts_by_label = Dict{String, Vec3{Float64}}()
    xs = Brillouin.cumdists.(kinter_cart.kpaths)
    xshift = zero(xs[1][1])
    for (ibranch, labels) in enumerate(kinter_cart.labels)
        xs[ibranch] .+= xshift
        for (ilab, x_idx) in enumerate(sort(collect(keys(labels))))
            label = labels[x_idx]
            # kinter (not cartesianized) holds the k point in crystal coordinates.
            kpts_by_label[String(label)] = Vec3{Float64}(kinter.kpaths[ibranch][x_idx])
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
    (; x, xticks, xlabels, kpts_by_label)
end

# Brillouin.jl labels may contain Γ and Unicode subscripts; QE/Wannier90 inputs want ASCII.
_ascii_klabel(label) = map(c -> c == 'Γ' ? 'G' : '₀' <= c <= '₉' ? Char(c - '₀' + '0') : c, label)

_fmt_kcrys(k) = join((@sprintf("%16.12f", abs(x) < 1e-12 ? 0.0 : x) for x in k), " ")

"""
    kpath_labels(kpts, plot_xdata) -> Vector{Pair{Int, String}}

Index into `kpts` and ASCII label (`Γ` -> `G`) of each high-symmetry point along the path, in path
order. At a branch break `X|U`, `X` and `U` are two consecutive points.
"""
function kpath_labels(kpts, plot_xdata)
    kbl = plot_xdata.kpts_by_label
    out = Pair{Int, String}[]
    ik = 1
    # Search forward from the previous label, so that a label visited more than once (e.g. Γ at
    # both ends) gets the right index.
    for xlabel in plot_xdata.xlabels, label in split(xlabel, "|")
        ik = findnext(k -> isapprox(k, kbl[label]; atol = 1e-8), kpts.vectors, ik)
        ik === nothing && error("high-symmetry point $label not found in kpts")
        push!(out, ik => _ascii_klabel(label))
    end
    out
end

"""
    write_kpath(prefix, kpts, plot_xdata; dir = ".")

Write the output of [`high_symmetry_kpath`](@ref) to `{prefix}_kpath.dat` in `dir`, one line per
k point: index, crystal coordinates, path coordinate `x` (1/Bohr), and for high-symmetry points
the ASCII label (`Γ` -> `G`). At a branch break `X|U`, `X` and `U` are consecutive lines with the
same `x`.

Columns 2-4 are a `pw.x` `K_POINTS crystal` list; the labeled lines give the Wannier90
`kpoint_path` segments and the tick positions of a band plot.
"""
function write_kpath(prefix, kpts, plot_xdata; dir = ".")
    label_at = Dict(kpath_labels(kpts, plot_xdata))
    open(joinpath(dir, "$(prefix)_kpath.dat"), "w") do io
        println(io, "# High-symmetry k path (ElectronPhonon.high_symmetry_kpath)")
        println(io, "# ik  k (crystal)  x (1/Bohr)  label")
        for (ik, k) in enumerate(kpts.vectors)
            @printf(io, "%6d %s %16.10f", ik, _fmt_kcrys(k), plot_xdata.x[ik])
            println(io, haskey(label_at, ik) ? " " * label_at[ik] : "")
        end
    end
    nothing
end
