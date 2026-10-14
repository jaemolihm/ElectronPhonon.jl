import Brillouin
import Bravais
using Printf

export high_symmetry_kpath
export write_kpath_files

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

"""
    write_kpath_files(prefix, kpts, plot_xdata; dir = ".")

Write the output of [`high_symmetry_kpath`](@ref) for use in DFT and Wannier90 band calculations:
- `{prefix}_kpath_qe.txt`: `K_POINTS crystal` card with all `kpts`, to append to a `pw.x` `bands` input.
- `{prefix}_kpath_wannier.txt`: `kpoint_path` block for a Wannier90 `.win` file (one line per segment).
- `{prefix}_kpath.labelinfo.dat`: one line per high-symmetry point in the format of Wannier90's
  `{seedname}_band.labelinfo.dat`: label, 1-based index into `kpts`, path coordinate `x` (1/Bohr),
  and the crystal coordinates.

Labels are written in ASCII (`Γ` -> `G`).
"""
function write_kpath_files(prefix, kpts, plot_xdata; dir = ".")
    kbl = plot_xdata.kpts_by_label
    fmt3(k) = join((@sprintf("%16.12f", abs(x) < 1e-12 ? 0.0 : x) for x in k), " ")

    open(joinpath(dir, "$(prefix)_kpath_qe.txt"), "w") do io
        println(io, "K_POINTS crystal")
        println(io, kpts.n)
        for k in kpts.vectors
            println(io, fmt3(k), "  1.0")
        end
    end

    # Segments: an xlabel "X|U" ends the current branch at X and starts the next one at U.
    open(joinpath(dir, "$(prefix)_kpath_wannier.txt"), "w") do io
        println(io, "begin kpoint_path")
        current = nothing
        for xlabel in plot_xdata.xlabels
            parts = split(xlabel, "|")
            if current !== nothing
                @printf(io, "%-4s %s    %-4s %s\n", _ascii_klabel(current), fmt3(kbl[current]),
                        _ascii_klabel(parts[1]), fmt3(kbl[parts[1]]))
            end
            current = String(parts[end])
        end
        println(io, "end kpoint_path")
    end

    # Each label is matched to its k point, searching forward from the previous label so that a
    # label visited more than once (e.g. Γ at both ends) gets the right index.
    open(joinpath(dir, "$(prefix)_kpath.labelinfo.dat"), "w") do io
        ik = 1
        for xlabel in plot_xdata.xlabels, label in split(xlabel, "|")
            ik = findnext(k -> isapprox(k, kbl[label]; atol = 1e-8), kpts.vectors, ik)
            ik === nothing && error("high-symmetry point $label not found in kpts")
            @printf(io, "%-4s %8d %16.10f %s\n", _ascii_klabel(label), ik, plot_xdata.x[ik], fmt3(kbl[label]))
        end
    end
    nothing
end
