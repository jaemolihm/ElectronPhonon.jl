# Interface for reading Quantum ESPRESSO input/output files.

export load_structure_from_qe

parse_qe_float(s) = parse(Float64, replace(String(s), r"[dD]" => "e"))

const _QE_CARDS = ("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS",
    "ADDITIONAL_K_POINTS", "CONSTRAINTS", "OCCUPATIONS", "ATOMIC_VELOCITIES", "ATOMIC_FORCES",
    "SOLVENTS", "HUBBARD")

_is_qe_card(line) = any(c -> startswith(uppercase(strip(line)), c), _QE_CARDS)

"""
    parse_qe_namelists(lines) -> Dict{String, String}

Collect all `key = value` assignments of the namelists (the part before the first card) into a
`Dict` with lowercase keys (e.g. `"celldm(1)"`). Values are kept as strings with quotes removed.
"""
function parse_qe_namelists(lines)
    vars = Dict{String, String}()
    for line in lines
        _is_qe_card(line) && break
        line = split(line, '!')[1]
        for chunk in split(line, ',')
            m = match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\(\s*\d+\s*\))?)\s*=\s*(\S+)", chunk)
            m === nothing && continue
            key = lowercase(replace(m.captures[1], r"\s" => ""))
            vars[key] = strip(m.captures[2], ['\'', '"'])
        end
    end
    vars
end

# Index of the card line and its lowercase option string (e.g. "{crystal}", "angstrom", "").
function _find_qe_card(lines, card)
    i = findfirst(line -> startswith(uppercase(strip(line)), card), lines)
    i === nothing && return nothing, ""
    i, lowercase(strip(lines[i])[length(card)+1:end])
end

"""
    qe_lattice_from_ibrav(ibrav, celldm) -> Mat3

Lattice vectors (columns, Bohr) following QE's `latgen` for
`celldm = [a, b/a, c/a, cos(bc), cos(ac), cos(ab)]`. Only the `ibrav` values below are implemented.
"""
function qe_lattice_from_ibrav(ibrav, celldm)
    a = celldm[1]
    if ibrav == 1  # simple cubic
        a1, a2, a3 = (a, 0, 0), (0, a, 0), (0, 0, a)
    elseif ibrav == 2  # fcc
        a1, a2, a3 = (-a/2, 0, a/2), (0, a/2, a/2), (-a/2, a/2, 0)
    elseif ibrav == 3  # bcc
        a1, a2, a3 = (a/2, a/2, a/2), (-a/2, a/2, a/2), (-a/2, -a/2, a/2)
    elseif ibrav == 4  # hexagonal
        a1, a2, a3 = (a, 0, 0), (-a/2, a*sqrt(3)/2, 0), (0, 0, a*celldm[3])
    elseif ibrav == 6  # simple tetragonal
        a1, a2, a3 = (a, 0, 0), (0, a, 0), (0, 0, a*celldm[3])
    elseif ibrav == 8  # simple orthorhombic
        a1, a2, a3 = (a, 0, 0), (0, a*celldm[2], 0), (0, 0, a*celldm[3])
    else
        error("unsupported ibrav=$ibrav")
    end
    Mat3{Float64}(a1..., a2..., a3...)
end

"""
    load_structure_from_qe(scf_file; compute_symmetry = true, dimension = 3) -> Structure

Build a [`Structure`](@ref) from a Quantum ESPRESSO `pw.x` input file (e.g. `scf.in`),
computing the symmetry with spglib. Supported:
- `ibrav = 0` with `CELL_PARAMETERS` in `alat`, `bohr` or `angstrom`;
- `ibrav = 1, 2, 3, 4, 6, 8` with either `celldm(i)` or `A`, `B`, `C` (Å);
- `ATOMIC_POSITIONS` in `crystal`, `alat`, `bohr` or `angstrom`.

Masses are taken from the `ATOMIC_SPECIES` card as written there.
This provides a way to load a `Structure` without going through `load_model`.
"""
function load_structure_from_qe(scf_file; compute_symmetry = true, dimension = 3)
    lines = readlines(scf_file)
    vars = parse_qe_namelists(lines)

    haskey(vars, "ibrav") || error("could not read ibrav from $scf_file")
    ibrav = parse(Int, vars["ibrav"])
    nat = parse(Int, vars["nat"])
    ntyp = parse(Int, vars["ntyp"])

    # celldm in QE convention: a in Bohr, then b/a, c/a, cosines.
    celldm = zeros(6)
    for i in 1:6
        key = "celldm($i)"
        haskey(vars, key) && (celldm[i] = parse_qe_float(vars[key]))
    end
    if haskey(vars, "a")
        celldm[1] == 0 || error("both celldm(1) and A given in $scf_file")
        A = parse_qe_float(vars["a"])
        celldm[1] = A * units.Å
        haskey(vars, "b") && (celldm[2] = parse_qe_float(vars["b"]) / A)
        haskey(vars, "c") && (celldm[3] = parse_qe_float(vars["c"]) / A)
        haskey(vars, "cosbc") && (celldm[4] = parse_qe_float(vars["cosbc"]))
        haskey(vars, "cosac") && (celldm[5] = parse_qe_float(vars["cosac"]))
        haskey(vars, "cosab") && (celldm[6] = parse_qe_float(vars["cosab"]))
    end

    if ibrav == 0
        icell, cell_unit = _find_qe_card(lines, "CELL_PARAMETERS")
        icell === nothing && error("ibrav = 0 but CELL_PARAMETERS card not found in $scf_file")
        vecs = [parse_qe_float.(split(strip(lines[icell+i]))[1:3]) for i in 1:3]
        lattice = Mat3{Float64}(vecs[1]..., vecs[2]..., vecs[3]...)
        if occursin("bohr", cell_unit)
            alat = norm(lattice[:, 1])
        elseif occursin("angstrom", cell_unit)
            lattice = lattice * units.Å
            alat = norm(lattice[:, 1])
        elseif occursin("alat", cell_unit)
            celldm[1] > 0 || error("CELL_PARAMETERS alat requires celldm(1) or A in $scf_file")
            alat = celldm[1]
            lattice = lattice * alat
        else
            error("CELL_PARAMETERS must specify alat, bohr or angstrom units in $scf_file")
        end
    else
        celldm[1] > 0 || error("could not read celldm(1) or A from $scf_file")
        alat = celldm[1]
        lattice = qe_lattice_from_ibrav(ibrav, celldm)
    end

    # ATOMIC_SPECIES card: ntyp lines of "label mass pseudopotential".
    ispecies, _ = _find_qe_card(lines, "ATOMIC_SPECIES")
    ispecies === nothing && error("ATOMIC_SPECIES card not found in $scf_file")
    masses = Dict{String, Float64}()
    for line in lines[ispecies+1:ispecies+ntyp]
        tok = split(strip(line))
        masses[tok[1]] = parse_qe_float(tok[2])
    end

    # ATOMIC_POSITIONS card: nat lines of "label x y z [if_pos]".
    ipos, pos_unit = _find_qe_card(lines, "ATOMIC_POSITIONS")
    ipos === nothing && error("ATOMIC_POSITIONS card not found in $scf_file")
    labels = String[]
    pos_raw = Vec3{Float64}[]
    for line in lines[ipos+1:ipos+nat]
        tok = split(strip(line))
        push!(labels, String(tok[1]))
        push!(pos_raw, Vec3{Float64}(parse_qe_float.(tok[2:4])...))
    end
    # Structure wants atom_pos in alat-unit Cartesian coordinates.
    if occursin("crystal", pos_unit)
        atom_pos = [lattice * x / alat for x in pos_raw]
    elseif occursin("bohr", pos_unit)
        atom_pos = pos_raw ./ alat
    elseif occursin("angstrom", pos_unit)
        atom_pos = pos_raw .* (units.Å / alat)
    elseif occursin("alat", pos_unit)
        atom_pos = pos_raw
    else
        error("ATOMIC_POSITIONS must specify crystal, alat, bohr or angstrom units in $scf_file")
    end

    mass = [masses[l] for l in labels]

    Structure(alat, lattice, mass, atom_pos, labels; compute_symmetry, dimension)
end
