# Interface for reading Quantum ESPRESSO input/output files.

parse_qe_float(s) = parse(Float64, replace(String(s), r"[dD]" => "e"))

"""
    load_structure_from_qe(scf_file) -> Structure

Build a [`Structure`](@ref) from a Quantum ESPRESSO `pw.x` input file (e.g. `scf.in`),
computing the symmetry with spglib. Only `ibrav = 1` (simple cubic) with
`ATOMIC_POSITIONS` given in `crystal` units is supported.

This provides a way to load a `Structure` without going through `load_model`.
"""
function load_structure_from_qe(scf_file)
    lines = readlines(scf_file)

    alat = 0.0
    ibrav = -999
    nat = 0
    ntyp = 0
    for line in lines
        for m in eachmatch(r"celldm\(1\)\s*=\s*([-+0-9.eEdD]+)", line)
            alat = parse_qe_float(m.captures[1])
        end
        for m in eachmatch(r"ibrav\s*=\s*(-?\d+)", line)
            ibrav = parse(Int, m.captures[1])
        end
        for m in eachmatch(r"nat\s*=\s*(\d+)", line)
            nat = parse(Int, m.captures[1])
        end
        for m in eachmatch(r"ntyp\s*=\s*(\d+)", line)
            ntyp = parse(Int, m.captures[1])
        end
    end
    ibrav == -999 && error("could not read ibrav from $scf_file")
    alat > 0 || error("could not read celldm(1) from $scf_file")

    # Simple cubic lattice; columns are the lattice vectors (Bohr).
    if ibrav == 1
        lattice = Mat3{Float64}(alat, 0.0, 0.0, 0.0, alat, 0.0, 0.0, 0.0, alat)
    else
        error("unsupported ibrav=$ibrav")
    end

    # ATOMIC_SPECIES card: ntyp lines of "label mass pseudopotential".
    ispecies = findfirst(line -> startswith(uppercase(strip(line)), "ATOMIC_SPECIES"), lines)
    ispecies === nothing && error("ATOMIC_SPECIES card not found in $scf_file")
    masses = Dict{String, Float64}()
    for line in lines[ispecies+1:ispecies+ntyp]
        tok = split(strip(line))
        masses[tok[1]] = parse_qe_float(tok[2])
    end

    # ATOMIC_POSITIONS card: nat lines of "label x y z" in crystal coordinates.
    ipos = findfirst(line -> startswith(uppercase(strip(line)), "ATOMIC_POSITIONS"), lines)
    ipos === nothing && error("ATOMIC_POSITIONS card not found in $scf_file")
    occursin("crystal", lowercase(lines[ipos])) || error("ATOMIC_POSITIONS must be given in 'crystal' units, other units not yet implemented")
    labels = String[]
    atom_pos_cryst = Vec3{Float64}[]
    for line in lines[ipos+1:ipos+nat]
        tok = split(strip(line))
        push!(labels, String(tok[1]))
        push!(atom_pos_cryst, Vec3{Float64}(parse_qe_float.(tok[2:4])...))
    end

    mass = [masses[l] for l in labels]

    # Structure wants atom_pos in alat-unit Cartesian coordinates.
    atom_pos = [lattice * x / alat for x in atom_pos_cryst]

    Structure(alat, lattice, mass, atom_pos, labels)
end
