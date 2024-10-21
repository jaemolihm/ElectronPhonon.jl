struct Structure
    # Lattice information
    alat          :: Float64        # Lattice parameter
    lattice       :: Mat3{Float64}  # lattice[:, i] is the i-th lattice vector in Bohr.
    recip_lattice :: Mat3{Float64}  # recip_lattice[:, i] is the i-th reciprocal lattice vector in 1/Bohr.
    volume        :: Float64        # Cell volume in Bohr^3. (=det(lattice))

    # Atom information
    mass        :: Vector{Float64}        # Atom mass in Rydberg units. (1 amu = 911.444)
    atom_pos    :: Vector{Vec3{Float64}}  # Atom position in alat units, Cartesian coordinates
    atom_labels :: Vector{String}

    # Symmetries
    symmetry :: Symmetry{Float64}

    function Structure(alat, lattice, mass, atom_pos, atom_labels; compute_symmetry = true)
        if length(mass) != length(atom_pos)
            error("Length of mass and atom_pos must be the same.")
        end
        if length(mass) != length(atom_labels)
            error("Length of mass and atom_labels must be the same.")
        end

        recip_lattice = inv(lattice') * 2π
        volume = abs(det(lattice))

        if compute_symmetry
            # Compute symmetry operations using Spglib
            atom_pos_crystal = Ref(lattice) .\ (atom_pos * alat)
            atoms_spglib = [label => [x for (l, x) in zip(atom_labels, atom_pos_crystal) if l == label] for label in atom_labels]
            symmetry = symmetry_operations(lattice, atoms_spglib)

        else
            # Trivial symmetry only.
            symmetry = identity_symmetry()
        end

        new(alat, lattice, recip_lattice, volume, repeat(mass, inner=3), atom_pos, atom_labels, symmetry)
    end
end

# Null structure
Structure(::Nothing) = Structure(1, I(3), [], [], []; compute_symmetry = false)
