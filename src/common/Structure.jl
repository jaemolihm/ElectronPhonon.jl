"""
    Structure(alat, lattice, mass, atom_pos, atom_labels; compute_symmetry = true, dimension = 3)

Lattice, atoms and symmetry of a crystal. `mass` and `atom_pos` are per atom; the stored
`mass` is expanded to three entries per atom (one per Cartesian displacement).

- `compute_symmetry = false` skips spglib and stores the identity operation only.
- `dimension < 3` keeps only the symmetry operations that do not mix the first `dimension`
  lattice directions with the rest (see [`restrict_symmetry_to_dimension`](@ref)), for a cell
  that is periodic in 3d but whose Hamiltonian is lower-dimensional. It does not change the
  lattice, and errors if the lattice mixes the two blocks.
"""
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

    function Structure(alat, lattice, mass, atom_pos, atom_labels; compute_symmetry = true,
                       dimension = 3)
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
            symmetry = symmetry_operations(lattice, atoms_spglib; dimension)

        else
            # Trivial symmetry only.
            symmetry = identity_symmetry()
        end

        new(alat, lattice, recip_lattice, volume, repeat(mass, inner=3), atom_pos, atom_labels, symmetry)
    end
end

# Null structure
Structure(::Nothing) = Structure(1, I(3), [], [], []; compute_symmetry = false)
