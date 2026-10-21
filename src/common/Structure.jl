"""
    Structure(alat, lattice, mass, atom_pos, atom_labels; dimension = 3)

Lattice and atoms of a crystal. `mass` and `atom_pos` are per atom; the stored
`mass` is expanded to three entries per atom (one per Cartesian displacement).

`dimension` (1, 2 or 3) is the number of periodic directions of the Hamiltonian, for a cell
that is periodic in 3d but whose Hamiltonian is lower-dimensional (a chain or sheet padded
with vacuum). It does not change the lattice. The constructor errors if the lattice mixes
the first `dimension` directions with the rest (see [`restrict_symmetry_to_dimension`](@ref)).

The symmetry group of the structure is `symmetry_operations(structure)`.
"""
struct Structure
    # Lattice information
    alat          :: Float64        # Lattice parameter
    lattice       :: Mat3{Float64}  # lattice[:, i] is the i-th lattice vector in Bohr.
    recip_lattice :: Mat3{Float64}  # recip_lattice[:, i] is the i-th reciprocal lattice vector in 1/Bohr.
    volume        :: Float64        # Cell volume in Bohr^3. (=det(lattice))

    # Number of periodic directions of the Hamiltonian (1, 2 or 3). The lattice is always 3d.
    dimension :: Int

    # Atom information
    mass        :: Vector{Float64}        # Atom mass in Rydberg units. (1 amu = 911.444)
    atom_pos    :: Vector{Vec3{Float64}}  # Atom position in alat units, Cartesian coordinates
    atom_labels :: Vector{String}

    function Structure(alat, lattice, mass, atom_pos, atom_labels; dimension = 3)
        if length(mass) != length(atom_pos)
            error("Length of mass and atom_pos must be the same.")
        end
        if length(mass) != length(atom_labels)
            error("Length of mass and atom_labels must be the same.")
        end
        check_lattice_blocks(lattice, dimension)

        recip_lattice = inv(lattice') * 2π
        volume = abs(det(lattice))

        new(alat, lattice, recip_lattice, volume, dimension, repeat(mass, inner=3), atom_pos, atom_labels)
    end
end

# Null structure
Structure(::Nothing) = Structure(1, I(3), [], [], [])

function Base.show(io::IO, ::MIME"text/plain", structure::Structure)
    println(io, "Structure: ", length(structure.atom_pos), " atoms ", structure.atom_labels)
    print(io, "  alat = ", structure.alat, " bohr, dimension = ", structure.dimension,
          ", volume = ", structure.volume, " bohr^3")
end
