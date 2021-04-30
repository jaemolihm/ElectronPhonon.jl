# Adapted from DFTK.jl external/spglib.jl and symmetry.jl
# TODO: Time-reversal symmetry

using spglib_jll
using StaticArrays
using LinearAlgebra

export symmetry_operations
export bzmesh_ir_wedge

# Routines for interaction with spglib
# Note: spglib/C uses the row-major convention, thus we need to perform transposes
#       between julia and spglib (https://spglib.github.io/spglib/variable.html)
const SPGLIB = spglib_jll.libsymspg

function spglib_get_error_message()
    error_code = ccall((:spg_get_error_code, SPGLIB), Cint, ())
    return unsafe_string(ccall((:spg_get_error_message, SPGLIB), Cstring, (Cint,), error_code))
end

"""
Convert the DFTK atoms datastructure into a tuple of datastructures for use with spglib.
`positions` contains positions per atom, `numbers` contains the mapping atom
to a unique number for each indistinguishable element, `spins` contains
the ``z``-component of the initial magnetic moment on each atom, `mapping` contains the
mapping of the `numbers` to the element objects in DFTK and `collinear` whether
the atoms mark a case of collinear spin or not. Notice that if `collinear` is false
then `spins` is garbage.
"""
function spglib_atoms(atoms, magnetic_moments=[])
    n_attypes = isempty(atoms) ? 0 : sum(length(positions) for (typ, positions) in atoms)
    spg_numbers = Vector{Cint}(undef, n_attypes)
    spg_spins = Vector{Cdouble}(undef, n_attypes)
    spg_positions = Matrix{Cdouble}(undef, 3, n_attypes)

    arbitrary_spin = false
    offset = 0
    nextnumber = 1
    mapping = Dict{Int, Any}()
    for (iatom, (el, positions)) in enumerate(atoms)
        mapping[nextnumber] = el

        # Default to zero magnetic moment unless this is a case of collinear magnetism
        for (ipos, pos) in enumerate(positions)
            # assign the same number to all elements with this position
            spg_numbers[offset + ipos] = nextnumber
            spg_positions[:, offset + ipos] .= pos

            if !isempty(magnetic_moments)
                magmom = magnetic_moments[iatom][2][ipos]
                spg_spins[offset + ipos] = magmom[3]
                !iszero(magmom[1:2]) && (arbitrary_spin = true)
            end
        end
        offset += length(positions)
        nextnumber += 1
    end

    collinear = !isempty(magnetic_moments) && !arbitrary_spin && !all(iszero, spg_spins)
    (positions=spg_positions, numbers=spg_numbers, spins=spg_spins,
     mapping=mapping, collinear=collinear)
end


function spglib_get_symmetry(lattice, atoms, magnetic_moments=[]; tol_symmetry=1e-5)
    lattice = Matrix{Float64}(lattice)  # spglib operates in double precision

    if isempty(atoms)
        # spglib doesn't like no atoms, so we default to
        # no symmetries (even though there are lots)
        return [Mat3{Int}(I)], [Vec3(zeros(3))]
    end

    # Ask spglib for symmetry operations and for irreducible mesh
    spg_positions, spg_numbers, spg_spins, _, collinear = spglib_atoms(atoms, magnetic_moments)

    max_ops = 384  # Maximal number of symmetry operations spglib searches for
    spg_rotations    = Array{Cint}(undef, 3, 3, max_ops)
    spg_translations = Array{Cdouble}(undef, 3, max_ops)
    if collinear
        spg_equivalent_atoms = Array{Cint}(undef, max_ops)
        spg_n_ops = ccall((:spg_get_symmetry_with_collinear_spin, SPGLIB), Cint,
                          (Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Cint, Ptr{Cdouble},
                           Ptr{Cdouble}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cdouble),
                          spg_rotations, spg_translations, spg_equivalent_atoms, max_ops, copy(lattice'),
                          spg_positions, spg_numbers, spg_spins, Cint(length(spg_numbers)), tol_symmetry)
    else
        spg_n_ops = ccall((:spg_get_symmetry, SPGLIB), Cint,
            (Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Cint, Cdouble),
            spg_rotations, spg_translations, max_ops, copy(lattice'), spg_positions, spg_numbers,
            Cint(length(spg_numbers)), tol_symmetry)
    end

    # If spglib does not find symmetries give an error
    if spg_n_ops == 0
        err_message = spglib_get_error_message()
        error("spglib failed to get the symmetries. Check your lattice, use a " *
              "uniform BZ mesh or disable symmetries. Spglib reported : " * err_message)
    end

    # Note: Transposes are performed to convert between spglib row-major to julia column-major
    Stildes = [Mat3{Int}(spg_rotations[:, :, i]') for i in 1:spg_n_ops]
    τtildes = [rationalize.(Vec3{Float64}(spg_translations[:, i]), tol=tol_symmetry)
               for i in 1:spg_n_ops]

    # Checks: (A Stilde A^{-1}) is unitary
    for Stilde in Stildes
        Scart = lattice * Stilde * inv(lattice)  # Form S in cartesian coords
        if maximum(abs, Scart'Scart - I) > tol_symmetry
            error("spglib returned bad symmetries: Non-unitary rotation matrix.")
        end
    end

    # Check (Stilde, τtilde) maps atoms to equivalent atoms in the lattice
    for (Stilde, τtilde) in zip(Stildes, τtildes)
        for (elem, positions) in atoms
            for coord in positions
                diffs = [rationalize.(Stilde * coord + τtilde - pos, tol=5*tol_symmetry)
                         for pos in positions]

                # If all elements of a difference in diffs is integer, then
                # Stilde * coord + τtilde and pos are equivalent lattice positions
                if !any(all(isinteger, d) for d in diffs)
                    error("spglib returned bad symmetries: Cannot map the atom at position " *
                          "$coord to another atom of the same element under the symmetry " *
                          "operation (Stilde, τtilde):\n" *
                          "($Stilde, $τtilde)")
                end
            end
        end
    end

    Stildes, τtildes
end

# function spglib_standardize_cell(lattice::AbstractArray{T}, atoms; correct_symmetry=true,
#                                  primitive=false, tol_symmetry=1e-5) where {T}
#     # Convert lattice and atoms to spglib and keep the mapping between our atoms
#     spg_lattice = copy(Matrix{Float64}(lattice)')
#     # and spglibs atoms
#     spg_positions, spg_numbers, spg_spins, atommapping = spglib_atoms(atoms)

#     # Ask spglib to standardize the cell (i.e. find a cell, which fits the spglib conventions)
#     num_atoms = ccall((:spg_standardize_cell, SPGLIB), Cint,
#       (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Cint, Cint, Cint, Cdouble),
#       spg_lattice, spg_positions, spg_numbers, length(spg_numbers), Cint(primitive),
#       Cint(!correct_symmetry), tol_symmetry)
#     spg_lattice = copy(spg_lattice')

#     newatoms = [(atommapping[iatom]
#                  => T.(spg_positions[findall(isequal(iatom), spg_numbers), :]))
#                 for iatom in unique(spg_numbers)]
#     Matrix{T}(spg_lattice), newatoms
# end

function spglib_get_stabilized_reciprocal_mesh(kgrid_size, rotations::Vector;
                                               is_shift=Vec3(0, 0, 0),
                                               is_time_reversal=false,
                                               qpoints=[Vec3(0.0, 0.0, 0.0)],
                                               isdense=false)
    spg_rotations = cat([copy(Cint.(S')) for S in rotations]..., dims=3)
    nkpt = prod(kgrid_size)
    mapping = Vector{Cint}(undef, nkpt)
    grid_address = Matrix{Cint}(undef, 3, nkpt)

    nrot = length(rotations)
    n_kpts = ccall((:spg_get_stabilized_reciprocal_mesh, SPGLIB), Cint,
      (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}),
      grid_address, mapping, [Cint.(kgrid_size)...], [Cint.(is_shift)...], Cint(is_time_reversal),
      Cint(nrot), spg_rotations, Cint(length(qpoints)), Vec3{Float64}.(qpoints))

    return n_kpts, Int.(mapping), [Vec3{Int}(grid_address[:, i]) for i in 1:nkpt]
end


"""
    symmetry_operations(lattice, atoms, magnetic_moments=[]; tol_symmetry=1e-5)
Compute the spatial symmetry operations of the system by calling spglib.
`atoms` should follow the format `Vector{Pair{String, Vector{Vector{Float64}}}}`.
String is an indicator for atom types. The Vector part is the list of atom positions in
the crystal coordinates.
"""
function symmetry_operations(lattice, atoms, magnetic_moments=[]; tol_symmetry=1e-5)
    symmetries = Vector{SymOp}()
    # Get symmetries from spglib
    Stildes, τtildes = spglib_get_symmetry(lattice, atoms, magnetic_moments;
                                           tol_symmetry=tol_symmetry)

    for isym = 1:length(Stildes)
        S = Stildes[isym]'                  # in fractional reciprocal coordinates
        τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
        τ = τ .- floor.(τ)
        @assert all(0 .≤ τ .< 1)
        push!(symmetries, (S, τ))
    end

    unique(symmetries)
end

"""
Implements a primitive search to find an irreducible subset of kpoints
amongst the provided kpoints.
"""
function find_irreducible_kpoints(kcoords, Stildes, τtildes)

    # This function is required because spglib sometimes flags kpoints
    # as reducible, where we cannot find a symmetry operation to
    # generate them from the provided irreducible kpoints. This
    # reimplements that part of spglib, with a possibly very slow
    # algorithm.

    # Flag which kpoints have already been mapped to another irred.
    # kpoint or which have been decided to be irreducible.
    kcoords_mapped = zeros(Bool, length(kcoords))
    kirreds = empty(kcoords)           # Container for irreducible kpoints
    ksymops = Vector{Vector{SymOp}}()  # Corresponding symops

    while !all(kcoords_mapped)
        # Select next not mapped kpoint as irreducible
        ik = findfirst(isequal(false), kcoords_mapped)
        push!(kirreds, kcoords[ik])
        thisk_symops = [identity_symop()]
        kcoords_mapped[ik] = true

        for jk in findall(.!kcoords_mapped)
            isym = findfirst(1:length(Stildes)) do isym
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                all(isinteger, kcoords[jk] - (Stildes[isym]' * kcoords[ik]))
            end

            if !isnothing(isym)  # Found a reducible kpoint
                kcoords_mapped[jk] = true
                S = Stildes[isym]'                  # in fractional reciprocal coordinates
                τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
                τ = τ .- floor.(τ)
                @assert all(0 .≤ τ .< 1)
                push!(thisk_symops, (S, τ))
            end
        end  # jk

        push!(ksymops, thisk_symops)
    end
    kirreds, ksymops
end

"""Bring kpoint coordinates into the range [0.0, 1.0)"""
function normalize_kpoint_coordinate(x::Real)
    x = x - floor(Int, x)
    @assert 0.0 ≤ x < 1.0
    x
end
normalize_kpoint_coordinate(k::AbstractVector) = normalize_kpoint_coordinate.(k)


@doc raw"""
     bzmesh_ir_wedge(kgrid_size, symmetries; tol_symmetry=1e-6)
Construct the irreducible wedge of a uniform Gamma-centered Brillouin zone mesh for sampling
``k``-Points. The function returns a `Kpoints` object.
"""
function bzmesh_ir_wedge(kgrid_size, symmetries; time_reversal=false, tol_symmetry=1e-6)
    itrevs = time_reversal ? (1, -1) : (1,)
    ntrev = length(itrevs)
    nsym = length(symmetries)

    weight_irr_int = Vector{Int}()
    k_irr = Vector{Vec3{Float64}}()
    found = zeros(Bool, kgrid_size...)

    # kz = i3 / kgrid_size[3] is the fastest index
    for (i3, i2, i1) in Iterators.product((0:n-1 for n in reverse(kgrid_size))...)
        # check if this k-point has already been found equivalent to another. If so, skip.
        found[i1+1, i2+1, i3+1] && continue

        k = Vec3{Rational{Int}}((i1, i2, i3) .// kgrid_size)

        # Check if there are equivalent k-point to the remaining k points
        # Also, count the number of symops that map k to itself.
        nsym_star = 0
        for symop in symmetries
            for itrev in itrevs
                Sk = mod.(itrev * symop[1] * k, 1)
                if Sk > k
                    i1, i2, i3 = Int.(Sk.data .* kgrid_size) .+ 1
                    found[i1, i2, i3] = true
                elseif Sk == k
                    nsym_star += 1
                else
                    # If Sk is not in the later k points and not itself, i) the grid breaks the
                    # symmetry, or ii) symmetry search has a problem.
                    error("Problem in symmetry of the irreducible k points")
                end
            end
        end
        push!(k_irr, k)
        push!(weight_irr_int, nsym * ntrev / nsym_star)
    end
    @assert sum(weight_irr_int) == prod(kgrid_size)
    weight_irr = weight_irr_int / prod(kgrid_size)
    k_irr, weight_irr
    Kpoints{Float64}(length(k_irr), k_irr, weight_irr, kgrid_size)
end