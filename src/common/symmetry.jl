# Adapted from DFTK.jl external/spglib.jl and symmetry.jl
# TODO: Time-reversal symmetry

# We follow the DFTK convention.
# The symmetry operations (S, τ) are reciprocal-space operations.
# The corresponding real-space operation r -> W * r + w satisfies S = W' and τ = -W^-1 w.
# See https://juliamolsim.github.io/DFTK.jl/dev/developer/symmetries for details.

using spglib_jll
using Spglib
using StaticArrays
using LinearAlgebra

export Symmetry
export symmetry_operations
export symmetry_is_subset
export symmetrize
export symmetrize_array
export symmetrize_array!

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
    # spg_positions = Matrix{Cdouble}(undef, 3, n_attypes)
    spg_positions = Vec3{Float64}[]

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
            # spg_positions[:, offset + ipos] .= pos
            push!(spg_positions, pos)

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

# Convert atom_pos_crys and atom_labels to Spglib atom data structure
function spglib_atoms(atom_pos_crys::Vector{T}, atom_labels::Vector{String}, magnetic_moments=[]) where T <: AbstractVector{FT} where FT
    atoms_dict = Dict{String, Vector{Vector{FT}}}()
    for (pos, label) in zip(atom_pos_crys, atom_labels)
        if haskey(atoms_dict, label)
            push!(atoms_dict[label], pos)
        else
            atoms_dict[label] = [pos]
        end
    end
    atoms = collect(atoms_dict)
    spglib_atoms(atoms, magnetic_moments)
end

"""
Returns crystallographic conventional cell according to the International Table of
Crystallography Vol A (ITA) in case `to_primitive=false`. If `to_primitive=true`
the primitive lattice is returned in the convention of the reference work of
Cracknell, Davies, Miller, and Love (CDML). Of note this has minor differences to
the primitive setting choice made in the ITA.
"""
function get_spglib_lattice(model; to_primitive=false)
    # TODO This drops magnetic moments!
    # TODO For time-reversal symmetry see the discussion in PR 496.
    #      https://github.com/JuliaMolSim/DFTK.jl/pull/496/files#r725203554
    #      Essentially this does not influence the standardisation,
    #      but it only influences the kpath.
    spg_positions, spg_numbers, _ = spglib_atoms(atom_pos_crys(model), model.atom_labels)
    structure = Spglib.Cell(model.lattice, spg_positions, spg_numbers)
    Matrix(Spglib.standardize_cell(structure; to_primitive).lattice)
end

function spglib_spacegroup_number(model)
    # Get spacegroup number according to International Tables for Crystallography (ITA)
    # TODO Time-reversal symmetry disabled? (not yet available in DFTK)
    # TODO Are magnetic moments passed?
    spg_positions, spg_numbers, _ = spglib_atoms(atom_pos_crys(model), model.atom_labels)
    structure = Spglib.Cell(model.lattice, spg_positions, spg_numbers)
    Spglib.get_dataset(structure).spacegroup_number
end

atom_pos_crys(model) = [inv(model.lattice) * (pos .* model.alat) for pos in model.atom_pos]

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

struct SymOp{FT}
    S::Mat3{Int}
    τ::Vec3{FT}
    Scart::Mat3{FT}
    τcart::Vec3{FT}
    is_inv::Bool
    is_tr::Bool
end

function Base.isapprox(s1::SymOp, s2::SymOp)
    s1.S == s2.S || return false
    for (τ1, τ2) in zip(s1.τ, s2.τ)
        τ1 - τ2 ≈ round.(Int, τ1 - τ2) || return false
    end
    s1.Scart ≈ s2.Scart || return false
    # τcart cannot be checked because it can differ by a lattice vector.
    # But we check τ so the result should be fine.
    s1.is_inv == s2.is_inv || return false
    s1.is_tr == s2.is_tr || return false
    return true
end
Base.one(::Type{SymOp{FT}}) where FT = SymOp(Mat3{Int}(I), zeros(Vec3{FT}), Mat3{FT}(I), zeros(Vec3{FT}), false, false)
Base.one(::T) where {T <: SymOp} = one(T)
Base.isone(s::T) where {T <: SymOp} = isapprox(s, one(T))

function Base.:*(op1::SymOp, op2::SymOp)
    S = op1.S * op2.S
    τ = op1.τ + op1.S' \ op2.τ
    Scart = op1.Scart * op2.Scart
    τcart = op1.τcart + op1.Scart' \ op2.τcart
    is_inv = xor(op1.is_inv, op2.is_inv)
    is_tr = xor(op1.is_tr, op2.is_tr)
    SymOp(S, τ, Scart, τcart, is_inv, is_tr)
end
Base.inv(op) = SymOp(Int.(inv(op.S)), -op.S'*op.τ, inv(op.Scart), -op.Scart'*op.τcart, op.is_inv, op.is_tr)

"""Symmetry operations"""
struct Symmetry{FT}
    "Number of symmetry operations. Maximum 96 (because of time reversal)."
    nsym::Int
    "Rotation matrix in reciprocal crystal coordinates"
    S::Vector{Mat3{Int}}
    "Fractional translation in real-space crystal coordinates"
    τ::Vector{Vec3{FT}}
    "Rotation matrix in reciprocal Cartesian coordinates"
    Scart::Vector{Mat3{FT}}
    "Fractional translation in real-space Cartesian coordinates (alat units)"
    τcart::Vector{Vec3{FT}}
    "true if the symmetry operation includes inversion (i.e. is an improper rotation)"
    is_inv::Vector{Bool}
    "true if the symmetry operation includes time reversal"
    is_tr::Vector{Bool}
    "true if the system have time reversal symmetry"
    time_reversal::Bool
end

function Symmetry(Ss_, τs_, time_reversal, lattice_)
    # FIXME: Complicated magnetic symmetry operations not implemented. Only grey groups
    # (time_reversal = true) or colorless groups (time_reversal = false)
    lattice = Mat3(lattice_)
    Ss = deepcopy(Ss_)
    τs = deepcopy(τs_)
    nsym = length(Ss)
    Scarts = [inv(lattice') * S * lattice' for S in Ss]
    τcarts = [lattice * τ for τ in τs]
    is_inv = [det(S) < 0 ? true : false for S in Ss]
    is_tr = [false for _ in 1:nsym]
    if time_reversal
        # Add TR * S for each spatial symmetry S.
        append!(Ss, Ss)
        append!(τs, τs)
        append!(Scarts, Scarts)
        append!(τcarts, τcarts)
        append!(is_inv, is_inv)
        append!(is_tr, [true for _ in 1:nsym])
        nsym *= 2
    end
    Symmetry(nsym, Ss, τs, Scarts, τcarts, is_inv, is_tr, time_reversal)
end

function Base.show(io::IO, obj::Symmetry)
    print(io, typeof(obj), "(nsym=$(obj.nsym))")
end

function Base.getindex(sym::Symmetry, i)
    1 <= i <= sym.nsym || throw(BoundsError(sym, i))
    SymOp(sym.S[i], sym.τ[i], sym.Scart[i], sym.τcart[i], sym.is_inv[i], sym.is_tr[i])
end
Base.firstindex(sym::Symmetry) = 1
Base.lastindex(sym::Symmetry) = sym.nsym

Base.iterate(sym::Symmetry, state=1) = state > sym.nsym ? nothing : (sym[state], state+1)
Base.length(sym::Symmetry) = sym.nsym
Base.keys(sym::Symmetry) = LinearIndices(1:sym.nsym)

"""Create symmetry object containing only identity"""
function identity_symmetry(::Type{FT}=Float64) where FT
    Symmetry(1, [Mat3{Int}(I)], [zeros(Vec3{FT})], [Mat3{FT}(I)], [zeros(Vec3{FT})], [false], [false], false)
end

# Check whether sym1 is a subset of sym2
function symmetry_is_subset(sym1, sym2)
    for s1 in sym1
        # Look for s1 in sym2. If not present, return false.
        any(s1 ≈ s2 for s2 in sym2) || return false
    end
    return true
end

function check_group(symops)
    is_approx_in_symops(s1) = any(s -> isapprox(s, s1), symops)
    is_approx_in_symops(one(symops[1])) || error("check_group: no identity element")
    for s in symops
        if !is_approx_in_symops(inv(s))
            error("check_group: symop $s with inverse $(inv(s)) is not in the group")
        end
        for s2 in symops
            if !is_approx_in_symops(s*s2) || !is_approx_in_symops(s2*s)
                error("check_group: product is not stable")
            end
        end
    end
    symops
end

"""
    symmetry_operations(lattice, atoms, magnetic_moments=[]; tol_symmetry=1e-5)
Compute the spatial symmetry operations of the system by calling spglib.
`atoms` should follow the format `Vector{Pair{String, Vector{Vector{Float64}}}}`.
String is an indicator for atom types. The Vector part is the list of atom positions in
the crystal coordinates.
"""
function symmetry_operations(lattice, atoms, magnetic_moments=[]; tol_symmetry=1e-5)
    # FIXME: is noncollinear symmetry implemented?
    Ss = Vector{Mat3{Int}}()
    τs = Vector{Vec3{Float64}}()
    # Get symmetries from spglib
    Ws, ws = spglib_get_symmetry(lattice, atoms, magnetic_moments;
                                           tol_symmetry=tol_symmetry)

    for isym = 1:length(Ws)
        S = Ws[isym]'             # in fractional reciprocal coordinates
        τ = -Ws[isym] \ ws[isym]  # in fractional real-space coordinates
        τ = τ .- floor.(τ)
        @assert all(0 .≤ τ .< 1)
        push!(Ss, S)
        push!(τs, τ)
    end
    time_reversal = magnetic_moments == [] ? true : false
    Symmetry(Ss, τs, time_reversal, lattice)
end

# """
# Implements a primitive search to find an irreducible subset of kpoints
# amongst the provided kpoints.
# """
# function find_irreducible_kpoints(kcoords, Stildes, τtildes)

#     # This function is required because spglib sometimes flags kpoints
#     # as reducible, where we cannot find a symmetry operation to
#     # generate them from the provided irreducible kpoints. This
#     # reimplements that part of spglib, with a possibly very slow
#     # algorithm.

#     # Flag which kpoints have already been mapped to another irred.
#     # kpoint or which have been decided to be irreducible.
#     kcoords_mapped = zeros(Bool, length(kcoords))
#     kirreds = empty(kcoords)           # Container for irreducible kpoints
#     ksymops = Vector{Vector{SymOp}}()  # Corresponding symops

#     while !all(kcoords_mapped)
#         # Select next not mapped kpoint as irreducible
#         ik = findfirst(isequal(false), kcoords_mapped)
#         push!(kirreds, kcoords[ik])
#         thisk_symops = [identity_symop()]
#         kcoords_mapped[ik] = true

#         for jk in findall(.!kcoords_mapped)
#             isym = findfirst(1:length(Stildes)) do isym
#                 # If the difference between kred and Stilde' * k == Stilde^{-1} * k
#                 # is only integer in fractional reciprocal-space coordinates, then
#                 # kred and S' * k are equivalent k-Points
#                 all(isinteger, kcoords[jk] - (Stildes[isym]' * kcoords[ik]))
#             end

#             if !isnothing(isym)  # Found a reducible kpoint
#                 kcoords_mapped[jk] = true
#                 S = Stildes[isym]'                  # in fractional reciprocal coordinates
#                 τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
#                 τ = τ .- floor.(τ)
#                 @assert all(0 .≤ τ .< 1)
#                 push!(thisk_symops, (S, τ))
#             end
#         end  # jk

#         push!(ksymops, thisk_symops)
#     end
#     kirreds, ksymops
# end

"""Bring kpoint coordinates into the range [0.0, 1.0)"""
function normalize_kpoint_coordinate(x::Real)
    x = x - floor(Int, x)
    x = x - floor(Int, x) # Single line can fail if x = -1e-18
    @assert 0.0 ≤ x < 1.0
    x
end
normalize_kpoint_coordinate(k::AbstractVector) = normalize_kpoint_coordinate.(k)


"""Bring kpoint coordinates into the range [-0.5, 0.5)"""
function normalize_kpoint_coordinate_centered(x::Real)
    normalize_kpoint_coordinate(x + 0.5) .- 0.5
end
normalize_kpoint_coordinate_centered(k::AbstractVector) = normalize_kpoint_coordinate_centered.(k)


function bzmesh_ir_wedge(ngrid, symmetry::Symmetry; ignore_time_reversal=false)
    Base.depwarn("Renamed. Use kpoints_grid instead", :bzmesh_ir_wedge)
    kpoints_grid_symmetry(ngrid, symmetry; ignore_time_reversal)
end

"""
    kpoints_grid(grid, symmetry::Symmetry; ignore_time_reversal=false)
Construct the irreducible wedge of a uniform Brillouin zone mesh for sampling ``k``-Points.
The mesh includes the Gamma point. Returns a `Kpoints` object.
- `ignore_time_reversal`: If true, ignore all symmetries involving time reversal.
"""
function kpoints_grid_symmetry(ngrid, symmetry::Symmetry; ignore_time_reversal=false)
    nsym_used = ignore_time_reversal ? sum(.!symmetry.is_tr) : symmetry.nsym
    weight_irr_int = Vector{Int}()
    k_irr = Vector{Vec3{Float64}}()
    found = zeros(Bool, ngrid...)

    # kz = i3 / ngrid[3] is the fastest index
    for (i3, i2, i1) in Iterators.product((0:n-1 for n in reverse(ngrid))...)
        # check if this k-point has already been found equivalent to another. If so, skip.
        found[i1+1, i2+1, i3+1] && continue

        k = Vec3{Rational{Int}}((i1, i2, i3) .// ngrid)

        # Check if there are equivalent k-point to the remaining k points
        # Also, count the number of symops that map k to itself.
        nsym_star = 0
        for (S, is_tr) in zip(symmetry.S, symmetry.is_tr)
            if ignore_time_reversal && is_tr
                continue
            end
            Sk = is_tr ? mod.(-S * k, 1) : mod.(S * k, 1)
            if Sk > k
                i1, i2, i3 = Int.(Sk.data .* ngrid) .+ 1
                found[i1, i2, i3] = true
            elseif Sk == k
                nsym_star += 1
            else
                # If Sk is not in the later k points and not itself, i) the grid breaks the
                # symmetry, or ii) symmetry search has a problem.
                error("Problem in symmetry of the irreducible k points")
            end
        end
        push!(k_irr, k)
        push!(weight_irr_int, nsym_used / nsym_star)
    end
    @assert sum(weight_irr_int) == prod(ngrid)
    weight_irr = weight_irr_int / prod(ngrid)
    k_irr, weight_irr
    GridKpoints(Kpoints{Float64}(length(k_irr), k_irr, weight_irr, ngrid))
end


# Symmetrization of tensors

"""
    symmetrize(tensor::StaticArray, symmetry::Symmetry; tr_odd=false, axial=false)
Symmetrize a tensor by applying all the symmetry and forming the average.
For array of StaticArrays, use `symmetrize.(arr, Ref(symmetry))`.
- `tr_odd`: true if the tensor is odd under time reversal.
- `axial`: true if the tensor is axial (a pseudotensor). Obtains additional -1 sign under inversion.
"""
symmetrize(arr, symmetry::Symmetry; tr_odd, axial) = error("Symmetrization Not implemented for this data type")

symmetrize(arr, symmetry::Nothing; tr_odd = false, axial = false) = arr

function symmetrize(scalar::Number, symmetry::Symmetry; tr_odd=false, axial=false)
    scalar_symm = scalar
    if tr_odd && any(symmetry.is_tr)
        scalar_symm = zero(scalar)
    end
    if axial && any(symmetry.is_inv)
        scalar_symm = zero(scalar)
    end
    scalar_symm
end

function symmetrize(vec::StaticVector, symmetry::Symmetry; tr_odd=false, axial=false)
    @assert size(vec) == (3,)
    vec_symm = zero(vec)
    for (S, is_inv, is_tr) in zip(symmetry.Scart, symmetry.is_inv, symmetry.is_tr)
        sign_coeff = 1
        if axial && is_inv
            sign_coeff *= -1
        end
        if tr_odd && is_tr
            sign_coeff *= -1
        end
        vec_symm = vec_symm + sign_coeff * S * vec
    end
    vec_symm = vec_symm / symmetry.nsym
    vec_symm
end

function symmetrize(mat::StaticMatrix, symmetry::Symmetry; tr_odd=false, axial=false)
    @assert size(mat) == (3, 3)
    mat_symm = zero(mat)
    for (S, is_inv, is_tr) in zip(symmetry.Scart, symmetry.is_inv, symmetry.is_tr)
        sign_coeff = 1
        if axial && is_inv
            sign_coeff *= -1
        end
        if tr_odd && is_tr
            sign_coeff *= -1
        end
        mat_symm = mat_symm + sign_coeff * S' * mat * S
    end
    mat_symm = mat_symm / symmetry.nsym
    mat_symm
end

"""
    symmetrize_array(arr::AbstractArray{T}, symmetry; order, tr_odd=false, axial=false)
Symmetrize a tensor by applying all the symmetry and forming the average.
# Inputs
- `order`: The order of the tensor. The first `order` indices of `arr` are the tensor indices. The remaining indices denote multiple tensors.
- `tr_odd`: true if the tensor is odd under time reversal.
- `axial`: true if the tensor is axial (a pseudotensor). Obtains additional -1 sign under inversion.
"""
function symmetrize_array(arr::AbstractArray{T}, symmetry; order, tr_odd=false, axial=false) where {T}
    @assert all(size(arr)[1:order] .== 3)
    arr_sym = zero(arr)
    @views if order == 0
        for i in eachindex(arr)
            arr_sym[i] = symmetrize(arr[i], symmetry; tr_odd, axial)
        end
    elseif order == 1
        for ind in CartesianIndices(size(arr)[order+1:end])
            arr_sym[:, ind] .= symmetrize(SVector{3}(arr[:, ind]), symmetry; tr_odd, axial)
        end
    elseif order == 2
        for ind in CartesianIndices(size(arr)[order+1:end])
            arr_sym[:, :, ind] .= symmetrize(SMatrix{3, 3}(arr[:, :, ind]), symmetry; tr_odd, axial)
        end
    else
        error("Order $order not implemented")
    end
    arr_sym
end

symmetrize_array(arr::AbstractArray, symmetry::Nothing; kwargs...) = arr

function symmetrize_array!(arr::AbstractArray, symmetry; kwargs...)
    arr_sym = symmetrize_array(arr, symmetry; kwargs...)
    arr .= arr_sym
end

# TODO: use == instead of isapprox for symop?
# TODO: Implement == (or isapprox) for Symmetry, and use it in test_hdf.jl



"""
    apply_symop(S :: SymOp, k :: Vec3, mode :: Symbol) -> Sk :: Vec3
Transform `k` (in reduced coordiantes by default) according to `S`.
- `mode == :position` or `:position_cartesian`: polar, time-reversal even (e.g. position)
- `mode == :momentum` or `:momentum_cartesian`: polar, time-reversal odd (e.g. momentum, velocity)
"""
@inline function apply_symop(S :: SymOp, k :: Vec3, mode :: Symbol, tr_apply_conj = true)
    # NOTE: Making tr_apply_conj a keyword argument would be nicer, but it is slow.
    if mode === :position
        # r -> W * r + w where S = W' and τ = -W^-1 w.
        return S.S' * (k - S.τ)
    elseif mode === :position_cartesian
        return S.Scart' * (k - S.τcart)
    elseif mode === :momentum
        Sk = S.S * k
        return S.is_tr ? (tr_apply_conj ? conj(-Sk) : -Sk) : Sk
    elseif mode === :momentum_cartesian
        Sk = S.Scart * k
        return S.is_tr ? (tr_apply_conj ? conj(-Sk) : -Sk) : Sk
    else
        throw(ArgumentError("Wrong mode $mode"))
    end
end
