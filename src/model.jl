
using FortranFiles
using Mmap

# Symmetry convention of ElectronPhonon.jl and Quantum ESPRESSO
# In QE, the symmetry operation is r -> r * S_QE - τ_QE.
# In our convention (we follow DFTK), we have r -> W * r + w, so W = S_QE^T, w = -τ_QE.
# Then, we store symmetries in reciprocal space, so we have
# S = W' = S_QE
# τ = -W^-1 * w = S_QE^T^-1 * τ_QE.

fortran_read_bool(f) = Bool(abs(read(f, Int32))) # In fortran file, 0 is false, -1 or +1 is true

struct SymmetryOperators{FT <: AbstractFloat, WannType <: AbstractWannierObject{FT}}
    symmetry::Symmetry{FT}
    operators::Vector{WannType}
end

"Tight-binding model for electron, phonon, and electron-phonon coupling.
All data is in coarse real-space grid."
Base.@kwdef mutable struct Model{FT <: AbstractFloat, WannType <: Union{Nothing, AbstractWannierObject{FT}}}
    # Crystal structure: lattice, atoms and `dimension`. `alat`, `lattice`, `recip_lattice`,
    # `volume`, `atom_pos`, `atom_labels` and `dimension` are read through it, e.g. `model.alat`.
    structure::Structure

    # Symmetry group of the Hamiltonian, used for k-reduction. It is a subgroup of
    # `symmetry_operations(structure)`, the group of the lattice and atoms.
    symmetry::Symmetry{FT}

    # Phonon mass in Rydberg units, one entry per phonon mode (length `nmodes`).
    mass::Vector{FT}

    nw::Int
    nmodes::Int

    # Wannier centers in Cartesian coordinates, bohr units
    wann_centers :: Vector{Vec3{FT}}

    # Long-range term in polar systems
    use_polar_dipole::Bool
    polar_phonon::Polar
    polar_eph::Polar

    el_ham::HostWannierObject{FT} # ELectron Hamiltonian
    # TODO: Use Hermiticity of hk

    el_ham_R::HostWannierObject{FT} # Electron Hamiltonian times R
    el_pos::HostWannierObject{FT} # Electron position (dipole)
    el_vel::Union{HostWannierObject{FT},Nothing} # ELectron velocity matrix
    el_velocity_mode::Symbol # :Direct or :BerryConnection. Default: :Direct
    # Two different ways to compute the velocity matrix.
    # If :Direct, use direct interpolation of dH/dk matrix elements (stored in el_vel).
    # If :BerryConnection, compute the Berry connection using el_pos and use Eq. (31) of
    # X. Wang et al, PRB 74 195118 (2006).
    # el_velocity_mode = :Direct          corresponds to vme = "dipole"  of
    # el_velocity_mode = :BerryConnection corresponds to vme = "wannier" of

    ph_dyn::HostWannierObject{FT} # Phonon dynamical matrix
    ph_dyn_R::HostWannierObject{FT} # Phonon dynamical matrix
    # TODO: Use real-valuedness of dyn_r
    # TODO: Use Hermiticity of dyn_q

    # electron-phonon coupling matrix in electron and phonon Wannier representation
    epmat::WannType
    # The crystal momentum that the outer R index of epmat couples. "k" or "q".
    epmat_outer_momentum::String
    # epmat_outer_momentum == :el : epmat.op_r is indexed as (i, j, nmodes, Rₚ, Rₑ)
    # epmat_outer_momentum == :ph : epmat.op_r is indexed as (i, j, nmodes, Rₑ, Rₚ)

    # Symmetry operators for Wannier functions
    el_sym::Union{SymmetryOperators{FT, HostWannierObject{FT}}, Nothing} = nothing
end

# Fields of `structure` read as `model.x`. Literal names fold to a single `getfield` chain.
@inline function Base.getproperty(model::Model, name::Symbol)
    if name === :alat || name === :lattice || name === :recip_lattice || name === :volume ||
       name === :atom_pos || name === :atom_labels || name === :dimension
        getfield(getfield(model, :structure), name)
    else
        getfield(model, name)
    end
end

Base.propertynames(::Model, private::Bool = false) =
    (fieldnames(Model)..., :alat, :lattice, :recip_lattice, :volume, :atom_pos, :atom_labels, :dimension)

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    println(io, typeof(model))
    println(io, "  structure: ", length(model.atom_pos), " atoms, dimension = ", model.dimension)
    println(io, "  nw = ", model.nw, ", nmodes = ", model.nmodes, ", nsym = ", model.symmetry.nsym)
    println(io, "  polar: ", model.polar_phonon.use ? "yes" : "no")
    print(io, "  epmat: ", model.epmat === nothing ? "none" : "outer momentum \"$(model.epmat_outer_momentum)\"")
end

function load_symmetry_operators_from_epw(folder)
    f = FortranFile(joinpath(folder, "epw_data_julia_symmetry.bin"), "r")

    alat = read(f, Float64)
    at_in_alat = read(f, (Float64, 3, 3))
    lattice = Mat3(at_in_alat .* alat)

    nsym = Int(read(f, Int32))
    symmetry_S = Int.(read(f, (Int32, 3, 3, nsym)))
    symmetry_τ = read(f, (Float64, 3, nsym))
    time_reversal = fortran_read_bool(f)

    # Symmetry conversion: see comment in the header.
    Ss = [Mat3(symmetry_S[:, :, i]) for i in 1:nsym]
    τs = [Ss[i]' \ Vec3(symmetry_τ[:, i]) for i in 1:nsym]

    symmetry = Symmetry(Ss, τs, time_reversal, lattice)

    nw = Int(read(f, Int32))
    nrr = Int(read(f, Int32))
    irvec_raw = convert.(Int, read(f, (Int32, 3, nrr)))
    irvec = reinterpret(Vec3{Int}, irvec_raw)[:]
    ind_el = sortperm(irvec, by=x->reverse(x))
    irvec = irvec[ind_el]

    operators = Vector{HostWannierObject{Float64}}()
    @views for isym = 1:symmetry.nsym
        symmetry_mel = read(f, (ComplexF64, nw, nw, nrr))
        symmetry_mel .= symmetry_mel[:, :, ind_el]
        push!(operators, WannierObject(irvec, reshape(symmetry_mel, (nw*nw, nrr))))
    end
    close(f)

    @assert symmetry.nsym == length(operators)

    SymmetryOperators(symmetry, operators)
end

function Base.show(io::IO, sym::SymmetryOperators)
    print(io, "$(typeof(sym)) with $(sym.symmetry.nsym) symmetries")
end



"""
    epmat_impose_hermiticity(model)

Impose Hermiticity of e-ph matrix elements: ``g_ij(Re, Rp) = g*_ji(-Re, Rp-Re)`` by
averaging the two terms.

In the momentum representation, this implies
``
g(k+q, -q)† = e(-i (k+q) Re) e(i q Rp) g(Re, Rp)†
            = e(-i k Re) e(i q (Rp - Re)) g(Re, Rp)†
            = e(i k Re) e(i q Rp) g(-Re, Rp-Re)†
            = e(i k Re) e(i q Rp) g(Re, Rp)
            = g(k, q)
``

Hermiticity is broken in the current implementation of EPW due to the use of asymmetric
Wigner-Seitz selection for e-ph matrix elements. We recover it by averaging epmat with its
Hermitian conjugate.
Hermiticity is not related to crystal symmetry; it should hold for Wannier functions
that break the crystal symmetry.
"""
function epmat_impose_hermiticity(model)
    epmat_impose_hermiticity(model.epmat, model.nw, model.nmodes, model.epmat_outer_momentum)
end

function epmat_impose_hermiticity(epmat, nw, nmodes, epmat_outer_momentum)
    @assert epmat_outer_momentum == "el"
    irvec_Re = epmat.irvec
    irvec_Rp = epmat.irvec_next

    g = Base.ReshapedArray(epmat.op_r, (nw, nw, nmodes, length(irvec_Rp), length(irvec_Re)), ());

    _dagger(A) = conj.(permutedims(A, (2, 1, 3)))

    # for Re in irvec, Rp in irvec_next
    Re = Vec3(0, 0, 0)
    @info sum(g)
    @views for Re in irvec_Re, Rp in irvec_Rp
        iRe1 = findfirst(x -> x == Re, irvec_Re)
        iRp1 = findfirst(x -> x == Rp, irvec_Rp)
        iRe2 = findfirst(x -> x == -Re, irvec_Re)
        iRp2 = findfirst(x -> x == Rp-Re, irvec_Rp)

        if iRe2 === nothing || iRp2 === nothing
            g[:, :, :, iRp1, iRe1] .= 0
        else
            A = (g[:, :, :, iRp1, iRe1] .+ _dagger(g[:, :, :, iRp2, iRe2])) ./ 2
            g[:, :, :, iRp1, iRe1] .= A
            g[:, :, :, iRp2, iRe2] .= _dagger(A)
        end
    end
    @info sum(g)
end
