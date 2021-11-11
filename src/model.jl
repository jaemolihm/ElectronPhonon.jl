
using FortranFiles

export load_model

# TODO: Implement Base.show(model::ModelEPW)

fortran_read_bool(f) = Bool(abs(read(f, Int32))) # In fortran file, 0 is false, -1 or +1 is true

struct SymmetryOperators{FT <: AbstractFloat, WannType <: AbstractWannierObject{FT}}
    symmetry::Symmetry{FT}
    operators::Vector{WannType}
end

"Tight-binding model for electron, phonon, and electron-phonon coupling.
All data is in coarse real-space grid."
Base.@kwdef mutable struct ModelEPW{FT <: AbstractFloat, WannType <: AbstractWannierObject{FT}}
    # Lattice information
    alat::FT # Lattice parameter
    # Lattice vector in Bohr. lattice[:, i] is the i-th lattice vector.
    lattice::Mat3{FT}
    recip_lattice::Mat3{FT}
    volume::FT

    # Symmetries
    symmetry::Symmetry{FT}

    # Atom information
    mass::Vector{FT}
    atom_pos::Vector{Vec3{FT}} # Atom position in alat units, Cartesian coordinates
    atom_labels::Vector{String}

    nw::Int
    nmodes::Int

    # Long-range term in polar systems
    use_polar_dipole::Bool
    polar_phonon::Polar{FT}
    polar_eph::Polar{FT}

    el_ham::WannierObject{FT} # ELectron Hamiltonian
    # TODO: Use Hermiticity of hk

    el_ham_R::WannierObject{FT} # Electron Hamiltonian times R
    el_pos::WannierObject{FT} # Electron position (dipole)
    el_vel::Union{WannierObject{FT},Nothing} # ELectron velocity matrix

    ph_dyn::WannierObject{FT} # Phonon dynamical matrix
    ph_dyn_R::WannierObject{FT} # Phonon dynamical matrix
    # TODO: Use real-valuedness of dyn_r
    # TODO: Use Hermiticity of dyn_q

    # electron-phonon coupling matrix in electron and phonon Wannier representation
    epmat::WannType
    # The crystal momentum that the outer R index of epmat couples. "k" or "q".
    epmat_outer_momentum::String

    # Symmetry operators for Wannier functions
    el_sym::Union{SymmetryOperators{FT, WannierObject{FT}}, Nothing} = nothing
end

"Read file and create ModelEPW object in the MPI root.
Broadcast to all other processors."
function load_model(folder::String, epmat_on_disk::Bool=false, tmpdir=nothing;
        epmat_outer_momentum="ph", load_symmetry_operators=false)
    # Read model from file
    if mpi_initialized()
        # FIXME: Read only in the root core, and then bcast.
        # The implementation below breaks if epmat size is large. MPI bcast of large array
        # with sizeof(array) is greater than typemax(Cint) was not possible.
        model = load_model_from_epw(folder, epmat_on_disk, tmpdir; epmat_outer_momentum, load_symmetry_operators)
        # if mpi_isroot(EPW.mpi_world_comm())
        #     model = load_model_from_epw(folder, epmat_on_disk, tmpdir)
        # else
        #     model = nothing
        # end
        # # Broadcast to all processors
        # model = mpi_bcast(model, EPW.mpi_world_comm())
    else
        model = load_model_from_epw(folder, epmat_on_disk, tmpdir; epmat_outer_momentum, load_symmetry_operators)
    end
    model
end

"""
Arguments:
epmat_on_disk
    If true, write epmat to file and read at each get_fourier! call.
    If false, load epmat to memory.
tmpdir
    Directory to write temporary binary files for epmat_on_disk=false calse.
epmat_outer_momentum
    Outer momentum that model.epmat couples to. "ph" (default) or "el".
"""
function load_model_from_epw(folder::String, epmat_on_disk::Bool=false, tmpdir=nothing;
        epmat_outer_momentum, load_symmetry_operators)
    T = Float64

    if epmat_on_disk && tmpdir === nothing
        error("If epmat_on_disk is true, tmpdir must be provided.")
    end
    if epmat_outer_momentum ∉ ["ph", "el"]
        throw(ArgumentError("epmat_outer_momentum must be ph or el."))
    end

    # Read binary data written by EPW and create ModelEPW object
    f = FortranFile(joinpath(folder, "epw_data_julia.bin"), "r")

    # Structure parameters
    alat = read(f, Float64)
    at_in_alat = read(f, (Float64, 3, 3))
    natoms = Int(read(f, Int32))
    atom_pos_arr = read(f, (Float64, 3, natoms))
    atom_labels = String.(trim.(read(f, (FString{3}, natoms))))
    atom_pos = reinterpret(Vec3{T}, atom_pos_arr)[:]

    lattice = Mat3{T}(at_in_alat .* alat)
    recip_lattice = 2T(π) * inv(lattice')
    volume = abs(det(lattice))

    # Symmetries
    nsym = Int(read(f, Int32))
    symmetry_S = Int.(read(f, (Int32, 3, 3, nsym)))
    symmetry_τ = read(f, (Float64, 3, nsym))
    time_reversal = fortran_read_bool(f)
    Ss = [Mat3(symmetry_S[:, :, i]) for i in 1:nsym]
    τs = [Vec3(symmetry_τ[:, i]) for i in 1:nsym]
    symmetry = Symmetry(Ss, τs, time_reversal, lattice)

    # Wannier parameters
    nw = Int(read(f, Int32))
    nmodes = Int(read(f, Int32))
    mass = read(f, (Float64, nmodes))

    nkc = tuple(convert.(Int, read(f, (Int32, 3)))...)
    nqc = tuple(convert.(Int, read(f, (Int32, 3)))...)

    # Polar parameters
    use_polar_dipole = fortran_read_bool(f)

    if use_polar_dipole
        ϵ_arr = read(f, (Float64, 3, 3))
        ϵ = Mat3{T}(ϵ_arr)
        Z_arr = read(f, (Float64, 3, 3, natoms))
        Z = [Mat3{Float64}(arr) for arr in eachslice(Z_arr, dims=3)]

        # EPW hard-coded parameters (see EPW/src/rigid_epw.f90)
        cutoff = T(14.0) # gmax
        η = T(1.0) # alph

        # Compute nxs for phonon dynamical matrix. See SUBROUTINE rgd_blk of EPW
        nxs_array = [0, 0, 0]
        for (i, n) in enumerate(nqc)
            if n > 1
                nxs_array[i] = floor(Int, sqrt(4*η*cutoff) / norm(recip_lattice[:, 2])) + 1
            end
        end
        nxs = tuple(nxs_array...)
        polar_phonon = Polar{T}(use=true; alat, volume, nmodes, recip_lattice, atom_pos, ϵ, Z, nxs, cutoff, η)

        # For e-ph coupling, nxs is nqc. See SUBROUTINE rgd_blk_epw_fine of EPW
        nxs = map(x -> 1 < x < 5 ? 5 : x, nqc) # quick-and-dirty fix for small nqc
        polar_eph = Polar{T}(use=true; alat, volume, nmodes, recip_lattice, atom_pos, ϵ, Z, nxs, cutoff, η)
    else
        # Set null objects
        polar_phonon = Polar{T}(nothing)
        polar_eph = Polar{T}(nothing)
    end

    # Electron Hamiltonian
    nrr_k = convert(Int, read(f, Int32))
    irvec_k = convert.(Int, read(f, (Int32, 3, nrr_k)))
    ham = read(f, (ComplexF64, nw, nw, nrr_k))
    pos = read(f, (ComplexF64, 3, nw, nw, nrr_k))

    # Electron velocity matrix. Optional
    has_velocity = fortran_read_bool(f)
    if has_velocity
        vel = read(f, (ComplexF64, 3, nw, nw, nrr_k))
    end

    # Phonon dynamical matrix
    nr_ph = convert(Int, read(f, Int32))
    irvec_ph = convert.(Int, read(f, (Int32, 3, nr_ph)))
    dyn_real = read(f, (Float64, nmodes, nmodes, nr_ph))
    dyn = complex(dyn_real)

    # Electron-phonon matrix
    nr_ep = convert(Int, read(f, Int32))
    irvec_ep = convert.(Int, read(f, (Int32, 3, nr_ep)))

    nr_el = nrr_k
    irvec_el = irvec_k

    # Transform R vectors from matrix to vector of StaticVectors
    irvec_el = reinterpret(Vec3{Int}, irvec_el)[:]
    irvec_ph = reinterpret(Vec3{Int}, irvec_ph)[:]
    irvec_ep = reinterpret(Vec3{Int}, irvec_ep)[:]

    # Sort R vectors using R[3], and then R[2], and then R[1].
    # This is useful for gridopt.
    ind_el = sortperm(irvec_el, by=x->reverse(x))
    ind_ph = sortperm(irvec_ph, by=x->reverse(x))
    ind_ep = sortperm(irvec_ep, by=x->reverse(x))

    irvec_el = irvec_el[ind_el]
    irvec_ph = irvec_ph[ind_ph]
    irvec_ep = irvec_ep[ind_ep]

    # Shuffle matrix elements accordingly
    ham = ham[:, :, ind_el]
    pos = pos[:, :, :, ind_el]
    dyn = dyn[:, :, ind_ph]

    # Electron-phonon coupling
    # This part is the bottleneck of this function.
    if epmat_on_disk
        empat_filename = "tmp_epmat.bin"

        # epmat stays on disk. Read each epmat for each ir and write to file.
        size_column = sizeof(ComplexF64)*nw^2*nmodes # Size of one column of data

        # Open file to write
        filename = joinpath(tmpdir, empat_filename)
        rm(filename, force=true)
        fw = open(filename, "w")

        for ir in 1:nr_ep
            # Read from Fortran file
            epmat_re_rp_read_ir = read(f, (ComplexF64, nw, nw, nr_el, nmodes))
            epmat_re_rp_ir = permutedims(epmat_re_rp_read_ir, [1, 2, 4, 3])

            # Shuffle irvec_el
            epmat_re_rp_ir = epmat_re_rp_ir[:, :, :, ind_el]

            # Write to file, as index epmat[:, :, :, :, ir_ep]
            ir_ep = findfirst(ind_ep .== ir) # Index of R point after sorting
            for ir_el in 1:nr_el
                if epmat_outer_momentum == "ph"
                    seek(fw, size_column * (nr_el * (ir_ep - 1) + ir_el - 1))
                else # epmat_outer_momentum == "el"
                    seek(fw, size_column * (nr_ep * (ir_el - 1) + ir_ep - 1))
                end
                write(fw, epmat_re_rp_ir[:,:,:,ir_el])
            end
        end
        close(fw)
    else
        # Read epmat to memory
        epmat_re_rp = zeros(ComplexF64, nw, nw, nmodes, nr_el, nr_ep)
        for ir in 1:nr_ep
            epmat_re_rp_read_ir = read(f, (ComplexF64, nw, nw, nr_el, nmodes))
            epmat_re_rp[:,:,:,:,ir] = permutedims(epmat_re_rp_read_ir, [1, 2, 4, 3])
        end
        epmat_re_rp = epmat_re_rp[:, :, :, ind_el, ind_ep]
        if epmat_outer_momentum == "el"
            epmat_re_rp = permutedims(epmat_re_rp, [1, 2, 3, 5, 4])
        end
    end
    close(f)

    # Reshape real-space matrix elements into 2-dimensional matrices
    # First index: all other indices
    # Second index: R vectors
    ham = reshape(ham, (nw*nw, nr_el))
    dyn = reshape(dyn, (nmodes*nmodes, nr_ph))
    el_ham = WannierObject(irvec_el, ham)
    ph_dyn = WannierObject(irvec_ph, dyn)

    # R * ham for electron and phonon velocity
    el_ham_R = wannier_object_multiply_R(el_ham, lattice)
    ph_dyn_R = wannier_object_multiply_R(ph_dyn, lattice)

    # Electron position (dipole) matrix elements
    pos2 = permutedims(pos, [2, 3, 1, 4])
    el_pos = WannierObject(irvec_el, reshape(pos2, (nw*nw*3, nr_el)))

    # Electron velocity matrix elements
    if has_velocity
        vel = vel[:, :, :, ind_el]
        vel2 = permutedims(vel, [2, 3, 1, 4])
        el_vel = WannierObject(irvec_el, reshape(vel2, (nw*nw*3, nr_el)))
    else
        el_vel = nothing
    end

    if epmat_on_disk
        if epmat_outer_momentum == "ph"
            epmat = DiskWannierObject(Float64, "epmat", nr_ep, irvec_ep, nw*nw*nmodes*nr_el,
                tmpdir, empat_filename, irvec_next=irvec_el)
        else # epmat_outer_momentum == "el"
            epmat = DiskWannierObject(Float64, "epmat", nr_el, irvec_el, nw*nw*nmodes*nr_ep,
                tmpdir, empat_filename, irvec_next=irvec_ep)
        end
    else
        if epmat_outer_momentum == "ph"
            epmat_re_rp = reshape(epmat_re_rp, (nw*nw*nmodes*nr_el, nr_ep))
            epmat = WannierObject(irvec_ep, epmat_re_rp, irvec_next=irvec_el)
        else # epmat_outer_momentum == "el"
            epmat_re_rp = reshape(epmat_re_rp, (nw*nw*nmodes*nr_ep, nr_el))
            epmat = WannierObject(irvec_el, epmat_re_rp, irvec_next=irvec_ep)
        end
    end

    if load_symmetry_operators
        el_sym = load_symmetry_operators_from_epw(folder)
        # Check el_sym.symmetry is a subset of model symmetry
        @assert symmetry_is_subset(el_sym.symmetry, symmetry) "el_sym.symmetry must be a subset of model symmetry"
    else
        el_sym = nothing
    end

    model = ModelEPW(;alat, lattice, recip_lattice, volume, nw, nmodes,
        mass, atom_pos, atom_labels, symmetry,
        use_polar_dipole, polar_phonon, polar_eph,
        el_ham, el_ham_R, el_pos, el_vel,
        ph_dyn, ph_dyn_R,
        epmat, epmat_outer_momentum,
        el_sym,
    )
end

function load_symmetry_operators_from_epw(folder)
    f = FortranFile(joinpath(folder, "epw_data_julia_symmetry.bin"), "r")

    alat = read(f, Float64)
    at_in_alat = read(f, (Float64, 3, 3))
    lattice = Mat3(at_in_alat .* alat)

    nsym = Int(read(f, Int32))
    symmetry_S = Int.(read(f, (Int32, 3, 3, nsym)))
    symmetry_τ = read(f, (Float64, 3, nsym))
    time_reversal = EPW.fortran_read_bool(f)

    Ss = [Mat3(symmetry_S[:, :, i]) for i in 1:nsym]
    τs = [Vec3(symmetry_τ[:, i]) for i in 1:nsym]

    # We do not use time reversal for symmetry matrices because off-diagonal elements in QME
    # breaks the time-reversal symmetry even for the linear response.
    symmetry = Symmetry(Ss, τs, false, lattice)

    nw = Int(read(f, Int32))
    nrr = Int(read(f, Int32))
    irvec_raw = convert.(Int, read(f, (Int32, 3, nrr)))
    irvec = reinterpret(Vec3{Int}, irvec_raw)[:]
    ind_el = sortperm(irvec, by=x->reverse(x))
    irvec = irvec[ind_el]

    operators = Vector{WannierObject{Float64}}()
    @views for isym in 1:nsym
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