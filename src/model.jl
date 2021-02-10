
using FortranFiles

import EPW.WanToBloch: get_ph_eigen!

export load_model
export get_ph_eigen!

"Tight-binding model for electron, phonon, and electron-phonon coupling.
All data is in coarse real-space grid."
Base.@kwdef struct ModelEPW{WannType <: AbstractWannierObject{Float64}}
    # Lattice information
    alat::Float64 # Lattice parameter
    # Lattice vector in Bohr. lattice[:, i] is the i-th lattice vector.
    lattice::Mat3{Float64}
    recip_lattice::Mat3{Float64}
    volume::Float64

    # Atom information
    mass::Array{Float64,1}
    atom_pos::Vector{Vec3{Float64}}

    nw::Int
    nmodes::Int

    # Long-range term in polar systems
    use_polar_dipole::Bool
    polar_phonon::Polar{Float64}
    polar_eph::Polar{Float64}

    el_ham::WannierObject{Float64} # ELectron Hamiltonian
    # TODO: Use Hermiticity of hk

    el_ham_R::WannierObject{Float64} # Electron Hamiltonian times R
    el_pos::WannierObject{Float64} # Electron position (dipole)

    ph_dyn::WannierObject{Float64} # Phonon dynamical matrix
    # TODO: Use real-valuedness of dyn_r
    # TODO: Use Hermiticity of dyn_q

    epmat::WannType # electron-phonon coupling matrix in electron and phonon Wannier representation
end

"Read file and create ModelEPW object in the MPI root.
Broadcast to all other processors."
function load_model(folder::String, epmat_on_disk::Bool=false, tmpdir=nothing)
    # Read model from file
    if mpi_initialized()
        # FIXME: Read only in the root core, and then bcast.
        # The implementation below breaks if epmat size is large. MPI bcast of large array
        # with sizeof(array) is greater than typemax(Cint) was not possible.
        model = load_model_from_epw(folder, epmat_on_disk, tmpdir)
        # if mpi_isroot(EPW.mpi_world_comm())
        #     model = load_model_from_epw(folder, epmat_on_disk, tmpdir)
        # else
        #     model = nothing
        # end
        # # Broadcast to all processors
        # model = mpi_bcast(model, EPW.mpi_world_comm())
    else
        model = load_model_from_epw(folder, epmat_on_disk, tmpdir)
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
"""
function load_model_from_epw(folder::String, epmat_on_disk::Bool=false, tmpdir=nothing)
    T = Float64

    if epmat_on_disk && tmpdir == nothing
        error("If epmat_on_disk is true, tmpdir must be provided.")
    end
    # Read binary data written by EPW and create ModelEPW object
    f = FortranFile(joinpath(folder, "epw_data_julia.bin"), "r")

    # Structure parameters
    alat = read(f, Float64)
    at_in_alat = read(f, (Float64, 3, 3))
    natoms = convert(Int, read(f, Int32))
    atom_pos_arr = read(f, (Float64, 3, natoms))
    atom_pos = reinterpret(Vec3{T}, atom_pos_arr)[:]

    lattice = Mat3{T}(at_in_alat .* alat)
    recip_lattice = 2T(π) * inv(lattice')
    volume = abs(det(lattice))

    # Wannier parameters
    nw = convert(Int, read(f, Int32))
    nmodes = convert(Int, read(f, Int32))
    mass = read(f, (Float64, nmodes))

    nkc = tuple(convert.(Int, read(f, (Int32, 3)))...)
    nqc = tuple(convert.(Int, read(f, (Int32, 3)))...)

    # Polar parameters
    use_polar_dipole_fortran = read(f, Int32) # 0 is false, +1 or -1 is true
    use_polar_dipole = convert(Bool, abs(use_polar_dipole_fortran))

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
        polar_phonon = Polar(alat=alat, volume=volume, recip_lattice=recip_lattice,
            atom_pos=atom_pos, ϵ=ϵ, Z=Z, nxs=nxs, cutoff=cutoff, η=η)

        # For e-ph coupling, nxs is nqc. See SUBROUTINE rgd_blk_epw_fine of EPW
        polar_eph = Polar(alat=alat, volume=volume, recip_lattice=recip_lattice,
            atom_pos=atom_pos, ϵ=ϵ, Z=Z, nxs=nqc, cutoff=cutoff, η=η)
    else
        # Set null objects
        polar_phonon = Polar(T)
        polar_eph = Polar(T)
    end

    # Electron Hamiltonian
    nrr_k = convert(Int, read(f, Int32))
    irvec_k = convert.(Int, read(f, (Int32, 3, nrr_k)))
    ham = read(f, (ComplexF64, nw, nw, nrr_k))
    pos = read(f, (ComplexF64, 3, nw, nw, nrr_k))

    # Phonon dynamical matrix
    nr_ph = convert(Int, read(f, Int32))
    irvec_ph = convert.(Int, read(f, (Int32, 3, nr_ph)))
    dyn_r_real = read(f, (Float64, nmodes, nmodes, nr_ph))
    dyn_r = complex(dyn_r_real)

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

    ham = ham[:, :, ind_el]
    dyn_r = dyn_r[:, :, ind_ph]

    # Electron-phonon coupling
    # This part is the bottleneck of this function.
    if epmat_on_disk
        empat_filename = "tmp_epmat.bin"

        # epmat stays on disk. Read each epmat for each ir and write to file.
        size_column =  sizeof(ComplexF64)*nw^2*nrr_k*nmodes # Size of one column of data

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

            # Write to file, as index epmat[:, :, :, :, ir_new]
            ir_new = findfirst(ind_ep .== ir) # Index of R point after sorting
            seek(fw, size_column * (ir_new - 1))
            write(fw, epmat_re_rp_ir)
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
    end
    close(f)

    # Reshape real-space matrix elements into 2-dimensional matrices
    # First index: all other indices
    # Second index: R vectors
    ham = reshape(ham, (nw*nw, nr_el))
    dyn_r = reshape(dyn_r, (nmodes*nmodes, nr_ph))
    el_ham = WannierObject(nr_el, irvec_el, ham)
    ph_dyn = WannierObject(nr_ph, irvec_ph, dyn_r)

    # R * ham for electron velocity
    ham_R = zeros(eltype(ham), (nw*nw, 3, nr_el))
    for ir = 1:nr_el
        @views for i = 1:3
            ham_R[:, i, ir] .= im .* ham[:, ir] .* dot(lattice[i, :], irvec_el[ir])
        end
    end
    el_ham_R = WannierObject(nr_el, irvec_el, reshape(ham_R, (nw*nw*3, nr_el)))

    # Electron position (dipole) matrix elements
    pos2 = permutedims(pos, [2, 3, 1, 4])
    el_pos = WannierObject(nr_el, irvec_el, reshape(pos2, (nw*nw*3, nr_el)))

    if epmat_on_disk
        epmat = DiskWannierObject(Float64, "epmat", nr_ep, irvec_ep, nw*nw*nmodes*nr_el,
            tmpdir, empat_filename)
    else
        epmat_re_rp = reshape(epmat_re_rp, (nw*nw*nmodes*nrr_k, nr_ep))
        epmat = WannierObject(nr_ep, irvec_ep, epmat_re_rp)
    end

    model = ModelEPW(alat=alat, lattice=lattice, recip_lattice=recip_lattice,
        volume=volume, nw=nw, nmodes=nmodes, mass=mass, atom_pos=atom_pos,
        use_polar_dipole=use_polar_dipole, polar_phonon=polar_phonon, polar_eph=polar_eph,
        el_ham=el_ham, el_ham_R=el_ham_R, el_pos=el_pos,
        ph_dyn=ph_dyn, epmat=epmat
    )

    model
end

# Redefine WanToBloch functions
"""
    get_ph_eigen!(values, vectors, model::ModelEPW, fourier_mode="normal")
"""
function get_ph_eigen!(values, vectors, model::ModelEPW, xq; fourier_mode="normal")
    if model.use_polar_dipole
        get_ph_eigen!(values, vectors, model.ph_dyn, model.mass, xq,
            model.polar_phonon, fourier_mode=fourier_mode)
    else
        get_ph_eigen!(values, vectors, model.ph_dyn, model.mass, xq,
            fourier_mode=fourier_mode)
    end
end
