
using FortranFiles

export load_model_from_epw

"Tight-binding model for electron, phonon, and electron-phonon coupling.
All data is in coarse real-space grid."
Base.@kwdef struct ModelEPW1{WannType <: AbstractWannierObject{Float64}}
    nw::Int
    nmodes::Int
    mass::Array{Float64,1}

    nr_el::Int

    el_ham::WannierObject{Float64}
    # TODO: Use Hermiticity of hk

    ph_dyn::WannierObject{Float64}
    # TODO: Use real-valuedness of dyn_r
    # TODO: Use Hermiticity of dyn_q

    "electron-phonon coupling matrix in electron and phonon Wannier representation"
    epmat::WannType
end
ModelEPW = ModelEPW1 # WARNING, only while developing.

"""
Arguments:
epmat_on_disk
    If true, write epmat to file and read at each get_fourier! call.
    If false, load epmat to memory.
tmpdir
    Directory to write temporary binary files for epmat_on_disk=false calse.
"""
function load_model_from_epw(folder::String, epmat_on_disk::Bool=false, tmpdir=nothing)
    if epmat_on_disk && tmpdir == nothing
        error("If epmat_on_disk is true, tmpdir must be provided.")
    end
    # Read binary data written by EPW and create ModelEPW object
    f = FortranFile(joinpath(folder, "epw_data_julia.bin"), "r")
    nw = convert(Int, read(f, Int32))
    nmodes = convert(Int, read(f, Int32))
    mass = read(f, (Float64, nmodes))

    # Electron Hamiltonian
    nrr_k = convert(Int, read(f, Int32))
    irvec_k = convert.(Int, read(f, (Int32, 3, nrr_k)))
    ham_r = read(f, (ComplexF64, nw, nw, nrr_k))

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

    ham_r = ham_r[:, :, ind_el]
    dyn_r = dyn_r[:, :, ind_ph]

    # Electron-phonon coupling
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
    ham_r = reshape(ham_r, (nw*nw, nrr_k))
    dyn_r = reshape(dyn_r, (nmodes*nmodes, nr_ph))
    el_ham = WannierObject(nr_el, irvec_el, ham_r)
    ph_dyn = WannierObject(nr_ph, irvec_ph, dyn_r)

    if epmat_on_disk
        epmat = DiskWannierObject1(Float64, "epmat", nr_ep, irvec_ep, nw*nw*nmodes*nr_el,
            tmpdir, empat_filename)
    else
        epmat_re_rp = reshape(epmat_re_rp, (nw*nw*nmodes*nrr_k, nr_ep))
        epmat = WannierObject(nr_ep, irvec_ep, epmat_re_rp)
    end

    model = ModelEPW(nw=nw, nmodes=nmodes, mass=mass,
        nr_el=nr_el,
        el_ham=el_ham, ph_dyn=ph_dyn, epmat=epmat
    )

    model
end
