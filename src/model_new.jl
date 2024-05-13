function pair_to_complex(pair)
    nums = parse.(Float64, split(strip(pair)[2:end-1], ","))
    return Complex(nums...)
end

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


function _parse_wigner(f, nr, dims1, dims2)
    irvec = zeros(Int, 3, nr)
    wslen = zeros(Float64, nr)
    ndegen = zeros(Int, dims1, dims2, nr)

    for ir in 1:nr
        line = split(readline(f))

        irvec[:, ir] .= parse.(Int, line[1:3])
        wslen[ir] = parse(Float64, line[4])

        for i in 1:dims1
            ndegen[i, :, ir] .= parse.(Int, split(readline(f)))
        end
    end

    irvec = reinterpret(Vec3{Int}, irvec)[:]

    return irvec, ndegen, wslen
end


"""
    epw_parse_structure(folder :: String) => Structure

Parse crystal.fmt file written by EPW.
"""
function epw_parse_structure(folder::String)

    f = open(joinpath(folder, "crystal.fmt"), "r")
    nat = parse(Int, readline(f))
    nmodes = parse(Int, readline(f))
    nelec, = parse.(Float64, split(readline(f)))
    at = Mat3(parse.(Float64, split(readline(f))))
    bg = Mat3(parse.(Float64, split(readline(f))))
    omega = parse(Float64, readline(f))
    alat = parse(Float64, readline(f))
    tau = parse.(Float64, split(readline(f)))
    amass = parse.(Float64, split(readline(f)))
    ityp = parse.(Int, split(readline(f)))
    close(f)

    # Convert to jl convention
    lattice = at .* alat
    recip_lattice = bg .* (2π / alat)
    @assert lattice' * recip_lattice ≈ 2π * I(3)

    mass = amass[ityp]
    atom_pos = reinterpret(Vec3{Float64}, reshape(tau, 3, 2))[:]
    atom_labels = string.(ityp)  # EPW does not store atom labels, use ityp instead

    return Structure(alat, lattice, mass, atom_pos, atom_labels)
end


function load_model_from_epw_new(
    folder :: String,
    outdir :: String,
    prefix :: String
    ;
    epmat_outer_momentum :: String = "el",
    load_epmat :: Bool = true,
    load_symmetry_operators :: Bool = false,
    )

    structure = ElectronPhonon.epw_parse_structure(folder)

    # Read Wigner-Seitz information
    f = open(joinpath(folder, "wigner.fmt"), "r")
    nr_el, nr_ph, nr_ep, dims, dims2 = parse.(Int, split(readline(f)))
    irvec_el, ndegen_el, wslen_el = _parse_wigner(f, nr_el, dims,  dims)
    irvec_ph, ndegen_ph, wslen_ph = _parse_wigner(f, nr_ph, dims2, dims2)
    irvec_ep, ndegen_ep, wslen_ep = _parse_wigner(f, nr_ep, dims,  dims2)
    close(f)


    # Read epwdata.fmt

    f = open(joinpath(folder, "epwdata.fmt"), "r")
    ef = parse(Float64, readline(f))
    nw, nr_el, nmodes, nr_ph, nr_ep = parse.(Int, split(readline(f)))
    zstar_epsi = parse.(Float64, split(readline(f)))

    ham = zeros(ComplexF64, nw, nw, nr_el)
    for i in 1:nw
        for j in 1:nw
            for ir in 1:nr_el
                ham[i, j, ir] = ElectronPhonon.pair_to_complex(readline(f))
            end
        end
    end
    if size(ndegen_el, 1) == 1
        for ir in 1:nr_el
            if ndegen_el[1, 1, ir] == 0
                ham[:, :, ir] .= 0
            else
                ham[:, :, ir] ./= ndegen_el[1, 1, ir]
            end
        end
    else
        for ir in 1:nr_el
            for j in 1:nw
                for i in 1:nw
                    if ndegen_el[i, j, ir] == 0
                        ham[i, j, ir] = 0
                    else
                        ham[i, j, ir] /= ndegen_el[i, j, ir]
                    end
                end
            end
        end
    end

    dyn = zeros(ComplexF64, nmodes, nmodes, nr_ph)
    for i in 1:nmodes
        for j in 1:nmodes
            for ir in 1:nr_ph
                dyn[i, j, ir] = ElectronPhonon.pair_to_complex(readline(f))
            end
        end
    end

    if size(ndegen_ph, 1) == 1
        @views for ir in 1:nr_ph
            if ndegen_ph[1, 1, ir] == 0
                dyn[:, :, ir] .= 0
            else
                dyn[:, :, ir] ./= ndegen_ph[1, 1, ir]
            end
        end
    else
        @views for ir in 1:nr_ph
            for j in 1:div(nmodes, 3)
                for i in 1:div(nmodes, 3)
                    inds1 = (1+3(i-1)) : 3i
                    inds2 = (1+3(j-1)) : 3j
                    if ndegen_ph[i, j, ir] == 0
                        dyn[inds1, inds2, ir] .= 0
                    else
                        dyn[inds1, inds2, ir] ./= ndegen_ph[i, j, ir]
                    end
                end
            end
        end
    end

    if norm(dyn) < eps(Float64)
        error("Dynamical matrix is not written in epwdata.fmt. Set lifc = .false. in EPW.
        (lifc = .true. is not yet implemented in the new parser.)")
    end

    close(f)

    Z = [Mat3(arr) for arr in eachslice(reshape(zstar_epsi[1:end-9], 3, 3, :); dims=3)]
    ϵ = Mat3(zstar_epsi[end-8:end])
    use_polar_dipole = any(norm.(Z) .> sqrt(eps(Float64)))

    if use_polar_dipole || isfile(joinpath(folder, "quadrupole.fmt"))
        # EPW hard-coded parameters (see EPW/src/rigid_f90)
        cutoff = 14.0  # gmax
        η = 1.0  # alph

        if isfile(joinpath(folder, "quadrupole.fmt"))
            Q = parse_epw_quadrupole_fmt(joinpath(folder, "quadrupole.fmt"))
        else
            Q = zeros(Vec3{Mat3{Float64}}, length(structure.atom_pos))
        end

        # Compute nxs for phonon dynamical matrix. See SUBROUTINE rgd_blk of EPW
        nxs = Tuple(floor(Int, sqrt(4*η*cutoff) / norm(structure.recip_lattice[:, i])) + 1 for i in 1:3)
        polar_phonon = Polar{Float64}(use=true; structure.alat, structure.volume, nmodes, structure.recip_lattice, structure.atom_pos, ϵ, Z, Q, nxs, cutoff, η)

        # Compute nxs for electron-phonon coupling. See SUBROUTINE rgd_blk_epw of EPW
        # FIXME: EPW shifts origin for each q, this is not implemented here (See SUBROUTINE rgd_blk_epw of EPW)
        nxs = Tuple(floor(Int, sqrt(4*η*cutoff) / norm(structure.recip_lattice[:, i])) + 1 for i in 1:3)
        polar_eph = Polar{Float64}(use=true; structure.alat, structure.volume, nmodes, structure.recip_lattice, structure.atom_pos, ϵ, Z, Q, nxs, cutoff, η)
    else
        # Set null objects
        polar_phonon = Polar{Float64}(nothing)
        polar_eph = Polar{Float64}(nothing)
    end


    # Position (dipole) and velocity matrix elements
    # FIXME: Implement, and then use el_velocity_mode = :Direct
    # FIXME: Apply ndegen
    pos = zeros(ComplexF64, 3, nw, nw, nr_el)
    vel = zeros(ComplexF64, 3, nw, nw, nr_el)
    # el_velocity_mode = :Direct
    el_velocity_mode = :BerryConnection


    # Sort R vectors using R[3], and then R[2], and then R[1].
    ind_el = sortperm(irvec_el, by=x->reverse(x))
    ind_ph = sortperm(irvec_ph, by=x->reverse(x))
    ind_ep = sortperm(irvec_ep, by=x->reverse(x))

    irvec_el = irvec_el[ind_el]
    irvec_ph = irvec_ph[ind_ph]
    irvec_ep = irvec_ep[ind_ep]
    ndegen_el = ndegen_el[:, :, ind_el]
    ndegen_ph = ndegen_ph[:, :, ind_ph]
    ndegen_ep = ndegen_ep[:, :, ind_ep]

    # Shuffle matrix elements accordingly
    ham = ham[:, :, ind_el]
    dyn = dyn[:, :, ind_ph]
    pos = pos[:, :, :, ind_el]
    vel = vel[:, :, :, ind_el]


    # Reshape real-space matrix elements into 2-dimensional matrices
    # First index: all other indices
    # Second index: R vectors
    ham = reshape(ham, (nw^2, nr_el))
    dyn = reshape(dyn, (nmodes^2, nr_ph))
    el_ham = WannierObject(irvec_el, ham)
    ph_dyn = WannierObject(irvec_ph, dyn)

    # R * ham for electron and phonon velocity
    el_ham_R = wannier_object_multiply_R(el_ham, structure.lattice)
    ph_dyn_R = wannier_object_multiply_R(ph_dyn, structure.lattice)

    # Electron position (dipole) and velocity matrix elements
    pos = permutedims(pos, [2, 3, 1, 4])
    vel = permutedims(vel, [2, 3, 1, 4])
    el_pos = WannierObject(irvec_el, reshape(pos, (nw^2 * 3, nr_el)))
    el_vel = WannierObject(irvec_el, reshape(vel, (nw^2 * 3, nr_el)))


    # TODO: load_symmetry_operators

    # Read electron-phonon coupling

    if load_epmat
        filename = joinpath(folder, outdir, "$prefix.epmatwp")
        ep = read_epmat(filename, nw, nmodes, nr_el, nr_ep, irvec_el, irvec_ep, ind_el, ind_ep, ndegen_el, ndegen_ep, epmat_outer_momentum)

    else
        # Do not read epmat
        ep = nothing
        epmat_outer_momentum = "nothing"
    end


    # Symmetry operators in Wannier function basis

    if load_symmetry_operators
        error("Not implemented")
    else
        el_sym = nothing
    end



    Model(; structure.alat, structure.lattice, structure.recip_lattice, structure.volume, nw, nmodes,
        structure.mass, structure.atom_pos, structure.atom_labels, structure.symmetry,
        use_polar_dipole, polar_phonon, polar_eph,
        el_ham, el_ham_R, el_pos, el_vel, el_velocity_mode,
        ph_dyn, ph_dyn_R,
        epmat = ep, epmat_outer_momentum,
        el_sym,
    )


end


function read_epmat(filename, nw, nmodes, nr_el, nr_ep, irvec_el, irvec_ep, ind_el, ind_ep, ndegen_el, ndegen_ep, epmat_outer_momentum)
    f = open(filename, "r")
    epmat = zeros(ComplexF64, nw, nw, nr_el, nmodes, nr_ep)
    @views for ir_ep in 1:nr_ep
        read!(f, epmat[:, :, :, :, ir_ep])
    end
    close(f)

    # Shuffle R vector order
    epmat = epmat[:, :, ind_el, :, ind_ep]

    # Apply Wigner-Seitz degeneracy
    if size(ndegen_ep, 1) == 1
        @views for ir_ep in 1:nr_ep
            if ndegen_ep[1, 1, ir_ep] == 0
                epmat[:, :, :, :, ir_ep] .= 0
            else
                epmat[:, :, :, :, ir_ep] ./= ndegen_ep[1, 1, ir_ep]
            end
        end
    
        @views for ir_el in 1:nr_el
            if ndegen_el[1, 1, ir_el] == 0
                epmat[:, :, ir_el, :, :] .= 0
            else
                epmat[:, :, ir_el, :, :] ./= ndegen_el[1, 1, ir_el]
            end
        end

    else
        @views for ir_ep in 1:nr_ep, imode in 1:nmodes, iw in 1:nw
            iatm = cld(imode, 3)
            if ndegen_ep[iw, iatm, ir_ep] == 0
                epmat[iw, :, :, imode, ir_ep] .= 0
            else
                epmat[iw, :, :, imode, ir_ep] ./= ndegen_ep[iw, iatm, ir_ep]
            end
        end
    
        @views for ir_el in 1:nr_el, jw in 1:nw, iw in 1:nw
            if ndegen_el[iw, jw, ir_el] == 0
                epmat[iw, jw, ir_el, :, :] .= 0
            else
                epmat[iw, jw, ir_el, :, :] ./= ndegen_el[iw, jw, ir_el]
            end
        end
    end

    # Index order
    # EPW : (iw, jw, iRe, imode, iRp)
    # EP.jl epmat_outer_momentum == "ph" : (iw, jw, imode, iRe, iRp)
    # EP.jl epmat_outer_momentum == "el" : (iw, jw, imode, iRp, iRe)

    if epmat_outer_momentum == "ph"
        epmat = permutedims(epmat, [1, 2, 4, 3, 5]);
        epmat = reshape(epmat, (nw^2 * nmodes * nr_el, nr_ep))
        ep = WannierObject(irvec_ep, epmat; irvec_next = irvec_el)

    else
        epmat = permutedims(epmat, [1, 2, 4, 5, 3]);
        epmat = reshape(epmat, (nw^2 * nmodes * nr_ep, nr_el))
        ep = WannierObject(irvec_el, epmat; irvec_next = irvec_ep)
    end
    
    return ep
end
