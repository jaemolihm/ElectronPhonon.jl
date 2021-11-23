using HDF5

"""
Debugging flags in `kwargs`
- `DEBUG_random_gauge`: Multiply random phases to the eigenstates at k+q to change the eigenstate gauge. (Default: false)
"""
function compute_electron_phonon_bte_data_coherence(model, btedata_prefix, window_k, window_kq, kpts,
        kqpts, qpts, nband, nband_ignore, nstates_base_k, nstates_base_kq, energy_conservation,
        average_degeneracy, symmetry, mpi_comm_k, mpi_comm_q, fourier_mode, qme_offdiag_cutoff;
        kwargs...)
    FT = Float64

    nw = model.nw
    nmodes = model.nmodes
    nk = kpts.n
    nq = qpts.n
    nkq = kqpts.n

    mpi_isroot() && println("Calculating electron and phonon states")
    g = nothing
    # TODO: parallelize this part
    @timing "hdf init" begin
        # Open HDF5 file for writing BTEdata
        filename = "$btedata_prefix.rank$(mpi_myrank(mpi_comm_k)).h5"
        rm(filename, force=true)
        fid_btedata = h5open(filename, "w")

        # Calculate initial (k) and final (k+q) electron states, write to HDF5 file
        mpi_isroot() && println("Calculating electron states at k")
        el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"], window_k, nband, nband_ignore; fourier_mode)
        # for (el, xk) in zip(el_k_save, kpts.vectors)
        #     set_gauge_to_diagonalize_velocity_matrix!(el, xk, 1, model)
        # end
        el_k_boltzmann, _ = electron_states_to_QMEStates(el_k_save, kpts, qme_offdiag_cutoff, nstates_base_k)
        g = create_group(fid_btedata, "initialstate_electron")
        dump_BTData(g, el_k_boltzmann)

        # mpi_isroot() && println("Calculating electron states at k+q")
        el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector"], window_kq, nband, nband_ignore; fourier_mode)
        # DEBUG: randomly change the eigenstate gauge at k+q so that the gauge is different from k
        if get(kwargs, :DEBUG_random_gauge, false) == true
            # Multiply random phase factor
            for el in el_kq_save, ib in 1:el.nband
                get_u(el)[:, ib] .*= cispi(2*rand())
            end
            # Swap degenerate eigenvectors
            for el in el_kq_save, ib in 1:el.nband-1
                if abs(el.e[el.rng[ib]] - el.e[el.rng[ib+1]]) < EPW.electron_degen_cutoff && rand() > 0.5
                    u = get_u(el)
                    u[:, ib], u[:, ib+1] = u[:, ib+1], u[:, ib]
                end
            end
        end
        el_kq_boltzmann, _ = electron_states_to_QMEStates(el_kq_save, kqpts, qme_offdiag_cutoff, nstates_base_kq)
        g = create_group(fid_btedata, "finalstate_electron")
        dump_BTData(g, el_kq_boltzmann)

        # Write phonon states to HDF5 file
        mpi_isroot() && println("Calculating phonon states")
        ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode)
        ph_boltzmann, _ = phonon_states_to_BTStates(ph_save, qpts)
        g = create_group(fid_btedata, "phonon")
        dump_BTData(g, ph_boltzmann)
    end

    # Write gauge matrices that map eigenstates in el_k_save to eigenstates in el_kq_save.
    # When not using symmetry, compute <u_mk|u_nk> (i.e. q=0).
    # When using symmetry, also compute <u_m,Sk|S|u_nk> for symmetry operations.
    # TODO: Optimize memory and disk usage by writing only nonzero matrix elements
    # TODO: Merge two cases
    mpi_isroot() && println("Calculating and writing gauge matrices")

    gauge = zeros(Complex{FT}, nband, nband, nk)
    is_degenerate = zeros(Bool, nband, nband, nk) # FIXME: Cannot use BitVector because HDF5.jl does not support it.

    if symmetry !== nothing
        # Write symmetry object to file
        g = create_group(fid_btedata, "gauge/symmetry")
        dump_BTData(g, symmetry)

        tmp_arr_full = zeros(Complex{FT}, nw, nw)
        sym_k = zeros(Complex{FT}, nw, nw)
        for isym = 1:symmetry.nsym
            # Find symmetry in model.el_sym
            isym_el = findfirst(s -> s ≈ symmetry[isym], model.el_sym.symmetry)

            gauge .= 0
            is_degenerate .= false
            @views for ik = 1:nk
                xk = kpts.vectors[ik]
                sxk = symmetry[isym].S * xk
                isk = xk_to_ik(sxk, kqpts)
                isk === nothing && continue # skip if Sk is not in kqpts

                rng_k = el_k_save[ik].rng
                rng_sk = el_kq_save[isk].rng
                e_k = el_k_save[ik].e
                e_sk = el_kq_save[isk].e

                # Compute symmetry gauge matrix: S_H = U†(Sk) * S_W * U(k)
                get_fourier!(sym_k, model.el_sym.operators[isym_el], xk, mode=fourier_mode)
                tmp_arr = view(tmp_arr_full, :, 1:el_k_save[ik].nband)
                mul!(tmp_arr, sym_k, get_u(el_k_save[ik]))
                mul!(gauge[rng_sk, rng_k, ik], get_u(el_kq_save[isk])', tmp_arr)

                # Set is_degenerate
                for ib in rng_k, jb in rng_sk
                    is_degenerate[jb, ib, ik] = abs(e_k[ib] - e_sk[jb]) < electron_degen_cutoff
                end
            end
            g = create_group(fid_btedata, "gauge/isym$isym")
            g["gauge_matrix"] = gauge
            g["is_degenerate"] = is_degenerate
        end
    else
        gauge = zeros(Complex{FT}, nband, nband, nk)
        @views for ik = 1:nk
            xk = kpts.vectors[ik]
            ik_kq = xk_to_ik(xk, kqpts)
            ik_kq === nothing && continue # skip if xk is not in kqpts

            rng_k = el_k_save[ik].rng
            rng_kq = el_kq_save[ik_kq].rng
            e_k = el_k_save[ik].e
            e_kq = el_kq_save[ik_kq].e

            # Compute gauge matrix: gauge = U†(k) * U(k)
            mul!(gauge[rng_kq, rng_k, ik], get_u(el_kq_save[ik_kq])', get_u(el_k_save[ik]))

            # Set is_degenerate
            for ib in rng_k, jb in rng_kq
                is_degenerate[jb, ib, ik] = abs(e_k[ib] - e_kq[jb]) < electron_degen_cutoff
            end
        end
        g = create_group(fid_btedata, "gauge")
        g["gauge_matrix"] = gauge
        g["is_degenerate"] = is_degenerate
    end


    # E-ph matrix in electron Wannier, phonon Bloch representation
    epdatas = [ElPhData{Float64}(nw, nmodes, nband, nband_ignore)]
    Threads.resize_nthreads!(epdatas)
    epobj_ekpR = WannierObject(model.epmat.irvec_next, zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))

    # Setup for collecting scattering processes
    max_nscat = nkq * nmodes * nband^2
    bt_mel = zeros(Complex{FT}, max_nscat)
    bt_econv_p = zeros(Bool, max_nscat)
    bt_econv_m = zeros(Bool, max_nscat) # FIXME: Cannot use BitVector because HDF5.jl does not support it.
    bt_ib = zeros(Int16, max_nscat) # For band indices, use Int16 assuming they do not exceed 32767
    bt_jb = zeros(Int16, max_nscat)
    bt_imode = zeros(Int16, max_nscat)
    bt_ik = zeros(Int, max_nscat)
    bt_ikq = zeros(Int, max_nscat)

    # Setup for collecting scattering processes
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k   points = $nk")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of k+q points = $nkq")
    println("MPI-k rank $(mpi_myrank(mpi_comm_k)), Number of q   points = $nq")
    flush(stdout)
    flush(stderr)

    nscat_tot = 0
    for ik in 1:nk
        if mod(ik, 100) == 0
            println("ik = $ik")
            # mpi_isroot() && println("ik = $ik")
            flush(stdout)
            flush(stderr)
        end
        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]

        for epdata in epdatas
            copyto!(epdata.el_k, el_k)
        end

        get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, get_u(el_k), fourier_mode)

        bt_nscat = 0

        # Threads.@threads :static for ikq in 1:nkq
        for ikq in 1:nkq
            tid = Threads.threadid()
            epdata = epdatas[tid]

            epdata.wtk = kpts.weights[ik]
            epdata.wtq = kqpts.weights[ikq]

            xkq = kqpts.vectors[ikq]

            # Find xq in qpts. Since xq can be shifted by a lattice vector, take xq from qpts.vectors
            iq = xk_to_ik(xkq - xk, qpts)
            xq = qpts.vectors[iq]

            # Copy saved electron and phonon states to epdata
            copyto!(epdata.ph, ph_save[iq])
            copyto!(epdata.el_kq, el_kq_save[ikq])

            el_kq = epdata.el_kq
            ph = epdata.ph

            # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
            check_energy_conservation_all(epdata, kqpts.ngrid, model.recip_lattice, energy_conservation...) || continue

            # Compute electron-phonon coupling
            get_eph_kR_to_kq!(epdata, epobj_ekpR, xq, fourier_mode)
            if any(abs.(xq) .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                model.polar_eph.use && epdata_compute_eph_dipole!(epdata)
            end
            epdata_set_g2!(epdata)

            # Average g2 over degenerate electron bands
            if average_degeneracy
                epdata_g2_degenerate_average!(epdata)
            end

            @timing "bt_push" @inbounds for imode in 1:nmodes, jb in el_kq.rng, ib in el_k.rng
                # Ignore negative-frequency mode
                epdata.ph.e[imode] < 0 && continue
                # Save only if the scattering satisfies energy conservation
                econv_p, econv_m = (check_energy_conservation(el_k, el_kq, ph, ib, jb, imode, sign_ph,
                kqpts.ngrid, model.recip_lattice, energy_conservation...) for sign_ph in (1, -1))
                if econv_p || econv_m
                    bt_nscat += 1
                    ω = epdata.ph.e[imode]
                    bt_mel[bt_nscat] = epdata.ep[jb, ib, imode] / sqrt(2ω)
                    bt_econv_p[bt_nscat] = econv_p
                    bt_econv_m[bt_nscat] = econv_m
                    bt_ib[bt_nscat] = ib
                    bt_jb[bt_nscat] = jb
                    bt_imode[bt_nscat] = imode
                    bt_ik[bt_nscat] = ik
                    bt_ikq[bt_nscat] = ikq
                end
                # TODO: Save bt_econv_p and bt_econv_m also in BTScattering
            end
        end # ikq

        @timing "bt_dump" @views begin
            g = create_group(fid_btedata, "scattering/ik$ik")
            g["mel"] = bt_mel[1:bt_nscat]
            g["econv_p"] = bt_econv_p[1:bt_nscat]
            g["econv_m"] = bt_econv_m[1:bt_nscat]
            g["ib"] = bt_ib[1:bt_nscat]
            g["jb"] = bt_jb[1:bt_nscat]
            g["imode"] = bt_imode[1:bt_nscat]
            g["ik"] = bt_ik[1:bt_nscat]
            g["ikq"] = bt_ikq[1:bt_nscat]
        end

        nscat_tot += bt_nscat
    end # ik

    close(fid_btedata)
    @info "nscat_tot = $nscat_tot"
    nothing
end

