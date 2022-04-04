using HDF5
using OffsetArrays

"""
Debugging flags in `kwargs`
- `DEBUG_random_gauge`: Multiply random phases to the eigenstates at k+q to change the eigenstate gauge. (Default: false)
- `compute_derivative`: Compute the covariant derivative operator and write to file.
- `max_derivative_order`: Maximum order of the finite-difference formula for the covariant derivative.
    All orders from 1 to `max_derivative_order` are computed.
"""
function compute_electron_phonon_bte_data_coherence(model, btedata_prefix, window_k, window_kq, kpts,
        kqpts, qpts, nband, nstates_base_k, nstates_base_kq, energy_conservation,
        average_degeneracy, symmetry, mpi_comm_k, mpi_comm_q, qme_offdiag_cutoff;
        fourier_mode, compute_derivative=false, max_derivative_order=1, kwargs...)
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
        quantities = ["eigenvalue", "eigenvector", "velocity"]
        compute_derivative && push!(quantities, "position")
        el_k_save = compute_electron_states(model, kpts, quantities, window_k; fourier_mode)
        # for (el, xk) in zip(el_k_save, kpts.vectors)
        #     set_gauge_to_diagonalize_velocity_matrix!(el, xk, 1, model)
        # end
        el_k_boltzmann = electron_states_to_QMEStates(el_k_save, kpts, qme_offdiag_cutoff, nstates_base_k)
        dump_BTData(create_group(fid_btedata, "initialstate_electron"), el_k_boltzmann)

        # mpi_isroot() && println("Calculating electron states at k+q")
        # To ensure gauge consistency between symmetry-equivalent k points, we explicitly compute
        # electron states only for k+q in the irreducible BZ and unfold them to the full BZ.
        kqpts_irr, ik_to_ikirr_isym_kq = fold_kpoints(kqpts, symmetry)
        el_kq_save_irr = compute_electron_states(model, kqpts_irr, ["eigenvalue", "eigenvector"], window_kq; fourier_mode)
        # DEBUG: randomly change the eigenstate gauge at k+q so that the gauge is different from k
        if get(kwargs, :DEBUG_random_gauge, false) == true
            # Multiply random phase factor
            for el in el_kq_save_irr, ib in el.rng
                el.u[:, ib] .*= cispi(2*rand())
            end
            # Swap degenerate eigenvectors
            for el in el_kq_save_irr, ib in el.rng[1:end-1]
                if abs(el.e[ib] - el.e[ib+1]) < EPW.electron_degen_cutoff && rand() > 0.5
                    el.u[:, ib], el.u[:, ib+1] = el.u[:, ib+1], el.u[:, ib]
                end
            end
        end
        el_kq_save = unfold_ElectronStates(model, el_kq_save_irr, kqpts_irr, kqpts, ik_to_ikirr_isym_kq, symmetry; fourier_mode)
        el_kq_save_irr !== el_kq_save && empty!(el_kq_save_irr) # This object is not used anymore.
        el_kq_boltzmann = electron_states_to_QMEStates(el_kq_save, kqpts, qme_offdiag_cutoff, nstates_base_kq)
        dump_BTData(create_group(fid_btedata, "finalstate_electron"), el_kq_boltzmann)
        # fid_btedata["finalstate_ik_to_ikirr_isym"] = _data_julia_to_hdf5(ik_to_ikirr_isym_kq)

        # Write phonon states to HDF5 file
        mpi_isroot() && println("Calculating phonon states")
        ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]; fourier_mode)
        ph_boltzmann, _ = phonon_states_to_BTStates(ph_save, qpts)
        dump_BTData(create_group(fid_btedata, "phonon"), ph_boltzmann)

        # Write symmetry information to HDF5 file if used
        if symmetry !== nothing
            dump_BTData(create_group(fid_btedata, "symmetry"), symmetry)
        end
    end

    # If using symmetry, unfold el to full k point grid, write to file.
    if symmetry !== nothing
        el_unfold, ik_to_ikirr_isym = unfold_QMEStates(el_k_boltzmann, model.el_sym.symmetry)
        dump_BTData(create_group(fid_btedata, "initialstate_electron_unfolded"), el_unfold)
        fid_btedata["ik_to_ikirr_isym"] = _data_julia_to_hdf5(ik_to_ikirr_isym)
    end

    if compute_derivative
        el_sym = symmetry !== nothing ? model.el_sym : nothing
        for order in 1:max_derivative_order
            g = create_group(fid_btedata, "covariant_derivative_order$order")
            bvec_data = finite_difference_vectors(model.recip_lattice, kpts.ngrid; order)
            if symmetry !== nothing
                compute_covariant_derivative_matrix(el_k_boltzmann, el_k_save, bvec_data, el_sym,
                                                    el_unfold, ik_to_ikirr_isym; hdf_group=g,
                                                    fourier_mode)
            else
                compute_covariant_derivative_matrix(el_k_boltzmann, el_k_save, bvec_data;
                                                    hdf_group=g, fourier_mode)
            end
        end
    end

    # Write gauge matrices that map eigenstates in el_k_save to eigenstates in el_kq_save.
    # When not using symmetry, compute <u_mk|u_nk> (i.e. q=0).
    # When using symmetry, also compute <u_m,Sk|S|u_nk> for symmetry operations.
    # TODO: Optimize memory and disk usage by writing only nonzero matrix elements
    # TODO: Merge two cases
    # TODO: Use unfolding of el_kq to simplify this part.
    # TODO: Cleanup gauge and gauge_self.
    mpi_isroot() && println("Calculating and writing gauge matrices")

    iband_min = minimum(el.rng.start for el in el_k_save if el.nband > 0)
    iband_max = maximum(el.rng.stop  for el in el_k_save if el.nband > 0)
    rng_max = iband_min:iband_max
    gauge = OffsetArray(zeros(Complex{FT}, nband, nband, nk), rng_max, rng_max, :)
    is_degenerate = OffsetArray(zeros(Bool, nband, nband, nk), rng_max, rng_max, :)

    @timing "gauge" if symmetry !== nothing
        # Write symmetry object to file
        g = create_group(fid_btedata, "gauge/symmetry")
        dump_BTData(g, symmetry)

        tmp_arr_full = zeros(Complex{FT}, nw, nw)
        tmp_arr2_full = zeros(Complex{FT}, nw, nw)
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

                el_k = el_k_save[ik]
                el_sk = el_kq_save[isk]
                rng_k = el_k.rng
                rng_sk = el_sk.rng

                # Compute symmetry gauge matrix: S_H = U†(Sk) * S_W * U(k) = <u(Sk)|S|u(k)>
                get_fourier!(sym_k, model.el_sym.operators[isym_el], xk; fourier_mode)
                tmp_arr = view(tmp_arr_full, :, rng_k)
                tmp_arr2 = view(tmp_arr2_full, rng_sk, rng_k)
                mul!(tmp_arr, sym_k, no_offset_view(el_k.u))
                mul!(tmp_arr2, no_offset_view(el_sk.u)', tmp_arr)
                gauge[el_sk.rng, el_k.rng, ik] .= tmp_arr2
                # FIXME: Perform SVD to make gauge completely unitary

                # Set is_degenerate. Use more loose tolerance because symmetry can be slightly
                # broken at the Hamiltonian level.
                for ib in rng_k, jb in rng_sk
                    is_degenerate[jb, ib, ik] = abs(el_k.e[ib] - el_sk.e[jb]) < 10 * unit_to_aru(:meV)
                end
            end
            g = create_group(fid_btedata, "gauge/isym$isym")
            dump_BTData(create_group(g, "gauge_matrix"), gauge)
            dump_BTData(create_group(g, "is_degenerate"), is_degenerate)
        end

        # Symmetry matrix elements for symmetry that maps k to itself: Sk = k.
        # Needed for symmetrization of quantities defined on the irreducible grid.
        is_degenerate_self = OffsetArray(zeros(Bool, nband, nband), rng_max, rng_max)
        ik_list = Int[]
        for ik = 1:nk
            xk = kpts.vectors[ik]
            el_k = el_k_save[ik]
            rng_k = el_k.rng

            # First, count the number of symops that satisfy Sk = k and S /= I.
            count_total = 0
            for symop in symmetry
                isone(symop) && continue # Skip identity
                sxk = symop.is_tr ? -symop.S * xk : symop.S * xk
                if normalize_kpoint_coordinate(xk) ≈ normalize_kpoint_coordinate(sxk)
                    count_total += 1
                end
            end

            # Skip this k point if no S other than identity maps k to itself.
            count_total == 0 && continue
            push!(ik_list, ik)

            isym_list = zeros(Int, count_total)
            gauge_list = OffsetArray(zeros(Complex{FT}, nband, nband, count_total),
                                    rng_max, rng_max, 1:count_total)

            icount = 0
            for isym = 1:symmetry.nsym
                symop = symmetry[isym]
                isone(symop) && continue # Skip identity
                # Find symmetry in model.el_sym
                isym_el = findfirst(s -> s ≈ symop, model.el_sym.symmetry)

                sxk = symop.is_tr ? -symop.S * xk : symop.S * xk
                normalize_kpoint_coordinate(xk) ≈ normalize_kpoint_coordinate(sxk) || continue

                icount += 1
                isym_list[icount] = isym

                # Compute symmetry gauge matrix: S_H = U†(Sk) * S_W * U(k) = <u(k)|S|u(k)>
                get_fourier!(sym_k, model.el_sym.operators[isym_el], xk; fourier_mode)
                tmp_arr = view(tmp_arr_full, :, rng_k)
                tmp_arr2 = view(tmp_arr2_full, rng_k, rng_k)
                mul!(tmp_arr, sym_k, no_offset_view(el_k.u))
                mul!(tmp_arr2, no_offset_view(el_k.u)', tmp_arr)
                gauge_list[el_k.rng, el_k.rng, icount] .= tmp_arr2
                # FIXME: Perform SVD to make gauge completely unitary
            end

            # Set is_degenerate_self. Use more loose tolerance because symmetry can be slightly
            # broken at the Hamiltonian level.
            for ib in rng_k, jb in rng_k
                is_degenerate_self[jb, ib] = abs(el_k.e[ib] - el_k.e[jb]) < 10 * unit_to_aru(:meV)
            end
            g = create_group(fid_btedata, "gauge_self/ik$ik")
            g["isym"] = isym_list
            dump_BTData(create_group(g, "gauge_matrix"), gauge_list)
            dump_BTData(create_group(g, "is_degenerate"), is_degenerate_self)
        end
        fid_btedata["gauge_self/ik_list"] = ik_list
    else
        tmp_arr_full = zeros(Complex{FT}, nw, nw)
        @views for ik = 1:nk
            xk = kpts.vectors[ik]
            ik_kq = xk_to_ik(xk, kqpts)
            ik_kq === nothing && continue # skip if xk is not in kqpts

            el_k = el_k_save[ik]
            el_kq = el_kq_save[ik_kq]
            rng_k = el_k.rng
            rng_kq = el_kq.rng

            # Compute gauge matrix: gauge = U†(k) * U(k)
            tmp_arr = view(tmp_arr_full, rng_kq, rng_k)
            mul!(tmp_arr, no_offset_view(el_kq.u)', no_offset_view(el_k.u))
            gauge[el_kq.rng, el_k.rng, ik] .= tmp_arr

            # Set is_degenerate
            for ib in rng_k, jb in rng_kq
                is_degenerate[jb, ib, ik] = abs(el_k.e[ib] - el_kq.e[jb]) < electron_degen_cutoff
            end
        end
        g = create_group(fid_btedata, "gauge")
        dump_BTData(create_group(g, "gauge_matrix"), gauge)
        dump_BTData(create_group(g, "is_degenerate"), is_degenerate)
    end


    # E-ph matrix in electron Wannier, phonon Bloch representation
    epdatas = [ElPhData{Float64}(nw, nmodes, nband)]
    Threads.resize_nthreads!(epdatas)
    epobj_ekpR = WannierObject(model.epmat.irvec_next, zeros(ComplexF64, (nw*nband*nmodes, length(model.epmat.irvec_next))))

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
            epdata.el_k = el_k
        end

        get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, no_offset_view(el_k.u); fourier_mode)

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
            epdata.ph = ph_save[iq]
            epdata.el_kq = el_kq_save[ikq]

            el_kq = epdata.el_kq
            ph = epdata.ph

            # If all bands and modes do not satisfy energy conservation, skip this (k, q) point pair.
            check_energy_conservation_all(epdata, kqpts.ngrid, model.recip_lattice, energy_conservation...) || continue

            # Compute electron-phonon coupling
            get_eph_kR_to_kq!(epdata, epobj_ekpR, xq; fourier_mode)
            @timing "dipole" if any(abs.(xq) .> 1.0e-8) && model.use_polar_dipole
                epdata_set_mmat!(epdata)
                model.polar_eph.use && epdata_compute_eph_dipole!(epdata)
            end

            # Skip calculation of g2 because g2 is not used.
            # epdata_set_g2!(epdata)
            # # Average g2 over degenerate electron bands
            # average_degeneracy && epdata_g2_degenerate_average!(epdata)

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

