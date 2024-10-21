function load_wfpt(folder, outdir, prefix, nw, nmodes, recip_lattice, alat)
    # Read real-space WFPT data

    # Read coarse q-points from file
    # Need to use the read q points because dwmat, dgmat, sthmat is computed for each q points
    # so the order of q points must be the same.

    xq = zeros(3)
    xqs_cart = Vec3{Float64}[]
    f = open(joinpath(folder, outdir, "$prefix.xqc1"), "r")
    while !eof(f)
        read!(f, xq)
        push!(xqs_cart, copy(xq))
    end
    close(f)

    # Convert from crystal to Cartesian
    xqs = (Ref(recip_lattice) .\ xqs_cart) .* (2π / alat)
    qpts_coarse = Kpoints(xqs)
    nqc = qpts_coarse.n

    # Read Wigner-Seitz information
    f = open(joinpath(folder, "wigner.fmt"), "r")
    nr_el, nr_ph, nr_ep, dims, dims2 = parse.(Int, split(readline(f)))
    irvec_el, ndegen_el, wslen_el = _parse_wigner(f, nr_el, dims,  dims)
    close(f)

    ind_el = sortperm(irvec_el, by=reverse)
    irvec_el = irvec_el[ind_el]
    ndegen_el = ndegen_el[:, :, ind_el]

    # Momentum matrix elements
    pmat = zeros(ComplexF64, 3, nw, nw, nr_el)
    f = open(joinpath(folder, outdir, "$prefix.cpmew1"), "r")
    read!(f, pmat)
    close(f)

    # Debye-Waller matrix elements
    dw = zeros(ComplexF64, nw, nw, nr_el, 3, nmodes)
    f = open(joinpath(folder, outdir, "$prefix.dwmatwe1"), "r")
    read!(f, dw)
    close(f)

    # Electron-phonon correction matrix elements
    dg = zeros(ComplexF64, nw, nw, nr_el, nmodes, nqc)
    f = open(joinpath(folder, outdir, "$prefix.dgmatwe1"), "r")
    read!(f, dg)
    close(f)

    # Sternheimer matrix elements
    sth = zeros(ComplexF64, nw, nw, nr_el, nmodes, nmodes, nqc)
    f = open(joinpath(folder, outdir, "$prefix.sthmatwe1"), "r")
    read!(f, sth)
    close(f)

    pmat = permutedims(pmat, [2, 3, 1, 4])[:, :, :, ind_el]
    dw = permutedims(dw, [1, 2, 4, 5, 3])[:, :, :, :, ind_el]
    dg = permutedims(dg, [1, 2, 4, 3, 5])[:, :, :, ind_el, :]
    sth = permutedims(sth, [1, 2, 4, 5, 3, 6])[:, :, :, :, ind_el, :]

    for ir in 1:nr_el
        for j in 1:nw
            for i in 1:nw
                @views if ndegen_el[i, j, ir] == 0
                    pmat[i, j, :, ir] .= 0
                    dw[i, j, :, :, ir] .= 0
                    dg[i, j, :, ir, :] .= 0
                    sth[i, j, :, :, ir, :] .= 0
                else
                    pmat[i, j, :, ir] ./= ndegen_el[i, j, ir]
                    dw[i, j, :, :, ir] ./= ndegen_el[i, j, ir]
                    dg[i, j, :, ir, :] ./= ndegen_el[i, j, ir]
                    sth[i, j, :, :, ir, :] ./= ndegen_el[i, j, ir]
                end
            end
        end
    end

    el_mom = WannierObject(irvec_el, reshape(pmat, :, nr_el))
    dwmat = WannierObject(irvec_el, reshape(dw, :, nr_el))
    dgmat = [WannierObject(irvec_el, reshape(dg[:, :, :, :, iq], :, nr_el)) for iq in 1:nqc]
    sthmat = [WannierObject(irvec_el, reshape(sth[:, :, :, :, :, iq], :, nr_el)) for iq in 1:nqc]

    (; el_mom, dwmat, dgmat, sthmat, qpts_coarse)
end;


function compute_debye_waller_active_space(model, kpts, el_mom, window; fourier_mode = "gridopt", wannier_gauge = true)
    # Compute active space Debye-Waller matrix element
    # If wannier_gauge is true, return the matrix element in the Wannier gauge
    # Otherwise, return in the Bloch gauge.

    (; nw, nmodes) = model

    # Debye-Waller is defined at q=0
    xq = Vec3(0., 0., 0.)

    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector"]; fourier_mode);

    # E-ph matrix in electron Bloch, phonon Wannier representation
    ep_ekpR_obj = WannierObject(model.epmat.irvec_next,
        zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))
    epmat = get_interpolator(model.epmat; fourier_mode)
    ep_ekpR = get_interpolator(ep_ekpR_obj; fourier_mode)


    epdata = ElPhData(nw, nmodes)
    epdata.ph = compute_phonon_states(model, ElectronPhonon.Kpoints(xq), ["eigenvalue", "eigenvector"])[1]

    dw_active = zeros(ComplexF64, nw, nw, 3, nmodes, kpts.n)

    mom_itp = get_interpolator(el_mom; fourier_mode)

    for (ik, xk) in enumerate(kpts.vectors)
        epdata.el_k  = el_k_save[ik]
        epdata.el_kq = el_k_save[ik]
        uk = epdata.el_k.u

        get_eph_RR_to_kR!(ep_ekpR_obj, epmat, xk, no_offset_view(uk))
        get_eph_kR_to_kq!(no_offset_view(epdata.ep), ep_ekpR, xq, I(model.nmodes), no_offset_view(uk))

        ElectronPhonon.set_velocity!(epdata.el_k, mom_itp, xk, :Direct)

        for imode in 1:nmodes, idir in 1:3
            for ib in 1:nw, jb in 1:nw, pb in 1:nw
                if window[1] <= epdata.el_k.e[pb] <= window[2]
                    dw_active[ib, jb, idir, imode, ik] += im * (
                        epdata.ep[pb, ib, imode]' * epdata.el_k.v[pb, jb][idir]
                        - epdata.el_k.v[pb, ib][idir]' * epdata.ep[pb, jb, imode] )
                end
            end
        end

        # Transform from Bloch to Wannier gauge
        if wannier_gauge
            @views for imode in 1:nmodes, idir in 1:3
                dw_active[:, :, idir, imode, ik] .= uk * dw_active[:, :, idir, imode, ik] * uk'
            end
        end
    end

    dw_active
end


function run_wfpt(folder, outdir, prefix, model, window_wfpt, kpts, occupation_params, wfpt_objs = nothing)
    fourier_mode = "gridopt"
    (; nw, nmodes) = model

    nchunks_threads = 2 * nthreads()

    if wfpt_objs === nothing
        @time el_mom, dwmat, dgmat, sthmat, qpts_coarse = ElectronPhonon.load_wfpt(folder, outdir, prefix, nw, nmodes, model.recip_lattice, model.alat)
    else
        (; el_mom, dwmat, dgmat, sthmat, qpts_coarse) = wfpt_objs
    end

    ph_save = compute_phonon_states(model, qpts_coarse, ["eigenvalue", "eigenvector", "eph_dipole_coeff"]);

    dw_active = ElectronPhonon.compute_debye_waller_active_space(model, kpts, el_mom, window_wfpt)

    dw_itps = get_interpolator_channel(dwmat; fourier_mode)
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector"]; fourier_mode);

    # E-ph matrix in electron Bloch, phonon Wannier representation
    ep_ekpR_obj = WannierObject(model.epmat.irvec_next,
        zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))
    epmat = get_interpolator(model.epmat; fourier_mode)
    ep_ekpRs = get_interpolator_channel(ep_ekpR_obj; fourier_mode)

    epdatas = Channel{ElPhData{Float64}}(nthreads())
    foreach(1:nthreads()) do _
        put!(epdatas, ElPhData{Float64}(nw, nmodes))
    end

    Σ_Fan_channel = [[zeros(nw, length(occupation_params.Tlist)) for _ in 1:kpts.n] for _ in 1:nchunks_threads]
    Σ_DW_channel = [[zeros(nw, length(occupation_params.Tlist)) for _ in 1:kpts.n] for _ in 1:nchunks_threads]

    for ik in 1:kpts.n
        if mod(ik, 10) == 0 && mpi_isroot()
            mpi_isroot() && @printf "ik = %5d / %5d\n" ik kpts.n
            flush(stdout)
            flush(stderr)
        end

        xk = kpts.vectors[ik]
        el_k = el_k_save[ik]
        uk = el_k.u

        get_eph_RR_to_kR!(ep_ekpR_obj, epmat, xk, no_offset_view(uk))

        for (id_chunk, iqs) in enumerate(chunks(1:qpts_coarse.n; n = nchunks_threads))
        # @threads for (id_chunk, iqs) in enumerate(chunks(1:qpts_coarse.n; n = nchunks_threads))
            epdata = take!(epdatas)
            ep_ekpR = take!(ep_ekpRs)
            dw_itp = take!(dw_itps)
            Σ_Fan_chunk = Σ_Fan_channel[id_chunk]
            Σ_DW_chunk = Σ_DW_channel[id_chunk]

            epdata.el_k = el_k

            for iq in iqs
                xq = qpts_coarse.vectors[iq]

                ph = ph_save[iq]
                dg_itp = get_interpolator(dgmat[iq]; fourier_mode = "normal")
                sth_itp = get_interpolator(sthmat[iq]; fourier_mode = "normal")

                el_kq = compute_electron_states(model, Kpoints(xk + xq), ["eigenvalue", "eigenvector"])[1]

                epdata.el_kq = el_kq
                epdata.ph = ph

                ukq = el_kq.u
                u_ph = ph_save[iq].u

                # Interpolate WFPT matrix elements from (Re, q) to (k, q)
                get_fourier!(dw_itp.out, dw_itp, xk)
                get_fourier!(dg_itp.out, dg_itp, xk)
                get_fourier!(sth_itp.out, sth_itp, xk)

                dg_k_tmp = reshape(dg_itp.out, nw, nw, nmodes)
                dg_k = zeros(ComplexF64, nw, nw, nmodes)
                @views for imode in 1:nmodes, jmode in 1:nmodes
                    dg_k[:, :, imode] .+= dg_k_tmp[:, :, jmode] .* u_ph[jmode, imode]
                end

                sth_k_tmp = reshape(sth_itp.out, nw, nw, nmodes, nmodes)
                sth_k = zeros(ComplexF64, nw, nw, nmodes)
                @views for imode in 1:nmodes, jmode in 1:nmodes, kmode in 1:nmodes
                    sth_k[:, :, imode] .+= sth_k_tmp[:, :, kmode, jmode] .* conj(u_ph[kmode, imode]) .* u_ph[jmode, imode]
                end

                dw_k_tmp = reshape(dw_itp.out, nw, nw, 3, nmodes)
                dw_k_tmp .-= view(dw_active, :, :, :, :, ik)  # Subtract active-space Debye-Waller contribution

                dw_k = zeros(ComplexF64, nw, nw, nmodes)
                @views for imode in 1:nmodes, jdir in 1:3, kmode in 1:nmodes
                    iatm = div(imode - 1, 3) + 1
                    jmode = 3 * (iatm - 1) + jdir
                    dw_k[:, :, kmode] .+= dw_k_tmp[:, :, jdir, imode] .* conj(u_ph[imode, kmode]) .* u_ph[jmode, kmode]
                end

                @views for imode in 1:nmodes
                    dg_k[ :, :, imode] .= ukq' * dg_k[ :, :, imode] * uk
                    sth_k[:, :, imode] .= uk'  * sth_k[:, :, imode] * uk
                    dw_k[ :, :, imode] .= uk'  * dw_k[ :, :, imode] * uk
                end

                # Add g*g contribution to the Sternheimer matrix
                get_eph_kR_to_kq!(epdata, ep_ekpR, xq)
                epdata_set_mmat!(epdata)
                epdata_compute_eph_dipole!(epdata)

                for imode in 1:nmodes, ib in 1:nw, jb in 1:nw, pb in 1:nw
                    ek1 = el_k.e[ib]
                    ek2 = el_k.e[jb]
                    ekq = el_kq.e[pb]
                    if ekq < window_wfpt[1] || ekq > window_wfpt[2]
                        if window_wfpt[1] <= ek1 <= window_wfpt[2] && window_wfpt[1] <= ek2 <= window_wfpt[2]
                            # sth_k[ib, jb, imode] += dg_k[pb, ib, imode]' * epdata.ep[pb, jb, imode] / (ek1 - ekq)
                            sth_k[ib, jb, imode] += epdata.ep[pb, ib, imode]' * epdata.ep[pb, jb, imode] / (ek1 - ekq)
                        end
                    end
                end

                # Compute rest-space contribution to the self-energy

                for (iocc, (; T)) in enumerate(occupation_params)
                    for imode in 1:nmodes
                        ωq = ph.e[imode]
                        ωq < ElectronPhonon.omega_acoustic && continue

                        nq = occ_boson(ωq, T)
                        coeff = (nq + 1/2) / 2ωq

                        for ib in el_k.rng
                            Σ_Fan_chunk[ik][ib, iocc] += coeff * 2 * real(sth_k[ib, ib, imode])
                            Σ_DW_chunk[ik][ib, iocc] += coeff * real(dw_k[ib, ib, imode])
                        end
                    end
                end
            end # iq

            put!(ep_ekpRs, ep_ekpR)
            put!(epdatas, epdata)
            put!(dw_itps, dw_itp)

        end # iq chunk
    end # ik

    Σ_Fan = sum(Σ_Fan_channel)
    Σ_DW = sum(Σ_DW_channel)

    Σ_Fan ./= qpts_coarse.n
    Σ_DW ./= qpts_coarse.n
    Σ = Σ_Fan + Σ_DW

    (; Σ, Σ_Fan, Σ_DW)
end



function compute_self_energy_active_DW(folder, outdir, prefix, model, window, kpts, qpts, occupation_params, wfpt_objs = nothing)
    fourier_mode = "gridopt"
    (; nw, nmodes) = model

    if wfpt_objs === nothing
        @time el_mom, dwmat, _, _ = ElectronPhonon.load_wfpt(folder, outdir, prefix, nw, nmodes, model.recip_lattice, model.alat)
    else
        (; el_mom, dwmat) = wfpt_objs
    end

    ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector"]);

    dw_active = ElectronPhonon.compute_debye_waller_active_space(model, kpts, el_mom, window; wannier_gauge=false)

    # dw_itp = get_interpolator(dwmat)
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector"]; fourier_mode);

    Σ = [zeros(nw, length(occupation_params.Tlist)) for _ in 1:kpts.n]
    dw_k = zeros(ComplexF64, nw, nw, nmodes)

    for iq in 1:qpts.n
        ph = ph_save[iq]

        for ik in 1:kpts.n
            # xk = kpts.vectors[ik]
            el_k = el_k_save[ik]

            # uk = el_k.u
            u_ph = ph.u

            # get_fourier!(dw_itp.out, dw_itp, xk)

            # dw_active is in the Wannier gauge, so we don't need to multiply uk
            dw_k_tmp = view(dw_active, :, :, :, :, ik)

            dw_k .= 0
            @views for imode in 1:nmodes, jdir in 1:3, kmode in 1:nmodes
                iatm = div(imode - 1, 3) + 1
                jmode = 3 * (iatm - 1) + jdir
                dw_k[:, :, kmode] .+= dw_k_tmp[:, :, jdir, imode] .* conj(u_ph[imode, kmode]) .* u_ph[jmode, kmode]
            end

            # Compute rest-space contribution to the self-energy

            for (iocc, (; T)) in enumerate(occupation_params)
                for imode in 1:nmodes
                    ωq = ph.e[imode]
                    ωq < ElectronPhonon.omega_acoustic && continue

                    nq = occ_boson(ωq, T)
                    coeff = (nq + 1/2) / 2ωq * qpts.weights[iq]

                    for ib in el_k.rng
                        Σ[ik][ib, iocc] += coeff * real(dw_k[ib, ib, imode])
                    end
                end
            end
        end
    end

    Σ
end
