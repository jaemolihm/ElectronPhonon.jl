using Test
using EPW
using LinearAlgebra
using HDF5

@testset "Transport electron QME" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model_el = load_model(folder, epmat_outer_momentum="el")
    model_ph = load_model(folder, epmat_outer_momentum="ph")

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)

    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    @testset "electron doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (15.0, 16.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        # Disable symmetry because symmetry is not yet implemented in QME
        symmetry = nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = false,
            average_degeneracy = false,
            run_for_qme = true,
        )

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = 1.0e20 * model_el.volume / unit_to_aru(:cm)^3,
            volume = model_el.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        fid = h5open(filename, "r")
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
        ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in; symmetry)
        _, mobility_qme_serta_SI = transport_print_mobility(out_qme.σ_serta, transport_params, do_print=false);
        _, mobility_qme_iter_SI = transport_print_mobility(out_qme.σ, transport_params, do_print=false);

        # For electron-doped BN, there is only 1 band, so the result is identical to BTE
        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = false,
            energy_conservation = energy_conservation,
        )
        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

        output_serta = EPW.run_serta(filename_btedata, transport_params, symmetry, model_el.recip_lattice);

        # LBTE
        bte_scat_mat, el_i_bte, el_f_bte, ph_bte = EPW.compute_bte_scattering_matrix(filename_btedata, transport_params, model_el.recip_lattice);
        inv_τ = output_serta.inv_τ;
        output_lbte = EPW.solve_electron_bte(el_i_bte, el_f_bte, bte_scat_mat, inv_τ, transport_params, symmetry)
        _, mobility_bte_serta_SI = transport_print_mobility(output_lbte.σ_serta_list, transport_params, do_print=false);
        _, mobility_bte_iter_SI = transport_print_mobility(output_lbte.σ_list, transport_params, do_print=false);

        # Test QME and BTE result is identical (because there is only one band)
        @test el_i.nband == 1
        @test all(el_i.ib1 .== 1)
        @test all(el_i.ib2 .== 1)
        @test μlist_qme ≈ transport_params.μlist
        @test mobility_qme_serta_SI ≈ output_serta.mobility_SI
        @test mobility_qme_serta_SI ≈ mobility_bte_serta_SI
        @test mobility_qme_iter_SI ≈ mobility_bte_iter_SI
    end

    @testset "hole doping" begin
        # Reference data created by Julia
        μlist_ref = [0.790917195763113, 0.7915721538246762, 0.7922179662218595]
        mobility_serta_ref = reshape(hcat([
            [ 154.94103397017716      0.1326859955044396    0.1324776220752463;
                0.1326877605488312  154.94094743849948     -0.1326439790408618;
                0.13247694551742326  -0.13264006226947253 154.94099562350138],
            [  87.36380425317321      0.16932578335065523   0.16924768386860323;
                0.16932646614600216  87.36377170542119     -0.1693094755988512;
                0.1692474234844571   -0.1693079534381468   87.36378975901164],
            [  54.80031000518976      0.13764845554718236   0.13761236019306125;
                0.1376487790868413   54.80029498314644     -0.13764064298926706;
                0.13761223715127288  -0.13763991994381577  54.80030325365559]]...), 3, 3, 3)
        mobility_iter_ref = reshape(hcat([
            [ 171.3724393444944      -0.938067462675716    -0.9382996790526787;
               -0.9380642534407789  171.37233875273688      0.9381074612202703;
               -0.9382968095805431    0.9381122409514259  171.3724092682189],
            [  98.33420466690691     -0.4819817558918622   -0.48207136396687617;
               -0.48198094701940075  98.33416520844746      0.48199785392067246;
               -0.482070570718436     0.4819993393961689   98.33419407518866],
            [  62.258325888820735    -0.28605374243001874  -0.2860959621291857;
               -0.28605357204272724  62.25830715818872      0.28606168396235326;
               -0.2860957139306533    0.28606231267524573  62.25832122844622]]...), 3, 3, 3)

        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = -1.0e21 * model_el.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2
        )

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        # Disable symmetry because symmetry is not yet implemented in QME
        symmetry = nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = false,
            average_degeneracy = false,
            run_for_qme = true,
        )

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        fid = h5open(filename, "r")
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
        ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in; symmetry)
        _, mobility_qme_serta_SI = transport_print_mobility(out_qme.σ_serta, transport_params, do_print=false)
        _, mobility_qme_iter_SI = transport_print_mobility(out_qme.σ, transport_params, do_print=false);

        @test transport_params.μlist ≈ μlist_ref
        @test mobility_qme_serta_SI ≈ mobility_serta_ref
        @test mobility_qme_iter_SI ≈ mobility_iter_ref
    end
end
