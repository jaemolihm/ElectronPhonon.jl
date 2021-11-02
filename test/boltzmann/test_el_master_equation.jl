using Test
using EPW
using LinearAlgebra
using HDF5

@testset "Transport electron QME" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el")

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

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = 1.0e20 * model.volume / unit_to_aru(:cm)^3,
            volume = model.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        # Disable symmetry because symmetry is not yet implemented in QME
        symmetry = nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model, nklist, nqlist,
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
        _, mobility_qme_serta_SI = transport_print_mobility(out_qme.σ_serta, transport_params, do_print=false);
        _, mobility_qme_iter_SI = transport_print_mobility(out_qme.σ, transport_params, do_print=false);

        # For electron-doped BN, there is only 1 band, so the result is identical to BTE
        # Calculate matrix elements
        @time EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = false,
            energy_conservation = energy_conservation,
        )
        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

        output_serta = EPW.run_serta(filename_btedata, transport_params, symmetry, model.recip_lattice);

        # LBTE
        bte_scat_mat, el_i_bte, el_f_bte, ph_bte = EPW.compute_bte_scattering_matrix(filename_btedata, transport_params, model.recip_lattice);
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
        μlist_ref = [0.7910477407091759, 0.7917002728887299, 0.7923440548581497]
        mobility_serta_ref = reshape(hcat([
            [ 157.54068371069306       0.0915436436160749    0.09153685519794288;
                0.09154327194860364  157.5406226667718      -0.0915364517867568;
                0.09153643992296691   -0.09153646828015147 157.54064354080072],
            [  89.79625493542417       0.16325211674102438   0.16324940485381034;
                0.1632519706664871    89.79623165980605     -0.16324879390216274;
                0.16324924143854314   -0.1632488004840559   89.79623954265853],
            [  56.612681325715194      0.140192601728844     0.14019126993125194;
                0.14019253203977936   56.6126705034983      -0.1401907682350232;
                0.1401911919179557    -0.14019077140042663  56.61267412807016]]...), 3, 3, 3)
        mobility_iter_ref = reshape(hcat([
            [ 174.0412199688272      -0.9656616189724784   -0.9656695135777413;
               -0.9656632094011806  174.0411499897708       0.9656688931573153;
               -0.965670193526415     0.9656695929271255  174.041175972123],
            [ 100.92254171104952     -0.49534539870529504  -0.4953487104558193;
               -0.49534644920597304 100.92251404577405      0.49534890423490796;
               -0.4953492742400059    0.49534941740468746 100.92252455539021],
            [  64.21026564170282     -0.29319692702881484  -0.29319861236581;
               -0.29319762273739086  64.21025245891612      0.29319892145085186;
               -0.2931990230615029    0.29319928275273494  64.2102575644252] ]...), 3, 3, 3)

        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model.volume,
            spin_degeneracy = 2
        )

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        # Disable symmetry because symmetry is not yet implemented in QME
        symmetry = nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model, nklist, nqlist,
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


@testset "Transport electron QME symmetry" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

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

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = 1.0e20 * model.volume / unit_to_aru(:cm)^3,
            volume = model.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        # Run without symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
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
        out_qme_wo_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in)

        # Run with symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = true,
            symmetry = model.el_sym.symmetry,
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
        out_qme_w_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in,
                                                symmetry=model.el_sym.symmetry; filename)

        _, mobility_serta_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_serta_w_sym = transport_print_mobility(out_qme_w_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_iter_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ, transport_params, do_print=false)
        _, mobility_iter_w_sym = transport_print_mobility(out_qme_w_symmetry.σ, transport_params, do_print=false)

        @test all(isapprox.(mobility_serta_w_sym, mobility_serta_wo_sym, atol=1e-2))
        @test all(isapprox.(mobility_iter_w_sym, mobility_iter_wo_sym, atol=0.2))
    end

    @testset "hole doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model.volume,
            spin_degeneracy = 2
        )

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        # Run without symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
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
        out_qme_wo_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in)

        # Run with symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = true,
            symmetry = model.el_sym.symmetry,
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
        out_qme_w_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in,
                                                symmetry=model.el_sym.symmetry; filename)


        _, mobility_serta_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_serta_w_sym = transport_print_mobility(out_qme_w_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_iter_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ, transport_params, do_print=false)
        _, mobility_iter_w_sym = transport_print_mobility(out_qme_w_symmetry.σ, transport_params, do_print=false)

        # There is some error bewteen w/o and w/ symmetry, possibly because Wannier functions
        # break the symmetry. So we use a large error tolerance.
        @test all(isapprox.(mobility_serta_w_sym, mobility_serta_wo_sym, atol=5.0))
        @test all(isapprox.(mobility_iter_w_sym, mobility_iter_wo_sym, atol=5.0))
    end
end
