using Test
using EPW
using LinearAlgebra

@testset "Transport electron QME" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)
    model.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)

    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    @testset "electron doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window_k  = (15.0, 16.0) .* unit_to_aru(:eV)
        window_kq = (15.0, 16.2) .* unit_to_aru(:eV)

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            nlist = fill(1.0e20 * model.volume / unit_to_aru(:cm)^3, length(Tlist)),
            volume = model.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        for use_irr_k in [false, true]
            symmetry = use_irr_k ? model.el_sym.symmetry : nothing

            # Calculate matrix elements
            @time EPW.run_transport(
                model, nklist, nqlist,
                fourier_mode = "gridopt",
                folder = tmp_dir,
                window_k  = window_k,
                window_kq = window_kq,
                energy_conservation = energy_conservation,
                use_irr_k = use_irr_k,
                average_degeneracy = false,
                run_for_qme = true,
            )

            filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
            qme_model = load_QMEModel(filename, transport_params)

            # Check that all bands are non-degenerate
            @test all(qme_model.el.ib1 .== qme_model.el.ib2)

            # Compute chemical potential
            bte_compute_μ!(qme_model)
            μlist_qme = copy(qme_model.transport_params.μlist)

            # Compute scattering matrix
            compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

            # Solve QME and compute mobility
            out_qme = solve_electron_linear_conductivity(qme_model)

            # For electron-doped BN, there is only 1 band, so the result is identical to BTE
            # Calculate matrix elements
            @time EPW.run_transport(
                model, nklist, nqlist,
                fourier_mode = "gridopt",
                folder = tmp_dir,
                window_k  = window_k,
                window_kq = window_kq,
                use_irr_k = use_irr_k,
                symmetry = symmetry, # Need to explicit set because we cannot use time-reversal in QME, but BTE uses it by default
                energy_conservation = energy_conservation,
            )
            filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

            output_serta = EPW.run_serta(filename_btedata, transport_params, symmetry, model.recip_lattice);

            # LBTE
            bte_scat_mat, el_i_bte, el_f_bte, ph_bte = EPW.compute_bte_scattering_matrix(filename_btedata, transport_params, model.recip_lattice);
            inv_τ = output_serta.inv_τ;
            output_lbte = EPW.solve_electron_bte(el_i_bte, el_f_bte, bte_scat_mat, inv_τ, transport_params, symmetry)

            # Test QME and BTE result is identical (because there is only one band)
            @test qme_model.el.nband == 1
            @test all(qme_model.el.ib1 .== 5)
            @test all(qme_model.el.ib2 .== 5)
            @test μlist_qme ≈ transport_params.μlist
            @test out_qme.σ_serta ≈ output_serta.σ rtol=1e-6
            @test out_qme.σ_serta ≈ output_lbte.σ_serta rtol=1e-6
            @test out_qme.σ ≈ output_lbte.σ rtol=1e-6
        end
    end

    @testset "hole doping" begin
        # Reference data created by Julia
        μlist_ref = [0.791047740709208, 0.8040244612441655, 0.8113446910209247]
        mobility_serta_ref = [
            157.6354565237407 -0.024334294722791496 0.16012775256003797;
            -0.024334666844653526 157.6922191946616 -0.13347446554112802;
            0.1601273377014441 -0.13347448239321733 157.30209186805777;;;
            1.4069873503115224 -0.007574207818182263 -0.0060226996625005885;
            -0.007574209728391237 1.4074531282894558 0.006246910593850275;
            -0.00602270178037118 0.006246910513135652 1.4042519283126582;;;
            3.0803672115307705 -0.013576857441663563 -0.01007660095710515;
            -0.01357686057918015 3.0814173018035906 0.010582434840712423;
            -0.010076604434956218 0.010582434708509504 3.0742021833356183
        ]
        mobility_iter_ref = [
            174.14790090064858 -1.0732787799627495 -0.8877777916355186;
            -1.0800152580951061 174.18723216123917 0.89559795136831;
            -0.9234529968364676 0.9331430224729136 173.8077973232189;;;
            1.5298854896165828 -0.012201177502010427 -0.010624217529115592;
            -0.012245718321866118 1.5302384666474702 0.010733015364539227;
            -0.01081941454863688 0.010921059105715543 1.5270500554100421;;;
            3.3453730764496385 -0.022822744791515446 -0.01925452455823818;
            -0.02292995919412514 3.346171428120513 0.019495853712917403;
            -0.01972239112896868 0.019965257758520913 3.33899866309352
        ]

        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            nlist = [-1.e21, -1.e20, -1.e19] .* model.volume ./ unit_to_aru(:cm)^3,
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
        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        bte_compute_μ!(qme_model)

        # Compute scattering matrix
        compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

        # Solve QME and compute mobility
        out_qme = solve_electron_linear_conductivity(qme_model)
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

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            nlist = fill(1.0e20 * model.volume / unit_to_aru(:cm)^3, length(Tlist)),
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
        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        bte_compute_μ!(qme_model)

        # Compute scattering matrix
        compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

        # Solve QME and compute mobility
        out_qme_wo_symmetry = solve_electron_linear_conductivity(qme_model)

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
        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        bte_compute_μ!(qme_model)

        # Compute scattering matrix
        compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

        # Solve QME and compute mobility
        out_qme_w_symmetry = solve_electron_linear_conductivity(qme_model)

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

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            nlist = fill(-1.0e21 * model.volume / unit_to_aru(:cm)^3, length(Tlist)),
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
        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        bte_compute_μ!(qme_model)

        # Compute scattering matrix
        compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

        # Solve QME and compute mobility
        out_qme_wo_symmetry = solve_electron_linear_conductivity(qme_model)

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
        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        bte_compute_μ!(qme_model)

        # Compute scattering matrix
        compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

        # Solve QME and compute mobility
        out_qme_w_symmetry = solve_electron_linear_conductivity(qme_model)

        _, mobility_serta_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_serta_w_sym = transport_print_mobility(out_qme_w_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_iter_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ, transport_params, do_print=false)
        _, mobility_iter_w_sym = transport_print_mobility(out_qme_w_symmetry.σ, transport_params, do_print=false)

        # There is some error bewteen w/o and w/ symmetry, possibly because Wannier functions
        # break the symmetry. So we use a large error tolerance.
        @test all(isapprox.(mobility_serta_w_sym, mobility_serta_wo_sym, atol=5.0))
        @test all(isapprox.(mobility_iter_w_sym, mobility_iter_wo_sym, atol=5.0))

        # Transport distribution function
        @testset "TDF" begin
            el = qme_model.el
            elist = range(minimum(el.e1) - 5e-3, maximum(el.e1) + 5e-3, length=101)
            de = elist[2] - elist[1]
            smearing = 10 * unit_to_aru(:meV)

            Σ_tdf = EPW.compute_transport_distribution_function(out_qme_w_symmetry; elist, smearing, symmetry=model.el_sym.symmetry)

            @test size(Σ_tdf) == (length(elist), 3, 3, length(transport_params.Tlist))
            @test dropdims(sum(Σ_tdf, dims=1), dims=1) .* de ≈ out_qme_w_symmetry.σ
        end
    end
end


# Randomly perturb the gauge of eigenstates by setting the debugging flag
# DEBUG_random_gauge to true. And check that the result does not change.
@testset "Transport electron QME gauge" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
    window = (10.5, 11.0) .* unit_to_aru(:eV)
    window_k  = window
    window_kq = window

    transport_params = ElectronTransportParams(
        Tlist = Tlist,
        nlist = fill(-1.0e21 * model.volume / unit_to_aru(:cm)^3, length(Tlist)),
        smearing = smearing,
        nband_valence = 4,
        volume = model.volume,
        spin_degeneracy = 2
    )

    nklist = (12, 12, 12)
    nqlist = (12, 12, 12)

    for use_irr_k in [false, true]
        symmetry = use_irr_k ? model.el_sym.symmetry : nothing

        out_qme = Dict()
        for random_gauge in [false, true]
            # Calculate matrix elements
            @time EPW.run_transport(
                model, nklist, nqlist,
                fourier_mode = "gridopt",
                folder = tmp_dir,
                window_k  = window_k,
                window_kq = window_kq,
                energy_conservation = energy_conservation,
                use_irr_k = use_irr_k,
                average_degeneracy = false,
                run_for_qme = true,
                DEBUG_random_gauge = random_gauge,
            )

            filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
            qme_model = load_QMEModel(filename, transport_params)

            # Compute chemical potential
            bte_compute_μ!(qme_model)

            # Compute scattering matrix
            compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

            # Solve QME and compute mobility
            out_qme[random_gauge] = solve_electron_linear_conductivity(qme_model)
        end

        @test out_qme[true].σ_serta ≈ out_qme[false].σ_serta
        @test out_qme[true].σ ≈ out_qme[false].σ
    end
end