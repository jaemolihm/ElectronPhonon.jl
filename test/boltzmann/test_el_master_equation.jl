using Test
using ElectronPhonon
using LinearAlgebra

@testset "Transport electron QME" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)
    model.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)

    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    @testset "electron doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * unit_to_aru(:meV))
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
            @time ElectronPhonon.run_transport(
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
            @time ElectronPhonon.run_transport(
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

            output_serta = ElectronPhonon.run_serta(filename_btedata, transport_params, symmetry, model.recip_lattice);

            # LBTE
            bte_scat_mat, el_i_bte, el_f_bte, ph_bte = ElectronPhonon.compute_bte_scattering_matrix(filename_btedata, transport_params, model.recip_lattice);
            inv_τ = output_serta.inv_τ;
            output_lbte = ElectronPhonon.solve_electron_bte(el_i_bte, el_f_bte, bte_scat_mat, inv_τ, transport_params, symmetry)

            # Test QME and BTE result is identical (because there is only one band)
            @test qme_model.el.nband == 1
            @test all(qme_model.el.ib1 .== 5)
            @test all(qme_model.el.ib2 .== 5)
            @test μlist_qme ≈ transport_params.μlist
            @test out_qme.σ_serta ≈ output_serta.σ rtol=1e-6
            @test out_qme.σ_serta ≈ output_lbte.σ_serta rtol=1e-6
            @test out_qme.σ ≈ output_lbte.σ rtol=1e-6

            # Test MRTA
            filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
            qme_model_mrta = load_QMEModel(filename, transport_params)
            bte_compute_μ!(qme_model_mrta)
            compute_qme_scattering_matrix!(qme_model_mrta, use_mrta=true, compute_Sᵢ=false)

            # Solve QME and compute mobility
            out_qme_mrta = solve_electron_linear_conductivity(qme_model_mrta)

            # Reference results from Julia
            @test real(out_qme.σ_serta[1, 1, 1]) ≈ 0.16875742960005255 atol=1e-8
        end
    end

    @testset "hole doping" begin
        # Reference data created by Julia
        μlist_ref = [0.791047740709208, 0.8040244612441655, 0.8113446910209247]
        mobility_serta_ref = [
            157.5406808551514 0.09154363832711741 0.09153685558777472;
            0.09154326665961728 157.5406198123421 -0.09153645164973326;
            0.09153644031272026 -0.09153646814307381 157.5406406743406;;;
            1.4062094442222162 -0.006599763356813851 -0.006599815034577345;
            -0.006599765264744874 1.4062089714045054 0.006599814195300255;
            -0.006599817154531968 0.006599814116469437 1.406209131066517;;;
            3.0786139148358562 -0.011378512178041362 -0.011378597191547437;
            -0.011378515311819822 3.0786131484296537 0.011378602153754133;
            -0.011378600672818094 0.01137860202464794 3.0786134058639627
        ]
        mobility_iter_ref = [
            174.0412168151312 -0.9656616055127596 -0.9656694943739863;
            -0.9656631962263525 174.0411468366693 0.9656688739311827;
            -0.9656701754727458 0.965669574878848 174.04117280729045;;;
            1.5290094496500093 -0.01125579797567985 -0.011255854637859133;
            -0.011255805310435078 1.5290089268019726 0.011255847785408347;
            -0.011255858453071137 0.011255851105435893 1.529009114359275;;;
            3.343405253278211 -0.02069377792463766 -0.020693875463777746;
            -0.02069379470997397 3.3434044032840866 0.020693866814168325;
            -0.02069388447158823 0.020693875164414996 3.343404719304864
        ]

        energy_conservation = (:Fixed, 4 * 80.0 * unit_to_aru(:meV))
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

        # Test without symmetry
        symmetry = nothing

        # Calculate matrix elements
        @time ElectronPhonon.run_transport(
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
        @test real.(mobility_qme_serta_SI) ≈ mobility_serta_ref
        @test real.(mobility_qme_iter_SI) ≈ mobility_iter_ref
    end
end


@testset "Transport electron QME symmetry" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)

    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    @testset "electron doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * unit_to_aru(:meV))
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
        @time ElectronPhonon.run_transport(
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
        @time ElectronPhonon.run_transport(
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
        energy_conservation = (:Fixed, 4 * 80.0 * unit_to_aru(:meV))
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
        @time ElectronPhonon.run_transport(
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
        @time ElectronPhonon.run_transport(
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

        # Test use_full_grid = true
        out_qme_w_symmetry_full_grid = solve_electron_linear_conductivity(qme_model, use_full_grid=true)
        @test out_qme_w_symmetry_full_grid.σ ≈ out_qme_w_symmetry.σ rtol=0.05
        @test out_qme_w_symmetry_full_grid.σ_serta ≈ out_qme_w_symmetry.σ_serta rtol=0.05
        @test (ElectronPhonon.unfold_QMEVector(out_qme_w_symmetry.δρ[1], qme_model, true, false)
               ≈ out_qme_w_symmetry_full_grid.δρ[1]) rtol=0.05
        @test (ElectronPhonon.unfold_QMEVector(out_qme_w_symmetry.δρ_serta[1], qme_model, true, false)
               ≈ out_qme_w_symmetry_full_grid.δρ_serta[1]) rtol=0.05

        # Transport distribution function
        @testset "TDF" begin
            el = qme_model.el
            elist = range(minimum(el.e1) - 5e-3, maximum(el.e1) + 5e-3, length=101)
            de = elist[2] - elist[1]
            smearing = 10 * unit_to_aru(:meV)

            Σ_tdf = ElectronPhonon.compute_transport_distribution_function(out_qme_w_symmetry; elist, smearing, symmetry=model.el_sym.symmetry)

            @test size(Σ_tdf) == (length(elist), 3, 3, length(transport_params.Tlist))
            @test dropdims(sum(Σ_tdf, dims=1), dims=1) .* de ≈ out_qme_w_symmetry.σ
        end
    end
end


# Randomly perturb the gauge of eigenstates by setting the debugging flag
# DEBUG_random_gauge to true. And check that the result does not change.
@testset "Transport electron QME gauge" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    energy_conservation = (:Fixed, 4 * 80.0 * unit_to_aru(:meV))
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

        out_bte = Dict()
        out_qme = Dict()
        for qme_offdiag_cutoff in [-1, ElectronPhonon.electron_degen_cutoff]
            for random_gauge in [false, true]
                # Calculate matrix elements
                @time ElectronPhonon.run_transport(
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
                    qme_offdiag_cutoff = qme_offdiag_cutoff,
                )

                filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
                qme_model = load_QMEModel(filename, transport_params)

                # Compute chemical potential
                bte_compute_μ!(qme_model, do_print=false)

                # Compute scattering matrix
                compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

                # Compute mobility
                if qme_offdiag_cutoff < 0
                    out_bte[random_gauge] = solve_electron_linear_conductivity(qme_model)
                else
                    out_qme[random_gauge] = solve_electron_linear_conductivity(qme_model)
                end
            end
        end

        # For QME (including coherence between degenerate bands), the random gauge
        # should not affect the conductivity.
        @test out_qme[true].σ_serta ≈ out_qme[false].σ_serta atol=1e-6
        @test out_qme[true].σ ≈ out_qme[false].σ atol=1e-6

        # For BTE (with qme_offdiag_cutoff < 0, only diagonal states), the random gauge
        # does affect the conductivity.
        if use_irr_k != true
            # When using symmetry, the conductivity matrix is symmetrized so the effect of the
            # random gauge is not seen when comparing BTE w/ and w/o random gauge.
            @test norm(out_bte[true].σ - out_bte[false].σ) > 5e-4
            @test norm(out_bte[true].σ - out_bte[false].σ) > 5e-4
        end
        @test norm(out_qme[true].σ_serta - out_bte[false].σ_serta) > 0.1
        @test norm(out_qme[true].σ - out_bte[false].σ) > 0.1
    end
end
