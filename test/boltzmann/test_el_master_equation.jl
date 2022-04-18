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
            157.54068371154364 0.09154363998690881 0.09153685724744312;
            0.09154326831940232 157.54062266873325 -0.09153645330939354;
            0.09153644197237673 -0.09153646980273679 157.54064353073207;;;
            1.4062094697183976 -0.006599763476475093 -0.006599815154239502;
            -0.0065997653844061584 1.4062089969006781 0.006599814314962396;
            -0.006599817274194147 0.00659981423613161 1.4062091565626926;;;
            3.078613970654638 -0.011378512384346727 -0.011378597397854403;
            -0.011378515518125227 3.0786132042484207 0.011378602360061158;
            -0.011378600879125093 0.011378602230955005 3.078613461682735
        ]
        mobility_iter_ref = [
            174.0412199698401 -0.9656616222990627 -0.9656695111918606;
            -0.9656632130104534 174.04114999137087 0.9656688907513974;
            -0.9656701922944877 0.9656695917020081 174.041175962007;;;
            1.5290094774671943 -0.011255798149911463 -0.01125585530478228;
            -0.01125580546743087 1.5290089545292673 0.011255847787678257;
            -0.011255859169984984 0.01125585117543175 1.5290091444848828;;;
            3.343405313641199 -0.02069377836452997 -0.020693874705709252;
            -0.02069379520071508 3.343404463866164 0.02069386770722072;
            -0.020693883572660783 0.020693875857152093 3.3434047740101334
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

        # Test without symmetry
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

        # Test use_full_grid = true
        out_qme_w_symmetry_full_grid = solve_electron_linear_conductivity(qme_model, use_full_grid=true)
        @test out_qme_w_symmetry_full_grid.σ ≈ out_qme_w_symmetry.σ rtol=0.05
        @test out_qme_w_symmetry_full_grid.σ_serta ≈ out_qme_w_symmetry.σ_serta rtol=0.05
        @test (EPW.unfold_QMEVector(out_qme_w_symmetry.δρ[1], qme_model, true, false)
               ≈ out_qme_w_symmetry_full_grid.δρ[1]) rtol=0.05
        @test (EPW.unfold_QMEVector(out_qme_w_symmetry.δρ_serta[1], qme_model, true, false)
               ≈ out_qme_w_symmetry_full_grid.δρ_serta[1]) rtol=0.05

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

        out_bte = Dict()
        out_qme = Dict()
        for qme_offdiag_cutoff in [-1, EPW.electron_degen_cutoff]
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