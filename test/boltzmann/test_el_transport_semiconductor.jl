using Test
using EPW
using LinearAlgebra

# TODO: Add test without polar_eph
# TODO: Add test with tetrahedron

@testset "Transport electron semiconductor SERTA" begin
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

        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
        )

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = 1.0e20 * model_el.volume / unit_to_aru(:cm)^3,
            volume = model_el.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))

        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        inv_τ = zeros(btmodel.el_i.n, length(transport_params.Tlist))
        EPW.compute_lifetime_serta!(inv_τ, btmodel, transport_params, model_el.recip_lattice)
        σ = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σ = symmetrize_array(σ, model_el.symmetry, order=2)
        _, mobility = transport_print_mobility(σ, transport_params, do_print=false)

        # Comparison with SERTA transport module (the non-Boltzmann one)
        transport_params_serta = ElectronTransportParams(
            Tlist = transport_params.Tlist,
            n = transport_params.n,
            smearing = (:Gaussian, transport_params.smearing[2]),
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2
        )

        # Run electron-phonon coupling calculation
        @time output = EPW.run_eph_outer_loop_q(
            model_ph, nklist, nqlist,
            fourier_mode="gridopt",
            window=window,
            transport_params=transport_params_serta,
        )

        _, mobility_serta = transport_print_mobility(output["transport_σ"], transport_params_serta, do_print=false)
        @test all(isapprox.(transport_params.μlist, transport_params_serta.μlist, atol=1e-7))
        @test all(isapprox.(mobility, mobility_serta, atol=2e-1))

        @testset "TDF" begin
            el = btmodel.el_i
            tdf_smearing = 10.0 * EPW.unit_to_aru(:meV)
            elist = range(minimum(el.e) - 3e-3, maximum(el.e) + 3e-3, length=1001)
            Σ_tdf = compute_transport_distribution_function(elist, tdf_smearing, el, inv_τ, transport_params, model_ph.symmetry)
            @test size(Σ_tdf) == (length(elist), 3, 3, length(transport_params.Tlist))
            @test dropdims(sum(Σ_tdf, dims=1), dims=1) .* (elist[2] - elist[1]) ≈ σ
        end
    end

    @testset "hole doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

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
        # FIXME: Here symmetry is not used (use_irr_k = false) because it changes the result a lot.
        #        Maybe symmetry is broken for e-ph coupling matrix elements.

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = -1.0e21 * model_el.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))

        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        inv_τ = zeros(btmodel.el_i.n, length(transport_params.Tlist))
        EPW.compute_lifetime_serta!(inv_τ, btmodel, transport_params, model_el.recip_lattice)
        σ = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        # σ = symmetrize_array(σ, model_el.symmetry, order=2)
        _, mobility = transport_print_mobility(σ, transport_params, do_print=false)

        # Comparison with SERTA transport module (the non-Boltzmann one)
        transport_params_serta = ElectronTransportParams(
            Tlist = transport_params.Tlist,
            n = transport_params.n,
            smearing = (:Gaussian, transport_params.smearing[2]),
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2
        )

        # Run electron-phonon coupling calculation
        @time output = EPW.run_eph_outer_loop_q(
            model_ph, nklist, nqlist,
            fourier_mode="gridopt",
            window=window,
            transport_params=transport_params_serta,
        )

        _, mobility_serta = transport_print_mobility(output["transport_σ"], transport_params_serta, do_print=false)
        @test all(isapprox.(transport_params.μlist, transport_params_serta.μlist, atol=1e-7))
        @test all(isapprox.(mobility, mobility_serta, atol=2e-1))
    end

    @testset "subgrid q" begin
        # Reference data created from Julia
        moblity_ref = cat([139.2223930040638, 63.903148701648945, 32.01401923743331] .* Ref(I(3))..., dims=3)

        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window_k  = (15.0, 15.8) .* unit_to_aru(:eV)
        window_kq = (15.0, 16.0) .* unit_to_aru(:eV)

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = 1.0e20 * model_el.volume / unit_to_aru(:cm)^3,
            volume = model_el.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        # Calculate matrix elements
        @time output = EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
        )
        filename_original = joinpath(tmp_dir, "btedata.rank0.h5")
        @test output.kpts.n == 3
        @test output.kqpts.n == 48
        @test output.qpts.n == 103

        subgrid_q_max = 0.15
        subgrid_scale = (2, 2, 2)

        for model in [model_ph, model_el]
            @time output_subgrid = run_transport_subgrid_q(
                model, output.kpts, output.qpts, output.nband, output.nband_ignore, subgrid_q_max, subgrid_scale;
                fourier_mode = "gridopt",
                window_k = window_k,
                window_kq = window_kq,
                folder = tmp_dir,
                energy_conservation = energy_conservation,
            )
            filename_subgrid = joinpath(tmp_dir, "btedata_subgrid.rank0.h5")

            @test output_subgrid.kpts.n == 3
            @test output_subgrid.kqpts.n == 216
            @test output_subgrid.qpts.n == 104
            @test output_subgrid.nband == output.nband
            @test output_subgrid.nband_ignore == output.nband_ignore

            # Calculate mobility
            output_serta_subgrid = EPW.run_serta_subgrid(filename_original, filename_subgrid, transport_params, model.symmetry, output.qpts, model.recip_lattice, do_print=false);

            @test output_serta_subgrid.mobility_SI ≈ moblity_ref
        end
    end
end

# Compare with EPW
@testset "Transport electron semiconductor EPW" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el")

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    @testset "electron doping" begin
        # Reference data created from EPW
        μlist_ref_epw = [15.359384, 15.353358, 15.347321] .* unit_to_aru(:eV)
        mobility_ref_epw_iter0 = reshape(hcat(0.484502E+03*I(3), 0.313252E+03*I(3), 0.224213E+03*I(3)), 3, 3, 3)
        mobility_ref_epw_iter1 = reshape(hcat(0.313101E+03*I(3), 0.199038E+03*I(3), 0.141522E+03*I(3)), 3, 3, 3)
        mobility_ref_epw_convd = reshape(hcat(0.357899E+03*I(3), 0.229641E+03*I(3), 0.164057E+03*I(3)), 3, 3, 3)

        Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
        smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (15.0, 15.8) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        # Calculate matrix elements
        @time output = EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
            average_degeneracy = true,
        )
        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = 1.0e20 * model.volume / unit_to_aru(:cm)^3,
            volume = model.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )
        @test transport_params.type === :Semiconductor

        # SERTA
        output_serta = EPW.run_serta(filename_btedata, transport_params, model.symmetry, model.recip_lattice);

        @test output.qpts.n == 58
        @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=2e-6 * unit_to_aru(:eV)))
        @test all(isapprox.(output_serta.mobility_SI, mobility_ref_epw_iter0, atol=1e-3))

        # LBTE
        @time bte_scat_mat, el_i, el_f, ph = EPW.compute_bte_scattering_matrix(filename_btedata, transport_params, model.recip_lattice);
        @test length(bte_scat_mat) == length(transport_params.Tlist)
        @test all(size.(bte_scat_mat) .== Ref((el_i.n, el_f.n)))

        inv_τ = output_serta.inv_τ;
        @time output_iter1 = EPW.solve_electron_bte(el_i, el_f, bte_scat_mat, inv_τ, transport_params, model.symmetry, max_iter=1);
        @time output_convd = EPW.solve_electron_bte(el_i, el_f, bte_scat_mat, inv_τ, transport_params, model.symmetry);

        _, mobility_serta = transport_print_mobility(output_iter1.σ_serta, transport_params, do_print=false)
        _, mobility_iter1 = transport_print_mobility(output_iter1.σ, transport_params, do_print=false)
        _, mobility_convd = transport_print_mobility(output_convd.σ, transport_params, do_print=false)

        @test mobility_serta ≈ output_serta.mobility_SI
        @test all(isapprox.(mobility_serta, mobility_ref_epw_iter0, atol=1e-3))
        @test all(isapprox.(mobility_iter1, mobility_ref_epw_iter1, atol=1e-3))
        @test all(isapprox.(mobility_convd, mobility_ref_epw_convd, atol=2e-3))
    end

    @testset "hole doping" begin
        # Reference data created from EPW
        μlist_ref_epw = [11.127307, 11.232810, 11.343188] .* unit_to_aru(:eV)
        mobility_ref_epw_iter0 = reshape(hcat([
            [0.125944E+01    0.865760E-01    0.296743E-01;
             0.865760E-01    0.122832E+01    0.143106E+00;
             0.296743E-01    0.143106E+00    0.107778E+01],
            [0.611312E+01    0.501977E+00    0.182462E+00;
             0.501977E+00    0.593903E+01    0.783224E+00;
             0.182462E+00    0.783224E+00    0.509630E+01],
            [0.101831E+02    0.905950E+00    0.337459E+00;
             0.905950E+00    0.987422E+01    0.137435E+01;
             0.337459E+00    0.137435E+01    0.837843E+01]]...), 3, 3, 3)
        mobility_ref_epw_iter1 = reshape(hcat([
            [0.132323E+01    0.824362E-01    0.255344E-01;
             0.824362E-01    0.129212E+01    0.147246E+00;
             0.255344E-01    0.147246E+00    0.114158E+01],
            [0.649693E+01    0.481153E+00    0.161639E+00;
             0.481153E+00    0.632284E+01    0.804048E+00;
             0.161639E+00    0.804048E+00    0.548011E+01],
            [0.109247E+02    0.868220E+00    0.299729E+00;
             0.868220E+00    0.106159E+02    0.141208E+01;
             0.299729E+00    0.141208E+01    0.912008E+01]]...), 3, 3, 3)
        mobility_ref_epw_convd = reshape(hcat([
            [0.132851E+01    0.818554E-01    0.249536E-01;
             0.818554E-01    0.129739E+01    0.147827E+00;
             0.249536E-01    0.147827E+00    0.114686E+01],
            [0.653845E+01    0.477030E+00    0.157516E+00;
             0.477030E+00    0.636435E+01    0.808171E+00;
             0.157516E+00    0.808171E+00    0.552163E+01],
            [0.110284E+02    0.858277E+00    0.289786E+00;
             0.858277E+00    0.107195E+02    0.142202E+01;
             0.289786E+00    0.142202E+01    0.922374E+01]]...), 3, 3, 3)

        Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
        smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
        energy_conservation = (:Fixed, 5 * 80.0 * EPW.unit_to_aru(:meV))
        energy_conservation = (:None, 0.0)
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        # Calculate matrix elements
        @time output = EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = false,
            energy_conservation = energy_conservation,
            average_degeneracy = true,
        )
        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = -1.0e15 * model.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model.volume,
            spin_degeneracy = 2,
        )
        @test transport_params.type === :Semiconductor

        # SERTA
        output_serta = EPW.run_serta(filename_btedata, transport_params, nothing, model.recip_lattice);
        @test output.qpts.n == 495
        @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=3e-6 * unit_to_aru(:eV)))
        @test all(isapprox.(output_serta.mobility_SI, mobility_ref_epw_iter0, atol=1e-3))

        # LBTE
        @time bte_scat_mat, el_i, el_f, ph = EPW.compute_bte_scattering_matrix(filename_btedata, transport_params, model.recip_lattice);
        @test length(bte_scat_mat) == length(transport_params.Tlist)
        @test all(size.(bte_scat_mat) .== Ref((el_i.n, el_f.n)))

        inv_τ = output_serta.inv_τ;
        @time output_iter1 = EPW.solve_electron_bte(el_i, el_f, bte_scat_mat, inv_τ, transport_params, nothing, max_iter=1);
        @time output_convd = EPW.solve_electron_bte(el_i, el_f, bte_scat_mat, inv_τ, transport_params, nothing, rtol=1e-7);
        # BTE does not converge for the default rtol = 1e-10, so I set rtol = 1e-7 to mimic EPW which uses atol = 1e-6.

        _, mobility_serta = transport_print_mobility(output_iter1.σ_serta, transport_params, do_print=false)
        _, mobility_iter1 = transport_print_mobility(output_iter1.σ, transport_params, do_print=false)
        _, mobility_convd = transport_print_mobility(output_convd.σ, transport_params, do_print=false)

        @test mobility_serta ≈ output_serta.mobility_SI
        @test all(isapprox.(mobility_serta, mobility_ref_epw_iter0, atol=1e-3))
        @test all(isapprox.(mobility_iter1, mobility_ref_epw_iter1, atol=2e-3))
        @test all(isapprox.(mobility_convd, mobility_ref_epw_convd, atol=2e-3))
    end
end

# For debugging: plot inverse lifetime
# begin
# plot(vec(output["ek"][2:4,:]), vec(output["transport_inv_τ"][:,:,1]), "o")
# plot(btmodel.el_i.e, inv_τ[:,1], "x")
# xlim([0.75, 0.81])
# axvline(transport_params.μlist[1])
# display(gcf()); clf()
# end