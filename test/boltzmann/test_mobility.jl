using Test
using EPW

# TODO: Add test without polar_eph
# TODO: Add test with tetrahedron

@testset "boltzmann cubicBN transport" begin
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

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = 1.0e20 * model_el.volume / unit_to_aru(:cm)^3,
            volume = model_el.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))

        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        inv_τ = zeros(Float64, btmodel.el_i.n, length(transport_params.Tlist))
        EPW.compute_lifetime_serta!(inv_τ, btmodel, transport_params, model_el.recip_lattice)
        σlist = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σlist = symmetrize_array(σlist, model_el.symmetry, order=2)
        mobility = transport_print_mobility(σlist, transport_params, do_print=false)

        # Comparison with SERTA transport module (the non-Boltzmann one)
        transport_params_serta = ElectronTransportParams{Float64}(
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

        mobility_serta = transport_print_mobility(output["transport_σlist"], transport_params_serta, do_print=false)
        @test all(isapprox.(transport_params.μlist, transport_params_serta.μlist, atol=1e-7))
        @test all(isapprox.(mobility, mobility_serta, atol=2e-1))

        # test transport_distribution_function
        el = btmodel.el_i
        tdf_smearing = 10.0 * EPW.unit_to_aru(:meV)
        elist = range(minimum(el.e) - 3e-3, maximum(el.e) + 3e-3, length=1001)
        Σ_tdf = compute_transport_distribution_function(elist, tdf_smearing, el, inv_τ, transport_params, model_ph.symmetry)
        @test size(Σ_tdf) == (length(elist), 3, 3, length(transport_params.Tlist))
        @test dropdims(sum(Σ_tdf, dims=1), dims=1) .* (elist[2] - elist[1]) ≈ σlist
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

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = -1.0e21 * model_el.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))

        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        inv_τ = zeros(Float64, btmodel.el_i.n, length(transport_params.Tlist))
        EPW.compute_lifetime_serta!(inv_τ, btmodel, transport_params, model_el.recip_lattice)
        σlist = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        # σlist = symmetrize_array(σlist, model_el.symmetry, order=2)
        mobility = transport_print_mobility(σlist, transport_params, do_print=false)

        # Comparison with SERTA transport module (the non-Boltzmann one)
        transport_params_serta = ElectronTransportParams{Float64}(
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

        mobility_serta = transport_print_mobility(output["transport_σlist"], transport_params_serta, do_print=false)
        @test all(isapprox.(transport_params.μlist, transport_params_serta.μlist, atol=1e-7))
        @test all(isapprox.(mobility, mobility_serta, atol=2e-1))
    end

    @testset "subgrid q" begin
        energy_conservation = (:Linear, 0.0)
        window_k  = (15.0, 15.8) .* unit_to_aru(:eV)
        window_kq = (15.0, 16.0) .* unit_to_aru(:eV)

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
        @test output.kpts.n == 3
        @test output.kqpts.n == 48
        @test output.qpts.n == 103

        subgrid_q_max = 0.15
        subgrid_scale = (2, 2, 2)

        @time output_subgrid = run_transport_subgrid_q(
            model_ph, output.kpts, output.qpts, output.nband, output.nband_ignore, subgrid_q_max, subgrid_scale;
            fourier_mode = "gridopt",
            window_k = window_k,
            window_kq = window_kq,
            folder = tmp_dir,
            energy_conservation = energy_conservation,
        )
        @test output_subgrid.kpts.n == 3
        @test output_subgrid.kqpts.n == 216
        @test output_subgrid.qpts.n == 104
        @test output_subgrid.nband == output.nband
        @test output_subgrid.nband_ignore == output.nband_ignore

        @time output_subgrid_el = run_transport_subgrid_q(
            model_el, output.kpts, output.qpts, output.nband, output.nband_ignore, subgrid_q_max, subgrid_scale;
            fourier_mode = "gridopt",
            window_k = window_k,
            window_kq = window_kq,
            folder = tmp_dir,
            energy_conservation = energy_conservation,
        )
        @test output_subgrid_el.kpts.n == 3
        @test output_subgrid_el.kqpts.n == 216
        @test output_subgrid_el.qpts.n == 104
        @test output_subgrid_el.nband == output.nband
        @test output_subgrid_el.nband_ignore == output.nband_ignore
    end
end

# Compare with EPW
@testset "boltzmann cubicBN transport epw" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model_el = load_model(folder, epmat_outer_momentum="el")

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    @testset "electron doping" begin
        # Reference data created from EPW
        μlist_ref_epw = [15.361083, 15.355056, 15.349019] .* unit_to_aru(:eV)
        mobility_ref_epw = reshape(hcat([
            [ 0.525206E+03    0.000000E+00    0.000000E+00;
              0.000000E+00    0.525206E+03    0.000000E+00;
              0.000000E+00    0.000000E+00    0.525206E+03],
            [ 0.339362E+03    0.000000E+00    0.000000E+00;
              0.000000E+00    0.339362E+03    0.000000E+00;
              0.000000E+00    0.000000E+00    0.339362E+03],
            [ 0.242791E+03    0.000000E+00    0.000000E+00;
              0.000000E+00    0.242791E+03    0.000000E+00;
              0.000000E+00    0.000000E+00    0.242791E+03]]...), 3, 3, 3)

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
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
            average_degeneracy = true,
        )

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = 1.0e20 * model_el.volume / unit_to_aru(:cm)^3,
            volume = model_el.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")
        output_serta = EPW.run_serta(filename_btedata, transport_params, model_el.symmetry, model_el.recip_lattice);

        @test output.qpts.n == 58
        @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=2e-6 * unit_to_aru(:eV)))
        @test all(isapprox.(output_serta.mobility_list, mobility_ref_epw, atol=1e-3))
    end

    @testset "hole doping" begin
        # Reference data created from EPW
        μlist_ref_epw = [11.127288, 11.232666, 11.342843] .* unit_to_aru(:eV)
        mobility_ref_epw = -reshape(hcat([
            [ 0.109805E+01  0.276227E-01  -0.139287E+00;
              0.276227E-01  0.103019E+01   0.177595E-01;
             -0.139287E+00  0.177595E-01   0.117709E+01],
            [ 0.542588E+01  0.171530E+00  -0.777768E+00;
              0.171530E+00  0.504036E+01   0.865870E-01;
             -0.777768E+00  0.865870E-01   0.587510E+01],
            [ 0.912517E+01  0.319938E+00  -0.137994E+01;
              0.319938E+00  0.843549E+01   0.142274E+00;
             -0.137994E+01  0.142274E+00   0.992912E+01]]...), 3, 3, 3)

        Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
        smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
        energy_conservation = (:Fixed, 5 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        # Calculate matrix elements
        @time output = EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = false,
            energy_conservation = energy_conservation,
            average_degeneracy = true,
        )
        # FIXME: Here symmetry is not used (use_irr_k = false) because it changes the result a lot.
        #        Maybe symmetry is broken for e-ph coupling matrix elements.

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = -1.0e15 * model_el.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2,
        )

        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")
        output_serta = EPW.run_serta(filename_btedata, transport_params, nothing, model_el.recip_lattice);

        @test output.qpts.n == 495
        @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=2e-6 * unit_to_aru(:eV)))
        @test all(isapprox.(output_serta.mobility_list, mobility_ref_epw, atol=3e-4))
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