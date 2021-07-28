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
        # Reference data created from EPW
        μlist_ref_epw = [15.361083, 15.355056, 15.349019] .* unit_to_aru(:eV)
        mobility_ref_epw_data = [
            [0.525206E+03    0.101254E-05   -0.105157E-07;
            0.101254E-05    0.525206E+03    0.136829E-05;
            -0.105157E-07    0.136829E-05    0.525206E+03],
            [0.339362E+03    0.340667E-05    0.274588E-05;
            0.340667E-05    0.339362E+03   -0.186869E-05;
            0.274588E-05   -0.186869E-05    0.339362E+03],
            [0.242791E+03    0.105480E-03    0.105008E-03;
            0.105480E-03    0.242791E+03   -0.104381E-03;
            0.105008E-03   -0.104381E-03    0.242791E+03],
        ]
        mobility_ref_epw = reshape(hcat(mobility_ref_epw_data...), 3, 3, 3)

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
        σlist = EPW.compute_mobility_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σlist = symmetrize_array(σlist, model_el.symmetry, order=2)
        mobility = transport_print_mobility(σlist, transport_params, do_print=false)

        @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=1e-7))
        @test all(isapprox.(mobility, mobility_ref_epw, atol=2e-1))

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
        σlist = EPW.compute_mobility_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
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

# For debugging: plot inverse lifetime
# begin
# plot(vec(output["ek"][2:4,:]), vec(output["transport_inv_τ"][:,:,1]), "o")
# plot(btmodel.el_i.e, inv_τ[:,1], "x")
# xlim([0.75, 0.81])
# axvline(transport_params.μlist[1])
# display(gcf()); clf()
# end