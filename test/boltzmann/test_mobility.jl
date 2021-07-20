using Test
using EPW
using NPZ

# TODO: Add test without polar_eph
# This file is currently just copied from test_cubicBN_transport.jl.
# Need to make some changes...

@testset "boltzmann cubicBN transport" begin
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

    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model_el = load_model(folder, epmat_outer_momentum="el")
    model_ph = load_model(folder, epmat_outer_momentum="ph")

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)

    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))

    @testset "electron doping" begin
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
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))

        EPW.bte_compute_μ!(transport_params, btmodel.el_i, model_el.volume, do_print=false)

        inv_τ = zeros(Float64, btmodel.el_i.n, length(transport_params.Tlist))
        EPW.compute_lifetime_serta!(inv_τ, btmodel, transport_params, model_el.recip_lattice)
        σlist = EPW.compute_mobility_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σlist = symmetrize_array(σlist, model_el.symmetry, order=2)
        mobility = EPW.transport_print_mobility(σlist, transport_params, model_el.volume, do_print=false)

        @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=1e-7))
        @test all(isapprox.(mobility, mobility_ref_epw, atol=2e-1))

        # Comparison with SERTA transport module (the non-Boltzmann one)
        transport_params_serta = TransportParams{Float64}(
            Tlist = transport_params.Tlist,
            n = transport_params.n,
            smearing = transport_params.smearing[2],
            carrier_type = "e",
            nband_valence = 4,
            spin_degeneracy = 2
        )

        # Run electron-phonon coupling calculation
        @time output = EPW.run_eph_outer_loop_q(
            model_ph, nklist, nqlist,
            fourier_mode="gridopt",
            window=window,
            transport_params=transport_params_serta,
        )

        mobility_serta = EPW.transport_print_mobility(output["transport_σlist"], transport_params_serta, model_ph.volume, do_print=false)
        @test all(isapprox.(transport_params.μlist, transport_params_serta.μlist, atol=1e-7))
        @test all(isapprox.(mobility, mobility_serta, atol=2e-1))
    end

    @testset "hole doping" begin
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
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))

        EPW.bte_compute_μ!(transport_params, btmodel.el_i, model_el.volume, do_print=false)

        inv_τ = zeros(Float64, btmodel.el_i.n, length(transport_params.Tlist))
        EPW.compute_lifetime_serta!(inv_τ, btmodel, transport_params, model_el.recip_lattice)
        σlist = EPW.compute_mobility_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        # σlist = symmetrize_array(σlist, model_el.symmetry, order=2)
        mobility = EPW.transport_print_mobility(σlist, transport_params, model_el.volume, do_print=false)

        # Comparison with SERTA transport module (the non-Boltzmann one)
        transport_params_serta = TransportParams{Float64}(
            Tlist = transport_params.Tlist,
            n = transport_params.n,
            smearing = transport_params.smearing[2],
            carrier_type = "h",
            nband_valence = 4,
            spin_degeneracy = 2
        )

        # Run electron-phonon coupling calculation
        @time output = EPW.run_eph_outer_loop_q(
            model_ph, nklist, nqlist,
            fourier_mode="gridopt",
            window=window,
            transport_params=transport_params_serta,
        )

        mobility_serta = EPW.transport_print_mobility(output["transport_σlist"], transport_params_serta, model_ph.volume, do_print=false)
        @test all(isapprox.(transport_params.μlist, transport_params_serta.μlist, atol=1e-7))
        @test all(isapprox.(mobility, mobility_serta, atol=2e-1))
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