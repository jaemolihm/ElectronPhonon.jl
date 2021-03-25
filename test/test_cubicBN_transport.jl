using Test
using EPW
using NPZ

# TODO: Add test without polar_eph

@testset "cubicBN transport" begin
    # Reference data created from EPW
    μlist_ref = [11.6637, 11.7570, 11.8509] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σlist_ref_data = [
        [ 0.000727453,-9.14361e-5,-6.53668e-6,
         -9.14361e-5,  1.15034e-5, 8.25108e-7,
         -6.53668e-6,  8.25108e-7, 6.18373e-8],
        [ 0.000110122,-1.36906e-5,-9.59891e-7,
         -1.36906e-5,  1.90487e-6, 1.60237e-7,
         -9.59891e-7,  1.60237e-7, 1.0241e-7],
        [ 2.22193e-5, -2.25559e-6,-1.63444e-7,
         -2.25559e-6,  1.04933e-6, 9.82076e-8,
         -1.63444e-7,  9.82076e-8, 4.88302e-7]
    ]
    transport_σlist_ref = reshape(hcat(transport_σlist_ref_data...), 3, 3, 3)

    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder)
    # model_disk = load_model(folder, true, folder)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = 80.0 * unit_to_aru(:meV)

    window_min = 10.9 * unit_to_aru(:eV)
    window_max = 11.9 * unit_to_aru(:eV)
    window = (window_min, window_max)

    nklist = (15, 15, 15)
    nqlist = (15, 15, 15)

    transport_params = TransportParams{Float64}(
        Tlist = Tlist,
        n = -1.0e15 * model.volume / unit_to_aru(:cm)^3,
        smearing = smearing,
        carrier_type = "h",
        nband_valence = 4,
        spin_degeneracy = 2
    )

    # Run electron-phonon coupling calculation
    @time output = EPW.run_eph_outer_loop_q(
        model, nklist, nqlist,
        fourier_mode="gridopt",
        window=window,
        transport_params=transport_params,
    )

    # EPW.transport_print_mobility(output["transport_σlist"], transport_params, model.volume)

    @test output["iband_min"] == 2
    @test output["iband_max"] == 4
    @test transport_params.μlist ≈ μlist_ref atol=1.e-5
    @test output["transport_σlist"] ≈ transport_σlist_ref atol=1.e-9

end
