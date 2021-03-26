using Test
using EPW
using NPZ

# TODO: Add test without polar_eph

@testset "cubicBN transport" begin
    # Reference data created from EPW
    μlist_ref = [11.6637, 11.7570, 11.8509] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σlist_ref_data = [
        [ 0.0007277045977865537, -9.146768638385551e-5, -6.538945892541539e-6,
         -9.146768638385552e-5,   1.1507356970359729e-5, 8.253931700230572e-7,
         -6.53894589254154e-6,    8.253931700230572e-7,  6.185872520204095e-8],
        [ 0.00011011471247628569,-1.368971748554584e-5, -9.598293254227398e-7,
         -1.3689717485545842e-5,  1.9047441221659623e-6, 1.6022725431498375e-7,
         -9.598293254227396e-7,   1.6022732760193538e-7, 1.0239805726826327e-7],
        [ 2.2221943050100873e-5, -2.2559080962660653e-6,-1.6344068322456315e-7,
         -2.2559080962660653e-6,  1.049317363049582e-6,  9.825701481924823e-8,
         -1.6344068322456313e-7,  9.825701481924826e-8,  4.882202079807381e-7]
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
    @test output["transport_σlist"] ≈ transport_σlist_ref atol=1.e-10
end
