using Test
using EPW

# TODO: Add test without polar_eph

@testset "cubicBN transport" begin
    # Reference data created from EPW
    μlist_ref = [11.138597, 11.247340, 11.357359] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σlist_ref_data = [
        [6.679703708189413e-9 -2.0436829784200155e-10 2.929302418762372e-10;
        -2.0436829784200176e-10 6.628794599955531e-9 -5.807371865799194e-10;
        2.929302418762373e-10 -5.807371865799194e-10 1.0247776600023677e-8],
        [1.272079939076327e-7 -4.028623691601275e-9 6.515938390733756e-9;
        -4.0286236916012755e-9 1.261459276891857e-7 -1.2685585188816758e-8;
        6.515938390733756e-9 -1.2685585188816758e-8 2.0366319596835616e-7],
        [4.6754483480973187e-7 -1.4969270849906877e-8 2.5731495540812904e-8;
        -1.496927084990687e-8 4.6350774299947803e-7 -4.976362345164354e-8;
        2.5731495540812884e-8 -4.976362345164354e-8 7.652329739886983e-7]
    ]
    transport_σlist_ref = reshape(hcat(transport_σlist_ref_data...), 3, 3, 3)

    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder)
    # model_disk = load_model(folder, true, folder)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = 80.0 * unit_to_aru(:meV)

    window_min = 10.5 * unit_to_aru(:eV)
    window_max = 11.3 * unit_to_aru(:eV)
    window = (window_min, window_max)

    nklist = (12, 12, 12)
    nqlist = (12, 12, 12)

    transport_params = ElectronTransportParams{Float64}(
        Tlist = Tlist,
        n = -1.0e15 * model.volume / unit_to_aru(:cm)^3,
        smearing = (:Gaussian, smearing),
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
    @test transport_params.μlist ≈ μlist_ref atol=1.e-7
    @test output["transport_σlist"] ≈ transport_σlist_ref atol=1.e-10
end
