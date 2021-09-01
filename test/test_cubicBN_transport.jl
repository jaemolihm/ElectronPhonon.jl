using Test
using EPW

# TODO: Add test without polar_eph

@testset "cubicBN transport" begin
    # Reference data created from EPW
    μlist_ref = [11.138597, 11.247340, 11.357359] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σlist_ref = reshape(hcat([
        [ 8.478527749363637e-11   6.261581766913868e-12  -5.523896618182155e-12;
          6.261581766913868e-12   8.733196256253513e-11  -1.449368646268298e-13;
         -5.523896618182155e-12  -1.449368646268298e-13   8.581432648259921e-11],
        [ 1.6171987938966909e-9   1.3743201152849928e-10  -1.150034380914425e-10;
          1.374320115284992e-10   1.671814431749074e-9    -6.41500786678983e-12;
         -1.1500343809144254e-10 -6.41500786678983e-12     1.6402245287651454e-9],
        [ 5.949496409097154e-9    5.404370270295094e-10   -4.421173565557497e-10;
          5.404370270295092e-10   6.1623475251836345e-9   -3.050986079126793e-11;
         -4.421173565557497e-10  -3.0509860791267884e-11   6.042418883930548e-9]]...), (3, 3, 3))
    mobility_ref = reshape(hcat([
        [-0.024341848385298275  -0.0017976997720365465  0.0015859104074518598;
         -0.0017976997720365465 -0.025073001524908968   4.161136565781236e-5;
          0.0015859104074518598  4.161136565781236e-5  -0.02463728829197615],
        [-0.46429768249417      -0.0394567227566647     0.03301748058818396;
         -0.039456722756664675  -0.4799778290404716     0.001841748396655468;
          0.03301748058818397    0.001841748396655468  -0.4709083696759164],
        [-1.708100083413528     -0.1551594399716512     0.12693186812529644;
         -0.15515943997165113   -1.7692096268340531     0.008759379311972432;
          0.12693186812529644    0.00875937931197242   -1.7347781206960147]]...), (3, 3, 3))

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
        volume = model.volume,
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
    mobility = transport_print_mobility(output["transport_σlist"], transport_params; do_print=false)

    @test output["iband_min"] == 2
    @test output["iband_max"] == 4
    @test transport_params.μlist ≈ μlist_ref atol=1.e-7
    @test output["transport_σlist"] ≈ transport_σlist_ref atol=1.e-10
    @test mobility ≈ mobility_ref
end
