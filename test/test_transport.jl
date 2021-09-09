using Test
using EPW

# TODO: Add test without polar_eph

@testset "transport cubicBN" begin
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
        [ 0.024341848385298275   0.0017976997720365465 -0.0015859104074518598;
          0.0017976997720365465  0.025073001524908968  -4.161136565781236e-5;
         -0.0015859104074518598 -4.161136565781236e-5   0.02463728829197615],
        [ 0.46429768249417       0.0394567227566647    -0.03301748058818396;
          0.039456722756664675   0.4799778290404716    -0.001841748396655468;
         -0.03301748058818397   -0.001841748396655468   0.4709083696759164],
        [ 1.708100083413528      0.1551594399716512    -0.12693186812529644;
          0.15515943997165113    1.7692096268340531    -0.008759379311972432;
         -0.12693186812529644   -0.00875937931197242    1.7347781206960147]]...), (3, 3, 3))

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
    @test mobility ≈ mobility_ref rtol=2e-5
end

@testset "transport Pb" begin
    # Reference data created from EPW
    μlist_ref_epw = [11.449807639297191, 11.479920504373881, 11.513292262946232] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σlist_ref = reshape(hcat([
        [2.1009148910410884     1.514425894002856e-8   1.3577962623538853e-7;
         1.5144258904349697e-8  2.100914882194067      2.7799404737957244e-7;
         1.3577962625322795e-7  2.77994047361733e-7    2.1009149651273433],
        [0.4670355892710652     2.645264165759879e-9   1.8803176589177695e-8;
         2.645264170219737e-9   0.4670355930413226     3.600603958193026e-8;
         1.8803176606947443e-8  3.600603959084998e-8   0.4670355990922323],
        [0.18348216693725655    9.294028961602618e-10  5.697353468048338e-9;
         9.294028928153685e-10  0.18348216908881146    1.0447758250842288e-8;
         5.697353468048338e-9   1.04477582530722e-8    0.1834821698292078]]...), (3, 3, 3))
    mobility_ref = reshape(hcat([
        [4.450042764704906      3.207773918413988e-8   2.8760096179988953e-7;
         3.2077739108566864e-8  4.4500427459656295     5.888317534650217e-7;
         2.87600961837676e-7    5.888317534272352e-7   4.4500429216303505],
        [0.9892491855610105     5.603053560984393e-9   3.9827857992226685e-8;
         5.60305357043102e-9    0.9892491935469631     7.62660194425332e-8;
         3.982785802986559e-8   7.626601946142647e-8   0.9892492063636684],
        [0.3886418687940784     1.9686102712634212e-9  1.2067821826847654e-8;
         1.968610264178452e-9   0.38864187335138406    2.212986885371047e-8;
         1.2067821826847654e-8  2.2129868858433747e-8  0.38864187491965096]]...), (3, 3, 3))

    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_Pb")

    model = load_model(folder)
    # model_disk = load_model(folder, true, folder)

    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K)
    smearing = 50.0 * unit_to_aru(:meV)

    e_fermi = 11.594123 * EPW.unit_to_aru(:eV)
    window  = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi

    nklist = (10, 10, 10)
    nqlist = (10, 10, 10)

    transport_params = ElectronTransportParams{Float64}(
        Tlist = Tlist,
        n = 4,
        volume = model.volume,
        smearing = (:Gaussian, smearing),
        nband_valence = 0,
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
    @test output["iband_max"] == 3
    @test transport_params.μlist ≈ μlist_ref_epw atol=1.e-7
    @test output["transport_σlist"] ≈ transport_σlist_ref atol=1.e-10
    @test mobility ≈ mobility_ref
end
