using Test
using EPW

# TODO: Add test without polar_eph

@testset "cubicBN transport" begin
    # Reference data created from EPW
    μlist_ref = [11.138597, 11.247340, 11.357359] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σlist_ref = reshape(hcat(
        [[8.386040777954402e-11  -2.565806813819003e-12   3.677571442017926e-12;
         -2.5658068138190042e-12  8.322127605245854e-11  -7.290848557790536e-12;
          3.677571442017928e-12  -7.290848557790533e-12   1.2865581129327742e-10],
         [1.5970333986816406e-9  -5.0578508251715215e-11  8.180401536779802e-11;
         -5.057850825171531e-11   1.5836998188712031e-9  -1.592609186531417e-10;
          8.180401536779802e-11  -1.5926091865314174e-10  2.556890603710858e-9],
         [5.8697921196906595e-9  -1.8793559810983822e-10  3.230449718901945e-10;
         -1.8793559810983855e-10  5.819108916930781e-9   -6.247565010552057e-10;
          3.230449718901945e-10  -6.247565010552059e-10   9.60711822658491e-9]]...), (3, 3, 3))
    mobility_ref = reshape(hcat(
        [[-0.02407631834255848    0.0007366429914985549 -0.0010558305535347456;
           0.0007366429914985553 -0.02389282365976858    0.0020932022096315974;
          -0.001055830553534746   0.0020932022096315965 -0.036937076164238046],
         [-0.45850819866555115    0.014521086865702788  -0.023485928197156693;
           0.014521086865702814  -0.45468012865418267    0.045723800760681194;
          -0.023485928197156693   0.0457238007606812    -0.7340831480795107],
         [-1.6852169864214068     0.05395629961502913   -0.09274619320523259;
           0.05395629961502923   -1.6706658417683165     0.179367556207582;
          -0.09274619320523259    0.17936755620758205   -2.7582031008711114]]...), (3, 3, 3))

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
    # Following tests are broken due to the use of gauge fixing for degenerate states
    @test_broken output["transport_σlist"] ≈ transport_σlist_ref atol=1.e-10
    @test_broken mobility ≈ mobility_ref
end
