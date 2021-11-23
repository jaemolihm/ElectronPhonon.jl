using Test
using EPW

# TODO: Add test without polar_eph

@testset "transport cubicBN" begin
    # Reference data created from EPW
    μlist_ref = [11.127307, 11.232810, 11.343188] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σ_ref = reshape(hcat([
        [ 4.3974489149690065e-9   3.057704386673255e-10  1.0923224611721906e-10;
          3.0577043866732556e-10  4.26880659233148e-9    4.951802817855756e-10;
          1.0923224611721906e-10  4.951802817855756e-10  3.753969439659786e-9],
        [ 2.1372873373058855e-8   1.7801428059108983e-9  6.798136162432943e-10;
          1.7801428059108983e-9   2.06140394633331e-8    2.7035137723385533e-9;
          6.798136162432944e-10   2.7035137723385533e-9  1.7750155065110347e-8],
        [ 3.563916570037439e-8    3.222579934628535e-9   1.269180882732624e-9;
          3.2225799346285356e-9   3.424099385634389e-8   4.735377493707785e-9;
          1.269180882732624e-9    4.735377493707785e-9   2.9181875864498524e-8]]...), (3, 3, 3))
    mobility_ref = reshape(hcat([
        [ 1.2625073354074245    0.08778667569148221 0.031360571697963704;
          0.08778667569148223   1.2255741318354918  0.1421662309652575;
          0.031360571697963704  0.1421662309652575  1.0777644143477636],
        [ 6.136150739663797     0.511078899151033   0.1951744508720589;
          0.511078899151033     5.91828956699082    0.7761786515208394;
          0.1951744508720589    0.7761786515208394  5.096068517825866],
        [10.232002462009381     0.9252025174314003  0.3643817598322462;
          0.9252025174314005    9.830587404465483   1.3595266112992335;
          0.3643817598322462    1.3595266112992335  8.378114914414631]]...), (3, 3, 3))

    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder)
    model.el_velocity_mode = :BerryConnection
    # model_disk = load_model(folder, true, folder)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = 80.0 * unit_to_aru(:meV)

    window_min = 10.5 * unit_to_aru(:eV)
    window_max = 11.3 * unit_to_aru(:eV)
    window = (window_min, window_max)

    nklist = (15, 15, 15)
    nqlist = (15, 15, 15)

    transport_params = ElectronTransportParams(
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
    _, mobility = transport_print_mobility(output["transport_σ"], transport_params; do_print=false)

    @test output["iband_min"] == 2
    @test output["iband_max"] == 4
    @test all(isapprox.(transport_params.μlist, μlist_ref, atol=3e-6 * unit_to_aru(:eV)))
    @test output["transport_σ"] ≈ transport_σ_ref atol=1.e-10
    @test mobility ≈ mobility_ref rtol=2e-5
end

@testset "transport Pb" begin
    # Reference data created from EPW
    μlist_ref_epw = [11.449807639297191, 11.479920504373881, 11.513292262946232] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σ_ref = reshape(hcat([
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
    model.el_velocity_mode = :BerryConnection
    # model_disk = load_model(folder, true, folder)

    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K)
    smearing = 50.0 * unit_to_aru(:meV)

    e_fermi = 11.594123 * EPW.unit_to_aru(:eV)
    window  = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi

    nklist = (10, 10, 10)
    nqlist = (10, 10, 10)

    transport_params = ElectronTransportParams(
        Tlist = Tlist,
        n = 4.0,
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
    _, mobility = transport_print_mobility(output["transport_σ"], transport_params; do_print=false)

    @test output["iband_min"] == 2
    @test output["iband_max"] == 3
    @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=2e-6 * unit_to_aru(:eV)))
    @test output["transport_σ"] ≈ transport_σ_ref atol=1.e-10
    @test mobility ≈ mobility_ref
end
