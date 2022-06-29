using Test
using EPW

# TODO: Add test without polar_eph

@testset "transport cubicBN" begin
    # Reference data created from EPW
    μlist_ref = [11.127307, 11.232810, 11.343188] .* unit_to_aru(:eV)

    # Reference data created from Julia
    transport_σ_ref = [4.397192544511247e-9    3.057526397413464e-10  1.092259219766887e-10;
                       3.057526397413463e-10   4.2685576381938534e-9  4.95151280158682e-10;
                       1.0922592197668872e-10  4.951512801586821e-10  3.753750593232604e-9;;;
                       2.1371627330847385e-8   1.780039171299877e-9   6.797742248663104e-10;
                       1.7800391712998775e-9   2.061283719795638e-8   2.703355411001862e-9;
                       6.797742248663112e-10   2.703355411001862e-9   1.7749120286842603e-8;;;
                       3.563708814159844e-8    3.222392333712107e-9   1.2691073202639052e-9;
                       3.222392333712107e-9    3.4238996985773257e-8  4.73510012228937e-9;
                       1.2691073202639066e-9   4.7351001222893704e-9  2.918017483918564e-8]
    mobility_ref = [ 1.262433731010478    0.08778156558934018  0.03135875602632175;
                     0.08778156558934017  1.2255026566768887   0.1421579045488031;
                     0.0313587560263218   0.1421579045488031   1.0777015831641632;;;
                     6.1357930010151795   0.5110491456600553   0.19516314160994908;
                     0.5110491456600555   5.917944396668989    0.7761331859900091;
                     0.19516314160994927  0.7761331859900091   5.095771433043573;;;
                    10.231405993166403    0.9251486570528965   0.36436063998625484;
                     0.9251486570528965   9.830014101273708    1.3594469777367564;
                     0.36436063998625484  1.359446977736756    8.377626548640242]

    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder)
    model.el_velocity_mode = :BerryConnection

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = 80.0 * unit_to_aru(:meV)

    window_min = 10.5 * unit_to_aru(:eV)
    window_max = 11.3 * unit_to_aru(:eV)
    window = (window_min, window_max)

    nklist = (15, 15, 15)
    nqlist = (15, 15, 15)

    transport_params = ElectronTransportParams(
        Tlist = Tlist,
        nlist = fill(-1.0e15 * model.volume / unit_to_aru(:cm)^3, length(Tlist)),
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
    @test output["transport_σ"] ≈ transport_σ_ref rtol=2e-5
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
    mobility_ref = [
        4.450042684021957 3.207698136099118e-8 2.876011293828138e-7;
        3.2076981285418175e-8 4.450042665281902 5.888319095166289e-7;
        2.876011293827769e-7 5.88831909441056e-7 4.450042840946738;;;
        0.9892491676248766 5.6029544020747825e-9 3.9827881055981323e-8;
        5.602954392628157e-9 0.9892491756107371 7.626603728100495e-8;
        3.982788104653469e-8 7.626603728100495e-8 0.9892491884274566;;;
        0.38864186174758036 1.9685809534593025e-9 1.2067828807856599e-8;
        1.9685809510976456e-9 0.38864186630486136 2.2129873753722652e-8;
        1.2067828805494904e-8 2.2129873746637645e-8 0.3886418678731324
    ]

    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_Pb")

    model = load_model(folder)
    model.el_velocity_mode = :BerryConnection

    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K)
    smearing = 50.0 * unit_to_aru(:meV)

    e_fermi = 11.594123 * EPW.unit_to_aru(:eV)
    window  = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi

    nklist = (10, 10, 10)
    nqlist = (10, 10, 10)

    transport_params = ElectronTransportParams(
        Tlist = Tlist,
        nlist = fill(4.0, length(Tlist)),
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
