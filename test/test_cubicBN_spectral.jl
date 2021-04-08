using Test
using EPW
using NPZ

@testset "cubicBN spectral" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder)
    model_disk = load_model(folder, true, folder)

    μ = 25.0 * unit_to_aru(:eV)
    Tlist = [200.0, 300.0] .* unit_to_aru(:K)
    smearing = 500.0 * unit_to_aru(:meV)

    window_min = 5.0 * unit_to_aru(:eV)
    window_max = 45.0 * unit_to_aru(:eV)
    window = (window_min, window_max)

    nklist = (5, 5, 5)
    xqs = reshape([0.1, 0.2, 0.4], (3, 1))
    qpts = EPW.Kpoints(xqs)

    phspec_params = PhononSpectralParams(
        μ = μ,
        Tlist = Tlist,
        smearing = smearing,
        degeneracy = 2.0,
        ωlist = [0.0, 0.1] .* unit_to_aru(:eV),
    )

    # Run electron-phonon coupling calculation
    @time output = EPW.run_eph_outer_loop_q(
        model, nklist, qpts,
        fourier_mode="normal",
        window=window,
        phspec_params=phspec_params,
    )

    @time output_gridopt = EPW.run_eph_outer_loop_q(
        model, nklist, qpts,
        fourier_mode="gridopt",
        window=window,
        phspec_params=phspec_params,
    )

    @time output_disk_gridopt = EPW.run_eph_outer_loop_q(
        model_disk, nklist, qpts,
        fourier_mode="gridopt",
        window=window,
        phspec_params=phspec_params,
    )

    @testset "WannierObject, normal" begin
        @test size(output["omega"]) == (6, 1)
        @test size(output["ph_green"]) == (2, 6, 2, 1)
        @test size(output["ph_selfen_dynamic"]) == (2, 6, 2, 1)
        @test size(output["ph_selfen_static"]) == (6, 2, 1)
        @test output["ph_selfen_static"] ≈ real.(output["ph_selfen_dynamic"][1, :, :, :])
        @test output["ph_green"][1,1,1,1] ≈ -476.10107160679075 - 84.36380057758412im
        @test output["ph_green"][2,4,2,1] ≈ -69.80496828863502 - 184.80079603184387im
        @test output["ph_selfen_dynamic"][1,1,1,1] ≈ -0.00751013463005002 - 0.00036085370536193207im
        @test output["ph_selfen_dynamic"][2,3,2,1] ≈ -0.013672290161049485 - 0.002974318480002591im
    end

    @testset "WannierObject, gridopt" begin
        for key in keys(output)
            @test output_gridopt[key] ≈ output[key]
        end
    end

    @testset "DiskWannierObject, gridopt" begin
        for key in keys(output)
            @test output_disk_gridopt[key] ≈ output[key]
        end
    end
end
