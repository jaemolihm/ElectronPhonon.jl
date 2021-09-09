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
        # Comparing with previous Julia calculation (not EPW)
        @test output["ph_green"][1,1,1,1] ≈ -475.7952489842732 - 85.19500012385838im
        @test output["ph_green"][2,4,2,1] ≈ -227.200162845575 - 380.47175530108007im
        @test output["ph_selfen_dynamic"][1,1,1,1] ≈ -0.005171810651802322 - 0.00036464326665328067im
        @test output["ph_selfen_dynamic"][2,3,2,1] ≈ -0.008272200993521831 - 0.0016227371488728394im
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
