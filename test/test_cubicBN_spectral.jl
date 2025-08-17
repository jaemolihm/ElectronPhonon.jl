using Test
using ElectronPhonon

@testset "cubicBN spectral" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model_from_epw_new(folder, "temp", "bn"; epmat_outer_momentum = "ph")
    # model_disk = load_model(folder; epmat_on_disk=true, tmpdir=folder)

    μ = 25.0 * unit_to_aru(:eV)
    Tlist = [200.0, 300.0] .* unit_to_aru(:K)
    smearing = 500.0 * unit_to_aru(:meV)

    window_min = 5.0 * unit_to_aru(:eV)
    window_max = 45.0 * unit_to_aru(:eV)
    window = (window_min, window_max)

    nklist = (5, 5, 5)
    xqs = reshape([0.1, 0.2, 0.4], (3, 1))
    qpts = ElectronPhonon.Kpoints(xqs)

    phspec_params = PhononSpectralParams(
        μ = μ,
        Tlist = Tlist,
        smearing = smearing,
        degeneracy = 2.0,
        ωlist = [0.0, 0.1] .* unit_to_aru(:eV),
    )

    # Run electron-phonon coupling calculation
    @time output = ElectronPhonon.run_eph_outer_loop_q(
        model, nklist, qpts,
        fourier_mode="normal",
        window=window,
        phspec_params=phspec_params,
    )

    @time output_gridopt = ElectronPhonon.run_eph_outer_loop_q(
        model, nklist, qpts,
        fourier_mode="gridopt",
        window=window,
        phspec_params=phspec_params,
    )

    # @time output_disk_gridopt = ElectronPhonon.run_eph_outer_loop_q(
    #     model_disk, nklist, qpts,
    #     fourier_mode="gridopt",
    #     window=window,
    #     phspec_params=phspec_params,
    # )

    @testset "WannierObject, normal" begin
        @test size(output["omega"]) == (6, 1)
        @test size(output["ph_green"]) == (2, 6, 2, 1)
        @test size(output["ph_selfen_dynamic"]) == (2, 6, 2, 1)
        @test size(output["ph_selfen_static"]) == (6, 2, 1)
        @test output["ph_selfen_static"] ≈ real.(output["ph_selfen_dynamic"][1, :, :, :])
        # Comparing with previous Julia calculation (not EPW)
        @test output["ph_green"][1,1,1,1] ≈ -473.7462896170925 - 90.54066555067868im
        @test output["ph_green"][2,4,2,1] ≈ -202.17360033173236 - 368.6411650451125im
        @test output["ph_selfen_dynamic"][1,1,1,1] ≈ -0.00531192372209388 - 0.00038919931243949887im
        @test output["ph_selfen_dynamic"][2,3,2,1] ≈ -0.00857591990430974 - 0.002064030220318118im
    end

    @testset "WannierObject, gridopt" begin
        for key in keys(output)
            key == "kpts" && continue
            @test output_gridopt[key] ≈ output[key]
        end
    end

    # @testset "DiskWannierObject, gridopt" begin
    #     for key in keys(output)
    #         key == "kpts" && continue
    #         @test output_disk_gridopt[key] ≈ output[key]
    #     end
    # end
end
