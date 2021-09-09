using Test
using EPW
using NPZ

# TODO: Add test with use_ws = false

@testset "cubicBN self-energy" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    # Load reference data (calculated from EPW)
    ek_ref = npzread(joinpath(folder, "el_energy.npy")) * unit_to_aru(:eV)
    el_imsigma_ref = imag(npzread(joinpath(folder, "el_imsigma.npy"))) * unit_to_aru(:eV)
    omega_ref = npzread(joinpath(folder, "ph_energy.npy")) * unit_to_aru(:eV)
    ph_imsigma_ref = npzread(joinpath(folder, "ph_imsigma.npy")) * unit_to_aru(:eV)

    model = load_model(folder)
    model_disk = load_model(folder, true, folder)

    μ = 25.0 * unit_to_aru(:eV)
    Tlist = [200.0, 300.0] .* unit_to_aru(:K)
    smearing = 500.0 * unit_to_aru(:meV)

    window_min = 5.0 * unit_to_aru(:eV)
    window_max = 45.0 * unit_to_aru(:eV)
    window = (window_min, window_max)

    nklist = (5, 5, 5)
    nqlist = (5, 5, 5)

    elself_params = ElectronSelfEnergyParams(
    μ = μ,
    Tlist = Tlist,
    smearing = smearing,
    )

    phself_params = PhononSelfEnergyParams(
    μ = μ,
    Tlist = Tlist,
    smearing = smearing,
    spin_degeneracy = 2.0
    )

    # Run electron-phonon coupling calculation
    @time output = EPW.run_eph_outer_loop_q(
        model, nklist, nqlist,
        fourier_mode="gridopt",
        window=window,
        elself_params=elself_params,
        phself_params=phself_params,
    )

    @time output_disk = EPW.run_eph_outer_loop_q(
        model_disk, nklist, nqlist,
        fourier_mode="gridopt",
        window=window,
        elself_params=elself_params,
        phself_params=phself_params,
    )

    @testset "WannierObject" begin
        iband_min = output["iband_min"]
        iband_max = output["iband_max"]
        @test iband_min == 2
        @test iband_max == 8

        @test ek_ref ≈ output["ek"][iband_min:iband_max, :] atol=2.e-4
        @test all(isapprox.(omega_ref, output["omega"], atol=1.e-8))
        @test all(isapprox.(ph_imsigma_ref, output["phself_imsigma"], atol=1.e-8))

        # Electron self-energy error can be large for states whose energy is near the window
        # boundary, with separation not much larger than the smearing.
        # So, we test only states is far from the window boundary.
        inds = vec((abs.(ek_ref .- window_min) .> smearing * 10)
                .& (abs.(ek_ref .- window_max) .> smearing * 10))
        errors = abs.(el_imsigma_ref - output["elself_imsigma"])
        @test maximum(reshape(errors, :, length(Tlist))[inds, :]) < 1.e-8
    end

    @testset "DiskWannierObject" begin
        iband_min = output_disk["iband_min"]
        iband_max = output_disk["iband_max"]
        @test iband_min == 2
        @test iband_max == 8

        @test ek_ref ≈ output_disk["ek"][iband_min:iband_max, :] atol=2.e-4
        @test all(isapprox.(omega_ref, output_disk["omega"], atol=1.e-8))
        @test all(isapprox.(ph_imsigma_ref, output_disk["phself_imsigma"], atol=1.e-8))

        # Electron self-energy error can be large for states whose energy is near the window
        # boundary, with separation not much larger than the smearing.
        # So, we test only states is far from the window boundary.
        inds = vec((abs.(ek_ref .- window_min) .> smearing * 10)
                .& (abs.(ek_ref .- window_max) .> smearing * 10))
        errors = abs.(el_imsigma_ref - output_disk["elself_imsigma"])
        @test maximum(reshape(errors, :, length(Tlist))[inds, :]) < 1.e-8
    end
end
