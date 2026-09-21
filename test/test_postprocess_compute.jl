using Test
using ElectronPhonon

@testset "compute bandstructure" begin
    model = _load_model_from_artifacts("cubicBN"; load_epmat = false)

    # The numerics behind plot_bandstructure. Since we use a band path, gridopt is not useful.
    kpts, plot_xdata = high_symmetry_kpath(model; kline_density=10)
    e_el = compute_eigenvalues_el(model, kpts; fourier_mode="normal")
    e_ph = compute_eigenvalues_ph(model, kpts; fourier_mode="normal")

    @test length(plot_xdata.x) == kpts.n
    @test size(e_el) == (model.nw, kpts.n)
    @test size(e_ph) == (model.nmodes, kpts.n)
    @test all(isfinite, e_el)
    @test all(isfinite, e_ph)
end

@testset "compute deformation pot." begin
    model = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "el")
    model_ph = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "ph")

    # Test that the function runs. No test for the correctness.
    outs = [compute_deformation_potential(model),
            compute_deformation_potential(model_ph),
            compute_deformation_potential(model, Vec3(0.0, 0.5, 0.0), band_rng=1:2, kline_density=15, include_polar=false)]

    for out in outs
        @test size(out.deformation_potential) == (model.nmodes, out.qpts.n)
        @test size(out.e_ph) == (model.nmodes, out.qpts.n)
        @test all(isfinite, out.deformation_potential)
        @test all(isfinite, out.e_ph)
    end
end
