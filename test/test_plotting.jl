using Test
using ElectronPhonon
using PyPlot

@testset "plotting" begin
    model = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "el")
    model_ph = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "ph")

    nfigs = length(PyPlot.get_fignums())

    out = plot_bandstructure(model; kline_density=10)
    @test out.fig isa PyPlot.Figure
    @test keys(out) == (:fig, :plotaxes, :kpts, :e_el, :e_ph, :plot_xdata)

    out = plot_deformation_potential(model)
    @test out.fig isa PyPlot.Figure
    @test keys(out) == (:fig, :e_ph, :deformation_potential, :qpts, :plot_xdata)

    out = plot_deformation_potential(model, Vec3(0.0, 0.5, 0.0); band_rng=1:2, kline_density=15, include_polar=false)
    @test out.fig isa PyPlot.Figure
    @test keys(out) == (:fig, :e_ph, :deformation_potential, :qpts, :plot_xdata)

    @test plot_decay(model.el_ham, model.lattice) isa PyPlot.Figure
    @test plot_decay(model) isa PyPlot.Figure
    @test plot_decay_eph(model) isa PyPlot.Figure
    @test plot_decay_eph(model_ph) isa PyPlot.Figure

    # test all figures are closed
    @test length(PyPlot.get_fignums()) == nfigs
end
