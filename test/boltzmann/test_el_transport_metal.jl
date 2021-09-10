using Test
using EPW
using LinearAlgebra

@testset "Transport electron metal" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_Pb")

    model_el = load_model(folder, epmat_outer_momentum="el")

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    # Reference data from EPW
    μlist_ref_epw = [11.449807639297191, 11.479920504373881, 11.513292262946232] .* unit_to_aru(:eV)
    σ_ref_epw_iter0 = reshape(hcat(0.966388E+05*I(3), 0.214829E+05*I(3), 0.843989E+04*I(3)), 3, 3, 3)
    σ_ref_epw_iter1 = reshape(hcat(0.107119E+06*I(3), 0.239672E+05*I(3), 0.943501E+04*I(3)), 3, 3, 3)
    σ_ref_epw_convd = reshape(hcat(0.110039E+06*I(3), 0.246179E+05*I(3), 0.968863E+04*I(3)), 3, 3, 3)

    # Parameters
    e_fermi = 11.594123 * EPW.unit_to_aru(:eV)
    window_k  = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi
    window_kq = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi
    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 50.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 10 * 50.0 * EPW.unit_to_aru(:meV))

    nklist = (10, 10, 10)
    nqlist = (10, 10, 10)

    # Calculate matrix elements
    @time output = EPW.run_transport(
        model_el, nklist, nqlist,
        fourier_mode = "gridopt",
        folder = tmp_dir,
        window_k  = window_k,
        window_kq = window_kq,
        use_irr_k = true,
        energy_conservation = energy_conservation,
        average_degeneracy = true,
    )
    filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")
    @test output.qpts.n == 723

    transport_params = ElectronTransportParams{Float64}(
        Tlist = Tlist,
        n = 4,
        smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
        nband_valence = 0,
        volume = model_el.volume,
        spin_degeneracy = 2
    )
    @test transport_params.type === :Metal

    # SERTA
    output_serta = EPW.run_serta(filename_btedata, transport_params, model_el.symmetry, model_el.recip_lattice)
    @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=2e-6 * unit_to_aru(:eV)))
    @test all(isapprox.(output_serta.σ_SI, σ_ref_epw_iter0, atol=0.2))

    # TODO: Test LBTE
    # @show maximum(abs.(output_serta.σ_SI .- σ_ref_epw_iter0))
    # @show abs.(output_serta.σ_SI .- σ_ref_epw_iter0) ./ output_serta.σ_SI

    # TODO: Test TDF
end