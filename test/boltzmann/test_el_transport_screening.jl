using Test
using EPW
using LinearAlgebra

@testset "transport with screening" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)
    model.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    energy_conservation = (:None, 0.0)
    window_k  = (15.0, 16.0) .* unit_to_aru(:eV)
    window_kq = (15.0, 16.0) .* unit_to_aru(:eV)

    nklist = (15, 15, 15)
    nqlist = (15, 15, 15)

    transport_params = ElectronTransportParams{Float64}(
        Tlist = [300.0] .* unit_to_aru(:K),
        nlist = [1.0e19] .* model.volume ./ unit_to_aru(:cm)^3,
        smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
        volume = model.volume,
        nband_valence = 4,
        spin_degeneracy = 2,
    )

    # Some arbitrary screening parameter
    screening_params = LindhardScreeningParams(
        degeneracy = 4,
        m_eff = 0.4,
        n = transport_params.nlist[1] / model.volume,
        ϵM = sum(diag(model.polar_phonon.ϵ)) / 3,
        smearing = 0.05 * unit_to_aru(:eV)
    )

    # Calculate matrix elements
    @time EPW.run_transport(
        model, nklist, nqlist,
        fourier_mode = "gridopt",
        folder = tmp_dir,
        window_k  = window_k,
        window_kq = window_kq,
        energy_conservation = energy_conservation,
        use_irr_k = use_irr_k,
        average_degeneracy = false,
        run_for_qme = true,
        screening_params = screening_params,
    )

    filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")

    qme_model = load_QMEModel(filename, transport_params);
    bte_compute_μ!(qme_model);
    compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true);

    # Calculate mobility
    out_linear = solve_electron_linear_conductivity(qme_model, qme_offdiag_cutoff=Inf, rtol=1e-5);
    @test out_linear.σ_serta ≈ 0.13543274066789665 * I(3)
    @test out_linear.σ ≈ 0.07161650214864869 * I(3)
end