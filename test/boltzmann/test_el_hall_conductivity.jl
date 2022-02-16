using Test
using EPW
using LinearAlgebra
using HDF5

@testset "QME transport Hall" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)
    model.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    window_k  = (10.5, 11.0) .* unit_to_aru(:eV)
    window_kq = (10.5, 11.0) .* unit_to_aru(:eV)
    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
    qme_offdiag_cutoff = EPW.electron_degen_cutoff

    transport_params = ElectronTransportParams(
        Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K),
        n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
        smearing = smearing,
        nband_valence = 4,
        volume = model.volume,
        spin_degeneracy = 2
    )

    nk = 20
    symmetry = nothing

    # Calculate matrix elements
    @time EPW.run_transport(
        model, (nk, nk, nk), (nk, nk, nk),
        fourier_mode = "gridopt",
        folder = tmp_dir,
        window_k  = window_k,
        window_kq = window_kq,
        energy_conservation = energy_conservation,
        use_irr_k = false,
        average_degeneracy = false,
        run_for_qme = true,
        compute_derivative = true,
    )

    filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
    fid = h5open(filename, "r")
    el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
    el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
    ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
    ∇ = EPW.load_covariant_derivative_matrix(fid["covariant_derivative"]);
    close(fid)

    bte_compute_μ!(transport_params, EPW.BTStates(el_i))

    @testset "constant RTA" begin
        inv_τ_constant = 10.0 * unit_to_aru(:meV)
        r_hall_ref_123 = [-0.11847158752557092, -0.14823502018049978, -0.15333964481975168]

        # Set scattering matrix with constant relaxation time
        S_out = [I(el_i.n) * (-inv_τ_constant + 0.0im) for _ in transport_params.Tlist]

        # Solve linear electrical conductivity
        out_linear = solve_electron_qme(transport_params, el_i, nothing, S_out; symmetry)

        # Solve linear Hall conductivity
        qme_model = EPW.QMEModel(; el=el_i, ∇=Vec3(∇), transport_params, S_out)
        out_hall = compute_linear_hall_conductivity(out_linear, qme_model);
        @test out_hall.r_hall[1, 2, 3, :] ≈ r_hall_ref_123 atol=1e-5
    end

    @testset "SERTA" begin
        r_hall_ref_123 = [-0.011250559739887816, -0.028690945110432325, -0.051159534325475216]

        # Calculate scattering matrix
        S_out, _ = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph, compute_S_in=false);

        # Set QMEModel
        qme_model = EPW.QMEModel(; el=el_i, ∇=Vec3(∇), transport_params, S_out);

        # Solve linear electrical conductivity
        out_linear = solve_electron_qme(transport_params, el_i, el_f, S_out; filename, symmetry);

        # Solve linear Hall conductivity
        out_hall = compute_linear_hall_conductivity(out_linear, qme_model);
        @test out_hall.r_hall[1, 2, 3, :] ≈ r_hall_ref_123 atol=1e-5
    end
end
