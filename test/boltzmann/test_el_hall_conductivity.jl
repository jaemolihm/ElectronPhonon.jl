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

    window = (10.5, 11.0) .* unit_to_aru(:eV)
    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
    qme_offdiag_cutoff = EPW.electron_degen_cutoff

    transport_params = ElectronTransportParams(
        Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K),
        n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
        smearing = smearing,
        nband_valence = 4,
        volume = model.volume,
        spin_degeneracy = 2
    )

    @testset "constant RTA" begin
        r_hall_ref_123 = [-0.11847158752557092, -0.14823502018049978, -0.15333964481975168]

        inv_τ_constant = 10.0 * unit_to_aru(:meV)
        nk = 20
        symmetry = nothing

        kpts, iband_min, iband_max, nstates_base = filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window; symmetry)
        nband = iband_max - iband_min + 1
        nband_ignore = iband_min - 1

        el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "velocity"], window, nband, nband_ignore);
        el, _ = EPW.electron_states_to_QMEStates(el_k_save, kpts, qme_offdiag_cutoff, nstates_base)
        v = [[v[a] for v in el.v] for a in 1:3]

        bte_compute_μ!(transport_params, EPW.BTStates(el))

        # Set scattering matrix with constant relaxation time
        S_out = [I(el.n) * (-inv_τ_constant + 0.0im) for _ in transport_params.Tlist]

        # Solve linear electrical conductivity
        out_linear = solve_electron_qme(transport_params, el, nothing, S_out; symmetry)

        # Solve linear Hall conductivity
        bvec_data = finite_difference_vectors(model.recip_lattice, el.kpts.ngrid)
        ∇ = Vec3(EPW.compute_covariant_derivative_matrix(el, el_k_save, bvec_data))
        qme_model = EPW.QMEModel(; el, ∇, transport_params, S_out)

        out_hall = compute_linear_hall_conductivity(out_linear, qme_model);
        @test out_hall.r_hall[1, 2, 3, :] ≈ r_hall_ref_123 atol=1e-5
    end
end
