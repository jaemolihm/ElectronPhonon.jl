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
    inv_τ_constant = 10.0 * unit_to_aru(:meV)

    # Reference result computed from Julia. Keys are (use_symmetry, method).
    r_hall_ref_123 = Dict(
        (false, :CRTA)  => [-0.11847158752557092,  -0.14823502018049978,  -0.15333964481975168],
        (false, :SERTA) => [-0.011250559739887816, -0.028690945110432325, -0.051159534325475216],
        (true,  :CRTA)  => [-0.1295862763932781,   -0.16238798092792733,  -0.16831395498549015],
        (true,  :SERTA) => [-0.01410358935362333,  -0.03482766355056225,  -0.060641233049638275],
    )
    μlist_ref = [0.789819320790235, 0.7904155994597073, 0.7910192438436332]

    # Without symmetry

    for use_symmetry in [false, true]
        symmetry = use_symmetry ? model.el_sym.symmetry : nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model, (nk, nk, nk), (nk, nk, nk),
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = use_symmetry,
            average_degeneracy = false,
            run_for_qme = true,
            compute_derivative = true,
        )
        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")

        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        bte_compute_μ!(qme_model)
        @test qme_model.transport_params.μlist ≈ μlist_ref

        for method in [:CRTA, :SERTA]
            # Calculate scattering matrix
            if method === :CRTA
                set_constant_qme_scattering_matrix!(qme_model, inv_τ_constant)
            elseif method === :SERTA
                compute_qme_scattering_matrix!(qme_model, compute_S_in=false)
            end

            # For CRTA, test whether unfolded scattering matrix is also proportional to identity.
            method === :CRTA && @test all(qme_model.S_out .≈ Ref(I(qme_model.el.n) * -inv_τ_constant))

            # Solve linear electrical conductivity
            out_linear = solve_electron_qme(qme_model)

            # Solve linear Hall conductivity
            out_hall = compute_linear_hall_conductivity(out_linear, qme_model)
            @test out_hall.r_hall[1, 2, 3, :] ≈ r_hall_ref_123[(use_symmetry, method)] atol=1e-5
        end
    end
end
