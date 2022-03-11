using Test
using EPW
using LinearAlgebra

@testset "QME transport Hall" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)
    model.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    window_k  = (10.6, 11.0) .* unit_to_aru(:eV)
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
        (false, :CRTA)  => [-0.7138963493098541, -0.6200701397844884, -0.5471086296788115],
        (false, :SERTA) => [-0.13556597124851688, -0.20984377305291413, -0.28992328236321735],
        (false,  :IBTE) => [-0.22015169630281456, -0.30888256624425386, -0.38911319773665887],
        (true,  :CRTA)  => [-0.7137123155282641, -0.6199872878487636, -0.5470982819277314],
        (true,  :SERTA) => [-0.13531696557166967, -0.20680990913882977, -0.2846293410021224],
        (true,  :IBTE) => [-0.22380405377467233, -0.30952331223248275, -0.3870892530602628],
    )
    μlist_ref = [0.7898193014825781, 0.7904144731751919, 0.7910106156685531]

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

        for method in [:CRTA, :SERTA, :IBTE]
            # Calculate scattering matrix
            if method === :CRTA
                set_constant_qme_scattering_matrix!(qme_model, inv_τ_constant)
            elseif method === :SERTA
                compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=false)
            elseif method === :IBTE
                compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)
            end

            # For CRTA, test whether unfolded scattering matrix is also proportional to identity.
            method === :CRTA && @test all(qme_model.Sₒ .≈ Ref(I(qme_model.el.n) * -inv_τ_constant))

            # Solve linear electrical conductivity
            out_linear = solve_electron_linear_conductivity(qme_model)

            # Solve linear Hall conductivity
            out_hall = solve_electron_hall_conductivity(out_linear, qme_model)
            if method === :CRTA || method === :SERTA
                @test out_hall.r_hall_serta[1, 2, 3, :] ≈ r_hall_ref_123[(use_symmetry, method)] atol=1e-5
                @test all(isnan.(out_hall.r_hall))
            elseif method === :IBTE
                @test out_hall.r_hall_serta[1, 2, 3, :] ≈ r_hall_ref_123[(use_symmetry, :SERTA)] atol=1e-5
                @test out_hall.r_hall[1, 2, 3, :] ≈ r_hall_ref_123[(use_symmetry, method)] atol=1e-5
            end
        end
    end
end
