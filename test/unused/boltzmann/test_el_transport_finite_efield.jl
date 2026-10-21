using Test
using ElectronPhonon
using LinearAlgebra

@testset "Transport finite E field" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    window_k  = (10.6, 11.0) .* unit_to_aru(:eV)
    window_kq = (10.5, 11.0) .* unit_to_aru(:eV)
    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 4 * 80.0 * unit_to_aru(:meV))
    qme_offdiag_cutoff = ElectronPhonon.electron_degen_cutoff

    transport_params = ElectronTransportParams(
        Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K),
        nlist = fill(-1.0e21 * model.volume / unit_to_aru(:cm)^3, 3),
        smearing = smearing,
        nband_valence = 4,
        volume = model.volume,
        spin_degeneracy = 2
    )

    nk = 20
    inv_τ_constant = 10.0 * unit_to_aru(:meV)

    # Reference result computed from Julia. Keys are (use_symmetry, method).
    # Calculated for E = [0.6, 0.8, -1.0] .* 1e-4.
    current_ref_1 = Dict(
        (false, :CRTA)  => [2.179980098541587e-5, 1.8332187496235162e-5, 1.639517578365892e-5],
        (false, :SERTA) => [2.074576794329549e-5, 1.176606981940157e-5, 7.268809861744779e-6],
        (false, :IBTE) =>  [4.118840229477912e-6, 3.870429064239907e-6, 4.253852487268902e-6],
        (true,  :CRTA)  => [2.179978209694831e-5, 1.8332179496101772e-5, 1.639517325048265e-5],
        (true,  :SERTA) => [2.0243411988064157e-5, 1.161690590054153e-5, 7.230682163663374e-6],
        (true,  :IBTE) =>  [6.91570493419106e-6, 5.383088989029682e-6, 4.765595999956389e-6],
    )

    for use_symmetry in [false, true]
        symmetry = use_symmetry ? model.el_sym.symmetry : nothing

        # Calculate matrix elements
        @time ElectronPhonon.run_transport(
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

        for method in [:CRTA, :SERTA, :IBTE]
            # Calculate scattering matrix
            if method === :CRTA
                set_constant_qme_scattering_matrix!(qme_model, inv_τ_constant)
            elseif method === :SERTA
                compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=false)
            elseif method === :IBTE
                compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)
            end

            # Solve linear electrical conductivity
            out_linear = solve_electron_linear_conductivity(qme_model)

            # Test very small E field: current should be E times linear conductivity
            E = rand(3) .* 1e-10
            out_E = solve_electron_finite_efield(qme_model, out_linear, E)

            current_linear = hcat([x * E for x in eachslice(out_linear.σ_serta, dims=3)]...)
            @test out_E.current_serta ≈ current_linear rtol=1e-3
            if method === :IBTE
                current_linear = hcat([x * E for x in eachslice(out_linear.σ, dims=3)]...)
                @test out_E.current ≈ current_linear rtol=1e-2
            end

            # Test large E field
            E = [0.6, 0.8, -1.0] .* 1e-4
            out_E = solve_electron_finite_efield(qme_model, out_linear, E)
            if method === :CRTA || method === :SERTA
                @test out_E.current_serta[1, :] ≈ current_ref_1[(use_symmetry, method)] atol=1e-10
                @test all(isnan.(out_E.current))
            elseif method === :IBTE
                @test out_E.current_serta[1, :] ≈ current_ref_1[(use_symmetry, :SERTA)] atol=1e-10
                @test out_E.current[1, :] ≈ current_ref_1[(use_symmetry, method)] atol=1e-10
            end
        end
    end
end
