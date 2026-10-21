using Test
using ElectronPhonon
using LinearAlgebra

# TODO: Add test for covariant derivative (same with and without symmetry)

@testset "QMEModel" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp = joinpath(folder, "tmp")
    mkpath(folder_tmp)

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    nk = 20
    nq = 20
    window_k  = (10.5, 11.0) .* unit_to_aru(:eV)
    window_kq = (10.4, 11.0) .* unit_to_aru(:eV)

    transport_params = ElectronTransportParams{Float64}(
        Tlist = [300.0] .* unit_to_aru(:K),
        nlist = [-1.0e16 * model.volume / unit_to_aru(:cm)^3],
        smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
        volume = model.volume,
        nband_valence = 4,
        spin_degeneracy = 2
    )

    @testset "without symmetry" begin
        @time output = ElectronPhonon.run_transport(
            model, (nk, nk, nk), (nq, nq, nq),
            fourier_mode = "gridopt",
            folder = folder_tmp,
            window_k  = window_k,
            window_kq = window_kq,
            average_degeneracy = false,
            run_for_qme = true,
            compute_derivative = true,
            use_irr_k = false,
        );


        filename = joinpath(folder_tmp, "btedata_coherence.rank0.h5")
        qme_model = load_QMEModel(filename, transport_params)
        bte_compute_μ!(qme_model)

        @test qme_model isa ElectronPhonon.QMEModel

        compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

        # Test multiply_Sᵢ for scalar and vector eltype are identical for QMEModel without symmetry
        v = QMEVector(qme_model.el, rand(Vec3{ComplexF64}, qme_model.el.n))
        v_1 = QMEVector(qme_model.el, [x[1] for x in v.data])
        Sᵢx = ElectronPhonon.multiply_Sᵢ(v, qme_model.Sᵢ_irr[1], qme_model)
        Sᵢx_1 = ElectronPhonon.multiply_Sᵢ(v_1, qme_model.Sᵢ_irr[1], qme_model)
        @test Sᵢx_1.data ≈ [x[1] for x in Sᵢx.data]
    end

    @testset "with symmetry" begin
        @time output = ElectronPhonon.run_transport(
            model, (nk, nk, nk), (nq, nq, nq),
            fourier_mode = "gridopt",
            folder = folder_tmp,
            window_k  = window_k,
            window_kq = window_kq,
            average_degeneracy = false,
            run_for_qme = true,
            compute_derivative = true,
            use_irr_k = true,
        );

        filename = joinpath(folder_tmp, "btedata_coherence.rank0.h5")
        qme_model = load_QMEModel(filename, transport_params)
        bte_compute_μ!(qme_model)

        @test qme_model isa ElectronPhonon.QMEIrreducibleKModel

        compute_qme_scattering_matrix!(qme_model, compute_Sᵢ=true)

        # Test whether multiply_Sᵢ and unfold_QMEVector commutes for a symmetric QMEVector.
        # if the underlying input QMEVector is symmetric. Concretely, we test
        # Case 1 : v_irr -> unfold -> v -> multiply_Sᵢ -> Sv, and
        # Case 2 : v_irr -> multiply_Sᵢ -> Sv_irr -> unfold -> Sv_irr_unfold.
        # Here, we can only test time-reversal odd and inversion even vectors because
        # multiply_Sᵢ uses these assumptions.
        trodd, invodd = true, false
        v_irr_raw = QMEVector(qme_model.el_irr, rand(Vec3{ComplexF64}, qme_model.el_irr.n))
        v_irr = ElectronPhonon.symmetrize_QMEVector(v_irr_raw, qme_model, trodd, invodd)
        v = ElectronPhonon.unfold_QMEVector(v_irr, qme_model, trodd, invodd)

        Sv_irr = ElectronPhonon.multiply_Sᵢ(v_irr, qme_model.Sᵢ_irr[1], qme_model)
        Sv = ElectronPhonon.multiply_Sᵢ(v, qme_model.Sᵢ_irr[1], qme_model)
        Sv_irr_unfold = ElectronPhonon.unfold_QMEVector(Sv_irr, qme_model, trodd, invodd)
        @test Sv.data ≈ Sv_irr_unfold.data
    end
end
