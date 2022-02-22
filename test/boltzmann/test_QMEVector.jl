using Test
using EPW
using HDF5
using LinearAlgebra

@testset "QMEVector" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder)

    # Setup QMEVector
    qme_offdiag_cutoff = 1.0 .* unit_to_aru(:eV)
    kpts = generate_kvec_grid(3, 3, 3)
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "velocity"]);
    el, _ = EPW.electron_states_to_QMEStates(el_k_save, kpts, qme_offdiag_cutoff, 0.)

    # Test basic operations
    x = QMEVector(el, rand(el.n))
    y = QMEVector(el, rand(ComplexF64, el.n))
    @test eltype(x) === eltype(x.data)
    @test eltype(y) === eltype(y.data)
    @test size(x) == el.n
    @test x[3] == x.data[3]
    # @test x[10:20] == x.data[10:20] # This does not currently work

    # Test basic arithmetic operations
    z = 2 * x - y / 3 + x * 0.5
    @test z.state === el
    @test z.data ≈ @. 2 * x.data - y.data / 3 + 0.5 * x.data
    @test length(z.data) == el.n
    A = rand(el.n, el.n)
    @test (A * y).state === el
    @test (A * y).data ≈ A * y.data
    @test (A \ y).data ≈ inv(A) * y.data

    # Test x * y as matrix multiplication
    @test data_ik(x * y, 1) ≈ data_ik(x, 1) * data_ik(y, 1)

    # Test get_velocity_as_QMEVector
    v = EPW.get_velocity_as_QMEVector(el)
    @test v isa Vec3
    @test eltype(v) <: QMEVector
    for i in 1:3
        @test v[i].state === el
        @test v[i].data ≈ [v[i] for v in el.v]
    end
end


@testset "QMEVector symmetry" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp = joinpath(folder, "tmp")
    mkpath(folder_tmp)

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    nk = 20
    nq = 20
    window_k  = (10.5, 11.0) .* unit_to_aru(:eV)
    window_kq = (10.4, 11.0) .* unit_to_aru(:eV)

    @time output = EPW.run_transport(
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

    transport_params = ElectronTransportParams{Float64}(
        Tlist = [300.0] .* unit_to_aru(:K),
        n = -1.0e16 * model.volume / unit_to_aru(:cm)^3,
        smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
        volume = model.volume,
        nband_valence = 4,
        spin_degeneracy = 2
    )

    filename = joinpath(folder_tmp, "btedata_coherence.rank0.h5")
    qme_model = load_QMEModel(filename, transport_params)
    bte_compute_μ!(qme_model)
    (; el_irr, el_f) = qme_model

    # Test that symmetrization applied twice is equivalent to symmetrization applied once.
    x_irr = QMEVector(el_irr, copy(el_irr.v))
    x_irr_symm = EPW.symmetrize_QMEVector(x_irr, qme_model, true, false)
    x_irr_symm2 = EPW.symmetrize_QMEVector(x_irr_symm, qme_model, true, false)
    @test norm(x_irr_symm2.data .- x_irr_symm.data) < norm(x_irr.data) * 1e-7

    # Test map from el_irr to el_f is the same with unfolding and rotation to el_f.
    unfold_map = EPW._qme_linear_response_unfold_map(el_irr, el_f, qme_model.filename);
    y1 = QMEVector(el_f, unfold_map * x_irr_symm.data)
    x_symm = EPW.unfold_QMEVector(x_irr_symm, qme_model, true, false)
    y2 = QMEVector(el_f, qme_model.el_to_el_f_sym_maps[1] * x_symm.data)
    @test norm(y1.data .- y2.data) < norm(y1.data) * 1e-7
end