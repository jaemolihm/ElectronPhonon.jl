using Test
using EPW
using HDF5
using LinearAlgebra

@testset "QMEVector" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp = joinpath(folder, "tmp")
    mkpath(folder_tmp)

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
