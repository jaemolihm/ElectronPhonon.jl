using Test
using ElectronPhonon
using LinearAlgebra

include("common_models_from_artifacts.jl")

@testset "QMEStates" begin
    model = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "ph")

    # Setup QMEVector
    qme_offdiag_cutoff = 1.0 .* unit_to_aru(:eV)
    kpts = kpoints_grid((3, 3, 3))
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "velocity"])
    el = ElectronPhonon.electron_states_to_QMEStates(el_k_save, kpts, qme_offdiag_cutoff, 0.)

    @testset "indmap" begin
        # Test indmap
        cnt = 0
        for ik in 1:el.kpts.n
            for ib2 in (minimum(el.ib2)-1):(maximum(el.ib2)+1)
                for ib1 in (minimum(el.ib1)-1):(maximum(el.ib1)+1)
                    i = ElectronPhonon.get_1d_index(el, ib1, ib2, ik)
                    if i == 0
                        @assert ElectronPhonon.hasstate(el, ib1, ib2, ik) == false
                    elseif 1 <= i <= el.n
                        @assert ElectronPhonon.hasstate(el, ib1, ib2, ik) == true
                        @assert (ik, ib1, ib2) == (el.ik[i], el.ib1[i], el.ib2[i])
                        cnt += 1
                    else
                        error("wrong i")
                    end
                end
            end
        end
        @assert cnt == el.n
    end

    @testset "QMEVector constructor" begin
        using OffsetArrays
        T = ComplexF64
        for v in (rand(T, el.n), view(rand(T, el.n+1), 2:el.n+1), OffsetArray(rand(T, el.n), :))
            @test QMEVector(el, v) isa AbstractVector
            @test QMEVector(el, v) isa QMEVector{T, Float64, typeof(v)}
        end
        @test_throws ArgumentError QMEVector(el, rand(T, el.n+1))
        @test_throws ArgumentError QMEVector(el, OffsetArray(rand(T, el.n), 2:1+el.n))
    end

    @testset "QMEVector" begin
        # Test basic operations
        x = QMEVector(el, rand(el.n))
        y = QMEVector(el, rand(ComplexF64, el.n))
        @test eltype(x) === eltype(x.data)
        @test eltype(y) === eltype(y.data)
        @test size(x) == (el.n,)
        @test x[3] == x.data[3]
        # @test x[10:20] == x.data[10:20] # This does not currently work

        # Test basic arithmetic operations
        z = 2 * x - y / 3 + x * 0.5
        @test z isa QMEVector
        @test z.state === el
        @test z.data ≈ @. 2 * x.data - y.data / 3 + 0.5 * x.data
        @test length(z.data) == el.n
        A = rand(el.n, el.n)
        @test (A * y).state === el
        @test (A * y).data ≈ A * y.data
        @test (A \ y).data ≈ inv(A) * y.data

        # Test broadcasting
        w = QMEVector(el, ComplexF64)
        @. w = x * y / 5 - 2 * x / y
        @test w isa QMEVector
        @test w.data ≈ @. x.data * y.data / 5 - 2 * x.data / y.data

        # Test x * y as matrix multiplication
        for ik in unique(el.ik)
            @test data_ik(x * y, ik) ≈ data_ik(x, ik) * data_ik(y, ik)
        end

        # Test QMEVector with view
        vdata = rand(el.n + 10)
        rng = 5+1:5+el.n
        vv = QMEVector(el, view(vdata, rng))
        @. vv = x * 2
        @test vv.data === @views vdata[rng]
        @test vdata[rng] ≈ x.data .* 2

        # Test get_velocity_as_QMEVector
        v = ElectronPhonon.get_velocity_as_QMEVector(el)
        @test v isa Vec3
        @test eltype(v) <: QMEVector
        for i in 1:3
            @test v[i].state === el
            @test v[i].data ≈ [v[i] for v in el.v]
        end

        # Test similar
        x = QMEVector(el, rand(el.n))
        for FT in (ComplexF64, ComplexF64, Float64, Float32)
            @test similar(x, FT) isa QMEVector{FT}
            @test similar(x, FT).state === x.state
        end
    end
end

# @testset "QMEVector symmetry" begin
#     BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
#     folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
#     folder_tmp = joinpath(folder, "tmp")
#     mkpath(folder_tmp)

#     model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

#     nk = 20
#     nq = 20
#     window_k  = (10.5, 11.0) .* unit_to_aru(:eV)
#     window_kq = (10.4, 11.0) .* unit_to_aru(:eV)

#     @time output = ElectronPhonon.run_transport(
#         model, (nk, nk, nk), (nq, nq, nq),
#         fourier_mode = "gridopt",
#         folder = folder_tmp,
#         window_k  = window_k,
#         window_kq = window_kq,
#         average_degeneracy = false,
#         run_for_qme = true,
#         compute_derivative = true,
#         use_irr_k = true,
#     );

#     transport_params = ElectronTransportParams{Float64}(
#         Tlist = [300.0] .* unit_to_aru(:K),
#         nlist = [-1.0e16 * model.volume / unit_to_aru(:cm)^3],
#         smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
#         volume = model.volume,
#         nband_valence = 4,
#         spin_degeneracy = 2
#     )

#     filename = joinpath(folder_tmp, "btedata_coherence.rank0.h5")
#     qme_model = load_QMEModel(filename, transport_params)
#     bte_compute_μ!(qme_model)
#     (; el_irr, el_f) = qme_model

#     # Test that symmetrization applied twice is equivalent to symmetrization applied once.
#     x_irr = QMEVector(el_irr, copy(el_irr.v))
#     x_irr_symm = ElectronPhonon.symmetrize_QMEVector(x_irr, qme_model, true, false)
#     x_irr_symm2 = ElectronPhonon.symmetrize_QMEVector(x_irr_symm, qme_model, true, false)
#     @test norm(x_irr_symm2.data .- x_irr_symm.data) < norm(x_irr.data) * 1e-7

#     # Test map from el_irr to el_f is the same with unfolding and rotation to el_f.
#     unfold_map, unfold_map_tr = ElectronPhonon._qme_linear_response_unfold_map(el_irr, el_f, qme_model.filename);
#     y1 = QMEVector(el_f, unfold_map * x_irr_symm.data) + QMEVector(el_f, unfold_map_tr * conj.(x_irr_symm.data))
#     x_symm = ElectronPhonon.unfold_QMEVector(x_irr_symm, qme_model, true, false)
#     y2 = QMEVector(el_f, qme_model.el_to_el_f_sym_maps[1] * x_symm.data)
#     @test norm(y1.data .- y2.data) < norm(y1.data) * 1e-7

#     # Test unfold_QMEVector is a copy if x.state === qme_model.el
#     x = QMEVector(qme_model.el, rand(ComplexF64, qme_model.el.n))
#     @test ElectronPhonon.unfold_QMEVector(x, qme_model, true, false).data ≈ x.data
# end
