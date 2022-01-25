using Test
using EPW
using HDF5
using LinearAlgebra

# TODO: Implement ξ term in compute_covariant_derivative_matrix and add test (see broken test below).

@testset "covariant derivative" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp = joinpath(folder, "tmp")
    mkpath(folder_tmp)

    model = load_model(folder)

    # The velocity computed by direct interpolation differ from the velocity of the Wannier
    # tight-binding model when the coarse k grid is not converged. So, we use BerryConnection
    # method.
    model.el_velocity_mode = :BerryConnection

    # Setup k points: grid spacing 1/nk, centered at xk0, include +- 2 points.
    xk0 = Vec3(0.0, 0.2, 0.0)
    nk = 240
    kpts_list = Vec3{Float64}[]
    for i in -2:2, j in -2:2, k in -2:2
        push!(kpts_list, xk0 .+ (i, j, k) ./ nk)
    end
    ik0 = findfirst(xk -> xk ≈ xk0, kpts_list)
    kpts = GridKpoints(Kpoints(length(kpts_list), kpts_list, ones(length(kpts_list)), (nk, nk, nk)))

    qme_offdiag_cutoff = 5.0 * unit_to_aru(:eV)

    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"])
    el, _ = EPW.electron_states_to_QMEStates(el_k_save, kpts, qme_offdiag_cutoff)

    bvec_data = finite_difference_vectors(model.recip_lattice, el.kpts.ngrid)
    ∇ = EPW.compute_covariant_derivative_matrix(el, el_k_save, bvec_data)

    # Test IO
    h5open(joinpath(folder_tmp, "covariant_derivative.h5"), "w") do f
        EPW.compute_covariant_derivative_matrix(el, el_k_save, bvec_data, f)
        ∇_from_file = EPW.load_covariant_derivative_matrix(f)
        @test ∇ ≈ ∇_from_file
    end

    # Test covariant derivative of the Hamiltonian operator. Compare with the analytic
    # solution which is the velocity operator.
    f = zeros(el.n)
    @. f[el.ib1 == el.ib2] = el.e1[el.ib1 == el.ib2]
    ∇f = ∇[1] * f
    ∇f_ik0_analytic = [x[1] for x in el_k_save[ik0].v]

    # Accumulate error for degenerate and nondegenerate pairs separately and check both.
    error_degen = 0.0
    error_nondegen = 0.0
    for i in 1:el.n
        if el.ik[i] == ik0
            if abs(el.e1[i] - el.e2[i]) < EPW.electron_degen_cutoff
                error_degen += abs(∇f[i] - ∇f_ik0_analytic[el.ib1[i], el.ib2[i]])
            else
                error_nondegen += abs(∇f[i] - ∇f_ik0_analytic[el.ib1[i], el.ib2[i]])
            end
        end
    end
    @test error_degen < 1e-3
    @test_broken error_nondegen < 1e-3 # Broken because ξ term is not implemented
end

@testset "covariant derivative bvec" begin
    # TODO: Implement finite_difference_vectors, add test for different alat and different lattice.
    alat = 0.9199392836280507
    recip_lattice = alat * [-1 1 -1; -1 1 1; 1 1 -1]
    ngrid = 10
    bvecs, bvecs_cart, wbs = finite_difference_vectors(recip_lattice, ngrid)
    @test length(bvecs) == 8
    @test bvecs_cart ≈ Ref(recip_lattice) .* bvecs
    @test all(round.(Int, b .* ngrid) ≈ b .* ngrid for b in bvecs)
    @test sum([b_cart * b_cart' .* wb for (b_cart, wb) in zip(bvecs_cart, wbs)]) ≈ I(3)
end