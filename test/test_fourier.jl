using ElectronPhonon
using Test
using Random

@testset "fourier" begin
    using ElectronPhonon: WannierObject

    irvec = [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, -2, 1]),
        Vec3{Int}([1, 3, -2]), Vec3{Int}([-1, 3, -2]),
        Vec3{Int}([-1, -3, 1])]
    nr = length(irvec)
    Random.seed!(123)
    op_r = randn(ComplexF64, 6, nr)

    # Constructor should throw error if irvec is not sorted
    @test_throws ErrorException obj = WannierObject(irvec, op_r)

    ind = sortperm(irvec, by=x->reverse(x))
    irvec = irvec[ind]
    op_r = op_r[:, ind]

    obj = WannierObject(irvec, op_r)

    xk1 = Vec3([0.3, -0.4, 0.7])
    xk2 = Vec3([0.3, -0.4, -0.5])
    op_k_1d = Array{ComplexF64,1}(undef, 6)
    op_k_normal = Array{ComplexF64,2}(undef, 2, 3)
    op_k_normal_2 = Array{ComplexF64,2}(undef, 2, 3)
    op_k_gridopt = Array{ComplexF64,2}(undef, 2, 3)

    obj_normal = get_interpolator(obj, fourier_mode="normal")
    obj_gridopt = get_interpolator(obj, fourier_mode="gridopt")
    get_fourier!(op_k_normal, obj_normal, xk1)
    get_fourier!(op_k_gridopt, obj_gridopt, xk1)
    @test op_k_normal ≈ op_k_gridopt
    @test obj_gridopt.gridopt.nr_23 == 4
    @test obj_gridopt.gridopt.nr_3 == 3

    get_fourier!(op_k_1d, obj_gridopt, xk1)
    @test vec(op_k_normal) ≈ op_k_1d

    get_fourier!(op_k_normal, obj_normal, xk2)
    get_fourier!(op_k_gridopt, obj_gridopt, xk2)
    @test op_k_normal ≈ op_k_gridopt

    # Update op_r
    op_r_new = randn(ComplexF64, 6, nr)
    obj_new = WannierObject(irvec, op_r_new)
    update_op_r!(obj, op_r_new)
    obj_new_normal = get_interpolator(obj_new, fourier_mode="normal")

    get_fourier!(op_k_normal, obj_normal, xk1)
    get_fourier!(op_k_gridopt, obj_gridopt, xk1)
    get_fourier!(op_k_normal_2, obj_new_normal, xk1)
    @test op_k_normal ≈ op_k_gridopt
    @test op_k_normal ≈ op_k_normal_2
end
