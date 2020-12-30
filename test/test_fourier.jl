using EPW
using Test
using Random

@testset "fourier" begin
    nr = 3
    irvec = [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, -2, 1]), Vec3{Int}([1, 3, -2])]
    Random.seed!(123)
    op_r = randn(ComplexF64, 6, 3)

    # Constructor should throw error if irvec is not sorted
    @test_throws ErrorException obj = WannierObject(4, irvec, op_r)
    @test_throws ErrorException obj = WannierObject(nr, irvec, op_r)

    ind = sortperm(irvec, by=x->reverse(x))
    irvec = irvec[ind]
    op_r = op_r[:, ind]

    obj = WannierObject(nr, irvec, op_r)

    xk1 = Vec3([0.3, -0.4, 0.7])
    xk2 = Vec3([0.3, -0.4, -0.5])
    op_k_1d = Array{ComplexF64,1}(undef, 6)
    op_k_normal = Array{ComplexF64,2}(undef, 2, 3)
    op_k_normal_2 = Array{ComplexF64,2}(undef, 2, 3)
    op_k_gridopt = Array{ComplexF64,2}(undef, 2, 3)

    get_fourier!(op_k_normal, obj, xk1, mode="normal")
    get_fourier!(op_k_gridopt, obj, xk1, mode="gridopt")
    @test op_k_normal ≈ op_k_gridopt

    phase = [cis(2π * sum(r .* xk1)) for r in irvec]
    get_fourier!(op_k_normal_2, obj, xk1, phase, mode="normal")
    @test op_k_normal ≈ op_k_normal_2

    get_fourier!(op_k_1d, obj, xk1, mode="gridopt")
    @test vec(op_k_normal) ≈ op_k_1d

    get_fourier!(op_k_normal, obj, xk2, mode="normal")
    get_fourier!(op_k_gridopt, obj, xk2, mode="gridopt")
    @test op_k_normal ≈ op_k_gridopt

    # Update op_r
    op_r_new = randn(ComplexF64, 6, 3)
    obj_new = WannierObject(nr, irvec, op_r_new)
    update_op_r!(obj, op_r_new)

    get_fourier!(op_k_normal, obj, xk1, mode="normal")
    get_fourier!(op_k_gridopt, obj, xk1, mode="gridopt")
    get_fourier!(op_k_normal_2, obj_new, xk1, mode="normal")
    @test op_k_normal ≈ op_k_gridopt
    @test op_k_normal ≈ op_k_normal_2

end
