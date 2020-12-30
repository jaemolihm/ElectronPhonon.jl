using EPW
using Test
using Random

@testset "fourier" begin
    nr = 3
    irvec = [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, -2, 1]), Vec3{Int}([1, 3, -2])]
    Random.seed!(123)
    op_r = randn(ComplexF64, 6, 3)

    # Constructor should throw error if irvec is not sorted
    @test_throws ErrorException obj = WannierObject(nr, irvec, op_r)

    ind = sortperm(irvec, by=x->reverse(x))
    irvec = irvec[ind]
    op_r = op_r[:, ind]

    obj = WannierObject(nr, irvec, op_r)

    xk1 = Vec3([0.3, -0.4, 0.7])
    xk2 = Vec3([0.3, -0.4, -0.5])
    op_k_1d = Array{ComplexF64,1}(undef, 6)
    op_k_normal = Array{ComplexF64,2}(undef, 2, 3)
    op_k_gridopt = Array{ComplexF64,2}(undef, 2, 3)

    get_fourier!(op_k_normal, obj, xk1; mode="normal")
    get_fourier!(op_k_gridopt, obj, xk1; mode="gridopt")
    @test op_k_normal ≈ op_k_gridopt

    get_fourier!(op_k_1d, obj, xk1; mode="gridopt")
    @test vec(op_k_normal) ≈ op_k_1d

    get_fourier!(op_k_normal, obj, xk2; mode="normal")
    get_fourier!(op_k_gridopt, obj, xk2; mode="gridopt")
    @test op_k_normal ≈ op_k_gridopt

end