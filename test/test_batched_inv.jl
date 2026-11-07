@testsnippet BatchedInv begin
    using ElectronPhonon: batched_inv!, _gauss_jordan_inv!
    using LinearAlgebra: LinearAlgebra, inv, norm, I
    using Random: MersenneTwister

    # Random complex batch whose first matrix has M[1,1] = 0 (needs a row swap), so the pivot
    # search matters.
    function inv_test_batch(n, nbatch)
        rng = MersenneTwister(7)
        M = randn(rng, ComplexF64, n, n, nbatch)
        n > 1 && (M[1, 1, 1] = 0)
        M
    end

    # Largest relative error of `batched_inv!` against `inv` on each matrix.
    function inv_max_relerr(X, M0)
        maximum(norm(X[:, :, b] - inv(M0[:, :, b])) / norm(inv(M0[:, :, b])) for b in axes(M0, 3))
    end
end

@testitem "batched_inv! host method vs inv (random batch, pivoting)" setup = [BatchedInv] begin
    for n in (1, 2, 4, 8)
        M0 = inv_test_batch(n, 37)
        X = similar(M0)
        batched_inv!(X, copy(M0))
        @test inv_max_relerr(X, M0) < 1e-11
    end
    # M[1,1] = 0 with nonzero off-diagonals: the first pivot must be swapped in
    M0 = inv_test_batch(4, 3)
    @test M0[1, 1, 1] == 0
    X = similar(M0)
    batched_inv!(X, copy(M0))
    @test norm(X[:, :, 1] * M0[:, :, 1] - I) < 1e-12
    # Real elements and a singular matrix
    Mr = randn(MersenneTwister(1), 3, 3, 5)
    Xr = similar(Mr)
    batched_inv!(Xr, copy(Mr))
    @test inv_max_relerr(Xr, Mr) < 1e-11
    Ms = copy(Mr); Ms[:, :, 2] .= 0
    @test_throws LinearAlgebra.SingularException batched_inv!(similar(Ms), Ms)
    # Only the host method accepts host arrays: a non-Array AbstractArray is a MethodError
    @test_throws MethodError batched_inv!(view(similar(Mr), :, :, 1:2:5), view(copy(Mr), :, :, 1:2:5))
    @test_throws DimensionMismatch batched_inv!(similar(Mr, 3, 3, 4), Mr)
end

@testitem "_gauss_jordan_inv! pivots on a tiny diagonal and reports a singular column" setup = [BatchedInv] begin
    # M[1, 1] = 0 and a small M[2, 2] force row exchanges in an elimination without pivoting
    for n in (2, 4, 8)
        M = randn(MersenneTwister(30 + n), ComplexF64, n, n)
        M[1, 1] = 0
        M[2, 2] = 1e-9
        Minv = inv(M)
        X = zeros(ComplexF64, n, n)
        @test _gauss_jordan_inv!(X, copy(M)) == 0
        @test norm(X - Minv) / norm(Minv) < 1e-10
        @test norm(M * X - I) < 1e-8
    end
    # Column 2 is a copy of column 1, so the pivot of column 2 vanishes
    M = [1.0 1.0 0.0; 2.0 2.0 1.0; 0.0 0.0 3.0]
    @test _gauss_jordan_inv!(zeros(3, 3), M) == 2
    @test_throws DimensionMismatch _gauss_jordan_inv!(zeros(3, 3), zeros(3, 2))
end

@testitem "batched_inv! CUDA method vs host" tags = [:gpu] setup = [BatchedInv] begin
    using CUDA
    if CUDA.functional()
        CUDA.allowscalar(false)
        for n in (2, 4, 8)
            M0 = inv_test_batch(n, 64)
            Xh = similar(M0)
            batched_inv!(Xh, copy(M0))
            Md = CuArray(M0)
            Xd = similar(Md)
            batched_inv!(Xd, Md)
            @test norm(Array(Xd) - Xh) / norm(Xh) < 1e-11
            @test inv_max_relerr(Array(Xd), M0) < 1e-11
        end
        # A strided device view must not fall through to the host method's scalar indexing
        Mv = CuArray(inv_test_batch(3, 6))
        @test_throws MethodError batched_inv!(similar(Mv), view(Mv, :, :, 1:2:6))
        Ms = CUDA.zeros(ComplexF64, 3, 3, 2)
        @test_throws LinearAlgebra.SingularException batched_inv!(similar(Ms), Ms)
    end
end
