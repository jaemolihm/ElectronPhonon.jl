using Test
using ElectronPhonon
using ElectronPhonon: WannierObject, Vec3
using LinearAlgebra

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

@testset "GPU batched Wannier interpolation" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU tests"
    else
        # Build a small model with a Hermitian H(k): enforce H(-R) = H(R)^†.
        nw = 6
        base = [Vec3(rand(-3:3, 3)...) for _ in 1:60]
        Rset = unique(vcat(base, [-r for r in base], [Vec3(0, 0, 0)]))
        blocks = Dict{Vec3{Int}, Matrix{ComplexF64}}()
        for r in Rset
            if haskey(blocks, -r)
                blocks[r] = blocks[-r]'
            else
                A = rand(ComplexF64, nw, nw)
                blocks[r] = (r == Vec3(0, 0, 0)) ? (A + A') / 2 : A
            end
        end
        irvec = sort(collect(Rset), by = x -> reverse(x))
        op_r = reduce(hcat, [vec(blocks[r]) for r in irvec])
        obj = WannierObject(irvec, op_r)

        kpts = [Vec3(rand(), rand(), rand()) for _ in 1:50]

        # --- to_device ---
        obj_gpu = to_device(obj)
        @test obj_gpu.op_r isa CuArray
        @test obj_gpu.ndata == obj.ndata
        @test Array(obj_gpu.op_r) ≈ obj.op_r

        # --- get_fourier_batched! device vs host ---
        Hk_cpu = zeros(ComplexF64, obj.ndata, length(kpts))
        get_fourier_batched!(Hk_cpu, get_interpolator(obj; fourier_mode="batched"), kpts)
        Hk_gpu = similar(obj_gpu.op_r, ComplexF64, obj.ndata, length(kpts))
        get_fourier_batched!(Hk_gpu, get_interpolator(obj_gpu; fourier_mode="batched"), kpts)
        @test Array(Hk_gpu) ≈ Hk_cpu

        # --- eigenvalues only: GPU vs CPU reference ---
        W_ref = get_el_eigen_valueonly_batched(obj, kpts)
        W_gpu = get_el_eigen_valueonly_batched(obj_gpu, kpts)
        @test W_gpu isa CuArray
        @test sort(Array(W_gpu), dims=1) ≈ sort(W_ref, dims=1)

        # --- eigenvalues + eigenvectors ---
        Wv_gpu, V_gpu = get_el_eigen_batched(obj_gpu, kpts)
        @test sort(Array(Wv_gpu), dims=1) ≈ sort(W_ref, dims=1)
        # Eigenvectors are gauge-dependent, so check the gauge-invariant reconstruction
        # H(k) ≈ V diag(w) V† for a few k-points.
        Wv = Array(Wv_gpu); V = Array(V_gpu); H = reshape(Hk_cpu, nw, nw, length(kpts))
        for ik in (1, 17, 50)
            @test V[:, :, ik] * Diagonal(Wv[:, ik]) * V[:, :, ik]' ≈ H[:, :, ik]
        end

        # --- batched eigensolve is not limited to nw ≤ 32 on this cuSOLVER ---
        let nw2 = 40, nk2 = 4
            H = CUDA.rand(ComplexF64, nw2, nw2, nk2)
            for k in 1:nk2; @views H[:, :, k] .= (H[:, :, k] + H[:, :, k]') / 2; end
            Wbig = Array(eigvals_batched(H))
            Hh = Array(H)
            for k in 1:nk2
                @test sort(Wbig[:, k]) ≈ sort(real(eigvals(Hermitian(Hh[:, :, k]))))
            end
        end
    end
end
