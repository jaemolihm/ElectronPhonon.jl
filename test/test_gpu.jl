using Test
using ElectronPhonon
using ElectronPhonon: WannierObject, Vec3, get_eph_RR_to_kR!, get_eph_kR_to_kq!
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

        # --- electron-phonon: batched drivers (CPU and GPU) vs per-k reference ---
        let nwe = 3, nmodes = 4, nr_el = 20, nr_ep = 15
            nband = nwe
            irvec_el = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_el], by = x -> reverse(x))
            irvec_ep = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_ep], by = x -> reverse(x))
            epmat_or = rand(ComplexF64, nwe^2 * nmodes * nr_ep, nr_el)
            epmat_obj = WannierObject(irvec_el, epmat_or; irvec_next = irvec_ep)
            uk  = rand(ComplexF64, nwe, nband)
            ukq = rand(ComplexF64, nwe, nband)
            u_ph = rand(ComplexF64, nmodes, nmodes)
            xk = Vec3(0.1, -0.4, 0.7); xq = Vec3(0.5, 0.2, -0.5)

            # per-k reference
            ref = WannierObject(irvec_ep, zeros(ComplexF64, nwe*nband*nmodes, nr_ep))
            get_eph_RR_to_kR!(ref, get_interpolator(epmat_obj; fourier_mode="normal"), xk, uk)
            ep_ref = zeros(ComplexF64, nband, nband, nmodes)
            get_eph_kR_to_kq!(ep_ref, get_interpolator(ref; fourier_mode="normal"), xq, u_ph, ukq)

            # batched drivers — CPU
            o_c = WannierObject(irvec_ep, zeros(ComplexF64, nwe*nband*nmodes, nr_ep))
            get_eph_RR_to_kR_batched!(o_c, get_interpolator(epmat_obj; fourier_mode="batched"), xk, uk)
            ep_c = zeros(ComplexF64, nband, nband, nmodes)
            get_eph_kR_to_kq_batched!(ep_c, get_interpolator(o_c; fourier_mode="batched"), xq, u_ph, ukq)
            @test ep_c ≈ ep_ref

            # batched drivers — GPU. Use CuArray (not cu) to keep Float64; cuBLAS only
            # reorders the GEMM summations, so it matches the CPU to ~1e-12.
            o_g = to_device(WannierObject(irvec_ep, zeros(ComplexF64, nwe*nband*nmodes, nr_ep)))
            get_eph_RR_to_kR_batched!(o_g, get_interpolator(to_device(epmat_obj); fourier_mode="batched"), xk, CuArray(uk))
            ep_g = CUDA.zeros(ComplexF64, nband, nband, nmodes)
            get_eph_kR_to_kq_batched!(ep_g, get_interpolator(o_g; fourier_mode="batched"), xq, CuArray(u_ph), CuArray(ukq))
            @test isapprox(Array(ep_g), ep_ref; rtol=1e-9)

            # list-batched drivers (many k / many q) vs the per-k reference
            nk2, nq2 = 5, 6
            ks = [Vec3(rand(3)...) for _ in 1:nk2]; ks[1] = xk
            qs = [Vec3(rand(3)...) for _ in 1:nq2]; qs[1] = xq
            uks  = cat([rand(ComplexF64, nwe, nband) for _ in 1:nk2]...; dims=3); uks[:, :, 1] .= uk
            uphs = cat([rand(ComplexF64, nmodes, nmodes) for _ in 1:nq2]...; dims=3); uphs[:, :, 1] .= u_ph
            ukqs = cat([rand(ComplexF64, nwe, nband) for _ in 1:nq2]...; dims=3); ukqs[:, :, 1] .= ukq

            epmat_g = get_interpolator(to_device(epmat_obj); fourier_mode="batched", batch_size=nk2)
            ep_all = CUDA.zeros(ComplexF64, nwe*nband*nmodes, nr_ep, nk2)
            get_eph_RR_to_kR_batched!(ep_all, epmat_g, ks, CuArray(uks))
            obj_k1 = to_device(WannierObject(irvec_ep, Array(ep_all[:, :, 1])))
            ep_kq_all = CUDA.zeros(ComplexF64, nband, nband, nmodes, nq2)
            get_eph_kR_to_kq_batched!(ep_kq_all, get_interpolator(obj_k1; fourier_mode="batched", batch_size=nq2),
                                      qs, CuArray(uphs), CuArray(ukqs))
            # q-index 1 used (xk, xq) → must match the reference
            @test isapprox(Array(ep_kq_all)[:, :, :, 1], ep_ref; rtol=1e-9)
        end
    end
end
