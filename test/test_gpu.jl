using Test
using ElectronPhonon
using ElectronPhonon: WannierObject, Vec3, get_eph_RR_to_kR!, get_eph_kR_to_kq!, get_eph_Rq_to_kq!, to_device
# Batched drivers / primitives are internal (unexported); import the ones the tests use.
using ElectronPhonon: eigvals_batched, eigen_batched, get_el_eigen_batched, get_el_eigen_valueonly_batched,
    get_el_velocity_direct_batched, get_eph_RR_to_kR_batched!, get_eph_kR_to_kq_batched!,
    get_eph_Rq_to_kq_batched!, eph_apply_rotations!, eph_apply_rotations_rqkq!, KRtoKQWorkspace, batched_gemm!
using LinearAlgebra

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# A mis-dispatch (a device view falling back to the generic scalar `batched_gemm!` instead of the
# GPU method) must fail loudly rather than silently limp — forbid scalar indexing on the device.
GPU_AVAILABLE && CUDA.allowscalar(false)

# Pb EPW model comes from a downloaded test artifact (see test/Artifacts.toml).
isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

"""
Validate the batched e-ph drivers against the per-k/q reference (`get_eph_RR_to_kR!` /
`get_eph_kR_to_kq!`) on a chosen backend. `to_dev` moves a `WannierObject` to the backend
(`identity` for CPU, `to_device` for GPU) and `arr_dev` moves a plain array (`identity` /
`CuArray`). Every batch element is checked. Full-band only (no energy window).
"""
function check_eph_batched(to_dev, arr_dev; rtol)
    nwe, nmodes, nr_el, nr_ep = 3, 4, 20, 15
    nband = nwe
    irvec_el = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_el], by = x -> reverse(x))
    irvec_ep = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_ep], by = x -> reverse(x))
    epmat_obj = WannierObject(irvec_el, rand(ComplexF64, nwe^2*nmodes*nr_ep, nr_el); irvec_next = irvec_ep)

    nk2, nq2 = 5, 6
    ks   = [Vec3(rand(3)...) for _ in 1:nk2]
    qs   = [Vec3(rand(3)...) for _ in 1:nq2]
    uks  = cat([rand(ComplexF64, nwe, nband) for _ in 1:nk2]...; dims = 3)
    uphs = cat([rand(ComplexF64, nmodes, nmodes) for _ in 1:nq2]...; dims = 3)
    ukqs = cat([rand(ComplexF64, nwe, nband) for _ in 1:nq2]...; dims = 3)
    ukqs_k = cat([rand(ComplexF64, nwe, nband) for _ in 1:nk2]...; dims = 3)  # k+q eigvecs, one per k

    # Independent per-k/q CPU references (ground truth). q-sweep uses k = ks[1].
    refs_RR = map(1:nk2) do ik
        r = WannierObject(irvec_ep, zeros(ComplexF64, nwe*nband*nmodes, nr_ep))
        get_eph_RR_to_kR!(r, get_interpolator(epmat_obj; fourier_mode="normal"), ks[ik], uks[:, :, ik])
        copy(r.op_r)
    end
    obj_ref1 = WannierObject(irvec_ep, copy(refs_RR[1]))
    ep_ref = zeros(ComplexF64, nband, nband, nmodes, nq2)
    for iq in 1:nq2
        get_eph_kR_to_kq!(view(ep_ref, :, :, :, iq), get_interpolator(obj_ref1; fourier_mode="normal"),
                          qs[iq], uphs[:, :, iq], ukqs[:, :, iq])
    end

    epmat_d = to_dev(epmat_obj)

    # list-batched RR→kR over all k — check every k slice (k is the MIDDLE axis)
    ep_all = arr_dev(zeros(ComplexF64, nwe*nband*nmodes, nk2, nr_ep))
    get_eph_RR_to_kR_batched!(ep_all, get_interpolator(epmat_d; fourier_mode="batched", batch_size=nk2), ks, arr_dev(uks))
    ep_all_h = Array(ep_all)
    for ik in 1:nk2
        @test isapprox(ep_all_h[:, ik, :], refs_RR[ik]; rtol)
    end

    # list-batched kR→kq over all q (fixed k = ks[1]) — check every slice
    obj_k1 = to_dev(WannierObject(irvec_ep, copy(refs_RR[1])))
    ep_kq_all = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq2))
    get_eph_kR_to_kq_batched!(ep_kq_all, get_interpolator(obj_k1; fourier_mode="batched", batch_size=nq2), qs, arr_dev(uphs), arr_dev(ukqs))
    ep_kq_h = Array(ep_kq_all)
    for iq in 1:nq2
        @test isapprox(ep_kq_h[:, :, :, iq], ep_ref[:, :, :, iq]; rtol)
    end

    # list-batched Rq→kq over all k (fixed q) — electron-Wannier / phonon-Bloch object interpolated
    # over R_el at each k, then rotated by uk (right) and ukq (left). Check every slice.
    eRpq_obj = WannierObject(irvec_el, rand(ComplexF64, nwe^2 * nmodes, nr_el))
    ep_rqkq_ref = zeros(ComplexF64, nband, nband, nmodes, nk2)
    for ik in 1:nk2
        get_eph_Rq_to_kq!(view(ep_rqkq_ref, :, :, :, ik), get_interpolator(eRpq_obj; fourier_mode="normal"),
                          ks[ik], uks[:, :, ik], ukqs_k[:, :, ik])
    end
    eRpq_d = to_dev(eRpq_obj)
    ep_rqkq = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nk2))
    get_eph_Rq_to_kq_batched!(ep_rqkq, get_interpolator(eRpq_d; fourier_mode="batched", batch_size=nk2),
                              ks, arr_dev(uks), arr_dev(ukqs_k))
    ep_rqkq_h = Array(ep_rqkq)
    for ik in 1:nk2
        @test isapprox(ep_rqkq_h[:, :, :, ik], ep_rqkq_ref[:, :, :, ik]; rtol)
    end

    # g2 fold: the driver can also write g2 = |ep|²/(2ω) in the same pass — the GPU fused kernel
    # writes it from registers, the CPU / large-nw path uses the generic broadcast. Check against
    # the independent per-q reference ep_ref. (Exercises both `eph_apply_rotations!` g2 paths.)
    ωq_g2  = rand(nmodes, nq2) .+ 0.5
    g2_out = arr_dev(zeros(nband, nband, nmodes, nq2))
    ep_g2  = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq2))
    get_eph_kR_to_kq_batched!(ep_g2, get_interpolator(obj_k1; fourier_mode="batched", batch_size=nq2),
        qs, arr_dev(uphs), arr_dev(ukqs); g2_out, ωq=arr_dev(ωq_g2))
    g2_ref = abs2.(ep_ref) ./ (2 .* reshape(ωq_g2, 1, 1, nmodes, nq2))
    @test isapprox(Array(g2_out), g2_ref; rtol)
end

@testset "batched e-ph drivers (CPU)" begin
    check_eph_batched(identity, identity; rtol=1e-10)
end

"""
`fourier_phase!` against a scalar `cispi(2 R·x)` reference, on the backend selected by `arr_dev`.
"""
function check_fourier_phase(arr_dev)
    nr, nk = 37, 11
    irvec = [Vec3(rand(-3:3, 3)...) for _ in 1:nr]
    xks   = [Vec3(rand(3)...) for _ in 1:nk]
    ref = [cispi(2 * (irvec[ir][1] * xks[ik][1] + irvec[ir][2] * xks[ik][2] +
                      irvec[ir][3] * xks[ik][3])) for ir in 1:nr, ik in 1:nk]

    irvec_mat = [Float64(irvec[ir][d]) for ir in 1:nr, d in 1:3]
    xkmat = [xks[ik][d] for d in 1:3, ik in 1:nk]
    phase = arr_dev(zeros(ComplexF64, nr, nk))
    ElectronPhonon.fourier_phase!(phase, arr_dev(irvec_mat), arr_dev(xkmat))
    # |phase| == 1, so an absolute bound is a relative bound.
    @test maximum(abs, Array(phase) .- ref) <= 4 * eps(Float64)
end

@testset "fourier_phase! (CPU)" begin
    check_fourier_phase(identity)
end

"""
The two `get_eph_kR_to_kq_batched!` methods and the k+q convention of `get_eph_RR_to_kR_batched!`,
on the backend given by `to_dev` / `arr_dev` (as in [`check_eph_batched`](@ref)).

1. The phase-taking method fed `fourier_phase!(qs)` reproduces the interpolator+`qs` method exactly.
2. Storing the kR intermediate in the k+q convention (`conj(exp(2πi R_p·x_k))`) and transforming at
   `x_{k+q}` reproduces the q-convention result, for a (k, k+q) pair whose `q = x_{k+q} - x_k` needs
   a mod-G reduction. This is the in-repo pin of the identity `exp(2πi R_p·q) =
   exp(2πi R_p·x_{k+q}) · conj(exp(2πi R_p·x_k))` for integer `R_p`.
"""
function check_eph_kq_convention(to_dev, arr_dev; rtol)
    nwe, nmodes, nr_el, nr_ep = 3, 4, 12, 15
    nband, nq = nwe, 9
    irvec_el = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_el], by = x -> reverse(x))
    irvec_ep = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_ep], by = x -> reverse(x))
    epmat_obj = to_dev(WannierObject(irvec_el, rand(ComplexF64, nwe^2*nmodes*nr_ep, nr_el);
                                     irvec_next = irvec_ep))
    itp_epmat = get_interpolator(epmat_obj; fourier_mode="batched", batch_size=1)

    # x_k and x_{k+q} on a 20³ grid; q = (x_{k+q} - x_k) mod G is what the loop feeds today.
    xk   = Vec3(0.75, -0.40, 0.35)
    xkqs = [Vec3(rand(-10:10, 3) ./ 20...) for _ in 1:nq]
    qs   = [Vec3(mod.(xkq .- xk, 1)...) for xkq in xkqs]
    @test any(i -> qs[i] != xkqs[i] - xk, 1:nq)   # at least one pair needs the mod-G reduction

    uk   = arr_dev(rand(ComplexF64, nwe, nband, 1))
    uphs = arr_dev(cat([rand(ComplexF64, nmodes, nmodes) for _ in 1:nq]...; dims=3))
    ukqs = arr_dev(cat([rand(ComplexF64, nwe, nband) for _ in 1:nq]...; dims=3))

    irvecp_mat = ElectronPhonon._irvec_matrix(irvec_ep, epmat_obj, Float64)
    ndata = nwe * nband * nmodes

    # (a) reference: q convention, interpolator + qs method
    ep_kR_q = arr_dev(zeros(ComplexF64, ndata, 1, nr_ep))
    get_eph_RR_to_kR_batched!(ep_kR_q, itp_epmat, [xk], uk)
    obj_q = to_dev(WannierObject(irvec_ep, Array(ep_kR_q)[:, 1, :]))
    ref = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq))
    get_eph_kR_to_kq_batched!(ref, get_interpolator(obj_q; fourier_mode="batched", batch_size=nq),
                              qs, uphs, ukqs)

    # (b) same phase, phase-taking method: must agree bit for bit
    phase_q = arr_dev(zeros(ComplexF64, nr_ep, nq))
    ElectronPhonon.fourier_phase!(phase_q, irvecp_mat,
                                  arr_dev([q[d] for d in 1:3, q in qs]))
    out_b = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq))
    get_eph_kR_to_kq_batched!(out_b, view(ep_kR_q, :, 1, :), phase_q, uphs, ukqs)
    @test Array(out_b) == Array(ref)

    # (c) k+q convention: fold conj(exp(2πi R_p·x_k)) into the child, transform at x_{k+q}
    P_k = arr_dev(zeros(ComplexF64, nr_ep, 1))
    ElectronPhonon.fourier_phase!(P_k, irvecp_mat, arr_dev(reshape([xk[d] for d in 1:3], 3, 1)))
    ep_kR_kq = arr_dev(zeros(ComplexF64, ndata, 1, nr_ep))
    get_eph_RR_to_kR_batched!(ep_kR_kq, itp_epmat, [xk], uk; kq_convention_phase = P_k)
    P_kq = arr_dev(zeros(ComplexF64, nr_ep, nq))
    ElectronPhonon.fourier_phase!(P_kq, irvecp_mat,
                                  arr_dev([xkq[d] for d in 1:3, xkq in xkqs]))
    out_c = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq))
    get_eph_kR_to_kq_batched!(out_c, view(ep_kR_kq, :, 1, :), P_kq, uphs, ukqs)
    @test isapprox(Array(out_c), Array(ref); rtol)
end

@testset "k+q convention identity (CPU)" begin
    check_eph_kq_convention(identity, identity; rtol=1e-13)
end

# Plumbing of the outer-q batched calculator payload (supports(_, EPDataKBatched) /
# run_calculator!(_, ::EPDataKBatched, ctx)) used by `run_eph_over_q_and_k(...; use_gpu=true)`.
# The per-q device lifecycle reuses the OuterIteration begin/end brackets (same as the CPU outer-q
# loop). Backend-agnostic, so no GPU is needed — checks the opt-in default and the MethodError path.
struct _PlainQCalc <: ElectronPhonon.AbstractCalculator end
mutable struct _OptInQCalc <: ElectronPhonon.AbstractCalculator; setup::Int; flush::Int; end
ElectronPhonon.supports(::_OptInQCalc, ::Type{ElectronPhonon.EPDataKBatched}) = true
ElectronPhonon.calculator_begin!(c::_OptInQCalc, ::ElectronPhonon.OuterIteration, ctx) = (c.setup += 1; nothing)
ElectronPhonon.calculator_end!(c::_OptInQCalc, ::ElectronPhonon.OuterIteration, ctx) = (c.flush += 1; nothing)

@testset "outer-q batched calculator hook plumbing" begin
    ctx = ElectronPhonon.LoopContext(ElectronPhonon.CPUBackend(), ElectronPhonon.SingleMode(), 1, 1:0, 4)
    # Default opts out; a calculator with no run_calculator! method for the payload is a MethodError.
    @test ElectronPhonon.supports(_PlainQCalc(), ElectronPhonon.EPDataKBatched) == false
    pl = ElectronPhonon.EPDataKBatched(nothing, nothing, nothing, nothing, nothing, nothing, nothing, 1)
    @test_throws MethodError ElectronPhonon.run_calculator!(_PlainQCalc(), pl, ctx)

    # Opt-in calculator: supports returns true and the per-iteration brackets dispatch to the override.
    c = _OptInQCalc(0, 0)
    @test ElectronPhonon.supports(c, ElectronPhonon.EPDataKBatched) == true
    ElectronPhonon.calculator_begin!(c, ElectronPhonon.OuterIteration(), ctx)
    ElectronPhonon.calculator_end!(c, ElectronPhonon.OuterIteration(), ctx)
    @test (c.setup, c.flush) == (1, 1)
end

@testset "Rq→kq rotation: GPU cuBLAS fallback (large nw²·nmodes)" begin
    # check_eph_batched exercises the FUSED rqkq kernel (nw²·nmodes ≤ _FUSED_RQKQ_MAX_NW2NM);
    # here nw=8, nmodes=12 ⇒ 768 > 512 forces the `invoke`-to-generic (cuBLAS two-GEMM) branch of
    # the CUDA `eph_apply_rotations_rqkq!`. Compare it to the generic CPU method (ground truth).
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU rqkq fallback test"
    else
        nw, nmodes, nk, nbandk, nbandkq = 8, 12, 10, 8, 8
        g   = rand(ComplexF64, nw * nw * nmodes, nk)
        uks = rand(ComplexF64, nw, nbandk, nk)
        ukqs = rand(ComplexF64, nw, nbandkq, nk)
        ep_cpu = zeros(ComplexF64, nbandkq, nbandk, nmodes, nk)
        eph_apply_rotations_rqkq!(ep_cpu, copy(g), uks, ukqs,
            zeros(ComplexF64, nbandkq, nw * nmodes, nk), zeros(ComplexF64, nw, nbandk, nmodes * nk))
        ep_gpu = CuArray(zeros(ComplexF64, nbandkq, nbandk, nmodes, nk))
        eph_apply_rotations_rqkq!(ep_gpu, CuArray(copy(g)), CuArray(uks), CuArray(ukqs),
            CuArray(zeros(ComplexF64, nbandkq, nw * nmodes, nk)),
            CuArray(zeros(ComplexF64, nw, nbandk, nmodes * nk)))
        @test isapprox(Array(ep_gpu), ep_cpu; rtol=1e-9)
    end
end

@testset "batched eigensolve (CPU)" begin
    # CPU eigen_batched: U must be eigenvectors of H — check eigenvalues vs LAPACK and the
    # gauge-invariant reconstruction H ≈ U·diag(E)·U† at a few k-points. (The GPU counterpart is
    # in "GPU batched Wannier interpolation".)
    nw, nk = 8, 5
    H = Array{ComplexF64,3}(undef, nw, nw, nk)
    for k in 1:nk; A = rand(ComplexF64, nw, nw); @views H[:, :, k] .= (A + A') / 2; end
    Hh = copy(H)   # eigen_batched overwrites its input
    E, U = eigen_batched(H)
    for k in (1, 3, 5)
        @test sort(E[:, k]) ≈ sort(real(eigvals(Hermitian(Hh[:, :, k]))))
        @test U[:, :, k] * Diagonal(E[:, k]) * U[:, :, k]' ≈ Hh[:, :, k]
    end
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
        obj_gpu = to_device(ElectronPhonon.gpu_backend(), obj)
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
        E_ref = get_el_eigen_valueonly_batched(get_interpolator(obj; fourier_mode="batched"), kpts)
        E_gpu = get_el_eigen_valueonly_batched(get_interpolator(obj_gpu; fourier_mode="batched"), kpts)
        @test E_gpu isa CuArray
        @test sort(Array(E_gpu), dims=1) ≈ sort(E_ref, dims=1)

        # --- eigenvalues + eigenvectors (CPU counterpart in "batched eigensolve (CPU)") ---
        Ev_gpu, U_gpu = get_el_eigen_batched(get_interpolator(obj_gpu; fourier_mode="batched"), kpts)
        @test sort(Array(Ev_gpu), dims=1) ≈ sort(E_ref, dims=1)
        # Eigenvectors are gauge-dependent, so check the gauge-invariant reconstruction
        # H(k) ≈ U diag(E) U† for a few k-points.
        Ev = Array(Ev_gpu); U = Array(U_gpu); H = reshape(Hk_cpu, nw, nw, length(kpts))
        for ik in (1, 17, 50)
            @test U[:, :, ik] * Diagonal(Ev[:, ik]) * U[:, :, ik]' ≈ H[:, :, ik]
        end

        # --- batched eigensolve is not limited to nw ≤ 32 on this cuSOLVER; check both the
        #     eigenvalues (vs LAPACK) and that the eigenvectors reconstruct H ≈ U·diag(E)·U† ---
        let nw2 = 40, nk2 = 4
            H = CUDA.rand(ComplexF64, nw2, nw2, nk2)
            for k in 1:nk2; @views H[:, :, k] .= (H[:, :, k] + H[:, :, k]') / 2; end
            Hh = Array(H)   # the batched solvers overwrite their input, so snapshot it first
            Ebig = Array(eigvals_batched(copy(H)))
            Eev, Uev = eigen_batched(copy(H)); Eev = Array(Eev); Uev = Array(Uev)
            for k in 1:nk2
                @test sort(Ebig[:, k]) ≈ sort(real(eigvals(Hermitian(Hh[:, :, k]))))
                @test Uev[:, :, k] * Diagonal(Eev[:, k]) * Uev[:, :, k]' ≈ Hh[:, :, k]
            end
        end

        # electron-phonon batched drivers on the GPU vs the per-k/q CPU reference
        check_eph_batched(obj -> to_device(ElectronPhonon.gpu_backend(), obj), CuArray; rtol=1e-9)

        # phase kernel and the k+q convention on the device
        check_fourier_phase(CuArray)
        check_eph_kq_convention(obj -> to_device(ElectronPhonon.gpu_backend(), obj), CuArray; rtol=1e-13)
    end
end

# Partial final q-batch: the GPU loop runs a batch narrower than the preallocated `nq_batch_max`
# by passing contiguous device VIEWS (`view(buf, :,:,:, 1:nq_batch)`) into
# `get_eph_kR_to_kq_batched!` and reusing the max-width workspace. This checks that path directly:
# the sliced-view result must match the full-width result, through BOTH `eph_apply_rotations!`
# branches — the fused kernel (`nw*nmodes ≤ _FUSED_ROT_MAX_NWNM`) and the cuBLAS
# `gemm_strided_batched!` path (above it), where a reshape of a view must stay a strided CuArray.
function check_eph_partial_view(nw, nmodes; rtol)
    nband, nr_ep, nq, m = nw, 8, 10, 7   # slice width m < full width nq
    irvec_ep = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_ep], by = x -> reverse(x))
    obj  = to_device(ElectronPhonon.gpu_backend(), WannierObject(irvec_ep, rand(ComplexF64, nw*nband*nmodes, nr_ep)))
    qs   = [Vec3(rand(3)...) for _ in 1:nq]
    uphs = CuArray(rand(ComplexF64, nmodes, nmodes, nq))
    ukqs = CuArray(rand(ComplexF64, nw, nband, nq))
    ws   = ElectronPhonon.KRtoKQWorkspace(obj.op_r, nw*nband*nmodes, nband, nband, nmodes, nq)

    full = CuArray(zeros(ComplexF64, nband, nband, nmodes, nq))
    get_eph_kR_to_kq_batched!(full, get_interpolator(obj; fourier_mode="batched", batch_size=nq),
                              qs, uphs, ukqs; ws)
    # Same call restricted to the first m q-points via views into the max-width buffers, reusing ws.
    part = CuArray(zeros(ComplexF64, nband, nband, nmodes, nq))
    get_eph_kR_to_kq_batched!(view(part, :, :, :, 1:m),
        get_interpolator(obj; fourier_mode="batched", batch_size=nq),
        view(qs, 1:m), view(uphs, :, :, 1:m), view(ukqs, :, :, 1:m); ws)
    @test isapprox(Array(view(part, :, :, :, 1:m)), Array(view(full, :, :, :, 1:m)); rtol)
end

@testset "GPU partial q-batch (views into max-width buffers)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU partial-batch test"
    else
        check_eph_partial_view(3, 4; rtol=1e-9)   # nw*nmodes = 12 ≤ 24 → fused kernel path
        check_eph_partial_view(6, 6; rtol=1e-9)   # nw*nmodes = 36 > 24 → cuBLAS strided path
    end
end


"""
The kR→kq GEMM strip (`_krkq_strip_width`): stacking `nk_b` outer-k into one tall
`eph_kR_to_kq_fourier!` and rotating each k off a q-strided slice of the result must reproduce the
`nk_b` separate skinny per-k `get_eph_kR_to_kq_batched!` calls. `nk` is chosen so the last strip is
partial, the case a `k`-batch that is not a multiple of `nk_b` produces in the loop.

At `nk_b = 1` on the GPU the GEMM shape and the kernel are identical to the per-k path, so the
result must be BITWISE equal (`exact`). At `nk_b > 1` it is not: a different `M` lets cuBLAS pick a
different kernel and reduction order — the in-repo precedent for k-batching moving last bits. On
the CPU even `nk_b = 1` moves last bits, because the strip hands the generic rotation a `SubArray`
whose `reshape` is a `ReshapedArray` and so takes the generic `mul!` instead of BLAS `gemm!`.
"""
function check_krkq_strip(arr_dev; nw, nmodes, nk, nk_b, rtol, exact = false)
    nband, nr_ep, nq = nw, 9, 11
    ndata = nw * nband * nmodes
    ep3   = arr_dev(rand(ComplexF64, ndata, nk, nr_ep))     # k in the middle: k-strips are row ranges
    ep2d  = reshape(ep3, ndata * nk, nr_ep)
    phase = arr_dev(rand(ComplexF64, nr_ep, nq))
    uphs  = arr_dev(rand(ComplexF64, nmodes, nmodes, nq))
    ukqs  = arr_dev(rand(ComplexF64, nw, nband, nq))
    ωq    = arr_dev(rand(nmodes, nq) .+ 0.5)

    # Reference: one skinny GEMM + rotation per k (the pre-strip code path).
    ref    = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq, nk))
    ref_g2 = arr_dev(zeros(nband, nband, nmodes, nq, nk))
    ws1 = KRtoKQWorkspace(ep3, ndata, nband, nband, nmodes, nq)
    for ik in 1:nk
        get_eph_kR_to_kq_batched!(view(ref, :, :, :, :, ik), view(ep3, :, ik, :), phase, uphs, ukqs;
            ws = ws1, g2_out = view(ref_g2, :, :, :, :, ik), ωq)
    end

    # Strip: one tall GEMM per strip of `nks ≤ nk_b` k, then the rotation per k.
    out    = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq, nk))
    out_g2 = arr_dev(zeros(nband, nband, nmodes, nq, nk))
    ws = KRtoKQWorkspace(ep3, ndata, nband, nband, nmodes, nq; nk_b)
    g5 = reshape(ws.g, nw, nband, nmodes, nk_b, nq)
    for sstart in 1:nk_b:nk
        nks  = min(nk_b, nk - sstart + 1)
        rows = ndata * (sstart - 1) + 1 : ndata * (sstart - 1 + nks)
        ElectronPhonon.eph_kR_to_kq_fourier!(view(ws.g, 1:ndata*nks, :), view(ep2d, rows, :), phase)
        for ik in sstart:(sstart + nks - 1)
            # Build the rotation's operand exactly as the loop does. `nk_b = 1` must give a dense
            # one — that is what lets a large-`nw` run, which has no fused kernel to fall back on,
            # take the strip code path at all; `nk_b > 1` gives a q-strided one, fused-kernel only.
            gk = ElectronPhonon._strip_g_slice(ws.g, g5, ik - sstart + 1, 1:nq)
            @test ElectronPhonon._is_dense(gk) == (nk_b == 1)
            eph_apply_rotations!(view(out, :, :, :, :, ik), gk,
                ukqs, uphs, ws.tmp; g2_out = view(out_g2, :, :, :, :, ik), ωq)
        end
    end

    if exact
        @test Array(out) == Array(ref)
        @test Array(out_g2) == Array(ref_g2)
    else
        @test isapprox(Array(out), Array(ref); rtol)
        @test isapprox(Array(out_g2), Array(ref_g2); rtol)
    end
end

@testset "kR→kq GEMM strip" begin
    # nk_b = 1 is the auto-disabled path (large ndata, or a k-batch of one).
    check_krkq_strip(identity; nw=3, nmodes=4, nk=3, nk_b=1, rtol=1e-13)
    # A strided per-k `g` is only readable by the CUDA fused kernel; the generic two-GEMM method
    # must say so rather than silently reshaping into a `ReshapedArray`.
    let nw = 3, nband = 3, nmodes = 4, nq = 5
        g_strided = view(reshape(rand(ComplexF64, nw*nband*nmodes*2, nq), nw, nband, nmodes, 2, nq),
                         :, :, :, 1, :)
        @test_throws AssertionError eph_apply_rotations!(
            zeros(ComplexF64, nband, nband, nmodes, nq), g_strided,
            rand(ComplexF64, nw, nband, nq), rand(ComplexF64, nmodes, nmodes, nq),
            zeros(ComplexF64, nband, nband * nmodes, nq))
    end
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU kR→kq strip test"
    else
        check_krkq_strip(CuArray; nw=3, nmodes=4, nk=3, nk_b=1, rtol=0, exact=true)
        # nw*nmodes = 12 ≤ 24 → fused rotation, so the q-strided strip slice is allowed.
        # nk = 7, nk_b = 3 → strips of 3, 3, 1: the last one is partial.
        check_krkq_strip(CuArray; nw=3, nmodes=4, nk=7, nk_b=3, rtol=1e-13)
        # nw*nmodes = 36 > 24 → nk_b is forced to 1 and the rotation is the cuBLAS two-GEMM branch,
        # which needs an operand it can take a pointer to. This is the combination a full-band
        # large-`nw` model ships with, and it is NOT the driver's nk_outer_batch_max = 1 clamp.
        @test ElectronPhonon._krkq_strip_width(; nw=6, nbandk_max=6, nmodes=6, nk_batch_max=256) == 1
        check_krkq_strip(CuArray; nw=6, nmodes=6, nk=3, nk_b=1, rtol=0, exact=true)
        # Same operand at a PARTIAL final q-tile: dense is not enough for cuBLAS, the `reshape`
        # that merges the band and mode axes must stay a device array (a `ReshapedArray` has no
        # pointer). The shapes above only cover the full-width tile.
        let nw = 6, nband = 6, nmodes = 6, nqmax = 5, nq = 3
            g2d = CuArray(zeros(ComplexF64, nw*nband*nmodes, nqmax))
            gk  = ElectronPhonon._strip_g_slice(g2d, reshape(g2d, nw, nband, nmodes, 1, nqmax), 1, 1:nq)
            @test size(gk) == (nw, nband, nmodes, nq)
            @test reshape(gk, nw, nband * nmodes, nq) isa CuArray
        end
    end
end


# A minimal AbstractCalculator that records the mode-resolved g2 and phonon frequency for
# every (ik, ikq). It mirrors what MigdalEliashberg's EliashbergCalculator reads
# (epstate.g2[m,n,imode], epstate.ph.e[imode]) but has no external dependency, so it can verify
# the GPU calculator loop (run_eph_over_k_and_kq use_gpu) against the CPU loop here.
mutable struct _RecordCalc <: ElectronPhonon.AbstractCalculator
    g2::Array{Float64,5}    # (nw, nw, nmodes, nk, nkq)
    ωq::Array{Float64,5}
    _RecordCalc() = new(zeros(0, 0, 0, 0, 0), zeros(0, 0, 0, 0, 0))
end
ElectronPhonon.supports(::_RecordCalc, ::Type{ElectronPhonon.OuterKLoop}) = true
ElectronPhonon.supports(::_RecordCalc, ::Type{ElectronPhonon.EPData}) = true
# Nothing per outer iteration; explicit no-op (there is no default bracket). CPU-only ⇒ SingleMode.
ElectronPhonon.calculator_begin!(::_RecordCalc, ::ElectronPhonon.OuterIteration, ctx) = nothing
ElectronPhonon.calculator_end!(::_RecordCalc, ::ElectronPhonon.OuterIteration, ctx) = nothing
function ElectronPhonon.setup_calculator!(c::_RecordCalc, kpts, qpts, el_states;
        el_states_kq, kqpts, nw, nmodes, kwargs...)
    c.g2 = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c.ωq = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c
end
ElectronPhonon.postprocess_calculator!(c::_RecordCalc; kwargs...) = c
function ElectronPhonon.run_calculator!(c::_RecordCalc, p::ElectronPhonon.EPData, ctx)
    (; epstate, ik, ikq) = p
    (; el_k, el_kq, ph) = epstate
    for imode in 1:ph.nmodes, n in el_k.rng, m in el_kq.rng
        c.g2[m, n, imode, ik, ikq] = epstate.g2[m, n, imode]
        c.ωq[m, n, imode, ik, ikq] = ph.e[imode]
    end
    c
end

# Device-native counterpart of `_RecordCalc`: opts into the batched payload so the GPU loop keeps
# the e-ph matrix on the device and calls `run_calculator!(::EPDataQBatched, ctx)` once per
# (k, batch). It records the same `g2 = |ep|²/2ω` and ωq as `_RecordCalc`, exercising the loop's
# device-native branch (supports(EPDataQBatched), ωq/ikqs staging, on-device g2) without MigdalEliashberg.
mutable struct _RecordCalcBatched <: ElectronPhonon.AbstractCalculator
    g2::Array{Float64,5}
    ωq::Array{Float64,5}
    _RecordCalcBatched() = new(zeros(0, 0, 0, 0, 0), zeros(0, 0, 0, 0, 0))
end
ElectronPhonon.supports(::_RecordCalcBatched, ::Type{ElectronPhonon.OuterKLoop}) = true
ElectronPhonon.supports(::_RecordCalcBatched, ::Type{ElectronPhonon.EPDataQBatched}) = true
# Device-native (GPU outer-k): that loop fires only the per-batch OuterIterationBatch bracket
# (BatchedMode); this calculator needs nothing there, so define an explicit no-op. The
# `OuterIteration` no-ops keep it usable under the CPU outer-k loop too, which does fire them.
ElectronPhonon.calculator_begin!(::_RecordCalcBatched, ::ElectronPhonon.OuterIteration, ctx) = nothing
ElectronPhonon.calculator_end!(::_RecordCalcBatched, ::ElectronPhonon.OuterIteration, ctx) = nothing
ElectronPhonon.calculator_begin!(::_RecordCalcBatched, ::ElectronPhonon.OuterIterationBatch, ctx) = nothing
ElectronPhonon.calculator_end!(::_RecordCalcBatched, ::ElectronPhonon.OuterIterationBatch, ctx) = nothing
function ElectronPhonon.setup_calculator!(c::_RecordCalcBatched, kpts, qpts, el_states;
        el_states_kq, kqpts, nw, nmodes, kwargs...)
    c.g2 = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c.ωq = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c
end
ElectronPhonon.postprocess_calculator!(c::_RecordCalcBatched; kwargs...) = c
# Computes g2 from `eps` itself (independent check of the fused-kernel matrix output). The GPU
# loop also folds g2 into the kRkq kernel and carries it in the payload; assert it matches the
# abs2/(2ω) recomputation — this guards the production g2 output in-package.
function ElectronPhonon.run_calculator!(c::_RecordCalcBatched, p::ElectronPhonon.EPDataQBatched, ctx)
    (; eps, g2s, ωqs, ik, ikqs, ibandk_offset) = p
    nbandkq, nbandk, nm, nqc = size(eps)
    g2dev = abs2.(eps) ./ (2 .* reshape(ωqs, 1, 1, nm, nqc))
    @assert maximum(abs, Array(g2s .- g2dev)) <= 1e-10 * maximum(abs, Array(g2dev))
    g2h = Array(g2dev)   # device → host (m,n,ν,j)
    ωh = Array(ωqs)
    ikqsh = Array(ikqs)
    for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        # ibandk_offset: the loop's k-side window projection offset (eps band n ↔ physical band ibandk_offset+n).
        c.g2[m, ibandk_offset + n, ν, ik, ikqsh[j]] = g2h[m, n, ν, j]
        c.ωq[m, ibandk_offset + n, ν, ik, ikqsh[j]] = ωh[ν, j]
    end
    c
end

@testset "GPU calculator loop (run_eph_over_k_and_kq use_gpu)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU calculator-loop test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        grid = (4, 4, 4)

        # NOTE on CPU-vs-GPU comparison: with use_gpu=true the SETUP eigensolve also runs on the
        # device (batched), which does NOT apply the EPW degeneracy gauge-fixing of the per-k CPU
        # path. So for degenerate bands/modes the GPU e-ph matrix / g2 differs from CPU by a gauge
        # (a unitary rotation within each degenerate subspace) — physically equivalent, but not
        # bit-identical, and largest on COARSE grids (more exact high-symmetry degeneracies; this
        # 4³ Pb grid is such a case). Eigenvalues are gauge-independent, so we compare ωq against
        # CPU; g2 correctness is checked GPU-vs-GPU, where all paths share the same GPU gauge.
        # CPU reference (gauge-independent ωq). Uses the non-batched host record calculator.
        cc = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cc], symmetry=nothing, progress_print_step=10^9)

        # GPU reference: the device-native batched calculator (single q-batch). The GPU path is
        # fully GPU (a non-batched calculator is rejected — see below), so the batched hook is the
        # GPU reference for the g2 comparisons. `_RecordCalcBatched` also independently recomputes
        # g2 = |ep|²/2ω from ep_kq and cross-checks the loop's folded g2, and `check_eph_batched`
        # above validates ep_kq itself — so the device g2 output is covered without a host path.
        cg = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cg], symmetry=nothing, use_gpu=true, progress_print_step=10^9)

        scale = maximum(abs, cg.g2)
        # Phonon frequencies are gauge-independent → must match the CPU path (eigenvalue precision).
        @test maximum(abs, cc.ωq .- cg.ωq) < 1e-6 * maximum(abs, cc.ωq)

        # GPU bit-faithfulness (shared gauge): multiple q-batches (nq_batch_max=7 → partial final
        # batch) must reproduce the single-batch GPU result exactly.
        cg7 = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cg7], symmetry=nothing, use_gpu=true, nq_batch_max=7, progress_print_step=10^9)
        @test maximum(abs, cg.g2 .- cg7.g2) < 1e-9 * scale
        @test cg.ωq == cg7.ωq

        # Outer-k batching (nk_outer_batch_max=5 forces a partial final k-batch) must agree too,
        # together with a partial q-batch.
        cbk = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cbk], symmetry=nothing, use_gpu=true,
            nk_outer_batch_max=5, nq_batch_max=7, progress_print_step=10^9)
        @test maximum(abs, cg.g2 .- cbk.g2) < 1e-9 * scale

        # nk_outer_batch_max = 1 clamps the kR→kq strip to nk_b = 1; the runs above use
        # nk_b = _krkq_strip_width(...) > 1 for this model, so both strip widths are exercised at
        # driver level. This is NOT the shape-derived auto-disable (large ndata / above the
        # fused-rotation threshold): the clamp degenerates the k-batch too, so the phase tile is
        # rebuilt per k and `ep_ekpR_all` is one k wide. That combination — nk_b = 1 with a full
        # k-batch and the cuBLAS rotation — is covered by the `kR→kq GEMM strip` testset above.
        cb1 = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cb1], symmetry=nothing, use_gpu=true,
            nk_outer_batch_max=1, progress_print_step=10^9)
        @test maximum(abs, cg.g2 .- cb1.g2) < 1e-9 * scale
        @test ElectronPhonon._krkq_strip_width(; nw = model.nw, nbandk_max = model.nw,
                  nmodes = model.nmodes, nk_batch_max = 256) > 1

        # Fully-GPU policy: a non-batched calculator (does not support EPDataQBatched) must be
        # rejected, not silently run on the host path.
        @test_throws Exception ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[_RecordCalc()], symmetry=nothing, use_gpu=true, progress_print_step=10^9)

        # Scope guards: the GPU path must reject out-of-scope options it does not implement (use a
        # batched calculator so the rejection is the scope guard, not the fully-GPU policy above).
        @test_throws Exception ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[_RecordCalcBatched()], symmetry=nothing, use_gpu=true,
            covariant_derivative_of_g=true, progress_print_step=10^9)
        @test_throws Exception ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[_RecordCalcBatched()], symmetry=nothing, use_gpu=true,
            energy_conservation=(:Fixed, 0.1), progress_print_step=10^9)
    end
end

# Outer-q analogue of `_RecordCalc`/`_RecordCalcBatched` above: a calculator for
# `run_eph_over_q_and_k` that accumulates a per-q, GAUGE-INVARIANT scalar
#   A[iq] = Σ_{m,n,ν,k} wtk[k] · |ep[m,n,ν,k]|²
# (the band-summed |g|² is invariant under the electron/phonon eigenvector gauge, so the
# degenerate-band gauge difference between the CPU LAPACK and GPU batched eigensolvers — see the
# note in the outer-k test above — does not matter here). This lets the SAME calculator validate
# both the CPU (`run_calculator!(::EPData)`) and GPU-batched (`run_calculator!(::EPDataKBatched)`) hooks of
# `run_eph_over_q_and_k` against each other directly, without restricting to eigenvalues only.
#
# Thread-safety: `run_calculator!` on the CPU path is called from multiple threads (one per
# k-chunk) for the SAME iq, so the per-q sum is accumulated into a per-chunk slot (`id_chunk`,
# passed by the driver) and reduced in `calculator_end!(::OuterIteration, ctx)`; no cross-thread races.
mutable struct _RecordCalcOuterQ <: ElectronPhonon.AbstractCalculator
    A::Vector{Float64}       # (nq,) final per-q gauge-invariant sum
    Achunk::Vector{Float64}  # (nchunks,) per-thread-chunk partial sum for the CURRENT q (CPU path)
    Adev::Any                # length-1 device accumulator for the CURRENT q (GPU path), or nothing
    _RecordCalcOuterQ() = new(zeros(0), zeros(0), nothing)
end
ElectronPhonon.supports(::_RecordCalcOuterQ, ::Type{ElectronPhonon.OuterQLoop}) = true
ElectronPhonon.supports(::_RecordCalcOuterQ, ::Type{ElectronPhonon.EPData}) = true
ElectronPhonon.supports(::_RecordCalcOuterQ, ::Type{ElectronPhonon.EPDataKBatched}) = true
ElectronPhonon.allowed_eph_phonon_basis(::_RecordCalcOuterQ) = [:eigenmode]
# Reads only the e-ph matrix (→ eigenvalues/eigenvectors); skips velocity/position.
ElectronPhonon.required_el_k_quantities(::_RecordCalcOuterQ) = ["eigenvalue", "eigenvector"]
function ElectronPhonon.setup_calculator!(c::_RecordCalcOuterQ, kpts, qpts, el_states;
        nchunks_threads=nthreads(), kwargs...)
    c.A = zeros(qpts.n)
    c.Achunk = zeros(nchunks_threads)
    c.Adev = nothing
    c
end
# Per-point path: zero this q's per-chunk accumulator. Batched path: (re)allocate and zero this q's
# length-1 device accumulator from ctx.backend. The loop shape is read from `ctx.mode`, not the backend.
function ElectronPhonon.calculator_begin!(c::_RecordCalcOuterQ, ::ElectronPhonon.OuterIteration, ctx)
    if ctx.mode isa ElectronPhonon.BatchedMode
        c.Adev = fill!(ElectronPhonon.alloc(ctx.backend, Float64, 1), 0.0)
    else
        fill!(c.Achunk, 0.0)
    end
    c
end
# Reduce this q's accumulator (per-chunk sum, or device→host copy) into A[iq] (iq = ctx.outer_index).
function ElectronPhonon.calculator_end!(c::_RecordCalcOuterQ, ::ElectronPhonon.OuterIteration, ctx)
    iq = ctx.outer_index
    c.A[iq] = c.Adev === nothing ? sum(c.Achunk) : Array(c.Adev)[1]
    c.Adev = nothing
    c
end
ElectronPhonon.postprocess_calculator!(c::_RecordCalcOuterQ; kwargs...) = c
# CPU path: epstate.ep is already an OffsetArray restricted to (el_kq.rng, el_k.rng, :), so summing
# all of it covers exactly the in-window (m, n, ν) triples.
function ElectronPhonon.run_calculator!(c::_RecordCalcOuterQ, p::ElectronPhonon.EPData, ctx)
    c.Achunk[p.id_chunk] += p.epstate.wtk * sum(abs2, p.epstate.ep)
    c
end
# GPU batched path: `ep_kq` is (nw,nw,nmodes,nkc) on the device, full-band (out-of-window bands already
# zeroed by the loop's eigenvector-column masking); the payload is trimmed to this batch's actual width
# `nkc` (no padded tail), so the reduction reads its size from `size(ep, 4)`. `sum` of a device
# broadcast expression reduces on-device and returns a host scalar without any scalar indexing
# (allowed under CUDA.allowscalar(false)).
function ElectronPhonon.run_calculator!(c::_RecordCalcOuterQ, p::ElectronPhonon.EPDataKBatched, ctx)
    ep, wtk = p.eps, p.wtk
    nkc = size(ep, 4)
    val = sum(abs2.(ep) .* reshape(wtk, 1, 1, 1, nkc))
    c.Adev .+= val
    c
end

@testset "run_eph_over_q_and_k CPU vs GPU equivalence" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping run_eph_over_q_and_k CPU-vs-GPU test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
        grid = (4, 4, 4)

        calc_cpu = _RecordCalcOuterQ()
        ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
            calculators=[calc_cpu], use_symmetry=false, keep_all_qpts=true,
            use_gpu=false, progress_print_step=10^9)

        calc_gpu = _RecordCalcOuterQ()
        ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
            calculators=[calc_gpu], use_symmetry=false, keep_all_qpts=true,
            use_gpu=true, progress_print_step=10^9)

        rdiff = maximum(abs, calc_cpu.A .- calc_gpu.A) / maximum(abs, calc_cpu.A)
        @info "run_eph_over_q_and_k CPU vs GPU" cpu_A=calc_cpu.A gpu_A=calc_gpu.A rdiff
        @test isapprox(calc_cpu.A, calc_gpu.A; rtol=1e-8)
    end
end

# F4: force a PARTIAL outer-q k-batch (small nk_batch_max) so the DECISION-9 payload trim is a real
# width-nk_batch < nk_batch_max trim, not the identity trim of the single-batch test above. nk=64,
# nk_batch_max=10 ⇒ 7 batches, the last of width 4 — the trimmed-view path is exercised end-to-end.
@testset "run_eph_over_q_and_k partial k-batch trim (CPU vs GPU, DECISION-9)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping run_eph_over_q_and_k partial-batch test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
        grid = (4, 4, 4)

        calc_cpu = _RecordCalcOuterQ()
        ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
            calculators=[calc_cpu], use_symmetry=false, keep_all_qpts=true,
            use_gpu=false, progress_print_step=10^9)

        calc_gpu = _RecordCalcOuterQ()
        ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
            calculators=[calc_gpu], use_symmetry=false, keep_all_qpts=true,
            use_gpu=true, nk_batch_max=10, progress_print_step=10^9)

        rdiff = maximum(abs, calc_cpu.A .- calc_gpu.A) / maximum(abs, calc_cpu.A)
        @info "run_eph_over_q_and_k partial k-batch (nk_batch_max=10) CPU vs GPU" rdiff
        @test isapprox(calc_cpu.A, calc_gpu.A; rtol=1e-8)
    end
end

@testset "GPU filter_kpoints with symmetry (IBZ reduction × use_gpu)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU filter_kpoints symmetry test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        # Fine-mesh Fermi level / 0.3 eV window, as in the anisotropic-ME (mp_mesh_k) pipeline.
        ef = 11.682221647 * ElectronPhonon.unit_to_aru(:eV)
        window = (ef - 0.3 * ElectronPhonon.unit_to_aru(:eV), ef + 0.3 * ElectronPhonon.unit_to_aru(:eV))
        # In filter_kpoints, `symmetry` (IBZ reduction, in kpoints_grid) and `use_gpu` (batched
        # eigensolve for the window test) are orthogonal: the IBZ k-set is built backend-independently,
        # and use_gpu only changes how the band eigenvalues are computed. The window test is discrete
        # (which bands fall inside), so it is robust to the ~1e-12 eigenvalue difference between the
        # cuSOLVER and CPU eigensolvers ⇒ identical ik_keep / band range / nelec_below, hence an
        # identical IBZ Kpoints object. (Eigenvectors / gauge are not involved here; cf. the g2
        # gauge caveat in the calculator-loop test above.)
        for nk in (12, 24)
            rsel = filter_electron_states((nk, nk, nk), model.nw, model.el_ham, window;
                symmetry = model.symmetry, use_gpu = false)
            gsel = filter_electron_states((nk, nk, nk), model.nw, model.el_ham, window;
                symmetry = model.symmetry, use_gpu = true)
            rk = rsel.kpts; gk = gsel.kpts
            @test gk.n == rk.n
            @test gk.ngrid == rk.ngrid
            @test gk.vectors == rk.vectors
            @test gk.weights == rk.weights
            @test band_range(gsel) == band_range(rsel)
            @test gsel.nstates_base == rsel.nstates_base
        end
    end
end

@testset "GPU compute_electron_states on the IBZ set (windowed)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU IBZ compute_electron_states test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        ef = 11.682221647 * ElectronPhonon.unit_to_aru(:eV)
        window = (ef - 0.3 * ElectronPhonon.unit_to_aru(:eV), ef + 0.3 * ElectronPhonon.unit_to_aru(:eV))
        # The anisotropic-ME outer states are the IBZ k-points from filter_kpoints (cf. the R1 test);
        # feed exactly that set to compute_electron_states and confirm the GPU eigensolve agrees with
        # CPU. Eigenvalues and the in-window band range are gauge-independent and must match to
        # eigenvalue precision. Eigenvectors (u_full) are NOT compared: the batched GPU eigensolve
        # does not apply the per-k EPW degeneracy gauge-fixing, so within Pb's cubic-degenerate
        # subspaces u_full differs by a (physically equivalent) unitary rotation — same caveat as the
        # velocity test below and the g2 gauge note in the calculator-loop test.
        kpts_ibz = filter_electron_states((24, 24, 24), model.nw, model.el_ham, window;
            symmetry = model.symmetry, use_gpu = false).kpts
        qv = ["eigenvalue", "eigenvector", "velocity", "position"]
        els_c = ElectronPhonon.compute_electron_states(model, kpts_ibz, qv, window; fourier_mode="gridopt")
        els_g = ElectronPhonon.compute_electron_states(model, kpts_ibz, qv, window; use_gpu=true)
        @test length(els_g) == length(els_c) == kpts_ibz.n
        demax = maximum(maximum(abs, els_c[ik].e_full .- els_g[ik].e_full) for ik in 1:kpts_ibz.n)
        escale = maximum(maximum(abs, els_c[ik].e_full) for ik in 1:kpts_ibz.n)
        @test demax < 1e-10 * escale
        @test all(els_c[ik].rng == els_g[ik].rng for ik in 1:kpts_ibz.n)
    end
end

@testset "compute_electron_states velocity (use_gpu)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU compute_electron_states velocity test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        kpts = ElectronPhonon.kpoints_grid((8, 8, 8))
        nk, nw = kpts.n, model.nw
        @assert model.el_velocity_mode === :BerryConnection  # Pb model_new

        qv = ["eigenvalue", "eigenvector", "velocity", "position"]
        els_c = ElectronPhonon.compute_electron_states(model, kpts, qv, (-Inf, Inf); fourier_mode="gridopt")
        els_g = ElectronPhonon.compute_electron_states(model, kpts, qv, (-Inf, Inf); use_gpu=true)

        # Eigenvalues are gauge-independent → must match the CPU path to eigenvalue precision.
        emax = maximum(maximum(abs, els_c[ik].e_full .- els_g[ik].e_full) for ik in 1:nk)
        @test emax < 1e-10 * maximum(maximum(abs, els_c[ik].e_full) for ik in 1:nk)

        # Strong, gauge-independent correctness gate: run the SAME device velocity path (el_ham_R
        # rotation + Berry term im*(e_i-e_j)*rbar) on the CPU eigenvectors and compare to the CPU
        # `get_el_velocity_berry_connection!`. Sharing the eigenvectors removes the degeneracy-gauge
        # difference, so this must match to machine precision (validates rotation + Berry math).
        ufc = zeros(ComplexF64, nw, nw, nk); ec = zeros(Float64, nw, nk)
        for ik in 1:nk; ufc[:, :, ik] .= els_c[ik].u_full; ec[:, ik] .= els_c[ik].e_full; end
        itp_v = get_interpolator(ElectronPhonon.to_device(ElectronPhonon.gpu_backend(), model.el_ham_R); fourier_mode="batched")
        v_dev = ElectronPhonon.get_el_velocity_direct_batched(itp_v, kpts.vectors, CuArray(ufc))
        itp_rbar = get_interpolator(ElectronPhonon.to_device(ElectronPhonon.gpu_backend(), model.el_pos); fourier_mode="batched")
        rbar_dev = ElectronPhonon.get_el_velocity_direct_batched(itp_rbar, kpts.vectors, CuArray(ufc))
        let E = CuArray(ec)
            v_dev .+= im .* (reshape(E, nw, 1, 1, nk) .- reshape(E, 1, nw, 1, nk)) .* rbar_dev
        end
        v_anchor = Array(v_dev)  # (nw, nw, 3, nk)
        gm = gs = 0.0
        for ik in 1:nk
            el = els_c[ik]
            for jb in el.rng, ib in el.rng, idir in 1:3
                gm = max(gm, abs(el.v[ib, jb][idir] - v_anchor[ib, jb, idir, ik]))
                gs = max(gs, abs(el.v[ib, jb][idir]))
            end
        end
        @test gm < 1e-11 * gs

        # vdiag is gauge-invariant for NON-degenerate bands (within a degenerate subspace it depends
        # on the gauge, which the batched GPU eigensolve does not fix). So compare CPU-vs-GPU vdiag
        # only on bands with no other band within 1e-6 Ha; these must match closely.
        vmax = vscale = 0.0; n_nondeg = 0
        for ik in 1:nk
            el = els_c[ik]; e = el.e_full
            for i in el.rng
                any(j -> j != i && abs(e[j] - e[i]) < 1e-6, el.rng) && continue
                n_nondeg += 1
                vmax = max(vmax, maximum(abs, els_c[ik].vdiag[i] .- els_g[ik].vdiag[i]))
                vscale = max(vscale, maximum(abs, els_c[ik].vdiag[i]))
            end
        end
        @test n_nondeg > 0
        @test vmax < 1e-8 * vscale

        # velocity_diagonal-only path: must equal the diagonal of the full-velocity result exactly,
        # and match CPU on non-degenerate bands.
        qd = ["eigenvalue", "eigenvector", "velocity_diagonal"]
        els_gd = ElectronPhonon.compute_electron_states(model, kpts, qd, (-Inf, Inf); use_gpu=true)
        els_cd = ElectronPhonon.compute_electron_states(model, kpts, qd, (-Inf, Inf); fourier_mode="gridopt")
        ddiag = 0.0
        for ik in 1:nk, i in els_g[ik].rng
            ddiag = max(ddiag, maximum(abs, els_gd[ik].vdiag[i] .- els_g[ik].vdiag[i]))
        end
        @test ddiag < 1e-13  # identical to real(diag(v)) of the full-velocity path
        vdm = vds = 0.0
        for ik in 1:nk
            el = els_cd[ik]; e = el.e_full
            for i in el.rng
                any(j -> j != i && abs(e[j] - e[i]) < 1e-6, el.rng) && continue
                vdm = max(vdm, maximum(abs, els_cd[ik].vdiag[i] .- els_gd[ik].vdiag[i]))
                vds = max(vds, maximum(abs, els_cd[ik].vdiag[i]))
            end
        end
        @test vdm < 1e-8 * vds
    end
end

@testset "compute_phonon_states velocity_diagonal (use_gpu)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU compute_phonon_states velocity test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        kpts = ElectronPhonon.kpoints_grid((8, 8, 8))
        nk, nm = kpts.n, model.nmodes

        qp = ["eigenvalue", "eigenvector", "velocity_diagonal"]
        pc = ElectronPhonon.compute_phonon_states(model, kpts, qp; fourier_mode="gridopt")
        pg = ElectronPhonon.compute_phonon_states(model, kpts, qp; use_gpu=true)

        # Phonon frequencies are gauge-independent → must match the CPU path.
        wm = maximum(maximum(abs, pc[ik].e .- pg[ik].e) for ik in 1:nk)
        @test wm < 1e-7 * maximum(maximum(abs, pc[ik].e) for ik in 1:nk)

        # vdiag = real(diag(u'·dD/dk·u))/(2ω). Gauge-invariant for non-degenerate modes; compare CPU
        # vs GPU on non-degenerate, non-Γ-acoustic modes (skip ω<1e-5 where /2ω blows up).
        vm = vs = 0.0; n_ok = 0
        for ik in 1:nk
            e = pc[ik].e
            for i in 1:nm
                e[i] < 1e-5 && continue
                any(j -> j != i && abs(e[j] - e[i]) < 1e-7, 1:nm) && continue
                n_ok += 1
                vm = max(vm, maximum(abs, pc[ik].vdiag[i] .- pg[ik].vdiag[i]))
                vs = max(vs, maximum(abs, pc[ik].vdiag[i]))
            end
        end
        @test n_ok > 0
        @test vm < 1e-8 * vs
    end
end


# Scatter round-trip: the device-resident scatter `eph_window_scatter!` (used by
# EliashbergCalculator's device path) must (1) write COLLISION-FREE — its non-collision invariant
# (distinct k → distinct outer state i, distinct k+q → distinct inner state f, so every target linear
# index is unique across the run) is what makes the atomic-free device writes correct — and (2) agree
# bit-for-bit between the generic (CPU) method and the CUDA kernel. Builds window-aware imaps (some
# out-of-window entries == 0) mimicking a small run and checks both the full-buffer (i0=0,
# ni_stride=n_i) and per-tile block-buffer (i0≠0, ni_stride=tile extent) addressings.
@testset "eph_window_scatter! round-trip (collision-free + CPU==CUDA)" begin
    using Random
    Random.seed!(20260717)
    FT = Float64
    nw = 4; nbandk = 3; nm = 2; nqc = 5; nkq = 8
    # Distinct positive global state indices, with some 0 (out-of-window) entries.
    imap_i_col = [0, 5, 2]                              # outer states for the nbandk projected bands
    n_i = 6
    imap_f = reshape(collect(1:nw*nkq), nw, nkq)        # distinct global inner states, all in-window
    imap_f[1, 2] = 0; imap_f[3, 5] = 0                  # a couple out-of-window
    n_f = nw * nkq
    ikqs = [2, 5, 7, 8, 3]                              # this chunk's (distinct) k+q indices

    # (1) Collision-free: the target linear indices of the in-window writes are all distinct.
    lins = Int[]
    for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nw
        i = imap_i_col[n]; f = imap_f[m, ikqs[j]]
        (i > 0 && f > 0) || continue
        push!(lins, ν + nm * (i - 1) + nm * n_i * (f - 1))
    end
    @test !isempty(lins)
    @test allunique(lins)

    g2vals = abs.(randn(nw, nbandk, nm, nqc))
    ωq = 0.01 .+ abs.(randn(nm, nqc))

    if GPU_AVAILABLE
        for (ni_stride, i0) in ((n_i, 0), (5, 1))       # full buffer, then a per-tile block buffer
            len = nm * ni_stride * n_f
            g2c = zeros(FT, len); ωc = zeros(FT, len)
            ElectronPhonon.eph_window_scatter!(g2c, ωc, g2vals, imap_i_col, imap_f, ikqs, ωq,
                nw, nbandk, nm, nqc, ni_stride, i0)
            g2g = CUDA.zeros(FT, len); ωg = CUDA.zeros(FT, len)
            ElectronPhonon.eph_window_scatter!(g2g, ωg, CUDA.CuArray(g2vals),
                CUDA.CuArray(imap_i_col), CUDA.CuArray(imap_f), CUDA.CuArray(ikqs), CUDA.CuArray(ωq),
                nw, nbandk, nm, nqc, ni_stride, i0)
            @test Array(g2g) == g2c                     # same integer indexing + copy ⇒ bit-identical
            @test Array(ωg) == ωc
        end

        # The outer-k driver always hands a contiguous k+q tile, so `ikqs` reaches the kernel as a
        # `UnitRange` (isbits, passed in the launch parameters) rather than a device array. Same
        # values, same result — but it is a different argument type through `cudaconvert`.
        rng_ikqs = 2:6
        len = nm * n_i * n_f
        g2c = zeros(FT, len); ωc = zeros(FT, len)
        ElectronPhonon.eph_window_scatter!(g2c, ωc, g2vals, imap_i_col, imap_f, rng_ikqs, ωq,
            nw, nbandk, nm, nqc, n_i, 0)
        g2g = CUDA.zeros(FT, len); ωg = CUDA.zeros(FT, len)
        ElectronPhonon.eph_window_scatter!(g2g, ωg, CUDA.CuArray(g2vals),
            CUDA.CuArray(imap_i_col), CUDA.CuArray(imap_f), rng_ikqs, CUDA.CuArray(ωq),
            nw, nbandk, nm, nqc, n_i, 0)
        @test Array(g2g) == g2c
        @test Array(ωg) == ωc
    end
end


# --- Stage 5: device-staging byte-accounting parity + memory-adaptive sizing (CPU-only) ------------
# Pins the staging byte functions against the byte formulas they replaced (the transition @assert of
# the plan, made durable). If a term drifts, these fail. No GPU needed.
using ElectronPhonon: plan_batch, _outer_k_staging_bytes, _outer_q_staging_bytes,
    estimate_device_memory, CPUBackend

# A calculator declaring per-point device scratch, to exercise the calculator-scratch term.
struct _ByteCalc <: ElectronPhonon.AbstractCalculator
    kbytes::Int
    qbytes::Int
end
ElectronPhonon.eph_batched_bytes_per_point(c::_ByteCalc, ::Type{ElectronPhonon.EPDataQBatched}; nw, nmodes) = c.kbytes
ElectronPhonon.eph_batched_bytes_per_point(c::_ByteCalc, ::Type{ElectronPhonon.EPDataKBatched}; nw, nmodes) = c.qbytes

# A stub backend with a settable free-memory budget, to drive the adaptive-width / fail-early paths
# on the CPU (no CUDA needed).
struct _StubBackend <: ElectronPhonon.AbstractBackend
    free::Int
end
ElectronPhonon.free_bytes(b::_StubBackend) = b.free

@testset "device-staging byte-accounting parity (Stage 5)" begin
    FT = Float64
    calcs = [_ByteCalc(1234, 5678)]

    @testset "outer-k parity" begin
        nw, nbandk_max, nmodes, nr_ep, nk, nkq, nq_grid, nk_batch_max = 7, 5, 6, 137, 90, 200, 64, 32
        ndata_epmat, nr_epmat = 2064, 43
        # Formulas reproduced inline (ground truth). The per-q term dropped the child interpolator
        # (cached_results / rdotk / xkmat) and `ikqs_dev` when the k+q-convention phase hoist made
        # them unnecessary; `24·nr_ep` (phase + rdotk) became `16·nr_ep` (the caller-owned P_kq tile).
        # `kRkq_ws.g` is the only term that carries the kR→kq strip width `nk_b`, so the `56` splits
        # into `40 + 16·nk_b`.
        exp_per_q(nk_b) = (40 + 16 * nk_b) * nw * nbandk_max * nmodes +
            16 * nr_ep + 16 * nmodes^2 + 8 * nmodes + 8 +
            sum(ElectronPhonon.eph_batched_bytes_per_point(c, ElectronPhonon.EPDataQBatched; nw, nmodes) for c in calcs)
        old_committed = 16 * nw^2 * nkq + (16 * nmodes^2 + 8 * nmodes) * nq_grid +
            16 * nw * nbandk_max * (nmodes * nr_ep + 1) * nk_batch_max
        # itp_epmat RR→kR interpolator Fourier scratch (2026-07-18), now rdotk-free.
        itp_epmat_term = (16 * ndata_epmat + 16 * nr_epmat + 24) * nk_batch_max
        # k+q-convention commitments: xk_dev + xkq_dev and P_k. The 1:nkq index vector is gone —
        # the payload's `ikqs` is the tile's `UnitRange`, passed in the kernel launch parameters.
        convention_term = 24 * (nk + nkq) + 16 * nr_ep * nk_batch_max
        for nk_b in (1, 4)
            per_point, committed = _outer_k_staging_bytes(; nw, nbandk_max, nmodes, nr_ep, nk, nkq,
                nq_grid, nk_batch_max, calculators = calcs, ndata_epmat, nr_epmat, nk_b, FT)
            @test per_point == exp_per_q(nk_b)
            @test committed == old_committed + itp_epmat_term + convention_term
        end
        # `nk_b = 1` (the default) must reproduce the pre-strip formula exactly.
        @test exp_per_q(1) == 56 * nw * nbandk_max * nmodes + 16 * nr_ep + 16 * nmodes^2 +
            8 * nmodes + 8 +
            sum(ElectronPhonon.eph_batched_bytes_per_point(c, ElectronPhonon.EPDataQBatched; nw, nmodes) for c in calcs)

        # The strip width itself: shape-derived, auto-off at large ndata and above the fused-rotation
        # threshold, and clamped by the outer-k batch width.
        @test ElectronPhonon._krkq_strip_width(; nw = 7, nbandk_max = 1, nmodes = 3,
                                               nk_batch_max = 256) == 7   # cld(128, 21)
        @test ElectronPhonon._krkq_strip_width(; nw = 7, nbandk_max = 1, nmodes = 3,
                                               nk_batch_max = 4) == 4     # clamped by the k-batch
        @test ElectronPhonon._krkq_strip_width(; nw = 32, nbandk_max = 4, nmodes = 24,
                                               nk_batch_max = 256) == 1   # ndata = 3072 > M_TARGET
        @test ElectronPhonon._krkq_strip_width(; nw = 12, nbandk_max = 1, nmodes = 3,
                                               nk_batch_max = 256) == 1   # nw·nmodes = 36 > 24
    end

    @testset "plan_batch memory-bound warning is opt-out" begin
        # A batch narrowed by free memory tells the user; a counterfactual query ("how wide would
        # the batch be at a different `per_point`?") must stay quiet.
        per_point, committed, cap = 1000, 10^6, 500
        b = _StubBackend(committed + 20 * per_point)          # memory-bound: 14 of the 500 asked
        @test plan_batch(b, per_point, committed, cap; warn = false) == 14
        @test_logs plan_batch(b, per_point, committed, cap; warn = false)
        @test_logs (:warn,) plan_batch(b, per_point, committed, cap)
    end

    @testset "outer-q parity" begin
        nw, nmodes, nr_el_ham, nr_ep_eRpq = 4, 21, 250, 419
        for use_polar_eph in (false, true)
            per_point, committed = _outer_q_staging_bytes(; nw, nmodes, nr_el_ham, nr_ep_eRpq,
                use_polar_eph, calculators = calcs, FT)
            # `16·nr` (interpolator core phase); the `8·nr` rdotk scratch is gone with the fused
            # `fourier_phase!` broadcast.
            old_per_k = 16 * nw^2 * (5 * nmodes + 8 + (use_polar_eph ? 1 : 0)) +
                16 * (nr_el_ham + nr_ep_eRpq) +
                sum(ElectronPhonon.eph_batched_bytes_per_point(c, ElectronPhonon.EPDataKBatched; nw, nmodes) for c in calcs)
            @test per_point == old_per_k
            @test committed == 0   # k side streamed: no whole-run device stack
        end
    end
end

@testset "plan_batch memory-adaptive sizing + fail-early (Stage 5)" begin
    # CPU backend: free is unbounded ⇒ batch = cap.
    @test plan_batch(CPUBackend(), 1600, 1600, 42; what = "cpu") == 42

    # Stub backend, tight memory: (free - committed) ÷ 10 * 7 ÷ per_point clamps the width.
    per_point, committed, free = 1600, 1600, 1_000_000
    expect = min(1000, max(1, ((free - committed) ÷ 10 * 7) ÷ per_point))
    @test plan_batch(_StubBackend(free), per_point, committed, 1000; what = "stub") == expect
    @test expect < 1000                                          # actually clamped by memory

    # Committed alone exceeds free ⇒ fail early (clear error, not an OOM mid-loop).
    @test_throws ErrorException plan_batch(_StubBackend(1000), 1600, 16000, 100; what = "stub-oom")
end
