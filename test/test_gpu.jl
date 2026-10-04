using Test
using ElectronPhonon
using ElectronPhonon: WannierObject, Vec3, get_eph_RR_to_kR!, get_eph_kR_to_kq!, get_eph_Rq_to_kq!, to_device
# Batched drivers / primitives are internal (unexported); import the ones the tests use.
using ElectronPhonon: eigvals_batched, eigen_batched, get_el_eigen_batched, get_el_eigen_valueonly_batched,
    get_el_velocity_direct_batched, get_eph_kR_to_kq_batched!, eph_rotate_kR_batched!,
    eph_apply_rotations!, eph_apply_rotations_rqkq!, batched_gemm!, get_fourier_batched!
using LinearAlgebra

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# `get_eph_kR_to_kq_batched!` takes the Fourier phase, not a q-list, because the driver builds one
# phase and reuses it over many k. The tests mostly have a q-list instead, so this wrapper does the
# phase build for them the way the production driver does — an explicit `irvec_mat` and a
# caller-owned destination, not the interpolator's internal scratch. It lives here rather than in
# `src` because no `src` caller needs it.
# Requires an in-memory parent (it reads `parent.op_r`), so no `DiskWannierObject`.
function kR_to_kq_from_qs!(ep_kq_all, backend, itp_ep_ekpR, qs, u_phs, ukqs; ws = (;),
                           g2_out = nothing, ωq = nothing)
    parent = itp_ep_ekpR.parent
    irvec_mat = ElectronPhonon._irvec_to_device_matrix(backend, parent.irvec, Float64)
    xkmat = ElectronPhonon.to_device(backend, [q[d] for d in 1:3, q in qs])
    phase = ElectronPhonon.build_fourier_phase!(
        ElectronPhonon.alloc(backend, ComplexF64, length(parent.irvec), length(qs)),
        irvec_mat, xkmat)
    @views get_eph_kR_to_kq_batched!(ep_kq_all, parent.op_r[1:parent.ndata, :], phase, u_phs, ukqs;
                                     ws..., g2_out, ωq)
end

# The RR→kR and Rq→kq steps as the engines run them: one batched Fourier transform over R_el at a
# k-list, then the rotation kernel. Like `kR_to_kq_from_qs!`, they allocate their own scratch.
function RR_to_kR_from_ks!(ep_ekpR_all, itp_epmat, ks, uks; additional_phase = nothing)
    g = similar(itp_epmat.parent.op_r, ComplexF64, itp_epmat.parent.ndata, length(ks))
    get_fourier_batched!(g, itp_epmat, ks)
    eph_rotate_kR_batched!(ep_ekpR_all, g, uks; additional_phase)
end
function Rq_to_kq_from_ks!(ep_kq_all, itp_epobj_eRpq, ks, uks, ukqs)
    nbandkq, nbandk, nmodes, nk = size(ep_kq_all)
    nw = size(uks, 1)
    op_r = itp_epobj_eRpq.parent.op_r
    g = similar(op_r, ComplexF64, nw^2 * nmodes, nk)
    get_fourier_batched!(g, itp_epobj_eRpq, ks)
    eph_apply_rotations_rqkq!(ep_kq_all, g, uks, ukqs, similar(op_r, ComplexF64, nbandkq, nw * nmodes, nk),
                              similar(op_r, ComplexF64, nw, nbandk, nmodes * nk))
end

# A mis-dispatch (a device view falling back to the generic scalar `batched_gemm!` instead of the
# GPU method) must fail loudly rather than silently limp — forbid scalar indexing on the device.
GPU_AVAILABLE && CUDA.allowscalar(false)

# Pb EPW model comes from a downloaded test artifact (see test/Artifacts.toml).
isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

"""
Validate the batched e-ph drivers against the per-k/q reference (`get_eph_RR_to_kR!` /
`get_eph_kR_to_kq!`) on `backend`. Every batch element is checked. Full-band only (no energy
window).
"""
function check_eph_batched(backend; rtol)
    to_dev(x) = ElectronPhonon.to_device(backend, x)
    arr_dev = to_dev
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

    # list-batched RR→kR over all k — check every column
    ep_all = arr_dev(zeros(ComplexF64, nwe*nband*nmodes, nr_ep, nk2))
    RR_to_kR_from_ks!(ep_all, get_interpolator(epmat_d; fourier_mode="batched", backend, batch_size=nk2), ks, arr_dev(uks))
    ep_all_h = Array(ep_all)
    for ik in 1:nk2
        @test isapprox(ep_all_h[:, :, ik], refs_RR[ik]; rtol)
    end

    # list-batched kR→kq over all q (fixed k = ks[1]) — check every slice
    obj_k1 = to_dev(WannierObject(irvec_ep, copy(refs_RR[1])))
    ep_kq_all = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq2))
    kR_to_kq_from_qs!(ep_kq_all, backend, get_interpolator(obj_k1; fourier_mode="batched", backend, batch_size=nq2), qs, arr_dev(uphs), arr_dev(ukqs))
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
    Rq_to_kq_from_ks!(ep_rqkq, get_interpolator(eRpq_d; fourier_mode="batched", backend, batch_size=nk2),
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
    kR_to_kq_from_qs!(ep_g2, backend, get_interpolator(obj_k1; fourier_mode="batched", backend, batch_size=nq2),
        qs, arr_dev(uphs), arr_dev(ukqs); g2_out, ωq=arr_dev(ωq_g2))
    g2_ref = abs2.(ep_ref) ./ (2 .* reshape(ωq_g2, 1, 1, nmodes, nq2))
    @test isapprox(Array(g2_out), g2_ref; rtol)
end

@testset "batched e-ph drivers (CPU)" begin
    check_eph_batched(ElectronPhonon.CPUBackend(); rtol=1e-10)
end

"""
`build_fourier_phase!` against a scalar `cispi(2 R·x)` reference, on `backend`.
"""
function check_fourier_phase(backend)
    arr_dev(x) = ElectronPhonon.to_device(backend, x)
    nr, nk = 37, 11
    irvec = [Vec3(rand(-3:3, 3)...) for _ in 1:nr]
    xks   = [Vec3(rand(3)...) for _ in 1:nk]
    ref = [cispi(2 * (irvec[ir][1] * xks[ik][1] + irvec[ir][2] * xks[ik][2] +
                      irvec[ir][3] * xks[ik][3])) for ir in 1:nr, ik in 1:nk]

    irvec_mat = [Float64(irvec[ir][d]) for ir in 1:nr, d in 1:3]
    xkmat = [xks[ik][d] for d in 1:3, ik in 1:nk]
    phase = arr_dev(zeros(ComplexF64, nr, nk))
    ElectronPhonon.build_fourier_phase!(phase, arr_dev(irvec_mat), arr_dev(xkmat))
    # |phase| == 1, so an absolute bound is a relative bound. The device compiler may contract
    # `R1*x1 + R2*x2 + R3*x3` into FMAs, which rounds differently from the host. For |R_d| ≤ 3 and
    # x_d ∈ [0, 1) the two sums differ by at most 3.11e-15 (the host rounds two products and two
    # partial sums, the FMA chain two partial sums; `R1*x1` is common), so the phases differ by at
    # most 2π·3.11e-15 = 1.95e-14. 2e-14 is that worst case, not a margin over a typical error.
    @test maximum(abs, Array(phase) .- ref) <= 2e-14
end

@testset "build_fourier_phase! (CPU)" begin
    check_fourier_phase(ElectronPhonon.CPUBackend())
end

"""
`get_eph_kR_to_kq_batched!` and the k+q convention of `eph_rotate_kR_batched!`, on `backend`
(as in [`check_eph_batched`](@ref)).

1. With the same `build_fourier_phase!(qs)` phase, the interpolator path (`kR_to_kq_from_qs!`, whose
   operand is the row-range view `op_r[1:ndata, :]`) and a contiguous operand agree bit for bit. On
   the device this pins that the CUDA extension keeps the view on cuBLAS.
2. Storing the kR intermediate in the k+q convention (`conj(exp(2πi R_p·x_k))`) and transforming at
   `x_{k+q}` reproduces the q-convention result, for a (k, k+q) pair whose `q = x_{k+q} - x_k` needs
   a mod-G reduction. This is the in-repo pin of the identity `exp(2πi R_p·q) =
   exp(2πi R_p·x_{k+q}) · conj(exp(2πi R_p·x_k))` for integer `R_p`.
"""
function check_eph_kq_convention(backend; rtol)
    to_dev(x) = ElectronPhonon.to_device(backend, x)
    arr_dev = to_dev
    nwe, nmodes, nr_el, nr_ep = 3, 4, 12, 15
    nband, nq = nwe, 9
    irvec_el = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_el], by = x -> reverse(x))
    irvec_ep = sort([Vec3(rand(-2:2, 3)...) for _ in 1:nr_ep], by = x -> reverse(x))
    epmat_obj = to_dev(WannierObject(irvec_el, rand(ComplexF64, nwe^2*nmodes*nr_ep, nr_el);
                                     irvec_next = irvec_ep))
    itp_epmat = get_interpolator(epmat_obj; fourier_mode="batched", backend, batch_size=1)

    # x_k and x_{k+q} on a 20³ grid; q = (x_{k+q} - x_k) mod G is what the loop feeds today.
    xk   = Vec3(0.75, -0.40, 0.35)
    xkqs = [Vec3(rand(-10:10, 3) ./ 20...) for _ in 1:nq]
    qs   = [Vec3(mod.(xkq .- xk, 1)...) for xkq in xkqs]
    @test any(i -> qs[i] != xkqs[i] - xk, 1:nq)   # at least one pair needs the mod-G reduction

    uk   = arr_dev(rand(ComplexF64, nwe, nband, 1))
    uphs = arr_dev(cat([rand(ComplexF64, nmodes, nmodes) for _ in 1:nq]...; dims=3))
    ukqs = arr_dev(cat([rand(ComplexF64, nwe, nband) for _ in 1:nq]...; dims=3))

    irvecp_mat = ElectronPhonon._irvec_to_device_matrix(backend, irvec_ep, Float64)
    ndata = nwe * nband * nmodes

    # (a) reference: q convention, interpolator + qs method
    ep_kR_q = arr_dev(zeros(ComplexF64, ndata, nr_ep, 1))
    RR_to_kR_from_ks!(ep_kR_q, itp_epmat, [xk], uk)
    obj_q = to_dev(WannierObject(irvec_ep, Array(ep_kR_q)[:, :, 1]))
    ref = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq))
    kR_to_kq_from_qs!(ref, backend, get_interpolator(obj_q; fourier_mode="batched", backend, batch_size=nq),
                              qs, uphs, ukqs)

    # (b) same phase, contiguous operand: must agree bit for bit
    phase_q = arr_dev(zeros(ComplexF64, nr_ep, nq))
    ElectronPhonon.build_fourier_phase!(phase_q, irvecp_mat,
                                  arr_dev([q[d] for d in 1:3, q in qs]))
    out_b = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq))
    get_eph_kR_to_kq_batched!(out_b, view(ep_kR_q, :, :, 1), phase_q, uphs, ukqs)
    @test Array(out_b) == Array(ref)

    # (c) k+q convention: fold conj(exp(2πi R_p·x_k)) into the child, transform at x_{k+q}.
    # `eph_rotate_kR_batched!` multiplies by whatever it is handed, so conjugate here.
    P_mk = arr_dev(zeros(ComplexF64, nr_ep, 1))
    ElectronPhonon.build_fourier_phase!(P_mk, irvecp_mat, arr_dev(reshape([xk[d] for d in 1:3], 3, 1)))
    P_mk .= conj.(P_mk)
    ep_kR_kq = arr_dev(zeros(ComplexF64, ndata, nr_ep, 1))
    RR_to_kR_from_ks!(ep_kR_kq, itp_epmat, [xk], uk; additional_phase = P_mk)
    P_kq = arr_dev(zeros(ComplexF64, nr_ep, nq))
    ElectronPhonon.build_fourier_phase!(P_kq, irvecp_mat,
                                  arr_dev([xkq[d] for d in 1:3, xkq in xkqs]))
    out_c = arr_dev(zeros(ComplexF64, nband, nband, nmodes, nq))
    get_eph_kR_to_kq_batched!(out_c, view(ep_kR_kq, :, :, 1), P_kq, uphs, ukqs)
    @test isapprox(Array(out_c), Array(ref); rtol)
end

@testset "k+q convention identity (CPU)" begin
    check_eph_kq_convention(ElectronPhonon.CPUBackend(); rtol=1e-13)
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

@testset "GPU batch_size budget" begin
    if GPU_AVAILABLE
        using ElectronPhonon: _default_batch_size, GPU_FOURIER_BATCH_BYTES
        gpu = ElectronPhonon.gpu_backend()
        # `16·(nr + ndata)` bytes per column, capped by `nk_hint`.
        for (nr, ndata) in ((617, 9), (617, 27), (2000, 49))
            @test _default_batch_size(gpu, nr, ndata) == fld(GPU_FOURIER_BATCH_BYTES, 16 * (nr + ndata))
            @test _default_batch_size(gpu, nr, ndata; nk_hint = 100) == 100
        end
        # A device backend has no per-thread channel: `get_interpolator_channel` would put
        # `nbuffers` budget-sized scratch buffers on the card, so it refuses one.
        err = try get_interpolator_channel(WannierObject([Vec3{Int}([0, 0, 0])],
                      randn(ComplexF64, 4, 1)); fourier_mode = "batched", backend = gpu)
        catch e; e end
        @test err isa ArgumentError
        @test occursin("host-only", err.msg)
        # A grid far below the budget gets a block no wider than the grid, not a budget-sized one.
        obj = ElectronPhonon.to_device(gpu, WannierObject(
            [Vec3{Int}([0, 0, 0]), Vec3{Int}([1, 0, 0])], randn(ComplexF64, 6, 2)))
        @test get_interpolator(obj; fourier_mode = "batched", backend = gpu,
                               nk_hint = 12).batch_size == 12
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
        # `nk_hint` everywhere: without it the GPU default spends the whole byte budget, which is
        # ~800 MB of scratch for these 50 k-points.
        gpu = ElectronPhonon.gpu_backend()
        itp_gpu() = get_interpolator(obj_gpu; fourier_mode="batched", backend = gpu,
                                     nk_hint = length(kpts))
        Hk_cpu = zeros(ComplexF64, obj.ndata, length(kpts))
        get_fourier_batched!(Hk_cpu, get_interpolator(obj; fourier_mode="batched"), kpts)
        Hk_gpu = similar(obj_gpu.op_r, ComplexF64, obj.ndata, length(kpts))
        get_fourier_batched!(Hk_gpu, itp_gpu(), kpts)
        @test Array(Hk_gpu) ≈ Hk_cpu

        # The block loop, on the device. Every grid in this file fits one block at the budgeted
        # default, so without an explicitly narrow `batch_size` nothing here would exercise the
        # `while start <= nk` path on the GPU at all.
        Hk_blk = similar(Hk_gpu)
        get_fourier_batched!(Hk_blk, get_interpolator(obj_gpu; fourier_mode="batched",
                                                      backend = gpu, batch_size = 7), kpts)
        @test length(kpts) > 7                      # the loop really runs more than once
        @test Array(Hk_blk) ≈ Array(Hk_gpu)

        # --- eigenvalues only: GPU vs CPU reference ---
        E_ref = get_el_eigen_valueonly_batched(get_interpolator(obj; fourier_mode="batched"), kpts)
        E_gpu = get_el_eigen_valueonly_batched(itp_gpu(), kpts)
        @test E_gpu isa CuArray
        @test sort(Array(E_gpu), dims=1) ≈ sort(E_ref, dims=1)

        # --- eigenvalues + eigenvectors (CPU counterpart in "batched eigensolve (CPU)") ---
        Ev_gpu, U_gpu = get_el_eigen_batched(itp_gpu(), kpts)
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
        check_eph_batched(ElectronPhonon.gpu_backend(); rtol=1e-9)

        # phase kernel and the k+q convention on the device
        check_fourier_phase(ElectronPhonon.gpu_backend())
        check_eph_kq_convention(ElectronPhonon.gpu_backend(); rtol=1e-13)
    end
end

# Partial final q-batch: the GPU loop runs a batch narrower than the preallocated `n_inner_tile`
# by passing contiguous device VIEWS (`view(buf, :,:,:, 1:nq_batch)`) into
# `get_eph_kR_to_kq_batched!`, its scratch `g` / `tmp` as views of the max-width buffers. This checks that path directly:
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
    ws   = (; g = CuArray{ComplexF64}(undef, nw*nband*nmodes, nq),
              tmp = CuArray{ComplexF64}(undef, nband, nband*nmodes, nq))

    full = CuArray(zeros(ComplexF64, nband, nband, nmodes, nq))
    kR_to_kq_from_qs!(full, ElectronPhonon.gpu_backend(),
        get_interpolator(obj; fourier_mode="batched", backend = ElectronPhonon.gpu_backend(), batch_size=nq),
        qs, uphs, ukqs; ws)
    # Same call restricted to the first m q-points via views into the max-width buffers and ws.
    part = CuArray(zeros(ComplexF64, nband, nband, nmodes, nq))
    kR_to_kq_from_qs!(view(part, :, :, :, 1:m), ElectronPhonon.gpu_backend(),
        get_interpolator(obj; fourier_mode="batched", backend = ElectronPhonon.gpu_backend(), batch_size=nq),
        view(qs, 1:m), view(uphs, :, :, 1:m), view(ukqs, :, :, 1:m);
        ws = (; g = view(ws.g, :, 1:m), tmp = view(ws.tmp, :, :, 1:m)))
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


@testset "eph_apply_rotations! rejects a non-dense g" begin
    # The two-GEMM rotation paths merge `g`'s band and mode axes with a `reshape`, so a strided `g`
    # is only readable by the CUDA fused kernel. The generic method must say so rather than silently
    # reshaping into a `ReshapedArray` the batched GEMMs cannot take a pointer to.
    let nw = 3, nband = 3, nmodes = 4, nq = 5
        g_strided = view(reshape(rand(ComplexF64, nw*nband*nmodes*2, nq), nw, nband, nmodes, 2, nq),
                         :, :, :, 1, :)
        @test !ElectronPhonon._is_dense(g_strided)
        @test_throws AssertionError eph_apply_rotations!(
            zeros(ComplexF64, nband, nband, nmodes, nq), g_strided,
            rand(ComplexF64, nw, nband, nq), rand(ComplexF64, nmodes, nmodes, nq),
            zeros(ComplexF64, nband, nband * nmodes, nq))
    end
end

@testset "kR→kq workspaces must have the block's exact extent" begin
    # A workspace array at a wider batch than the block (the full buffer rather than a view of its
    # leading columns) fails the size assertion instead of being used past the block.
    let nw = 3, nband = 3, nmodes = 2, nr = 4, nq = 5
        ndata = nw * nband * nmodes
        ep = zeros(ComplexF64, nband, nband, nmodes, nq)
        ep_kR, phase = rand(ComplexF64, ndata, nr), rand(ComplexF64, nr, nq)
        uphs, ukqs = rand(ComplexF64, nmodes, nmodes, nq), rand(ComplexF64, nw, nband, nq)
        @test_throws AssertionError get_eph_kR_to_kq_batched!(ep, ep_kR, phase, uphs, ukqs;
            g = rand(ComplexF64, ndata, nq + 2), tmp = rand(ComplexF64, nband, nband * nmodes, nq))
    end
end


# A minimal AbstractCalculator that records the mode-resolved g2 = |ep|²/2ω and phonon frequency for
# every (ik, ikq) at physical bands, from the blocks of `run_eph_over_k_and_kq`. It mirrors what
# MigdalEliashberg's G2Calculator reads but has no external dependency.
mutable struct _RecordCalc <: ElectronPhonon.AbstractCalculator
    g2::Array{Float64,5}    # (nw, nw, nmodes, nk, nkq)
    ωq::Array{Float64,5}
    _RecordCalc() = new(zeros(0, 0, 0, 0, 0), zeros(0, 0, 0, 0, 0))
end
ElectronPhonon.supports(::_RecordCalc, ::Type{ElectronPhonon.OuterKLoop}) = true
# The loop always provides `e`, `u` and the e-ph matrix elements, which is all this calculator
# reads, so it defines no `required_el_quantities` / `required_ph_quantities`.
ElectronPhonon.calculator_begin!(::_RecordCalc, ctx) = nothing
ElectronPhonon.calculator_end!(::_RecordCalc, ctx) = nothing
function ElectronPhonon.setup_calculator!(c::_RecordCalc, backend, els_k, els_kq, phs; nw, nmodes,
        kwargs...)
    c.g2 = zeros(nw, nw, nmodes, els_k.nk, els_kq.nk)
    c.ωq = zeros(nw, nw, nmodes, els_k.nk, els_kq.nk)
    c
end
ElectronPhonon.postprocess_calculator!(c::_RecordCalc; kwargs...) = c
function ElectronPhonon.run_calculator!(c::_RecordCalc, p::ElectronPhonon.EPBlock{ElectronPhonon.OuterKLoop}, ctx)
    (; ep, phs, ik, ikq) = p
    g2h = Array(abs2.(ep) ./ (2 .* reshape(phs.e, 1, 1, size(ep, 3), size(ep, 4))))
    ωh = Array(phs.e)
    offk, nbk = Array(p.els_k.iband_offset)[1], Array(p.els_k.nband)[1]
    offkq, nbkq = Array(p.els_kq.iband_offset), Array(p.els_kq.nband)
    for (j, ikq_j) in enumerate(ikq), ν in axes(ep, 3), n in 1:nbk, m in 1:nbkq[j]
        c.g2[offkq[j] + m, offk + n, ν, ik, ikq_j] = g2h[m, n, ν, j]
        c.ωq[offkq[j] + m, offk + n, ν, ik, ikq_j] = ωh[ν, j]
    end
    c
end

struct _QOnlyCalc <: ElectronPhonon.AbstractCalculator end
ElectronPhonon.supports(::_QOnlyCalc, ::Type{ElectronPhonon.OuterQLoop}) = true

@testset "batched calculator loop (run_eph_over_k_and_kq)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU calculator-loop test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        grid = (4, 4, 4)

        # NOTE on CPU-vs-GPU comparison: on a GPU backend the SETUP eigensolve also runs on the
        # device (batched), which does NOT apply the EPW degeneracy gauge-fixing of the per-k CPU
        # path. So for degenerate bands/modes the GPU e-ph matrix / g2 differs from CPU by a gauge
        # (a unitary rotation within each degenerate subspace) — physically equivalent, but not
        # bit-identical, and largest on COARSE grids (more exact high-symmetry degeneracies; this
        # 4³ Pb grid is such a case). Eigenvalues are gauge-independent, so we compare ωq against
        # CPU; g2 correctness is checked GPU-vs-GPU, where all paths share the same GPU gauge.
        cc = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cc], symmetry=nothing, progress_print_step=10^9)

        # GPU reference: a single q-tile.
        cg = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cg], symmetry=nothing, backend=ElectronPhonon.gpu_backend(),
            progress_print_step=10^9)

        scale = maximum(abs, cg.g2)
        # Phonon frequencies are gauge-independent → must match the CPU path (eigenvalue precision).
        @test maximum(abs, cc.ωq .- cg.ωq) < 1e-6 * maximum(abs, cc.ωq)

        # GPU bit-faithfulness (shared gauge): multiple q-tiles (n_inner_tile=7 → partial final
        # tile) must reproduce the single-tile GPU result.
        cg7 = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cg7], symmetry=nothing, backend=ElectronPhonon.gpu_backend(),
            n_inner_tile=7, progress_print_step=10^9)
        @test maximum(abs, cg.g2 .- cg7.g2) < 1e-9 * scale
        @test cg.ωq == cg7.ωq

        # Outer-k batching (n_outer_batch=5 forces a partial final k-batch) must agree too,
        # together with a partial q-tile.
        cbk = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cbk], symmetry=nothing, backend=ElectronPhonon.gpu_backend(),
            n_outer_batch=5, n_inner_tile=7, progress_print_step=10^9)
        @test maximum(abs, cg.g2 .- cbk.g2) < 1e-9 * scale

        # A degenerate outer-k batch (n_outer_batch = 1) must agree too: the k+q-convention
        # phase tile is then rebuilt per k and `ep_ekpR_all` is one k wide, i.e. the k-batch reuse
        # factor the loop is built around drops to 1.
        cb1 = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cb1], symmetry=nothing, backend=ElectronPhonon.gpu_backend(),
            n_outer_batch=1, progress_print_step=10^9)
        @test maximum(abs, cg.g2 .- cb1.g2) < 1e-9 * scale

        # A calculator that does not support the outer-k order is rejected, not silently skipped.
        @test_throws ArgumentError ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[_QOnlyCalc()], symmetry=nothing, backend=ElectronPhonon.gpu_backend(),
            progress_print_step=10^9)

        # Energy conservation is a CPU feature, refused on the GPU.
        @test_throws "CPUBackend feature" ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[_RecordCalc()], symmetry=nothing, backend=ElectronPhonon.gpu_backend(),
            energy_conservation_tol=0.1, progress_print_step=10^9)
    end
end

# The batched outer-k loop holds no device-specific code, so it runs on a `CPUBackend` with any
# tiling: the CUDA-free coverage of the block construction, the k+q-convention phase build and the
# q-tiling.
@testset "outer-k CPU: default widths == small tiles (_RecordCalc)" begin
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
    grid = (4, 4, 4)

    cpt = _RecordCalc()
    ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
        calculators=[cpt], symmetry=nothing, progress_print_step=10^9, verbosity=0)

    # n_inner_tile below nkq forces multiple q-tiles (plan_batch returns the cap verbatim on CPU).
    cba = _RecordCalc()
    ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
        calculators=[cba], symmetry=nothing, backend=ElectronPhonon.CPUBackend(),
        n_inner_tile=7, n_outer_batch=5, progress_print_step=10^9, verbosity=0)

    # Same backend and same eigensolve on both arms, so there is no gauge difference and g2 itself
    # can be compared — only the tiling separates them.
    scale = maximum(abs, cpt.g2)
    @test scale > 0
    @test maximum(abs, cpt.g2 .- cba.g2) < 1e-10 * scale
    @test cpt.ωq == cba.ωq
end

# Outer-q analogue of `_RecordCalc`: a calculator for `run_eph_over_q_and_k` that accumulates a
# per-q, GAUGE-INVARIANT scalar
#   A[iq] = Σ_{m,n,ν,k} wtk[k] · |ep[m,n,ν,k]|²
# over the in-window (m, n) of each k (the band-summed |g|² is invariant under the electron/phonon
# eigenvector gauge, so the degenerate-band gauge difference between the CPU LAPACK and GPU batched
# eigensolvers — see the note in the outer-k test above — does not matter here). One accumulator per
# q of the outer batch and thread chunk (the chunks of one q run concurrently), on the backend.
mutable struct _RecordCalcOuterQ <: ElectronPhonon.AbstractCalculator
    A::Vector{Float64}       # (nq,) final per-q gauge-invariant sum
    Adev::Any                # (n_outer_batch, nchunks) accumulator of the current batch, on the backend
    _RecordCalcOuterQ() = new(zeros(0), nothing)
end
ElectronPhonon.supports(::_RecordCalcOuterQ, ::Type{ElectronPhonon.OuterQLoop}) = true
ElectronPhonon.allowed_eph_phonon_basis(::_RecordCalcOuterQ) = [:eigenmode]
function ElectronPhonon.setup_calculator!(c::_RecordCalcOuterQ, backend, els_k, els_kq, phs;
        n_outer_batch, nchunks_threads, kwargs...)
    c.A = zeros(phs.nq)
    c.Adev = ElectronPhonon.alloc(backend, Float64, n_outer_batch, nchunks_threads)
    c
end
ElectronPhonon.calculator_begin!(c::_RecordCalcOuterQ, ctx) = (fill!(c.Adev, 0.0); c)
function ElectronPhonon.calculator_end!(c::_RecordCalcOuterQ, ctx)
    c.A[ctx.batch] .= vec(sum(Array(c.Adev); dims = 2))[1:length(ctx.batch)]
    c
end
ElectronPhonon.postprocess_calculator!(c::_RecordCalcOuterQ; kwargs...) = c
# `sum` of a device broadcast expression reduces on-device and returns a host scalar without any
# scalar indexing (allowed under CUDA.allowscalar(false)). Entries past a window are padding, so the
# reduction selects with `ifelse`.
function ElectronPhonon.run_calculator!(c::_RecordCalcOuterQ, p::ElectronPhonon.EPBlock{ElectronPhonon.OuterQLoop}, ctx)
    (; ep, wtk) = p
    nbkq, nbk, _, nkc = size(ep)
    inwin = (reshape(1:nbkq, nbkq, 1, 1, 1) .<= reshape(p.els_kq.nband, 1, 1, 1, nkc)) .&
            (reshape(1:nbk, 1, nbk, 1, 1) .<= reshape(p.els_k.nband, 1, 1, 1, nkc))
    val = sum(ifelse.(inwin, abs2.(ep), 0.0) .* reshape(wtk, 1, 1, 1, nkc))
    ib = p.iq - first(ctx.batch) + 1
    view(c.Adev, ib:ib, ctx.chunk) .+= val
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
            calculators=[calc_cpu], symmetry = nothing, 
            progress_print_step=10^9)

        calc_gpu = _RecordCalcOuterQ()
        ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
            calculators=[calc_gpu], symmetry = nothing, 
            backend=ElectronPhonon.gpu_backend(), progress_print_step=10^9)

        rdiff = maximum(abs, calc_cpu.A .- calc_gpu.A) / maximum(abs, calc_cpu.A)
        @info "run_eph_over_q_and_k CPU vs GPU" cpu_A=calc_cpu.A gpu_A=calc_gpu.A rdiff
        @test isapprox(calc_cpu.A, calc_gpu.A; rtol=1e-8)
    end
end

# The outer-q loop has no CUDA-only callee either, so it runs on a `CPUBackend` with any k-batch
# width.
@testset "run_eph_over_q_and_k CPU: default width == partial k-batch (D8)" begin
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
    grid = (4, 4, 4)

    calc_pt = _RecordCalcOuterQ()
    ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
        calculators=[calc_pt], symmetry = nothing, 
        progress_print_step=10^9, verbosity=0)

    # n_inner_tile below nk forces a partial final k-batch, so the block's width-nk trim is real.
    calc_ba = _RecordCalcOuterQ()
    ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
        calculators=[calc_ba], symmetry = nothing, 
        backend=ElectronPhonon.CPUBackend(), n_inner_tile=10,
        progress_print_step=10^9, verbosity=0)

    @test maximum(abs, calc_pt.A) > 0
    rdiff = maximum(abs, calc_pt.A .- calc_ba.A) / maximum(abs, calc_pt.A)
    @info "run_eph_over_q_and_k CPU, k-batch 10 vs default (Pb 4³)" rdiff
    @test isapprox(calc_pt.A, calc_ba.A; rtol=1e-10)
end

# A PARTIAL outer-q k tile (small n_inner_tile), so the block is a real trim to the tile's width, not
# the identity trim of the single-tile test above. nk=64, n_inner_tile=10 ⇒ 7 tiles, the last of
# width 4.
@testset "run_eph_over_q_and_k partial k tile (CPU vs GPU)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping run_eph_over_q_and_k partial-batch test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
        grid = (4, 4, 4)

        calc_cpu = _RecordCalcOuterQ()
        ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
            calculators=[calc_cpu], symmetry = nothing, 
            progress_print_step=10^9)

        calc_gpu = _RecordCalcOuterQ()
        ElectronPhonon.run_eph_over_q_and_k(model, grid, grid;
            calculators=[calc_gpu], symmetry = nothing, 
            backend=ElectronPhonon.gpu_backend(), n_inner_tile=10, progress_print_step=10^9)

        rdiff = maximum(abs, calc_cpu.A .- calc_gpu.A) / maximum(abs, calc_cpu.A)
        @info "run_eph_over_q_and_k partial k-batch (n_inner_tile=10) CPU vs GPU" rdiff
        @test isapprox(calc_cpu.A, calc_gpu.A; rtol=1e-8)
    end
end

@testset "GPU filter_kpoints with symmetry (IBZ reduction × backend)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU filter_kpoints symmetry test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        # Fine-mesh Fermi level / 0.3 eV window, as in the anisotropic-ME (mp_mesh_k) pipeline.
        ef = 11.682221647 * ElectronPhonon.unit_to_aru(:eV)
        window = (ef - 0.3 * ElectronPhonon.unit_to_aru(:eV), ef + 0.3 * ElectronPhonon.unit_to_aru(:eV))
        # In filter_kpoints, `symmetry` (IBZ reduction, in kpoints_grid) and `backend` (batched
        # eigensolve for the window test) are orthogonal: the IBZ k-set is built backend-independently,
        # and the backend only changes how the band eigenvalues are computed. The window test is discrete
        # (which bands fall inside), so it is robust to the ~1e-12 eigenvalue difference between the
        # cuSOLVER and CPU eigensolvers ⇒ identical ik_keep / band range / nelec_below, hence an
        # identical IBZ Kpoints object. (Eigenvectors / gauge are not involved here; cf. the g2
        # gauge caveat in the calculator-loop test above.)
        for nk in (12, 24)
            rsel = filter_electron_states((nk, nk, nk), model.nw, model.el_ham, window;
                symmetry = model.symmetry, backend = ElectronPhonon.CPUBackend())
            gsel = filter_electron_states((nk, nk, nk), model.nw, model.el_ham, window;
                symmetry = model.symmetry, backend = ElectronPhonon.gpu_backend())
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
            symmetry = model.symmetry).kpts
        qv = ["eigenvalue", "eigenvector", "velocity", "position"]
        els_c = ElectronPhonon.compute_electron_states(model, kpts_ibz, qv, window; fourier_mode="gridopt")
        els_g = ElectronPhonon.compute_electron_states(model, kpts_ibz, qv, window;
            backend=ElectronPhonon.gpu_backend())
        @test length(els_g) == length(els_c) == kpts_ibz.n
        demax = maximum(maximum(abs, els_c[ik].e_full .- els_g[ik].e_full) for ik in 1:kpts_ibz.n)
        escale = maximum(maximum(abs, els_c[ik].e_full) for ik in 1:kpts_ibz.n)
        @test demax < 1e-10 * escale
        @test all(els_c[ik].rng == els_g[ik].rng for ik in 1:kpts_ibz.n)
    end
end

@testset "compute_electron_states velocity (GPU backend)" begin
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU compute_electron_states velocity test"
    else
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum="el")
        kpts = ElectronPhonon.kpoints_grid((8, 8, 8))
        nk, nw = kpts.n, model.nw
        @assert model.el_velocity_mode === :BerryConnection  # Pb model_new

        qv = ["eigenvalue", "eigenvector", "velocity", "position"]
        els_c = ElectronPhonon.compute_electron_states(model, kpts, qv, (-Inf, Inf); fourier_mode="gridopt")
        els_g = ElectronPhonon.compute_electron_states(model, kpts, qv, (-Inf, Inf);
            backend=ElectronPhonon.gpu_backend())

        # Eigenvalues are gauge-independent → must match the CPU path to eigenvalue precision.
        emax = maximum(maximum(abs, els_c[ik].e_full .- els_g[ik].e_full) for ik in 1:nk)
        @test emax < 1e-10 * maximum(maximum(abs, els_c[ik].e_full) for ik in 1:nk)

        # Strong, gauge-independent correctness gate: run the SAME device velocity path (el_ham_R
        # rotation + Berry term im*(e_i-e_j)*rbar) on the CPU eigenvectors and compare to the CPU
        # `get_el_velocity_berry_connection!`. Sharing the eigenvectors removes the degeneracy-gauge
        # difference, so this must match to machine precision (validates rotation + Berry math).
        ufc = zeros(ComplexF64, nw, nw, nk); ec = zeros(Float64, nw, nk)
        for ik in 1:nk; ufc[:, :, ik] .= els_c[ik].u_full; ec[:, ik] .= els_c[ik].e_full; end
        itp_v = get_interpolator(ElectronPhonon.to_device(ElectronPhonon.gpu_backend(), model.el_ham_R);
            fourier_mode="batched", backend = ElectronPhonon.gpu_backend(), nk_hint = nk)
        v_dev = ElectronPhonon.get_el_velocity_direct_batched(itp_v, kpts.vectors, CuArray(ufc))
        itp_rbar = get_interpolator(ElectronPhonon.to_device(ElectronPhonon.gpu_backend(), model.el_pos);
            fourier_mode="batched", backend = ElectronPhonon.gpu_backend(), nk_hint = nk)
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
        els_gd = ElectronPhonon.compute_electron_states(model, kpts, qd, (-Inf, Inf);
            backend=ElectronPhonon.gpu_backend())
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
                ni_stride, i0)
            g2g = CUDA.zeros(FT, len); ωg = CUDA.zeros(FT, len)
            ElectronPhonon.eph_window_scatter!(g2g, ωg, CUDA.CuArray(g2vals),
                CUDA.CuArray(imap_i_col), CUDA.CuArray(imap_f), CUDA.CuArray(ikqs), CUDA.CuArray(ωq),
                ni_stride, i0)
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
            n_i, 0)
        g2g = CUDA.zeros(FT, len); ωg = CUDA.zeros(FT, len)
        ElectronPhonon.eph_window_scatter!(g2g, ωg, CUDA.CuArray(g2vals),
            CUDA.CuArray(imap_i_col), CUDA.CuArray(imap_f), rng_ikqs, CUDA.CuArray(ωq),
            n_i, 0)
        @test Array(g2g) == g2c
        @test Array(ωg) == ωc
    end
end


# --- plan_batch: memory-adaptive sizing (CPU-only) ---------------------------------------------
using ElectronPhonon: plan_batch, CPUBackend

# A stub backend with a settable free-memory budget, to drive the adaptive-width / fail-early paths
# on the CPU (no CUDA needed).
struct _StubBackend <: ElectronPhonon.AbstractBackend
    free::Int
end
ElectronPhonon.free_bytes(b::_StubBackend) = b.free

@testset "plan_batch memory-bound warning is opt-out" begin
    # A batch narrowed by free memory tells the user; a counterfactual query ("how wide would
    # the batch be at a different `per_point`?") must stay quiet.
    per_point, committed, cap = 1000, 10^6, 500
    b = _StubBackend(committed + 20 * per_point)          # memory-bound: 14 of the 500 asked
    @test plan_batch(b, per_point, committed, cap; warn = false) == 14
    @test_logs plan_batch(b, per_point, committed, cap; warn = false)
    @test_logs (:warn,) plan_batch(b, per_point, committed, cap)
end

# The estimate plans with the loop's own byte counts; on a CPU backend the tile is the cap.
@testset "estimate_device_memory" begin
    for (mom, loop) in (("el", :outer_k), ("ph", :outer_q))
        est = ElectronPhonon.estimate_device_memory(_load_model_from_artifacts("pb";
            epmat_outer_momentum = mom); nk = 64, nkq = 64, nchunks_threads = 1)
        @test est.loop == loop && est.committed > 0 && est.per_pair > 0 && est.batch == 64
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
