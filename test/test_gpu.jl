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

    # list-batched RR→kR over all k — check every column
    ep_all = arr_dev(zeros(ComplexF64, nwe*nband*nmodes, nr_ep, nk2))
    get_eph_RR_to_kR_batched!(ep_all, get_interpolator(epmat_d; fourier_mode="batched", batch_size=nk2), ks, arr_dev(uks))
    ep_all_h = Array(ep_all)
    for ik in 1:nk2
        @test isapprox(ep_all_h[:, :, ik], refs_RR[ik]; rtol)
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

# Plumbing of the outer-q batched calculator hook (allow_eph_batched_q / run_calculator_batched_q!
# and the per-q lifecycle hooks) used by `run_eph_outer_q(...; use_gpu=true)`. Backend-agnostic, so
# no GPU is needed — it only checks the opt-in defaults and the not-implemented error paths.
struct _PlainQCalc <: ElectronPhonon.AbstractCalculator end
mutable struct _OptInQCalc <: ElectronPhonon.AbstractCalculator; setup::Int; flush::Int; end
ElectronPhonon.allow_eph_batched_q(::_OptInQCalc) = true
ElectronPhonon.setup_calculator_batched_q!(c::_OptInQCalc; kwargs...) = (c.setup += 1; nothing)
ElectronPhonon.flush_calculator_batched_q!(c::_OptInQCalc; kwargs...) = (c.flush += 1; nothing)

@testset "outer-q batched calculator hook plumbing" begin
    # Default opts out; the lifecycle hooks are no-ops; the main hook errors if unimplemented.
    @test ElectronPhonon.allow_eph_batched_q(_PlainQCalc()) == false
    @test ElectronPhonon.setup_calculator_batched_q!(_PlainQCalc(); iq=1, proto=zeros(1)) === nothing
    @test ElectronPhonon.flush_calculator_batched_q!(_PlainQCalc(); iq=1) === nothing
    @test_throws Exception ElectronPhonon.run_calculator_batched_q!(
        _PlainQCalc(), nothing, nothing, nothing, nothing, nothing, nothing, nothing, 1)

    # Opt-in calculator: hook returns true and the lifecycle hooks dispatch to the override.
    c = _OptInQCalc(0, 0)
    @test ElectronPhonon.allow_eph_batched_q(c) == true
    ElectronPhonon.setup_calculator_batched_q!(c; iq=1, proto=zeros(1), k_chunk_size=4)
    ElectronPhonon.flush_calculator_batched_q!(c; iq=1)
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
        check_eph_batched(to_device, CuArray; rtol=1e-9)
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
    obj  = to_device(WannierObject(irvec_ep, rand(ComplexF64, nw*nband*nmodes, nr_ep)))
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


# A minimal AbstractCalculator that records the mode-resolved g2 and phonon frequency for
# every (ik, ikq). It mirrors what MigdalEliashberg's EliashbergCalculator reads
# (epdata.g2[m,n,imode], epdata.ph.e[imode]) but has no external dependency, so it can verify
# the GPU calculator loop (run_eph_over_k_and_kq use_gpu) against the CPU loop here.
mutable struct _RecordCalc <: ElectronPhonon.AbstractCalculator
    g2::Array{Float64,5}    # (nw, nw, nmodes, nk, nkq)
    ωq::Array{Float64,5}
    _RecordCalc() = new(zeros(0, 0, 0, 0, 0), zeros(0, 0, 0, 0, 0))
end
ElectronPhonon.allow_eph_outer_k(::_RecordCalc) = true
function ElectronPhonon.setup_calculator!(c::_RecordCalc, kpts, qpts, el_states;
        el_states_kq, kqpts, nw, nmodes, kwargs...)
    c.g2 = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c.ωq = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c
end
ElectronPhonon.setup_calculator_inner!(c::_RecordCalc; kwargs...) = c
ElectronPhonon.postprocess_calculator_inner!(c::_RecordCalc; kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_RecordCalc; kwargs...) = c
function ElectronPhonon.run_calculator!(c::_RecordCalc, epdata, ik, iq, ikq; kwargs...)
    (; el_k, el_kq, ph) = epdata
    for imode in 1:ph.nmodes, n in el_k.rng, m in el_kq.rng
        c.g2[m, n, imode, ik, ikq] = epdata.g2[m, n, imode]
        c.ωq[m, n, imode, ik, ikq] = ph.e[imode]
    end
    c
end

# Device-native counterpart of `_RecordCalc`: opts into the batched hook so the GPU loop keeps
# the e-ph matrix on the device and calls `run_calculator_batched!` once per (k, batch). It
# records the same `g2 = |ep|²/2ω` and ωq as `_RecordCalc`, exercising the loop's device-native
# branch (allow_eph_batched, ωq/ikqs staging, on-device g2) without MigdalEliashberg.
mutable struct _RecordCalcBatched <: ElectronPhonon.AbstractCalculator
    g2::Array{Float64,5}
    ωq::Array{Float64,5}
    _RecordCalcBatched() = new(zeros(0, 0, 0, 0, 0), zeros(0, 0, 0, 0, 0))
end
ElectronPhonon.allow_eph_outer_k(::_RecordCalcBatched) = true
ElectronPhonon.allow_eph_batched(::_RecordCalcBatched) = true
function ElectronPhonon.setup_calculator!(c::_RecordCalcBatched, kpts, qpts, el_states;
        el_states_kq, kqpts, nw, nmodes, kwargs...)
    c.g2 = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c.ωq = zeros(nw, nw, nmodes, kpts.n, kqpts.n)
    c
end
ElectronPhonon.setup_calculator_inner!(c::_RecordCalcBatched; kwargs...) = c
ElectronPhonon.postprocess_calculator_inner!(c::_RecordCalcBatched; kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_RecordCalcBatched; kwargs...) = c
# Computes g2 from `ep_kq` itself (independent check of the fused-kernel ep_kq output). The GPU
# loop now also folds g2 into the kRkq kernel and passes it as `g2=`; when present, assert it
# matches the abs2/(2ω) recomputation — this guards the production g2 output in-package.
function ElectronPhonon.run_calculator_batched!(c::_RecordCalcBatched, ep_kq, ωq, ik, ikqs;
        g2=nothing, ibandk_offset=0)
    nbandkq, nbandk, nm, nqc = size(ep_kq)
    g2dev = abs2.(ep_kq) ./ (2 .* reshape(ωq, 1, 1, nm, nqc))
    g2 === nothing || @assert maximum(abs, Array(g2 .- g2dev)) <= 1e-10 * maximum(abs, Array(g2dev))
    g2h = Array(g2dev)   # device → host (m,n,ν,j)
    ωh = Array(ωq)
    ikqsh = Array(ikqs)
    for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        # ibandk_offset: the loop's k-side window projection offset (ep_kq band n ↔ physical band ibandk_offset+n).
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

        # Outer-k batching (nk_batch_max=5 forces a partial final k-batch) must agree too,
        # together with a partial q-batch.
        cbk = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cbk], symmetry=nothing, use_gpu=true,
            nk_batch_max=5, nq_batch_max=7, progress_print_step=10^9)
        @test maximum(abs, cg.g2 .- cbk.g2) < 1e-9 * scale

        # Fully-GPU policy: a non-batched calculator (no allow_eph_batched) must be rejected, not
        # silently run on the host path.
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
            rk, rbmin, rbmax, rnel = filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window;
                symmetry = model.symmetry, use_gpu = false)
            gk, gbmin, gbmax, gnel = filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window;
                symmetry = model.symmetry, use_gpu = true)
            @test gk.n == rk.n
            @test gk.ngrid == rk.ngrid
            @test gk.vectors == rk.vectors
            @test gk.weights == rk.weights
            @test (gbmin, gbmax) == (rbmin, rbmax)
            @test gnel == rnel
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
        kpts_ibz = filter_kpoints((24, 24, 24), model.nw, model.el_ham, window;
            symmetry = model.symmetry, use_gpu = false)[1]
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
        itp_v = get_interpolator(ElectronPhonon.to_device(model.el_ham_R); fourier_mode="batched")
        v_dev = ElectronPhonon.get_el_velocity_direct_batched(itp_v, kpts.vectors, CuArray(ufc))
        itp_rbar = get_interpolator(ElectronPhonon.to_device(model.el_pos); fourier_mode="batched")
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
