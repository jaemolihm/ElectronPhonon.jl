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

    # single-k / single-q drivers (element 1)
    o1 = to_dev(WannierObject(irvec_ep, zeros(ComplexF64, nwe*nband*nmodes, nr_ep)))
    get_eph_RR_to_kR_batched!(o1, get_interpolator(epmat_d; fourier_mode="batched"), ks[1], arr_dev(uks[:, :, 1]))
    @test isapprox(Array(o1.op_r), refs_RR[1]; rtol)
    ep1 = arr_dev(zeros(ComplexF64, nband, nband, nmodes))
    get_eph_kR_to_kq_batched!(ep1, get_interpolator(o1; fourier_mode="batched"), qs[1], arr_dev(uphs[:, :, 1]), arr_dev(ukqs[:, :, 1]))
    @test isapprox(Array(ep1), ep_ref[:, :, :, 1]; rtol)

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
end

@testset "batched e-ph drivers (CPU)" begin
    check_eph_batched(identity, identity; rtol=1e-10)
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

        # electron-phonon batched drivers on the GPU vs the per-k/q CPU reference
        check_eph_batched(to_device, CuArray; rtol=1e-9)
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
    @inbounds for imode in 1:ph.nmodes, n in el_k.rng, m in el_kq.rng
        c.g2[m, n, imode, ik, ikq] = epdata.g2[m, n, imode]
        c.ωq[m, n, imode, ik, ikq] = ph.e[imode]
    end
    c
end

# Device-native counterpart of `_RecordCalc`: opts into the batched hook so the GPU loop keeps
# the e-ph matrix on the device and calls `run_calculator_batched!` once per (k, chunk). It
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
function ElectronPhonon.run_calculator_batched!(c::_RecordCalcBatched, ep_kq, ωq, ik, ikqs)
    nbandkq, nbandk, nm, nqc = size(ep_kq)
    g2h = Array(abs2.(ep_kq) ./ (2 .* reshape(ωq, 1, 1, nm, nqc)))   # device → host (m,n,ν,j)
    ωh = Array(ωq)
    ikqsh = Array(ikqs)
    @inbounds for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        c.g2[m, n, ν, ik, ikqsh[j]] = g2h[m, n, ν, j]
        c.ωq[m, n, ν, ik, ikqsh[j]] = ωh[ν, j]
    end
    c
end

@testset "GPU calculator loop (run_eph_over_k_and_kq use_gpu)" begin
    PB = "/mnt/home/jlihm/ceph/superconductivity/Pb/tutorial/1_epw/"
    if !GPU_AVAILABLE
        @info "CUDA not available/functional — skipping GPU calculator-loop test"
    elseif !isdir(PB)
        @info "Pb model data not found at $PB — skipping GPU calculator-loop test"
    else
        model = ElectronPhonon.load_model_from_epw_new(PB, "temp", "pb"; epmat_outer_momentum="el")
        grid = (4, 4, 4)

        cc = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cc], symmetry=nothing, progress_print_step=10^9)

        cg = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cg], symmetry=nothing, use_gpu=true, progress_print_step=10^9)

        scale = maximum(abs, cc.g2)
        @test maximum(abs, cc.g2 .- cg.g2) < 1e-9 * scale
        @test cc.ωq == cg.ωq   # ωq comes from the same precomputed phonon states

        # Chunked q-batches (q_batch_size=7 forces a partial final chunk) must agree too.
        cg7 = _RecordCalc()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cg7], symmetry=nothing, use_gpu=true, q_batch_size=7, progress_print_step=10^9)
        @test maximum(abs, cc.g2 .- cg7.g2) < 1e-9 * scale

        # Device-native path (batched hook: g2 formed and kept on the device) must match the
        # host CPU path, both for a single batch and for chunked (partial final) batches.
        cb = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cb], symmetry=nothing, use_gpu=true, progress_print_step=10^9)
        @test maximum(abs, cc.g2 .- cb.g2) < 1e-9 * scale
        @test cc.ωq == cb.ωq

        cb7 = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cb7], symmetry=nothing, use_gpu=true, q_batch_size=7, progress_print_step=10^9)
        @test maximum(abs, cc.g2 .- cb7.g2) < 1e-9 * scale

        # Outer-k tiling (k_batch_size=5 forces a partial final k-tile) must agree too,
        # together with a partial q-chunk.
        cbk = _RecordCalcBatched()
        ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[cbk], symmetry=nothing, use_gpu=true,
            k_batch_size=5, q_batch_size=7, progress_print_step=10^9)
        @test maximum(abs, cc.g2 .- cbk.g2) < 1e-9 * scale

        # Scope guards: the GPU path must reject out-of-scope options it does not implement.
        @test_throws Exception ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[_RecordCalc()], symmetry=nothing, use_gpu=true,
            covariant_derivative_of_g=true, progress_print_step=10^9)
        @test_throws Exception ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid;
            calculators=[_RecordCalc()], symmetry=nothing, use_gpu=true,
            energy_conservation=(:Fixed, 0.1), progress_print_step=10^9)
    end
end
