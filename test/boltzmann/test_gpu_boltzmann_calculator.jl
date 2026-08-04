using Test
using ElectronPhonon
const EP = ElectronPhonon
using Random

@testset "bte_scattering_increments (shared core) pinned values" begin
    # (sₒ, sᵢ) for methods 1..6 at a fixed in-window input, pinned from the validated
    # implementation as a regression guard. Input: ek, ekq, ωq, g2, wtq, μ, T, η =
    #   0.01, -0.005, 0.008, 1e-3, 0.5, 0.002, 0.01, 0.005  (atomic units).
    ref = Dict(
        1 => (0.057312033235413576, 0.056224157281476575),
        2 => (0.05827514910329083,  0.05529493824267435),
        3 => (0.04360686568446893,  0.08472285346130072),
        4 => (0.08781344455188772,  0.04207212358135953),
        5 => (0.06188108814500812,  0.059703185425524365),
        6 => (0.030909988400384957, 0.029822112446447974),
    )
    ek, ekq, ωq, g2, wtq, μ, T, η = 0.01, -0.005, 0.008, 1e-3, 0.5, 0.002, 0.01, SmearingType(:Gaussian, 0.005)
    for method in 1:6
        sₒ, sᵢ = EP.bte_scattering_increments(method, ek, ekq, ωq, g2, wtq, μ, T, η)
        @test sₒ ≈ ref[method][1] rtol=1e-12
        @test sᵢ ≈ ref[method][2] rtol=1e-12
    end
    # δ-underflow guard: huge energy mismatch ⇒ exact zero (no 0·Inf NaN even for Method5)
    s = EP.bte_scattering_increments(5, 10.0, -10.0, 0.01, 1e-3, 1.0, 0.01, 0.01, SmearingType(:Gaussian, 0.005))
    @test all(isfinite, s) && s == (0.0, 0.0)
end

# Both `bte_window_accumulate!` methods — the generic host one (which serves the CPU+batched
# validation configuration) and the CUDA kernel — against an independent CPU reference. The reference
# is a self-contained per-(m, n, iq) loop over the same shared `bte_scattering_increments`; it is
# deliberately NOT the implementation under test on either backend, so it independently pins both to
# ~machine eps, block-tile `i0` and out-of-window `imap == 0` included.
const _CUDA_OK = (get(ENV, "EP_TEST_CUDA", "1") == "1") && try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

# Independent CPU reference for the Sₒ/Sᵢ accumulation (NOT a production method, and deliberately not
# shared with `src`: keeping it duplicated is what makes it an oracle for the generic host method that
# now lives in src/boltzmann/boltzmann_calculator.jl as well as for the CUDA kernel).
function _bte_accumulate_ref!(So, Si, g2vals, ωqmat, imap_i_at_k, imap_f, ikqs, e_i, e_f, wf,
        μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nmodes, nq_batch, i0)
    nT = length(μs)
    for iq_batch in 1:nq_batch, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_at_k[n]; i > 0 || continue
        ikq = ikqs[iq_batch]; f = imap_f[m, ikq]; f > 0 || continue
        ek = e_i[i]; ekq = e_f[f]; wtq = wf[f]   # per-final-state weight
        for iT in 1:nT
            sₒ = 0.0; sᵢ = 0.0
            for ν in 1:nmodes
                ωq = ωqmat[ν, iq_batch]; ωq < ω_cutoff && continue
                sₒ_ν, sᵢ_ν = EP.bte_scattering_increments(method, ek, ekq, ωq,
                    g2vals[m, n, ν, iq_batch], wtq, μs[iT], Ts[iT], ηs[iT])
                sₒ += sₒ_ν; sᵢ += sᵢ_ν
            end
            So[i, iT] += sₒ
            Si[i - i0, f, iT] = sᵢ
        end
    end
    (So, Si)
end

# The two accumulate fixtures below run on either backend: `arr` moves an input array onto it
# (`identity` for the host, `CUDA.CuArray` for the device) and `zdev(T, dims...)` allocates the zeroed
# output buffers there. Both fixtures compare against `_bte_accumulate_ref!` above.
function check_bte_accumulate_methods(arr, zdev)
    Random.seed!(7)
    FT=Float64; nw=4; nmodes=3; nq_batch=6; nT=2
    ikqs = collect(1:nq_batch)
    imap_i_at_k = collect(1:nw)
    imap_f = reshape(collect(1:nw*nq_batch), nw, nq_batch)
    n_i=nw; n_f=nw*nq_batch
    e_i=0.01randn(n_i); e_f=0.01randn(n_f); wf=abs.(0.1randn(n_f)).+0.01  # per-final-state weight
    g2vals=abs.(randn(nw,nw,nmodes,nq_batch)).*1e-3; ωqmat=(0.5 .+ abs.(randn(nmodes,nq_batch))).*1e-2
    μs=FT[0.0,0.002]; Ts=FT[0.01,0.02]; ωcut=FT(1e-6)
    ηs = [SmearingType(:Gaussian, FT(x)) for x in [0.005, 0.005]]
    for method in 1:6
        So=zeros(n_i,nT); Si=zeros(n_i,n_f,nT)
        _bte_accumulate_ref!(So,Si,g2vals,ωqmat,imap_i_at_k,imap_f,ikqs,e_i,e_f,wf,
            μs,Ts,ηs,method,ωcut,nw,nw,nmodes,nq_batch,0)
        Sod=zdev(FT,n_i,nT); Sid=zdev(FT,n_i,n_f,nT)
        EP.bte_window_accumulate!(Sod,Sid,arr(g2vals),arr(ωqmat),
            arr(imap_i_at_k),arr(imap_f),arr(ikqs),
            arr(e_i),arr(e_f),arr(wf),
            arr(μs),arr(Ts),arr(ηs),method,ωcut,nw,nw,nmodes,nq_batch,0)
        @test Array(Sod) ≈ So rtol=1e-10
        @test Array(Sid) ≈ Si rtol=1e-10
    end
end

function check_bte_accumulate_tile(arr, zdev)
    Random.seed!(11)
    FT=Float64; nw=4; nmodes=2; nq_batch=4; nT=1
    ikqs = collect(1:nq_batch)
    # Some bands out-of-window (imap==0); in-window outer states live in a tile i0+1:i0+ni.
    i0=3; ni=4                          # global outer states 4..7 land in tile rows 1..4
    imap_i_at_k = [0, 4, 6, 7]           # band 1 out-of-window; others in-tile (global i)
    n_i_global = 10
    imap_f = [ (m+ (kq-1)*nw) % 7 == 0 ? 0 : (m + (kq-1)*nw) for m in 1:nw, kq in 1:nq_batch ]  # scatter some 0s
    n_f = nw*nq_batch
    e_i=0.01randn(n_i_global); e_f=0.01randn(n_f); wf=abs.(0.1randn(n_f)).+0.01  # per-final-state weight
    g2vals=abs.(randn(nw,nw,nmodes,nq_batch)).*1e-3; ωqmat=(0.5 .+ abs.(randn(nmodes,nq_batch))).*1e-2
    μs=FT[0.0]; Ts=FT[0.01]; ωcut=FT(1e-6)
    ηs = [SmearingType(:Gaussian, FT(x)) for x in [0.005]]
    for method in (1,5,6)
        So=zeros(n_i_global,nT); Si=zeros(ni,n_f,nT)
        _bte_accumulate_ref!(So,Si,g2vals,ωqmat,imap_i_at_k,imap_f,ikqs,e_i,e_f,wf,
            μs,Ts,ηs,method,ωcut,nw,nw,nmodes,nq_batch,i0)
        Sod=zdev(FT,n_i_global,nT); Sid=zdev(FT,ni,n_f,nT)
        EP.bte_window_accumulate!(Sod,Sid,arr(g2vals),arr(ωqmat),
            arr(imap_i_at_k),arr(imap_f),arr(ikqs),
            arr(e_i),arr(e_f),arr(wf),
            arr(μs),arr(Ts),arr(ηs),method,ωcut,nw,nw,nmodes,nq_batch,i0)
        @test Array(Sod) ≈ So rtol=1e-10
        @test Array(Sid) ≈ Si rtol=1e-10
        # out-of-window outer band 1 contributes nowhere; global rows 1..3,8..10 stay zero
        @test all(So[setdiff(1:n_i_global, 4:7), :] .== 0)
    end
end

# Host method: no CUDA needed. This is the method the CPU+batched configuration runs.
@testset "bte_window_accumulate! host method vs CPU reference" begin
    check_bte_accumulate_methods(identity, (T, dims...) -> zeros(T, dims...))
    check_bte_accumulate_tile(identity, (T, dims...) -> zeros(T, dims...))
end

if _CUDA_OK
    @testset "bte_window_accumulate! CUDA kernel vs CPU reference" begin
        check_bte_accumulate_methods(CUDA.CuArray, CUDA.zeros)
        check_bte_accumulate_tile(CUDA.CuArray, CUDA.zeros)
    end
else
    @info "CUDA not functional — skipping GPU bte_window_accumulate! test"
end

# End-to-end BoltzmannCalculator: the same calculator over a full pass of run_eph_over_k_and_kq must
# produce the same Sₒ/Sᵢ in every (backend, batched) configuration. Sₒ (the SERTA lifetime) is
# gauge-invariant so it agrees to ~machine eps; this is the real cross-check that the batched loop +
# its scatter match the per-point host path. Pb (metal) artifact model.
#
# NOTE: a green CPU+batched arm is NOT GPU coverage. It exercises the batched control flow, the tiling
# brackets, the payload construction and the calculator's batched `run_calculator!` on host arrays, but
# none of the CUDA kernels (`_bte_window_accumulate_kernel!`, `_window_scatter_kernel!`, the fused
# rotation kernel, `CUBLAS.gemm_strided_batched!`, the cuSOLVER batched eigensolve). Those are only
# covered by the `_CUDA_OK` arms.
@testset "end-to-end BTE: per-point vs batched (Pb)" begin
    model = _load_model_from_artifacts("pb")   # nw=4, nmodes=3; loads the e-ph matrix
    eV = EP.unit_to_aru(:eV); K = EP.unit_to_aru(:K); meV = EP.unit_to_aru(:meV)
    μ = 11.68eV; window = (μ - 0.5eV, μ + 0.5eV)
    mkcalc() = BoltzmannCalculator{Float64}(;
        occ = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0, μlist = μ,
            volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
        smearing_list = [SmearingType(:Gaussian, 100.0 * meV)], occupation_method = 5)
    runbte(grid, backend, batched; nq_batch_max = nothing, nk_outer_batch_max = 256) =
        (c = mkcalc(); EP.run_eph_over_k_and_kq(model, grid, grid;
            calculators = [c], symmetry = nothing, window_k = window, window_kq = window,
            fourier_mode = "gridopt", backend, batched, nq_batch_max, nk_outer_batch_max,
            progress_print_step = 10^9, verbosity = 0); c)

    cc = runbte((6, 6, 6), EP.CPUBackend(), false)
    @test length(cc.Sₒ[1]) > 0
    @test all(isfinite, stack(cc.Sₒ)) && all(isfinite, stack(cc.Sᵢ))

    # CPU + batched (no CUDA needed): the batched loop on host arrays must reproduce the per-point
    # result on the SAME grid, to summation order. It is slow by construction (serial k-batches,
    # `mul!`-loop `batched_gemm!`), so the grid is kept at 6³ — measured 0.04 s there, versus 0.06 s
    # for the per-point arm, small enough not to need a separate smaller grid. (4³ is NOT usable: this
    # ±0.5 eV window keeps zero k-points on that grid, and an empty selection cannot build a q-grid.)
    #
    # The two batch caps tile DIFFERENT axes, and both must be set explicitly here because on a
    # `CPUBackend` `plan_batch` returns the requested cap verbatim (`free_bytes` is unbounded):
    #   * `nq_batch_max` tiles the per-q device STAGING inside one k-batch. Without it every per-q
    #     buffer would be sized to the whole k+q grid.
    #   * `nk_outer_batch_max` tiles the outer-k axis, which is what the calculator's Sᵢ
    #     `TiledDeviceOutput` is tiled over — so it, not `nq_batch_max`, is what makes ntiles > 1 and
    #     drives `tile_begin!`/`tile_download!` more than once with a NONZERO `tile_offset`. The
    #     default 256 exceeds nk = 66 here, which would leave the whole run in a single tile at
    #     `i0 == 0` and never exercise the `Sᵢ[iT][i0+1:i0+ni, :] .= host[1:ni, :, iT]` bookkeeping.
    #     20 gives 4 tiles (20+20+20+6).
    @testset "CPU+batched == CPU+per-point" begin
        c_ba = runbte((6, 6, 6), EP.CPUBackend(), true; nq_batch_max = 7, nk_outer_batch_max = 20)
        # Guard the tiling itself: `tile_free!` resets `tile_i0` at postprocess, so the offset cannot be
        # read back after the run — assert its precondition instead. More outer k than the cap ⇒ several
        # k-batches ⇒ several Sᵢ tiles at nonzero `tile_offset`. If a future change to the grid or the
        # default cap collapses this to one tile, this fails instead of silently losing the coverage.
        @test c_ba.el_i.kpts.n > 20        # 66 outer k at cap 20 ⇒ 4 tiles
        @test stack(c_ba.Sₒ) ≈ stack(cc.Sₒ) rtol = 1e-9
        @test stack(c_ba.Sᵢ) ≈ stack(cc.Sᵢ) rtol = 1e-9
        # Measured 2026-08-03 (Pb 6³): Sₒ 6.1e-16, Sᵢ 1.3e-15 — pure batched-GEMM reassociation.
        @info "CPU+batched vs CPU+per-point (Pb 6³)" Sₒ_reldev =
            maximum(abs, stack(c_ba.Sₒ) .- stack(cc.Sₒ)) / maximum(abs, stack(cc.Sₒ)) Sᵢ_reldev =
            maximum(abs, stack(c_ba.Sᵢ) .- stack(cc.Sᵢ)) / maximum(abs, stack(cc.Sᵢ))
    end

    if _CUDA_OK
        # `batched` omitted: it derives from the backend, which is the production GPU spelling.
        cg = runbte((6, 6, 6), EP.gpu_backend(), nothing)
        # rtol, not ==: the setup eigensolve differs by a degeneracy gauge on the device, and Sₒ is an
        # atomic fold there (not bitwise reproducible run-to-run; measured 5e-16 relative at 6³, while
        # Sᵢ IS bitwise reproducible). Measured CPU-vs-GPU at 6³: Sₒ 1.8e-13, Sᵢ 2.5e-13.
        @test stack(cg.Sₒ) ≈ stack(cc.Sₒ) rtol = 1e-9
        @test stack(cg.Sᵢ) ≈ stack(cc.Sᵢ) rtol = 1e-9
        # Explicit `batched = false` on a GPU backend: rejected at the driver entry.
        @test_throws ArgumentError runbte((6, 6, 6), EP.gpu_backend(), false)
    end
end
