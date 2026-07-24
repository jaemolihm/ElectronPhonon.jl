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

# Device `bte_window_accumulate!` (CUDA kernel) vs an independent CPU reference. There is no generic
# CPU `bte_window_accumulate!` method (the CPU BoltzmannCalculator never batches — it accumulates per
# (k, q) via the `run_calculator!(::EPData)` host loop), so the reference here is a self-contained
# per-(m, n, iq) loop over the same shared `bte_scattering_increments`. It independently pins the
# device kernel to ~machine eps, block-tile `i0` and out-of-window `imap == 0` included.
const _CUDA_OK = (get(ENV, "EP_TEST_CUDA", "1") == "1") && try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

# Non-batched CPU reference for the device kernel's Sₒ/Sᵢ accumulation (not a production method).
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

let
    if _CUDA_OK
        @testset "bte_window_accumulate! CUDA kernel vs CPU reference" begin
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
                Sog=CUDA.zeros(FT,n_i,nT); Sig=CUDA.zeros(FT,n_i,n_f,nT)
                EP.bte_window_accumulate!(Sog,Sig,CUDA.CuArray(g2vals),CUDA.CuArray(ωqmat),
                    CUDA.CuArray(imap_i_at_k),CUDA.CuArray(imap_f),CUDA.CuArray(ikqs),
                    CUDA.CuArray(e_i),CUDA.CuArray(e_f),CUDA.CuArray(wf),
                    CUDA.CuArray(μs),CUDA.CuArray(Ts),CUDA.CuArray(ηs),method,ωcut,nw,nw,nmodes,nq_batch,0)
                @test Array(Sog) ≈ So rtol=1e-10
                @test Array(Sig) ≈ Si rtol=1e-10
            end
        end

        @testset "bte_window_accumulate! out-of-window (imap==0) + block-tile (i0≠0)" begin
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
                Sog=CUDA.zeros(FT,n_i_global,nT); Sig=CUDA.zeros(FT,ni,n_f,nT)
                EP.bte_window_accumulate!(Sog,Sig,CUDA.CuArray(g2vals),CUDA.CuArray(ωqmat),
                    CUDA.CuArray(imap_i_at_k),CUDA.CuArray(imap_f),CUDA.CuArray(ikqs),
                    CUDA.CuArray(e_i),CUDA.CuArray(e_f),CUDA.CuArray(wf),
                    CUDA.CuArray(μs),CUDA.CuArray(Ts),CUDA.CuArray(ηs),method,ωcut,nw,nw,nmodes,nq_batch,i0)
                @test Array(Sog) ≈ So rtol=1e-10
                @test Array(Sig) ≈ Si rtol=1e-10
                # out-of-window outer band 1 contributes nowhere; global rows 1..3,8..10 stay zero
                @test all(So[setdiff(1:n_i_global, 4:7), :] .== 0)
            end
        end
    else
        @info "CUDA not functional — skipping GPU bte_window_accumulate! test"
    end
end

# End-to-end BoltzmannCalculator: the same calculator run on CPU (use_gpu=false) and GPU
# (use_gpu=true) over a full pass of run_eph_over_k_and_kq must produce the same Sₒ/Sᵢ. Sₒ (the
# SERTA lifetime) is gauge-invariant so it agrees to ~machine eps; this is the real cross-check that
# the device path (batched loop + device scatter) matches the host path. Pb (metal) artifact model.
@testset "end-to-end BTE: CPU vs GPU (Pb)" begin
    model = _load_model_from_artifacts("pb")   # nw=4, nmodes=3; loads the e-ph matrix
    eV = EP.unit_to_aru(:eV); K = EP.unit_to_aru(:K); meV = EP.unit_to_aru(:meV)
    μ = 11.68eV; window = (μ - 0.5eV, μ + 0.5eV)
    mkcalc() = BoltzmannCalculator{Float64}(;
        occ = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0, μlist = μ,
            volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
        smearing_list = [SmearingType(:Gaussian, 100.0 * meV)], occupation_method = 5)
    runbte(use_gpu) = (c = mkcalc(); EP.run_eph_over_k_and_kq(model, (6, 6, 6), (6, 6, 6);
        calculators = [c], symmetry = nothing, window_k = window, window_kq = window,
        fourier_mode = "gridopt", use_gpu, progress_print_step = 10^9); c)

    cc = runbte(false)
    @test length(cc.Sₒ[1]) > 0
    @test all(isfinite, stack(cc.Sₒ)) && all(isfinite, stack(cc.Sᵢ))
    if _CUDA_OK
        cg = runbte(true)
        @test stack(cg.Sₒ) ≈ stack(cc.Sₒ) rtol = 1e-9
        @test stack(cg.Sᵢ) ≈ stack(cc.Sᵢ) rtol = 1e-9
    end
end
