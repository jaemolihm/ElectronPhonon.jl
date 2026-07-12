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
    for method in 1:6
        sₒ, sᵢ = EP.bte_scattering_increments(method, 0.01, -0.005, 0.008, 1e-3, 0.5, 0.002, 0.01, 0.005)
        @test sₒ ≈ ref[method][1] rtol=1e-12
        @test sᵢ ≈ ref[method][2] rtol=1e-12
    end
    # δ-underflow guard: huge energy mismatch ⇒ exact zero (no 0·Inf NaN even for Method5)
    s = EP.bte_scattering_increments(5, 10.0, -10.0, 0.01, 1e-3, 1.0, 0.01, 0.01, 0.005)
    @test all(isfinite, s) && s == (0.0, 0.0)
end

# Generic CPU `bte_window_scatter!` vs the CUDA kernel — same shared core, must agree to ~machine eps.
const _CUDA_OK = (get(ENV, "EP_TEST_CUDA", "1") == "1") && try
    @eval import CUDA
    CUDA.functional()
catch
    false
end
let
    if _CUDA_OK
        @testset "bte_window_scatter! CPU generic == CUDA kernel" begin
            Random.seed!(7)
            FT=Float64; nw=4; nm=3; nqc=6; nT=2
            ikqs = collect(1:nqc)
            imap_i_col = collect(1:nw)
            imap_f = reshape(collect(1:nw*nqc), nw, nqc)
            n_i=nw; n_f=nw*nqc
            e_i=0.01randn(n_i); e_f=0.01randn(n_f); wq=fill(1/nqc,nqc)
            g2vals=abs.(randn(nw,nw,nm,nqc)).*1e-3; ωqmat=(0.5 .+ abs.(randn(nm,nqc))).*1e-2
            μs=FT[0.0,0.002]; Ts=FT[0.01,0.02]; ηs=FT[0.005,0.005]; ωcut=FT(1e-6)
            for method in 1:6
                So=zeros(n_i,nT); Si=zeros(n_i,n_f,nT)
                EP.bte_window_scatter!(So,Si,g2vals,ωqmat,imap_i_col,imap_f,ikqs,e_i,e_f,wq,
                    μs,Ts,ηs,method,ωcut,nw,nw,nm,nqc,nT; i0=0)
                Sog=CUDA.zeros(FT,n_i,nT); Sig=CUDA.zeros(FT,n_i,n_f,nT)
                EP.bte_window_scatter!(Sog,Sig,CUDA.CuArray(g2vals),CUDA.CuArray(ωqmat),
                    CUDA.CuArray(imap_i_col),CUDA.CuArray(imap_f),CUDA.CuArray(ikqs),
                    CUDA.CuArray(e_i),CUDA.CuArray(e_f),CUDA.CuArray(wq),
                    CUDA.CuArray(μs),CUDA.CuArray(Ts),CUDA.CuArray(ηs),method,ωcut,nw,nw,nm,nqc,nT; i0=0)
                @test Array(Sog) ≈ So rtol=1e-10
                @test Array(Sig) ≈ Si rtol=1e-10
            end
        end

        @testset "bte_window_scatter! out-of-window (imap==0) + block-tile (i0≠0)" begin
            Random.seed!(11)
            FT=Float64; nw=4; nm=2; nqc=4; nT=1
            ikqs = collect(1:nqc)
            # Some bands out-of-window (imap==0); in-window outer states live in a tile i0+1:i0+ni.
            i0=3; ni=4                          # global outer states 4..7 land in tile rows 1..4
            imap_i_col = [0, 4, 6, 7]           # band 1 out-of-window; others in-tile (global i)
            n_i_global = 10
            imap_f = [ (m+ (kq-1)*nw) % 7 == 0 ? 0 : (m + (kq-1)*nw) for m in 1:nw, kq in 1:nqc ]  # scatter some 0s
            n_f = nw*nqc
            e_i=0.01randn(n_i_global); e_f=0.01randn(n_f); wq=fill(1/nqc,nqc)
            g2vals=abs.(randn(nw,nw,nm,nqc)).*1e-3; ωqmat=(0.5 .+ abs.(randn(nm,nqc))).*1e-2
            μs=FT[0.0]; Ts=FT[0.01]; ηs=FT[0.005]; ωcut=FT(1e-6)
            for method in (1,5,6)
                So=zeros(n_i_global,nT); Si=zeros(ni,n_f,nT)
                EP.bte_window_scatter!(So,Si,g2vals,ωqmat,imap_i_col,imap_f,ikqs,e_i,e_f,wq,
                    μs,Ts,ηs,method,ωcut,nw,nw,nm,nqc,nT; i0=i0)
                Sog=CUDA.zeros(FT,n_i_global,nT); Sig=CUDA.zeros(FT,ni,n_f,nT)
                EP.bte_window_scatter!(Sog,Sig,CUDA.CuArray(g2vals),CUDA.CuArray(ωqmat),
                    CUDA.CuArray(imap_i_col),CUDA.CuArray(imap_f),CUDA.CuArray(ikqs),
                    CUDA.CuArray(e_i),CUDA.CuArray(e_f),CUDA.CuArray(wq),
                    CUDA.CuArray(μs),CUDA.CuArray(Ts),CUDA.CuArray(ηs),method,ωcut,nw,nw,nm,nqc,nT; i0=i0)
                @test Array(Sog) ≈ So rtol=1e-10
                @test Array(Sig) ≈ Si rtol=1e-10
                # out-of-window outer band 1 contributes nowhere; global rows 1..3,8..10 stay zero
                @test all(So[setdiff(1:n_i_global, 4:7), :] .== 0)
            end
        end
    else
        @info "CUDA not functional — skipping GPU bte_window_scatter! test"
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
        smearing = [(:Gaussian, 100.0 * meV)], occupation_method = 5)
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
