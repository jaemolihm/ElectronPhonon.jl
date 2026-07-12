using Test
using ElectronPhonon
const EP = ElectronPhonon
using Random

# Direct (independent) reference for the shared BTE scattering core, transcribed from the upstream
# `dev_BTE` reference implementation. Guards against regressions in `bte_scattering_increments`
# (the one physics implementation shared by the CPU loop and GPU kernel).
function _ref_increment(method, ek, ekq, ωq, g2, wtq, μ, T, η)
    occf(e) = 1/(exp(e/T)+1)
    df(e)   = -1/(2+exp(e/T)+exp(-e/T))/T
    nb(e)   = e == 0 ? 0.0 : 1/expm1(e/T)
    δ(Δ)    = exp(-(Δ/η)^2)/sqrt(π)/η
    Δe1 = ek-ekq+ωq; Δe2 = ek-ekq-ωq
    δ1 = δ(Δe1); δ2 = δ(Δe2)
    (δ1 < eps(δ1) && δ2 < eps(δ2)) && return (0.0, 0.0)
    nq = nb(ωq); f_k = occf(ek-μ); f_kq = occf(ekq-μ)
    if method == 1
        f1o=nq+f_kq; f2o=nq+1-f_kq; f1i=nq+f_k; f2i=nq+1-f_k
    elseif method == 2
        po=f_kq*(1-f_kq)/(f_k*(1-f_k)); f1o=po*(nq+1-f_k); f2o=po*(nq+f_k)
        pi=f_k*(1-f_k)/(f_kq*(1-f_kq)); f1i=pi*(nq+1-f_kq); f2i=pi*(nq+f_kq)
    elseif method == 3
        po=(1-f_kq)/(1-f_k); f1o=po*nq; f2o=po*(nq+1)
        pi=(1-f_k)/(1-f_kq); f1i=pi*nq; f2i=pi*(nq+1)
    elseif method == 4
        po=f_kq/f_k; f1o=po*(nq+1); f2o=po*nq
        pi=f_k/f_kq; f1i=pi*(nq+1); f2i=pi*nq
    elseif method == 5
        f1o=sqrt(nq*(nq+1)*df(ekq-μ)/df(ek-μ)); f2o=f1o
        f1i=sqrt(nq*(nq+1)*df(ek-μ)/df(ekq-μ)); f2i=f1i
    else
        no=nb(ekq-ek); f1o=no+f_kq; f2o=-(no+f_kq)
        ni=nb(ek-ekq); f1i=ni+f_k;  f2i=-(ni+f_k)
    end
    pref = 2π*wtq*g2
    ((δ1*f1o+δ2*f2o)*pref, (δ2*f1i+δ1*f2i)*pref)
end

@testset "bte_scattering_increments (shared core) matches dev_BTE reference" begin
    Random.seed!(42)
    for _ in 1:200, method in 1:6
        ek = 0.02randn(); ekq = 0.02randn(); ωq = 0.005 + 0.01rand()
        g2 = 1e-3rand(); wtq = rand(); μ = 0.01randn(); T = 0.005 + 0.02rand(); η = 0.002 + 0.01rand()
        sₒ, sᵢ = EP.bte_scattering_increments(method, ek, ekq, ωq, g2, wtq, μ, T, η)
        rₒ, rᵢ = _ref_increment(method, ek, ekq, ωq, g2, wtq, μ, T, η)
        @test sₒ ≈ rₒ rtol=1e-12
        @test sᵢ ≈ rᵢ rtol=1e-12
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
