using Test
using ElectronPhonon
const EP = ElectronPhonon
using Random
using SparseArrays: nnz

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

# Both `bte_window_accumulate!` methods — the generic host one (which serves a run on
# a CPU backend) and the CUDA kernel — against an independent CPU reference. The reference
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
function _bte_accumulate_ref!(So, Si, epvals, ωqmat, imap_i_at_k, imap_f, ikqs, e_i, e_f, wf,
        μs, Ts, ηs, method, ω_cutoff, nbandkq, nbandk, nmodes, npairs, i0)
    nT = length(μs)
    for ipair in 1:npairs, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_at_k[n]; i > 0 || continue
        ikq = ikqs[ipair]; f = imap_f[m, ikq]; f > 0 || continue
        ek = e_i[i]; ekq = e_f[f]; wtq = wf[f]   # per-final-state weight
        for iT in 1:nT
            sₒ = 0.0; sᵢ = 0.0
            for ν in 1:nmodes
                ωq = ωqmat[ν, ipair]; ωq < ω_cutoff && continue
                g2 = abs2(epvals[m, n, ν, ipair]) / (2ωq)
                sₒ_ν, sᵢ_ν = EP.bte_scattering_increments(method, ek, ekq, ωq,
                    g2, wtq, μs[iT], Ts[iT], ηs[iT])
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
    FT=Float64; nw=4; nmodes=3; npairs=6; nT=2
    ikqs = collect(1:npairs)
    imap_i_at_k = collect(1:nw)
    imap_f = reshape(collect(1:nw*npairs), nw, npairs)
    n_i=nw; n_f=nw*npairs
    e_i=0.01randn(n_i); e_f=0.01randn(n_f); wf=abs.(0.1randn(n_f)).+0.01  # per-final-state weight
    epvals=randn(ComplexF64,nw,nw,nmodes,npairs).*0.03; ωqmat=(0.5 .+ abs.(randn(nmodes,npairs))).*1e-2
    μs=FT[0.0,0.002]; Ts=FT[0.01,0.02]; ωcut=FT(1e-6)
    ηs = [SmearingType(:Gaussian, FT(x)) for x in [0.005, 0.005]]
    for method in 1:6
        So=zeros(n_i,nT); Si=zeros(n_i,n_f,nT)
        _bte_accumulate_ref!(So,Si,epvals,ωqmat,imap_i_at_k,imap_f,ikqs,e_i,e_f,wf,
            μs,Ts,ηs,method,ωcut,nw,nw,nmodes,npairs,0)
        Sod=zdev(FT,n_i,nT); Sid=zdev(FT,n_i,n_f,nT)
        EP.bte_window_accumulate!(Sod,Sid,arr(epvals),arr(ωqmat),
            arr(imap_i_at_k),arr(imap_f),arr(ikqs),
            arr(e_i),arr(e_f),arr(wf),
            arr(μs),arr(Ts),arr(ηs),method,ωcut,0)
        @test Array(Sod) ≈ So rtol=1e-10
        @test Array(Sid) ≈ Si rtol=1e-10
    end
end

function check_bte_accumulate_tile(arr, zdev)
    Random.seed!(11)
    FT=Float64; nw=4; nmodes=2; npairs=4; nT=1
    ikqs = collect(1:npairs)
    # Some bands out-of-window (imap==0); in-window outer states live in a tile i0+1:i0+ni.
    i0=3; ni=4                          # global outer states 4..7 land in tile rows 1..4
    imap_i_at_k = [0, 4, 6, 7]           # band 1 out-of-window; others in-tile (global i)
    n_i_global = 10
    imap_f = [ (m+ (kq-1)*nw) % 7 == 0 ? 0 : (m + (kq-1)*nw) for m in 1:nw, kq in 1:npairs ]  # scatter some 0s
    n_f = nw*npairs
    e_i=0.01randn(n_i_global); e_f=0.01randn(n_f); wf=abs.(0.1randn(n_f)).+0.01  # per-final-state weight
    epvals=randn(ComplexF64,nw,nw,nmodes,npairs).*0.03; ωqmat=(0.5 .+ abs.(randn(nmodes,npairs))).*1e-2
    μs=FT[0.0]; Ts=FT[0.01]; ωcut=FT(1e-6)
    ηs = [SmearingType(:Gaussian, FT(x)) for x in [0.005]]
    for method in (1,5,6)
        So=zeros(n_i_global,nT); Si=zeros(ni,n_f,nT)
        _bte_accumulate_ref!(So,Si,epvals,ωqmat,imap_i_at_k,imap_f,ikqs,e_i,e_f,wf,
            μs,Ts,ηs,method,ωcut,nw,nw,nmodes,npairs,i0)
        Sod=zdev(FT,n_i_global,nT); Sid=zdev(FT,ni,n_f,nT)
        EP.bte_window_accumulate!(Sod,Sid,arr(epvals),arr(ωqmat),
            arr(imap_i_at_k),arr(imap_f),arr(ikqs),
            arr(e_i),arr(e_f),arr(wf),
            arr(μs),arr(Ts),arr(ηs),method,ωcut,i0)
        @test Array(Sod) ≈ So rtol=1e-10
        @test Array(Sid) ≈ Si rtol=1e-10
        # out-of-window outer band 1 contributes nowhere; global rows 1..3,8..10 stay zero
        @test all(So[setdiff(1:n_i_global, 4:7), :] .== 0)
    end
end

# Host method: no CUDA needed. This is the method a run on a CPU backend uses.
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
# produce the same Sₒ/Sᵢ in every (backend, tiling) configuration. Sₒ (the SERTA lifetime) is
# gauge-invariant so it agrees to ~machine eps. Pb (metal) artifact model.
#
# NOTE: a green CPU arm is NOT GPU coverage. It exercises the batched control flow, the tiling
# brackets, the payload construction and the calculator's batched `run_calculator!` on host arrays, but
# none of the CUDA kernels (`_bte_window_accumulate_kernel!`, `_window_scatter_kernel!`, the fused
# rotation kernel, `CUBLAS.gemm_strided_batched!`, the cuSOLVER batched eigensolve). Those are only
# covered by the `_CUDA_OK` arms.
@testset "end-to-end BTE: CPU tilings and GPU (Pb)" begin
    model = _load_model_from_artifacts("pb")   # nw=4, nmodes=3; loads the e-ph matrix
    eV = EP.unit_to_aru(:eV); K = EP.unit_to_aru(:K); meV = EP.unit_to_aru(:meV)
    μ = 11.68eV; window = (μ - 0.5eV, μ + 0.5eV)
    mkcalc(Si_format = :dense) = BoltzmannCalculator{Float64}(;
        occ = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0, μlist = μ,
            volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
        smearing_list = [SmearingType(:Gaussian, 100.0 * meV)], occupation_method = 5, Si_format)
    runbte(grid, backend; n_inner_tile = nothing, n_outer_batch = 256, Si_format = :dense) =
        (c = mkcalc(Si_format); EP.run_eph_over_k_and_kq(model, grid, grid;
            calculators = [c], symmetry = nothing, window_k = window, window_kq = window,
            backend, n_inner_tile, n_outer_batch,
            progress_print_step = 10^9, verbosity = 0); c)

    cc = runbte((6, 6, 6), EP.CPUBackend())
    @test length(cc.Sₒ[1]) > 0
    @test all(isfinite, stack(cc.Sₒ)) && all(isfinite, stack(cc.Sᵢ))

    # CPU with small tiles (no CUDA needed) must reproduce the default-width result on the SAME grid,
    # to summation order. The grid is kept at 6³. (4³ is NOT usable: this ±0.5 eV window keeps zero k-points on that grid, and an
    # empty selection cannot build a q-grid.)
    #
    # The two batch caps tile DIFFERENT axes, and both must be set explicitly here because on a
    # `CPUBackend` `plan_batch` returns the requested cap verbatim (`free_bytes` is unbounded):
    #   * `n_inner_tile` tiles the per-q device STAGING inside one k-batch. Without it every per-q
    #     buffer would be sized to the whole k+q grid.
    #   * `n_outer_batch` tiles the outer-k axis, which is what the calculator's Sᵢ
    #     `TiledDeviceOutput` is tiled over — so it, not `n_inner_tile`, is what makes ntiles > 1 and
    #     drives `tile_begin!`/`tile_download!` more than once with a NONZERO `tile_offset`. The
    #     default 256 exceeds nk = 66 here, which would leave the whole run in a single tile at
    #     `i0 == 0` and never exercise the `Sᵢ[iT][i0+1:i0+ni, :] .= host[1:ni, :, iT]` bookkeeping.
    #     20 gives 4 tiles (20+20+20+6).
    @testset "CPU small tiles == CPU default widths" begin
        c_ba = runbte((6, 6, 6), EP.CPUBackend(); n_inner_tile = 7, n_outer_batch = 20)
        # Guard the tiling itself: `tile_free!` resets `tile_i0` at postprocess, so the offset cannot be
        # read back after the run — assert its precondition instead. More outer k than the cap ⇒ several
        # k-batches ⇒ several Sᵢ tiles at nonzero `tile_offset`. If a future change to the grid or the
        # default cap collapses this to one tile, this fails instead of silently losing the coverage.
        @test c_ba.el_i.kpts.n > 20        # 66 outer k at cap 20 ⇒ 4 tiles
        @test stack(c_ba.Sₒ) ≈ stack(cc.Sₒ) rtol = 1e-9
        @test stack(c_ba.Sᵢ) ≈ stack(cc.Sᵢ) rtol = 1e-9
        @info "CPU small tiles vs CPU default widths (Pb 6³)" Sₒ_reldev =
            maximum(abs, stack(c_ba.Sₒ) .- stack(cc.Sₒ)) / maximum(abs, stack(cc.Sₒ)) Sᵢ_reldev =
            maximum(abs, stack(c_ba.Sᵢ) .- stack(cc.Sᵢ)) / maximum(abs, stack(cc.Sᵢ))
    end

    if _CUDA_OK
        cg = runbte((6, 6, 6), EP.gpu_backend())
        # rtol, not ==: the setup eigensolve differs by a degeneracy gauge on the device, and Sₒ is an
        # atomic fold there (not bitwise reproducible run-to-run; measured 5e-16 relative at 6³, while
        # Sᵢ IS bitwise reproducible). Measured CPU-vs-GPU at 6³: Sₒ 1.8e-13, Sᵢ 2.5e-13.
        @test stack(cg.Sₒ) ≈ stack(cc.Sₒ) rtol = 1e-9
        @test stack(cg.Sᵢ) ≈ stack(cc.Sᵢ) rtol = 1e-9
    end

    # Si_format = :csr stores the same Sᵢ, as the nonzeros of its transpose, over several tiles.
    @testset "Si_format = :csr == :dense" begin
        for backend in (_CUDA_OK ? (EP.CPUBackend(), EP.gpu_backend()) : (EP.CPUBackend(),))
            c_dense = runbte((6, 6, 6), backend; n_outer_batch = 20)
            c_csr = runbte((6, 6, 6), backend; n_outer_batch = 20, Si_format = :csr)
            @test isempty(c_csr.Sᵢ) && size(c_csr.Sᵢᵀ[1]) == reverse(size(c_dense.Sᵢ[1]))
            @test Matrix(transpose(c_csr.Sᵢᵀ[1])) == c_dense.Sᵢ[1]
            @test nnz(c_csr.Sᵢᵀ[1]) == count(!iszero, c_dense.Sᵢ[1])
            # The solver takes the CSR output as `transpose.(Sᵢᵀ)`: the same solve as on its dense
            # form, with the same Sₒ (Sₒ is a device atomic fold, not bitwise across runs).
            solve(scat_mat) = EP.solve_electron_bte(c_csr.el_i, c_csr.el_f, scat_mat, stack(c_csr.Sₒ),
                                                    c_csr.occ; solver = :fixed_point)
            r_csr, r_dense = solve(transpose.(c_csr.Sᵢᵀ)), solve(Matrix.(transpose.(c_csr.Sᵢᵀ)))
            @test r_csr.σ ≈ r_dense.σ rtol = 1e-12
            @test !(r_dense.σ ≈ r_dense.σ_serta)
        end
    end
end
