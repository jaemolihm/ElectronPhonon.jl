@testsnippet GpuBoltzmannCalculator begin
    const EP = ElectronPhonon
    using Random

    # `bte_window_accumulate!` on the tiles of a real outer-k run with ragged band counts, the generic host
    # method (a CPU backend) and the CUDA kernel, against a per-item reference loop over the same shared
    # `bte_scattering_increments` written from the segment table alone (`_items_by_loops`): the mode sum
    # inside, `Sₒ` accumulated over the pairs of the tile, `Sᵢ` written once per (i, f), window maps with
    # out-of-window bands, a batch offset `i0`.
    const _CUDA_OK = try
        @eval import CUDA
        CUDA.functional()
    catch
        false
    end

    include("../tile_reference.jl")

    function _bte_accumulate_ref(h, e_i, e_f, wf, μs, Ts, ηs, method, ω_cutoff, i0, ni)
        n_f = length(e_f); nT = length(μs)
        So = zeros(length(e_i), nT); Si = zeros(ni, n_f, nT)
        s_o = Dict{NTuple{3, Int}, Vector{Float64}}(); s_i = Dict{NTuple{3, Int}, Vector{Float64}}()
        for (l, m, n, ν, j) in _items_by_loops(h.seg, h.nm)
            i, f = h.imap_k[n, h.ik], h.imap_kq[m, h.ikq[j]]
            (i > 0 && f > 0) || continue
            so = get!(() -> zeros(nT), s_o, (j, m, n)); si = get!(() -> zeros(nT), s_i, (j, m, n))
            ω = h.e[ν, j]
            ω < ω_cutoff && continue
            for iT in 1:nT
                sₒ, sᵢ = EP.bte_scattering_increments(method, e_i[i], e_f[f], ω, abs2(h.ep[l]) / (2ω), wf[f],
                    μs[iT], Ts[iT], ηs[iT])
                so[iT] += sₒ; si[iT] += sᵢ
            end
        end
        for ((j, m, n), so) in s_o
            i, f = h.imap_k[n, h.ik], h.imap_kq[m, h.ikq[j]]
            So[i, :] .+= so
            Si[i - i0, f, :] .= s_i[(j, m, n)]
        end
        (So, Si)
    end

    function check_bte_accumulate_tiles(backend)
        eV = EP.unit_to_aru(:eV)
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        window = (11.68eV - 0.5eV, 11.68eV + 3eV)
        calc = _TileCollector()
        eng = EP.OuterKEngine(model, (6, 6, 6), (6, 6, 6); backend, calculators = [calc], window_k = window,
            window_kq = window, inner_loop_kq = true, n_inner_tile = 16, verbosity = 0, symmetry = nothing)
        EP.setup_calculator!(calc, eng.backend, eng.els_k, eng.els_kq, eng.phs; order = EP.OuterKLoop(), eng.sel_k,
            eng.sel_kq, nchunks_threads = 1, eng.n_outer_batch, eng.n_inner_tile, verbosity = 0)
        EP.stage1!(eng, 1:1)
        n_i = maximum(Array(eng.imap_k)); n_f = maximum(Array(eng.imap_kq))
        Random.seed!(7)
        e_i = 0.01randn(n_i); e_f = 0.01randn(n_f); wf = abs.(0.1randn(n_f)) .+ 0.01
        μs = [0.0, 0.002]; Ts = [0.01, 0.02]; ωcut = 1e-6
        ηs = [SmearingType(:Gaussian, x) for x in (0.005, 0.005)]
        to_dev = x -> EP.to_device(backend, x)
        ntiles = 0
        for inner in Iterators.partition(1:eng.kqpts.n, eng.n_inner_tile), method in (1, 5, 6)
            let tile = EP.stage2!(eng, 1, inner)
                tile === nothing && continue
                h = _tile_host(tile)
                is_k = filter(>(0), h.imap_k[:, h.ik])
                # the full buffer, then the rows of this k only (a batch offset)
                for (i0, ni) in ((0, n_i), (minimum(is_k) - 1, maximum(is_k) - minimum(is_k) + 1))
                    So_ref, Si_ref = _bte_accumulate_ref(h, e_i, e_f, wf, μs, Ts, ηs, method, ωcut, i0, ni)
                    So = to_dev(zeros(n_i, 2)); Si = to_dev(zeros(ni, n_f, 2))
                    EP.bte_window_accumulate!(So, Si, tile, to_dev(e_i), to_dev(e_f), to_dev(wf), to_dev(μs),
                        to_dev(Ts), to_dev(ηs), method, ωcut, i0)
                    @test Array(So) ≈ So_ref rtol = 1e-10
                    @test Array(Si) ≈ Si_ref rtol = 1e-10
                    @test any(!iszero, So_ref)
                end
                ntiles += 1
            end
        end
        @test ntiles > 0
    end
end

@testitem "bte_scattering_increments (shared core) pinned values" setup = [Common, GpuBoltzmannCalculator] begin
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

# Host method: no CUDA needed. This is the method a run on a CPU backend uses.
@testitem "bte_window_accumulate! host method vs per-item reference" setup = [Common, GpuBoltzmannCalculator] begin
    check_bte_accumulate_tiles(EP.CPUBackend())
end

@testitem "bte_window_accumulate! CUDA kernel vs per-item reference" tags = [:gpu] setup = [Common, GpuBoltzmannCalculator] begin
    check_bte_accumulate_tiles(EP.gpu_backend())
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
@testitem "end-to-end BTE: CPU tilings and GPU (Pb)" setup = [Common, GpuBoltzmannCalculator] begin
    model = _load_model_from_artifacts("pb")   # nw=4, nmodes=3; loads the e-ph matrix
    eV = EP.unit_to_aru(:eV); K = EP.unit_to_aru(:K); meV = EP.unit_to_aru(:meV)
    μ = 11.68eV; window = (μ - 0.5eV, μ + 0.5eV)
    mkcalc() = BoltzmannCalculator{Float64}(;
        occ = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0, μlist = μ,
            volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
        smearing_list = [SmearingType(:Gaussian, 100.0 * meV)], occupation_method = 5)
    runbte(grid, backend; n_inner_tile = nothing, n_outer_batch = 256) =
        (c = mkcalc(); EP.run_eph_over_k_and_kq(model, grid, grid;
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
end
