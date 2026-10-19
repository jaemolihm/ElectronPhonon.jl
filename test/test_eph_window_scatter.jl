@testsnippet EphWindowScatter begin
    using ElectronPhonon: OuterKEngine, OuterKLoop, CPUBackend, gpu_backend, stage1!, stage2!, setup_calculator!,
        gather_pair_table!, dedup_phonon_frequencies, phonon_index, unit_to_aru, OuterQEngine, OuterQLoop, compute_electron_states, Kpoints
    using Random

    # CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
    # tests when it is unavailable or non-functional (e.g. CPU-only CI).
    const WINDOW_SCATTER_GPU_AVAILABLE = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end

    include("tile_reference.jl")
end

# The flat tile's accessors and scatters, on the tiles of a real outer-k run with ragged band counts, against
# loops written from the segment table alone.
@testitem "tile accessors and window scatters" setup = [Common, EphWindowScatter] begin
    eV = unit_to_aru(:eV)
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    window = (11.68eV - 0.5eV, 11.68eV + 3eV)
    grid = (6, 6, 6)
    backends = Any[CPUBackend()]
    WINDOW_SCATTER_GPU_AVAILABLE && push!(backends, gpu_backend())
    for backend in backends
        calc = _TileCollector()
        eng = OuterKEngine(model, grid, grid; backend, calculators = [calc], window_k = window,
            window_kq = window, inner_loop_kq = true, n_inner_tile = 16, verbosity = 0, symmetry = nothing)
        setup_calculator!(calc, eng.backend, eng.els_k, eng.els_kq, eng.phs; order = OuterKLoop(), eng.sel_k,
            eng.sel_kq, nchunks_threads = 1, eng.n_outer_batch, eng.n_inner_tile, verbosity = 0)
        stage1!(eng, 1:1)
        n_i = maximum(Array(eng.imap_k)); n_f = maximum(Array(eng.imap_kq))
        nclasses = 0
        for inner in Iterators.partition(1:eng.kqpts.n, eng.n_inner_tile)
            let tile = stage2!(eng, 1, inner)
                tile === nothing && continue
                h = _tile_host(tile)
                nclasses = max(nclasses, size(h.seg, 2))
                items = _items_by_loops(h.seg, h.nm)
                @test length(items) == length(h.ep)
                # accessors on `device_tile`, through device broadcasts, against the loops
                to_dev = x -> ElectronPhonon.to_device(backend, x)
                dt = Ref(ElectronPhonon.device_tile(tile))
                @test Array(ElectronPhonon.item_indices.(dt, to_dev(first.(items)))) ==
                    [(m, n, ν, j) for (_, m, n, ν, j) in items]
                entries = unique(((m, n, j) for (_, m, n, _, j) in items))
                got = Array(ElectronPhonon.entry_indices.(dt, to_dev(collect(eachindex(entries)))))
                @test [(m, n, j) for (m, n, j, _, _) in got] == entries
                @test all(zip(got, entries)) do ((_, _, _, l1, stride), (m, n, j))
                    all(ν -> items[l1 + (ν - 1) * stride][2:5] == (m, n, ν, j), 1:h.nm)
                end
                @test [(l, m, n, ν, j) for (l, m, n, ν, j) in ElectronPhonon.eachitem(tile)] == items
                @test [(m, n, j) for (_, m, n, j, _, _) in ElectronPhonon.eachentry(tile)] == entries
                # the state maps of every item
                @test Array(ElectronPhonon.state_k.(dt, to_dev([n for (_, _, n, _, _) in items]),
                    to_dev([j for (_, _, _, _, j) in items]))) == [h.imap_k[n, h.ik] for (_, _, n, _, _) in items]
                @test Array(ElectronPhonon.state_kq.(dt, to_dev([m for (_, m, _, _, _) in items]),
                    to_dev([j for (_, _, _, _, j) in items]))) == [h.imap_kq[m, h.ikq[j]] for (_, m, _, _, j) in items]
                # the scatters: the full buffer, then a per-batch buffer of the rows this k holds
                is_k = filter(>(0), h.imap_k[:, h.ik])
                for (ni_stride, i0) in ((n_i, 0), (maximum(is_k) - minimum(is_k) + 1, minimum(is_k) - 1))
                    g2_ref, re_ref, im_ref = _scatter_reference(h, n_i, n_f, i0, ni_stride)
                    len = h.nm * ni_stride * n_f
                    g2, re, im_out = (to_dev(zeros(len)) for _ in 1:3)
                    ElectronPhonon.eph_window_scatter!(g2, tile, ni_stride, i0)
                    ElectronPhonon.eph_window_scatter_reim!(re, im_out, tile, ni_stride, i0)
                    @test any(!iszero, g2_ref)
                    @test Array(g2) ≈ vec(g2_ref) rtol = 1e-14
                    @test Array(re) == vec(re_ref) && Array(im_out) == vec(im_ref)
                end
            end
        end
        # vacuity guard: the fixture's tiles have more than one band class
        @test nclasses > 1
    end
end

# The accessors on the outer-q tile (per-pair `ik` / `iq` on the backend, several q per tile, k+q solved per
# tile) through device broadcasts: the energies of the gathered states against `compute_electron_states` at the
# pair's k and k+q, the eigenvectors normalized, and the state maps against the engine's.
@testitem "outer-q tile accessors" setup = [Common, EphWindowScatter] begin
    eV = unit_to_aru(:eV)
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
    window = (11.68eV - 0.5eV, 11.68eV + 3eV); grid = (4, 4, 4)
    backends = Any[CPUBackend()]
    WINDOW_SCATTER_GPU_AVAILABLE && push!(backends, gpu_backend())
    for backend in backends
        calc = _TileCollector()
        eng = ElectronPhonon.OuterQEngine(model, grid, grid; backend, calculators = [calc], window_k = window,
            window_kq = window, n_inner_tile = 48, n_outer_batch = 4, verbosity = 0, symmetry = nothing)
        setup_calculator!(calc, eng.backend, eng.els_k, eng.els_kq, eng.phs; order = ElectronPhonon.OuterQLoop(),
            eng.sel_k, eng.sel_kq, nchunks_threads = 1, eng.n_outer_batch, eng.n_inner_tile, verbosity = 0)
        stage1!(eng, 1:4)
        to_dev = x -> ElectronPhonon.to_device(backend, x)
        nclasses = 0
        let tile = stage2!(eng, 1:4, 1:12)
            nclasses = size(tile.seg_host, 2)
            seg, nm = tile.seg_host, tile.phs.nmodes
            n_pairs = size(tile.xkmat, 2)
            xks, xqs = tile.xk, tile.xq
            iks = Array(tile.ik)
            @test length(unique(Array(tile.iq))) > 1
            dt = Ref(ElectronPhonon.device_tile(tile))
            # every (n, j) of the tile's classes, as the loops of the segment table list them
            bands = [(j, n, m) for c in axes(seg, 2) for jj in 1:seg[ElectronPhonon.SEG_NPAIRS, c]
                     for j in (seg[ElectronPhonon.SEG_PAIR, c] + jj,)
                     for n in 1:seg[ElectronPhonon.SEG_BK, c] for m in 1:seg[ElectronPhonon.SEG_BKQ, c]]
            js, ns, ms = to_dev([b[1] for b in bands]), to_dev([b[2] for b in bands]), to_dev([b[3] for b in bands])
            e_k, e_kq = Array(ElectronPhonon.e_k.(dt, ns, js)), Array(ElectronPhonon.e_kq.(dt, ms, js))
            u_k = Array(ElectronPhonon.u_k.(dt, reshape(to_dev(collect(1:model.nw)), :, 1), reshape(ns, 1, :),
                reshape(js, 1, :)))
            u_kq = Array(ElectronPhonon.u_kq.(dt, reshape(to_dev(collect(1:model.nw)), :, 1), reshape(ms, 1, :),
                reshape(js, 1, :)))
            @test all(x -> isapprox(x, 1; atol = 1e-12), vec(sum(abs2, u_k; dims = 1)))
            @test all(x -> isapprox(x, 1; atol = 1e-12), vec(sum(abs2, u_kq; dims = 1)))
            # the energies of the pair's states, band by band of its window
            ref_k = compute_electron_states(model, Kpoints(n_pairs, collect(xks), ones(n_pairs), (1, 1, 1)),
                ["eigenvalue"], window; fourier_mode = "normal")
            ref_kq = compute_electron_states(model, Kpoints(n_pairs, [xks[j] + xqs[j] for j in 1:n_pairs],
                ones(n_pairs), (1, 1, 1)), ["eigenvalue"], window; fourier_mode = "normal")
            @test all(eachindex(bands)) do i
                j, n, m = bands[i]
                abs(e_k[i] - ref_k[j].e_full[ref_k[j].rng][n]) < 1e-9 &&
                    abs(e_kq[i] - ref_kq[j].e_full[ref_kq[j].rng][m]) < 1e-9
            end
            # the state maps: the pair's own k, and no k+q selection here
            @test Array(ElectronPhonon.state_k.(dt, ns, js)) == [Array(tile.imap_k)[b[2], iks[b[1]]] for b in bands]
        end
        @test nclasses > 1
    end
end

# `gather_pair_table!` expands a per-q table onto state pairs through the q index of their k points,
# against a plain triple loop: a Float64 table (frequencies) and an Int32 one (ME's frequency ids),
# a rectangular q index (IBZ outer k, full inner k+q) and k indices with repeats (several bands of
# one k point).
@testitem "gather_pair_table!" setup = [EphWindowScatter] begin
    Random.seed!(11)
    nm, nq, nk, nkq = 3, 9, 4, 6
    iq_kk = Int32.(rand(1:nq, nk, nkq))
    iks_i, iks_f = [1, 1, 3, 4, 2], [6, 2, 2, 5, 1, 3, 3]
    for table in (rand(nm, nq), Int32.(rand(1:100, nm, nq)))
        out = zeros(eltype(table), nm, length(iks_i), length(iks_f))
        gather_pair_table!(out, table, iq_kk, iks_i, iks_f)
        @test out == [table[ν, iq_kk[iks_i[i], iks_f[f]]]
                      for ν in 1:nm, i in eachindex(iks_i), f in eachindex(iks_f)]
        if WINDOW_SCATTER_GPU_AVAILABLE
            CUDA.allowscalar(false)
            out_d = CUDA.zeros(eltype(table), size(out)...)
            gather_pair_table!(out_d, CuArray(table), CuArray(iq_kk), CuArray(iks_i),
                               view(CuArray(iks_f), :))
            @test Array(out_d) == out
        end
    end
    @test_throws DimensionMismatch gather_pair_table!(zeros(nm, 2, 7), rand(nm, nq), iq_kk, iks_i,
                                                      iks_f)
    @test_throws DimensionMismatch gather_pair_table!(zeros(nm + 1, 5, 7), rand(nm, nq), iq_kk,
                                                      iks_i, iks_f)
end

# `dedup_phonon_frequencies` groups frequencies by the bin `floor(ω / bin)` and stores the bin centre.
@testitem "dedup_phonon_frequencies" setup = [EphWindowScatter] begin
    Random.seed!(12)
    cutoff = 1e-6
    @testset "bin edges" begin
        # A power-of-two bin makes j*bin, its neighbours and the centre exactly representable, so the
        # keys are known by hand: j*bin -> j, prevfloat -> j-1, nextfloat -> j, (j+1/2)*bin -> j.
        bin = 2.0^-30
        j = 5000
        ω = [j * bin, prevfloat(j * bin), nextfloat(j * bin), (j + 0.5) * bin,
             (j + 1) * bin, prevfloat((j + 1) * bin), 0.5 * cutoff]
        idx, ωvals = dedup_phonon_frequencies(ω, cutoff; bin)
        @test ωvals == [0.0, (j + 0.5) * bin, (j - 0.5) * bin, (j + 1.5) * bin]
        @test idx == Int32[2, 3, 2, 2, 4, 2, 1]
        # Values do not depend on the order of the input.
        perm = randperm(length(ω))
        _, ωvals_p = dedup_phonon_frequencies(ω[perm], cutoff; bin)
        @test sort(ωvals_p) == sort(ωvals)
    end
    @testset "bin 1e-12: error bound, order independence, bin = 0" begin
        # Clusters of near-equal frequencies (spread 1e-15) at 40 centres, repeated, plus sub-cutoff modes.
        centres = 1e-4 .+ 4e-3 .* rand(40)
        ωq = reshape([centres[rand(1:40)] + 1e-15 * randn() for _ in 1:3, _ in 1:500], 3, 500)
        ωq[1, 1:10] .= 0.2cutoff
        bin = 1e-12
        idx, ωvals = dedup_phonon_frequencies(ωq, cutoff; bin)
        above = ωq .>= cutoff
        @test all(idx[.!above] .== 1) && all(idx[above] .>= 2)
        @test maximum(abs.(ωvals[idx[above]] .- ωq[above])) <= bin / 2 + eps(maximum(ωq))
        @test length(ωvals) - 1 <= 2 * length(centres)    # a cluster splits at most at one edge
        @test length(ωvals) - 1 < length(unique(ωq[above]))
        perm = randperm(length(ωq))
        _, ωvals_p = dedup_phonon_frequencies(reshape(vec(ωq)[perm], size(ωq)), cutoff; bin)
        @test sort(ωvals_p) == sort(ωvals)
        # bin = 0 is the serial exact-equality grouping.
        idx0, ωvals0 = dedup_phonon_frequencies(ωq, cutoff; bin = 0)
        ref = unique(vec(ωq)[vec(above)])
        @test ωvals0 == [0.0; ref]
        @test all(ωvals0[idx0] .== ifelse.(above, ωq, 0.0))
        @test_throws ArgumentError dedup_phonon_frequencies(ωq, cutoff; bin = -1e-12)
        @test_throws ArgumentError dedup_phonon_frequencies([1e4], cutoff; bin = 1e-12)
    end
    @testset "phonon_index" begin
        nm, nq, nk, nkq = 3, 9, 4, 6
        ωph = 1e-4 .+ 4e-3 .* rand(nm, nq)
        ωph[1, 3] = 0.1cutoff
        iq_kk = Int32.(rand(1:nq, nk, nkq))
        iks_i, iks_f = [1, 1, 3, 4, 2], [6, 2, 2, 5, 1, 3, 3]
        idx, ωvals = phonon_index(ωph, iq_kk, iks_i, iks_f, cutoff)
        pair_ω = zeros(nm, length(iks_i), length(iks_f))
        gather_pair_table!(pair_ω, ωph, iq_kk, iks_i, iks_f)
        _, ωvals_q = dedup_phonon_frequencies(ωph, cutoff)
        @test ωvals == ωvals_q
        @test maximum(abs.(ωvals[idx] .- ifelse.(pair_ω .>= cutoff, pair_ω, 0.0))) <= 0.5 * ElectronPhonon.omega_bin + eps()
        idx0, ωvals0 = phonon_index(ωph, iq_kk, iks_i, iks_f, cutoff; bin = 0)
        @test ωvals0[idx0] == ifelse.(pair_ω .>= cutoff, pair_ω, 0.0)
        @test_throws ArgumentError phonon_index(ωph, iq_kk .+ Int32(nq), iks_i, iks_f, cutoff)
    end
    if WINDOW_SCATTER_GPU_AVAILABLE
        # The grouping is host-only (as the phonon table is); no EP device code forms keys. This checks
        # that CUDA's `floor(Int64, ω / bin)` broadcast agrees with the host key, edge values included.
        bin = 1e-12
        j = 3_000_000_000
        ω = vcat([j * bin, prevfloat(j * bin), nextfloat(j * bin), (j + 0.5) * bin,
                  prevfloat((j + 1) * bin)], 1e-4 .+ 4e-3 .* rand(10_000))
        CUDA.allowscalar(false)
        @test Array(floor.(Int64, CuArray(ω) ./ bin)) == floor.(Int64, ω ./ bin)
    end
end
