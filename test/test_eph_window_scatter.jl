using Test
using ElectronPhonon
using ElectronPhonon: eph_window_scatter!, eph_window_scatter_reim!, gather_pair_table!
using Random

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const WINDOW_SCATTER_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Independent reference for both scatters: the same window lookup written as a dense
# (nm, n_i, n_f) write, without the linear-index arithmetic the implementations use. `w` holds the
# block frequency `ωq[ν, j]` at each written slot, zero elsewhere.
function _scatter_reference(ep, ωq, imap_i_col, imap_f, ikqs, n_i, n_f, i0)
    nbandkq, nbandk, nm, npairs = size(ep)
    re = zeros(nm, n_i, n_f); im_out = zeros(nm, n_i, n_f); w = zeros(nm, n_i, n_f)
    for j in 1:npairs, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[j]]
        (i > 0 && f > 0) || continue
        re[ν, i - i0, f] = real(ep[m, n, ν, j])
        im_out[ν, i - i0, f] = imag(ep[m, n, ν, j])
        w[ν, i - i0, f] = ωq[ν, j]
    end
    (re, im_out, w)
end

@testset "eph_window_scatter_reim!" begin
    Random.seed!(7)
    nbandkq, nbandk, nm, nkq, npairs = 3, 2, 4, 7, 5
    # One outer band in the window and one outside it, so the `i > 0` branch is exercised.
    imap_i_col, n_i = [3, 0], 3
    # Each (band, k+q point) that is in the window gets its own inner state index, which is the
    # non-collision invariant the scatters rely on: distinct k+q -> distinct f, so no two writes
    # can target the same slot. The q points of one batch are distinct k+q points for the same
    # reason, hence `randperm` rather than `rand`.
    imap_f = zeros(Int, nbandkq, nkq)
    n_f = 0
    for j in eachindex(imap_f)
        if rand() < 0.75
            n_f += 1
            imap_f[j] = n_f
        end
    end
    ikqs = randperm(nkq)[1:npairs]
    ep = randn(ComplexF64, nbandkq, nbandk, nm, npairs)
    ωq = rand(nm, npairs) .+ 0.5

    N = nm * n_i * n_f
    re, im_out = (zeros(N), zeros(N))
    eph_window_scatter_reim!(re, im_out, ep, imap_i_col, imap_f, ikqs, n_i, 0)
    re_ref, im_ref, w_ref = _scatter_reference(ep, ωq, imap_i_col, imap_f, ikqs, n_i, n_f, 0)
    @test reshape(re, nm, n_i, n_f) == re_ref
    @test reshape(im_out, nm, n_i, n_f) == im_ref
    # Vacuity guard: the window maps must leave some slots written and some untouched, or the
    # equalities above would hold for an implementation that writes nothing.
    w = vec(w_ref)
    @test 0 < count(!iszero, w) < N

    # The relation the four-channel caller is built on: `(re² + im²)/(2ω)` is `abs2(ep)/(2ω)`, the
    # value today's `eph_window_scatter!` writes, and it is exact -- `abs2` is that sum of squares
    # and both sides divide by the same `2ω`. So a `Re/Im` sweep can reproduce a `g2` sweep bit for
    # bit, which is what lets a generic-vertex run be regression-tested against a TRS one.
    g2 = zeros(N)
    eph_window_scatter!(g2, abs2.(ep) ./ (2 .* reshape(ωq, 1, 1, nm, npairs)),
                        imap_i_col, imap_f, ikqs, n_i, 0)
    written = w .!= 0
    @test (re[written].^2 .+ im_out[written].^2) ./ (2 .* w[written]) == g2[written]
    @test all(iszero, g2[.!written])

    # Per-batch buffer: the output holds only the current outer-k batch, so global state `i` lands
    # at local row `i - i0`. Here that is the single row of a one-state batch.
    i0, ni_stride = 2, 1
    re_b, im_b = (zeros(nm * ni_stride * n_f) for _ in 1:2)
    eph_window_scatter_reim!(re_b, im_b, ep, imap_i_col, imap_f, ikqs, ni_stride, i0)
    re_bref, im_bref, _ = _scatter_reference(ep, ωq, imap_i_col, imap_f, ikqs, ni_stride, n_f, i0)
    @test reshape(re_b, nm, ni_stride, n_f) == re_bref
    @test reshape(im_b, nm, ni_stride, n_f) == im_bref

    @testset "GPU" begin
        if WINDOW_SCATTER_GPU_AVAILABLE
            # The CUDA method must agree with the generic one bit for bit: the writes are
            # collision-free, so the thread order cannot change any value.
            to_dev(x) = CuArray(x)
            re_d, im_d = (CUDA.zeros(Float64, N) for _ in 1:2)
            eph_window_scatter_reim!(re_d, im_d, to_dev(ep), to_dev(imap_i_col),
                                     to_dev(imap_f), to_dev(ikqs), n_i, 0)
            @test Array(re_d) == re
            @test Array(im_d) == im_out

            # The per-batch slot arithmetic, which the CUDA kernel spells out a second time by
            # hand: the same case as the CPU arm above, so a transcription slip in either copy of
            # `lin = ν + nm(i-i0-1) + nm·ni_stride(f-1)` shows up here.
            re_bd, im_bd = (CUDA.zeros(Float64, nm * ni_stride * n_f) for _ in 1:2)
            eph_window_scatter_reim!(re_bd, im_bd, to_dev(ep), to_dev(imap_i_col),
                                     to_dev(imap_f), to_dev(ikqs), ni_stride, i0)
            @test Array(re_bd) == re_b
            @test Array(im_bd) == im_b

            # Production hands this the batch's own q slice and a tile of a larger output buffer,
            # both as CONTIGUOUS views -- which of a `CuArray` are `CuArray`s again, so they take
            # the method below. (A genuinely strided output `SubArray` would not: it falls through
            # to the generic method and scalar-indexes device memory.) What this checks is the
            # offset: the helper writes inside its view and leaves the surrounding buffer alone.
            ep_pad = CUDA.zeros(ComplexF64, nbandkq, nbandk, nm, npairs + 2)
            copyto!(view(ep_pad, :, :, :, 1:npairs), ep)
            pads = [CUDA.zeros(Float64, 3N) for _ in 1:2]
            views = [view(pad, N+1:2N) for pad in pads]
            eph_window_scatter_reim!(views[1], views[2],
                                     view(ep_pad, :, :, :, 1:npairs), to_dev(imap_i_col),
                                     to_dev(imap_f), to_dev(ikqs), n_i, 0)
            @test Array(pads[1]) == vcat(zeros(N), re, zeros(N))
            @test Array(pads[2]) == vcat(zeros(N), im_out, zeros(N))
        else
            @info "CUDA not functional - skipping the GPU eph_window_scatter_reim! test"
        end
    end
end

# The scatters loop over the extents of their block values unchecked (`@inbounds` on the host, no
# bounds checks in the device kernels). A block value buffer at its full width, handed over with
# the k+q list and frequencies of fewer points, must be refused rather than read past the block.
@testset "scatters refuse block arrays of mismatched extent" begin
    nbandkq, nbandk, nm, npairs, nkq = 3, 2, 2, 4, 6
    imap_i_col, imap_f = [1, 2], reshape(collect(1:nbandkq * nkq), nbandkq, nkq)
    ikqs, ωq = collect(1:npairs), fill(0.01, nm, npairs)
    full = rand(nbandkq, nbandk, nm, npairs + 2)
    N = nm * 2 * nbandkq * nkq
    @test_throws DimensionMismatch eph_window_scatter!(zeros(N), full, imap_i_col, imap_f, ikqs, 2, 0)
    @test_throws DimensionMismatch eph_window_scatter_reim!(zeros(N), zeros(N), complex.(full),
        imap_i_col, imap_f, ikqs, 2, 0)
    @test_throws DimensionMismatch ElectronPhonon.bte_window_accumulate!(zeros(2, 1),
        zeros(2, nbandkq * nkq, 1), full, ωq, imap_i_col, imap_f, ikqs, zeros(2),
        zeros(nbandkq * nkq), ones(nbandkq * nkq), [0.0], [0.01],
        [SmearingType(:Gaussian, 0.005)], 5, 0.0, 0)
end

# `gather_pair_table!` expands a per-q table onto state pairs through the q index of their k points,
# against a plain triple loop: a Float64 table (frequencies) and an Int32 one (ME's frequency ids),
# a rectangular q index (IBZ outer k, full inner k+q) and k indices with repeats (several bands of
# one k point).
@testset "gather_pair_table!" begin
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
