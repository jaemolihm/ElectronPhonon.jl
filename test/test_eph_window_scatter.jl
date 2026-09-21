using Test
using ElectronPhonon
using ElectronPhonon: eph_window_scatter!, eph_window_scatter_reim!
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
# (nm, n_i, n_f) write, without the linear-index arithmetic the implementations use.
function _scatter_reference(ep, ωq, imap_i_col, imap_f, ikqs, n_i, n_f, i0)
    nbandkq, nbandk, nm, nq_batch = size(ep)
    re = zeros(nm, n_i, n_f); im_out = zeros(nm, n_i, n_f); w = zeros(nm, n_i, n_f)
    for j in 1:nq_batch, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
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
    nbandkq, nbandk, nm, nkq, nq_batch = 3, 2, 4, 7, 5
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
    ikqs = randperm(nkq)[1:nq_batch]
    ep = randn(ComplexF64, nbandkq, nbandk, nm, nq_batch)
    ωq = rand(nm, nq_batch) .+ 0.5

    N = nm * n_i * n_f
    re, im_out, w = (zeros(N), zeros(N), zeros(N))
    eph_window_scatter_reim!(re, im_out, w, ep, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq, nbandk, nm, nq_batch, n_i, 0)
    re_ref, im_ref, w_ref = _scatter_reference(ep, ωq, imap_i_col, imap_f, ikqs, n_i, n_f, 0)
    @test reshape(re, nm, n_i, n_f) == re_ref
    @test reshape(im_out, nm, n_i, n_f) == im_ref
    @test reshape(w, nm, n_i, n_f) == w_ref
    # Vacuity guard: the window maps must leave some slots written and some untouched, or the
    # equalities above would hold for an implementation that writes nothing.
    @test 0 < count(!iszero, w) < N

    # The relation the four-channel caller is built on: `(re² + im²)/(2ω)` is `abs2(ep)/(2ω)`, the
    # value today's `eph_window_scatter!` writes, and it is exact -- `abs2` is that sum of squares
    # and both sides divide by the same `2ω`. So a `Re/Im` sweep can reproduce a `g2` sweep bit for
    # bit, which is what lets a generic-vertex run be regression-tested against a TRS one.
    g2, w2 = (zeros(N), zeros(N))
    eph_window_scatter!(g2, w2, abs2.(ep) ./ (2 .* reshape(ωq, 1, 1, nm, nq_batch)),
                        imap_i_col, imap_f, ikqs, ωq,
                        nbandkq, nbandk, nm, nq_batch, n_i, 0)
    written = w .!= 0
    @test (re[written].^2 .+ im_out[written].^2) ./ (2 .* w[written]) == g2[written]
    @test w == w2

    # `ωq_out === nothing` writes the same Re/Im and touches no frequency buffer: the second sweep
    # of a two-run caller, which already has the frequencies from the first.
    re_noω, im_noω = (zeros(N), zeros(N))
    eph_window_scatter_reim!(re_noω, im_noω, nothing, ep, imap_i_col, imap_f, ikqs, nothing,
                             nbandkq, nbandk, nm, nq_batch, n_i, 0)
    @test re_noω == re
    @test im_noω == im_out

    # Per-batch buffer: the output holds only the current outer-k batch, so global state `i` lands
    # at local row `i - i0`. Here that is the single row of a one-state batch.
    i0, ni_stride = 2, 1
    re_b, im_b, w_b = (zeros(nm * ni_stride * n_f) for _ in 1:3)
    eph_window_scatter_reim!(re_b, im_b, w_b, ep, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq, nbandk, nm, nq_batch, ni_stride, i0)
    re_bref, im_bref, w_bref = _scatter_reference(ep, ωq, imap_i_col, imap_f, ikqs, ni_stride,
                                                  n_f, i0)
    @test reshape(re_b, nm, ni_stride, n_f) == re_bref
    @test reshape(im_b, nm, ni_stride, n_f) == im_bref
    @test reshape(w_b, nm, ni_stride, n_f) == w_bref

    @testset "GPU" begin
        if WINDOW_SCATTER_GPU_AVAILABLE
            # The CUDA method must agree with the generic one bit for bit: the writes are
            # collision-free, so the thread order cannot change any value.
            to_dev(x) = CuArray(x)
            re_d, im_d, w_d = (CUDA.zeros(Float64, N) for _ in 1:3)
            eph_window_scatter_reim!(re_d, im_d, w_d, to_dev(ep), to_dev(imap_i_col),
                                     to_dev(imap_f), to_dev(ikqs), to_dev(ωq),
                                     nbandkq, nbandk, nm, nq_batch, n_i, 0)
            @test Array(re_d) == re
            @test Array(im_d) == im_out
            @test Array(w_d) == w

            # The per-batch slot arithmetic, which the CUDA kernel spells out a second time by
            # hand: the same case as the CPU arm above, so a transcription slip in either copy of
            # `lin = ν + nm(i-i0-1) + nm·ni_stride(f-1)` shows up here.
            re_bd, im_bd, w_bd = (CUDA.zeros(Float64, nm * ni_stride * n_f) for _ in 1:3)
            eph_window_scatter_reim!(re_bd, im_bd, w_bd, to_dev(ep), to_dev(imap_i_col),
                                     to_dev(imap_f), to_dev(ikqs), to_dev(ωq),
                                     nbandkq, nbandk, nm, nq_batch, ni_stride, i0)
            @test Array(re_bd) == re_b
            @test Array(im_bd) == im_b
            @test Array(w_bd) == w_b

            # Production hands this the batch's own q slice and a tile of a larger output buffer,
            # both as CONTIGUOUS views -- which of a `CuArray` are `CuArray`s again, so they take
            # the method below. (A genuinely strided output `SubArray` would not: it falls through
            # to the generic method and scalar-indexes device memory.) What this checks is the
            # offset: the helper writes inside its view and leaves the surrounding buffer alone.
            ep_pad = CUDA.zeros(ComplexF64, nbandkq, nbandk, nm, nq_batch + 2)
            copyto!(view(ep_pad, :, :, :, 1:nq_batch), ep)
            pads = [CUDA.zeros(Float64, 3N) for _ in 1:3]
            views = [view(pad, N+1:2N) for pad in pads]
            eph_window_scatter_reim!(views[1], views[2], views[3],
                                     view(ep_pad, :, :, :, 1:nq_batch), to_dev(imap_i_col),
                                     to_dev(imap_f), to_dev(ikqs), to_dev(ωq),
                                     nbandkq, nbandk, nm, nq_batch, n_i, 0)
            @test Array(pads[1]) == vcat(zeros(N), re, zeros(N))
            @test Array(pads[2]) == vcat(zeros(N), im_out, zeros(N))
            @test Array(pads[3]) == vcat(zeros(N), w, zeros(N))

            # `nothing` is a singleton type, so the no-frequency launch compiles to its own kernel.
            re_n, im_n = (CUDA.zeros(Float64, N) for _ in 1:2)
            eph_window_scatter_reim!(re_n, im_n, nothing, to_dev(ep), to_dev(imap_i_col),
                                     to_dev(imap_f), to_dev(ikqs), nothing,
                                     nbandkq, nbandk, nm, nq_batch, n_i, 0)
            @test Array(re_n) == re
            @test Array(im_n) == im_out
        else
            @info "CUDA not functional - skipping the GPU eph_window_scatter_reim! test"
        end
    end
end
