using Test
using ElectronPhonon
using ElectronPhonon: gpu_backend, on_backend, Vec3, CPUBackend, copy_batched_electron_states!,
    compute_electron_states_batched!, unit_to_aru, BatchedWannierInterpolator,
    get_fourier_batched!, eigen_batched, inside_window
using OffsetArrays: no_offset_view

const BATCHED_ELECTRON_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Every quantity of `els` at its in-window bands against the `ElectronState`s `ref`, with
# `isequal`; returns the number of k points that differ. Box entries past `nband` are undefined and not read.
function _batched_electron_mismatches(els, ref)
    off, nband = Array(els.iband_offset), Array(els.nband)
    count(eachindex(ref)) do ik
        el = ref[ik]; nb = el.nband
        ok = nband[ik] == nb && (nb == 0 || off[ik] == first(el.rng) - 1)
        els.e === nothing || (ok &= isequal(Array(els.e)[1:nb, ik], collect(el.e)))
        els.u === nothing || (ok &= isequal(Array(els.u)[:, 1:nb, ik], collect(no_offset_view(el.u))))
        els.vdiag === nothing || (ok &= isequal(Array(els.vdiag)[:, 1:nb, ik],
                                              reshape(reinterpret(Float64, collect(el.vdiag)), 3, nb)))
        for (x, y) in ((els.v, el.v), (els.rbar, el.rbar))
            x === nothing && continue
            ok &= isequal(Array(x)[:, 1:nb, 1:nb, ik],
                          reshape(reinterpret(ComplexF64, collect(no_offset_view(y))), 3, nb, nb))
        end
        !ok
    end
end

@testset "BatchedElectronState" begin
    model_pb = _load_model_from_artifacts("pb"; load_epmat = false)
    # Nonzero position matrix elements, which the pb model does not have.
    model_bn = _load_model_from_artifacts("cubicBN"; load_epmat = false)
    kpts = kpoints_grid((4, 4, 4))
    e_F = 11.68 * unit_to_aru(:eV)
    # Narrow: one band per k at the surviving k. Wide: 0 to 2 bands per k, offsets 0 and 1.
    window_narrow = (e_F - 0.2 * unit_to_aru(:eV), e_F + 0.2 * unit_to_aru(:eV))
    window_wide = (e_F - 2.0 * unit_to_aru(:eV), e_F + 0.5 * unit_to_aru(:eV))
    sel = filter_electron_states((12, 12, 12), model_pb.nw, model_pb.el_ham, window_narrow;
                                 fourier_mode = "gridopt")
    per_point_name = Dict(:e => "eigenvalue", :u => "eigenvector",
        :vdiag => "velocity_diagonal", :v => "velocity", :rbar => "position")
    per_point(qs) = [per_point_name[x] for x in qs]
    quantity_lists = ([:e], [:e, :u], [:e, :u, :vdiag], [:e, :u, :v, :vdiag],
                      [:e, :u, :vdiag, :v, :rbar], [:u, :rbar], [:vdiag])

    @testset "host builder == compute_electron_states" begin
        # The per-point builder is the oracle: both run the same per-k kernels on the same
        # in-window eigenvectors, so every stored quantity equals its `ElectronState` field bit for
        # bit, for a selection, an energy window (band-ragged) and the full band.
        nbad = 0
        for m in (model_pb, model_bn), mode in (:Direct, :BerryConnection),
                fourier_mode in ("normal", "gridopt"), quantities in quantity_lists
            m.el_velocity_mode = mode
            inputs = m === model_pb ? (((sel,), ()), ((kpts,), (window_wide,)), ((kpts,), ())) :
                                      (((kpts,), ()),)
            for (args, window) in inputs
                els = compute_electron_states_batched(m, args..., quantities, window...; fourier_mode)
                ref = compute_electron_states(m, args..., per_point(quantities), window...;
                                              fourier_mode)
                nbad += _batched_electron_mismatches(els, ref)
            end
            m.el_velocity_mode = :Direct
        end
        @test nbad == 0
        # The comparison has teeth: the windows are band-ragged, and cubicBN's position matrix
        # elements are not zero.
        els = compute_electron_states_batched(model_pb, kpts, [:e], window_wide)
        @test extrema(els.nband) == (0, 2) && extrema(els.iband_offset) == (0, 1) && els.nband_max == 2
        @test maximum(abs, compute_electron_states_batched(model_bn, kpts, [:rbar]).rbar) > 0.1

        # With a cache: the eigenpairs are copied out per k, exactly as the per-point builder does.
        cache = electron_eigenpairs(model_pb, kpts; fourier_mode = "normal")
        @test all(quantity_lists) do quantities
            els = compute_electron_states_batched(model_pb, kpts, quantities, window_wide;
                                                eigenpairs = cache)
            ref = compute_electron_states(model_pb, kpts, per_point(quantities), window_wide;
                                          eigenpairs = cache)
            _batched_electron_mismatches(els, ref) == 0
        end
    end

    @testset "fields, conversions, checks" begin
        els = compute_electron_states_batched(model_pb, sel, [:e, :vdiag])
        @test els.kpts === sel.kpts && els.nk == sel.kpts.n && els.nw == model_pb.nw
        @test els.u === nothing && els.v === nothing && els.rbar === nothing
        @test occursin("BatchedElectronState{Float64}(nw = 4, nband_max = 1", sprint(show, els))

        # `BandStates` from the batched states equals the one from the per-point states.
        bs = BandStates(els, sel)
        bs_ref, _ = electron_states_to_BandStates(
            compute_electron_states(model_pb, sel, ["eigenvalue", "velocity_diagonal"]), sel)
        @test bs.es == bs_ref.es && bs.vs == bs_ref.vs && bs.iks == bs_ref.iks
        @test isempty(BandStates(compute_electron_states_batched(model_pb, sel, [:e]), sel).vs)
        fb = electron_states_to_FilteredBandStates(kpts,
            compute_electron_states_batched(model_pb, kpts, [:e], window_wide), 0.0; nw = 4)
        fb_ref = electron_states_to_FilteredBandStates(kpts,
            compute_electron_states(model_pb, kpts, ["eigenvalue"], window_wide), 0.0; nw = 4)
        @test fb.iks == fb_ref.iks && fb.ibands == fb_ref.ibands && fb.n == 24

        @test_throws "unknown electron quantities [:velocity]" compute_electron_states_batched(
            model_pb, kpts, [:velocity])
        @test_throws "duplicates" compute_electron_states_batched(model_pb, kpts, [:e, :e])
        @test_throws "e must be (nband_max, nk)" BatchedElectronState{Float64}(4, 1, 216, sel.kpts,
            els.iband_offset, els.nband, els.e[:, 1:2], nothing, nothing, nothing, nothing)
    end

    @testset "empty container, copy and the in-tile builder" begin
        els = compute_electron_states_batched(model_pb, kpts, [:e, :u], window_wide)
        inds = [5, 2, 64, 17]
        tile = BatchedElectronState(CPUBackend(), 4, els.nband_max, 6, [:e, :u])
        @test tile.kpts === nothing && tile.nk == 6 && tile.vdiag === nothing
        copy_batched_electron_states!(tile, els, inds)
        # The copy takes the undefined padding along with the rest of a column, hence isequal.
        @test isequal(tile.e[:, 1:4], els.e[:, inds]) && isequal(tile.u[:, :, 1:4], els.u[:, :, inds])
        @test tile.iband_offset[1:4] == els.iband_offset[inds] && tile.nband[1:4] == els.nband[inds]
        @test_throws "cannot copy 4 points" copy_batched_electron_states!(
            BatchedElectronState(CPUBackend(), 4, 4, 6, [:e, :u]), els, inds)

        # Solved into a tile: the bands of the window, moved to local bands 1:nband, of the same
        # batched solve, in a box as wide as the largest window. Off-grid points, more tile than
        # points.
        xks = kpts.vectors[1:50] .+ Ref(Vec3(0.013, 0.02, -0.01))
        backends = Any[CPUBackend()]
        BATCHED_ELECTRON_GPU_AVAILABLE && push!(backends, gpu_backend())
        for backend in backends, window in (window_wide, window_narrow, (-Inf, Inf))
            nw = model_pb.nw
            tile = BatchedElectronState(backend, nw, nw, 53, [:e, :u])
            itp = BatchedWannierInterpolator(ElectronPhonon.to_device(backend, model_pb.el_ham);
                                             backend, batch_size = tile.nk)
            out = compute_electron_states_batched!(tile, itp, ElectronPhonon.alloc(backend, ComplexF64,
                nw^2, tile.nk), model_pb, xks, window)
            hk = ElectronPhonon.alloc(backend, ComplexF64, nw^2, length(xks))
            get_fourier_batched!(hk, itp, xks)
            E, U = Array.(eigen_batched(reshape(hk, nw, nw, :)))
            e, u = Array(out.e), Array(out.u)
            off, nband = Array(out.iband_offset), Array(out.nband)
            @test out.nk == length(xks) && out.nband_max == max(maximum(nband), 1)
            # The box is the leading memory of the tile's buffers.
            @test vec(Array(out.u)) == vec(Array(tile.u))[1:length(out.u)]
            @test all(eachindex(xks)) do j
                r = inside_window(E[:, j], window...)
                issorted(E[:, j]) && nband[j] == length(r) && (isempty(r) || off[j] == first(r) - 1) &&
                    e[1:nband[j], j] == E[r, j] && u[:, 1:nband[j], j] == U[:, r, j]
            end
        end
        itp = BatchedWannierInterpolator(model_pb.el_ham; batch_size = 6)
        @test_throws "nband_max = nw = 4" compute_electron_states_batched!(
            BatchedElectronState(CPUBackend(), 4, 2, 6, [:e, :u]), itp, zeros(ComplexF64, 16, 6),
            model_pb, xks[1:2], window_wide)
    end

    @testset "GPU" begin
        if BATCHED_ELECTRON_GPU_AVAILABLE
            CUDA.allowscalar(false)
            backend = gpu_backend()
            # The device builder runs the per-point device arm's solve and rotations without the
            # copy back, so it equals it bit for bit.
            nbad = 0
            for mode in (:Direct, :BerryConnection), quantities in ([:e], [:e, :u],
                    [:e, :u, :vdiag], [:vdiag])
                model_pb.el_velocity_mode = mode
                for (args, window) in (((sel,), ()), ((kpts,), (window_wide,)), ((kpts,), ()))
                    els = compute_electron_states_batched(model_pb, args..., quantities, window...;
                                                        backend)
                    @test all(x -> x === nothing || on_backend(backend, x),
                              (els.iband_offset, els.nband, els.e, els.u, els.vdiag))
                    ref = compute_electron_states(model_pb, args..., per_point(quantities),
                                                  window...; backend)
                    nbad += _batched_electron_mismatches(els, ref)
                end
                model_pb.el_velocity_mode = :Direct
            end
            @test nbad == 0
            cache = electron_eigenpairs(model_pb, kpts; backend)
            els = compute_electron_states_batched(model_pb, kpts, [:e, :u, :vdiag], window_wide;
                                                backend, eigenpairs = cache)
            @test _batched_electron_mismatches(els, compute_electron_states(model_pb, kpts,
                per_point([:e, :u, :vdiag]), window_wide; backend, eigenpairs = cache)) == 0

            # A host container streamed into a device tile, and a device one copied in place.
            b_host = compute_electron_states_batched(model_pb, kpts, [:e, :u], window_wide)
            inds = [5, 2, 64, 17]
            for els_src in (b_host, compute_electron_states_batched(model_pb, kpts, [:e, :u],
                                                                window_wide; backend))
                tile = BatchedElectronState(backend, 4, els_src.nband_max, 6, [:e, :u])
                copy_batched_electron_states!(tile, els_src, inds)
                @test Array(tile.nband)[1:4] == Array(els_src.nband)[inds]
                @test isequal(Array(tile.u)[:, :, 1:4], Array(els_src.u)[:, :, inds])
            end
            @test_throws "quantities [:v] are not supported" compute_electron_states_batched(
                model_pb, kpts, [:e, :v]; backend)
        end
    end
end
