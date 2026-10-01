using Test
using ElectronPhonon
using ElectronPhonon: gpu_backend, on_backend, Vec3, alloc_tile, stage!, quantity_arrays,
    CPUBackend, unfold_rule, builder_scratch, builder_scratch_bytes, compute_electron_states_batched!,
    unit_to_aru, BatchedWannierInterpolator, get_fourier_batched!, eigen_batched, _band_window
using OffsetArrays: no_offset_view

const BATCHED_ELECTRON_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Every quantity of `b` at its in-window bands against the `ElectronState`s `ref`, with `isequal`;
# returns the number of k points that differ. Box entries past `nband` are undefined and not read.
function _batched_electron_mismatches(b, ref)
    off, nband = Array(b.iband_offset), Array(b.nband)
    count(eachindex(ref)) do ik
        el = ref[ik]; nb = el.nband
        ok = nband[ik] == nb && (nb == 0 || off[ik] == first(el.rng) - 1)
        q = keys(b.qty)
        :e ∈ q && (ok &= isequal(Array(b.e)[1:nb, ik], collect(el.e)))
        :u ∈ q && (ok &= isequal(Array(b.u)[:, 1:nb, ik], collect(no_offset_view(el.u))))
        :vdiag ∈ q && (ok &= isequal(Array(b.vdiag)[:, 1:nb, ik],
                                     reshape(reinterpret(Float64, collect(el.vdiag)), 3, nb)))
        for name in (:v, :rbar)
            name ∈ q || continue
            ok &= isequal(Array(getproperty(b, name))[:, 1:nb, 1:nb, ik],
                reshape(reinterpret(ComplexF64, collect(no_offset_view(getproperty(el, name)))),
                        3, nb, nb))
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
                b = compute_electron_states_batched(m, args..., quantities, window...; fourier_mode)
                ref = compute_electron_states(m, args..., per_point(quantities), window...;
                                              fourier_mode)
                nbad += _batched_electron_mismatches(b, ref)
            end
            m.el_velocity_mode = :Direct
        end
        @test nbad == 0
        # The comparison has teeth: the windows are band-ragged, and cubicBN's position matrix
        # elements are not zero.
        b = compute_electron_states_batched(model_pb, kpts, [:e], window_wide)
        @test extrema(b.nband) == (0, 2) && extrema(b.iband_offset) == (0, 1) && b.nbox == 2
        @test maximum(abs, compute_electron_states_batched(model_bn, kpts, [:rbar]).rbar) > 0.1

        # With a cache: the eigenpairs are copied out per k, exactly as the per-point builder does.
        cache = electron_eigenpairs(model_pb, kpts; fourier_mode = "normal")
        @test all(quantity_lists) do quantities
            b = compute_electron_states_batched(model_pb, kpts, quantities, window_wide;
                                                eigenpairs = cache)
            ref = compute_electron_states(model_pb, kpts, per_point(quantities), window_wide;
                                          eigenpairs = cache)
            _batched_electron_mismatches(b, ref) == 0
        end
    end

    @testset "quantities by name, conversions, checks" begin
        b = compute_electron_states_batched(model_pb, sel, [:e, :vdiag])
        @test b.kpts === sel.kpts && b.sel === sel && b.n == sel.kpts.n && b.nw == model_pb.nw
        @test_throws "quantity :u was not requested" b.u
        @test quantity_arrays(b) === (b.e, b.vdiag, b.iband_offset, b.nband)
        @test occursin("BatchedElectronState{Float64}(nw = 4, nbox = 1", sprint(show, b))

        # `BandStates` from the batched states equals the one from the per-point states.
        bs, _ = electron_states_to_BandStates(b, sel)
        bs_ref, _ = electron_states_to_BandStates(
            compute_electron_states(model_pb, sel, ["eigenvalue", "velocity_diagonal"]), sel)
        @test bs.es == bs_ref.es && bs.vs == bs_ref.vs && bs.iks == bs_ref.iks
        @test isempty(electron_states_to_BandStates(
            compute_electron_states_batched(model_pb, sel, [:e]), sel)[1].vs)
        fb = electron_states_to_FilteredBandStates(kpts,
            compute_electron_states_batched(model_pb, kpts, [:e], window_wide), 0.0; nw = 4)
        fb_ref = electron_states_to_FilteredBandStates(kpts,
            compute_electron_states(model_pb, kpts, ["eigenvalue"], window_wide), 0.0; nw = 4)
        @test fb.iks == fb_ref.iks && fb.ibands == fb_ref.ibands && fb.n == 24

        @test unfold_rule(:e) === :none && unfold_rule(:u) === :gauge &&
              unfold_rule(:vdiag) === :cartesian && unfold_rule(:v) === :cartesian
        @test_throws "no symmetry unfolding rule" unfold_rule(:rbar)

        @test_throws "unknown electron quantity :velocity" compute_electron_states_batched(
            model_pb, kpts, [:velocity])
        @test_throws "duplicates" compute_electron_states_batched(model_pb, kpts, [:e, :e])
        @test_throws "quantity :e must be a Float64 array of size (1, 216)" BatchedElectronState{
            Float64}(4, 1, 216, sel.kpts, sel, b.iband_offset, b.nband, (; e = b.e[:, 1:2]))
        @test_throws "unknown electron quantity :nband" BatchedElectronState{Float64}(4, 1, 216,
            sel.kpts, sel, b.iband_offset, b.nband, (; nband = b.e))
    end

    @testset "tile, stage! and the in-tile builder" begin
        b = compute_electron_states_batched(model_pb, kpts, [:e, :u], window_wide)
        inds = [5, 2, 64, 17]
        tile = alloc_tile(b, CPUBackend(), 6)
        @test tile.kpts === nothing && tile.sel === nothing && tile.nbox == b.nbox
        stage!(tile, b, inds)
        # The gather copies the undefined padding along with the rest of a column, hence isequal.
        @test isequal(tile.e[:, 1:4], b.e[:, inds]) && isequal(tile.u[:, :, 1:4], b.u[:, :, inds])
        @test tile.iband_offset[1:4] == b.iband_offset[inds] && tile.nband[1:4] == b.nband[inds]
        @test_throws "(nw, nbox) = (4, 4), the source (4, 2)" stage!(alloc_tile(
            compute_electron_states_batched(model_pb, kpts, [:e, :u]), CPUBackend(), 6), b, inds)
        @test_throws "cannot stage" stage!(tile,
            compute_phonon_states_batched(model_pb, kpts, [:e, :u]), inds)

        # Solved into a tile: the bands of the window, moved to local bands 1:nband, of the same
        # batched solve. Off-grid points, more tile than points.
        xks = kpts.vectors[1:50] .+ Ref(Vec3(0.013, 0.02, -0.01))
        backends = Any[CPUBackend()]
        BATCHED_ELECTRON_GPU_AVAILABLE && push!(backends, gpu_backend())
        for backend in backends, window in (window_wide, window_narrow, (-Inf, Inf))
            nw = model_pb.nw
            tile = alloc_tile(compute_electron_states_batched(model_pb, kpts, [:e, :u]), backend,
                              53)
            scratch = builder_scratch(model_pb, backend, [:e, :u], tile.n)
            compute_electron_states_batched!(tile, scratch, model_pb, xks, [:e, :u], window)
            itp = BatchedWannierInterpolator(ElectronPhonon.to_device(backend, model_pb.el_ham);
                                             backend, batch_size = tile.n)
            hk = ElectronPhonon.alloc(backend, ComplexF64, nw^2, length(xks))
            get_fourier_batched!(hk, itp, xks)
            E, U = Array.(eigen_batched(reshape(hk, nw, nw, :)))
            e, u = Array(tile.e), Array(tile.u)
            off, nband = Array(tile.iband_offset), Array(tile.nband)
            @test all(eachindex(xks)) do j
                r = _band_window(E[:, j], window)
                issorted(E[:, j]) && nband[j] == length(r) && (isempty(r) || off[j] == first(r) - 1) &&
                    e[1:nband[j], j] == E[r, j] && u[:, 1:nband[j], j] == U[:, r, j]
            end
        end
        @test builder_scratch_bytes(model_pb, [:e, :u], 10) > 0
        @test_throws "box width nw = 4" compute_electron_states_batched!(alloc_tile(b, CPUBackend(), 6),
            builder_scratch(model_pb, CPUBackend(), [:e, :u], 6), model_pb, xks[1:2], [:e, :u],
            window_wide)
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
                    b = compute_electron_states_batched(model_pb, args..., quantities, window...;
                                                        backend)
                    @test all(x -> on_backend(backend, x), quantity_arrays(b))
                    ref = compute_electron_states(model_pb, args..., per_point(quantities),
                                                  window...; backend)
                    nbad += _batched_electron_mismatches(b, ref)
                end
                model_pb.el_velocity_mode = :Direct
            end
            @test nbad == 0
            cache = electron_eigenpairs(model_pb, kpts; backend)
            b = compute_electron_states_batched(model_pb, kpts, [:e, :u, :vdiag], window_wide;
                                                backend, eigenpairs = cache)
            @test _batched_electron_mismatches(b, compute_electron_states(model_pb, kpts,
                per_point([:e, :u, :vdiag]), window_wide; backend, eigenpairs = cache)) == 0

            # A host container streamed into a device tile, and a device one gathered in place.
            b_host = compute_electron_states_batched(model_pb, kpts, [:e, :u], window_wide)
            inds = [5, 2, 64, 17]
            for src in (b_host, compute_electron_states_batched(model_pb, kpts, [:e, :u],
                                                                window_wide; backend))
                tile = alloc_tile(src, backend, 6)
                stage!(tile, src, inds)
                @test Array(tile.nband)[1:4] == Array(src.nband)[inds]
                @test isequal(Array(tile.u)[:, :, 1:4], Array(src.u)[:, :, inds])
            end
            @test_throws "quantity :v is not supported" compute_electron_states_batched(
                model_pb, kpts, [:e, :v]; backend)
        end
    end
end
