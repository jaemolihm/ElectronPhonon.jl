using Test
using ElectronPhonon
using ElectronPhonon: Vec3, electron_degen_cutoff, electron_eigenpairs, _eigenpair_index,
    gpu_backend, to_device, AbstractBackend, CPUBackend
using LinearAlgebra
using OffsetArrays: no_offset_view

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const EIGENPAIRS_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# Every field a run fills, compared with `==`: `v`/`rbar`/`occupation` are window views, so take
# them through `no_offset_view` (their axes are covered by `rng`).
# The `quantities == ["eigenvalue"]` branch is the one case that cannot be bitwise equal to its
# no-cache counterpart: without a cache it runs the value-only LAPACK driver, while a cache holds
# the eigenvalues of the full eigensolve, and the two agree only to round-off. What the cache branch
# must do exactly is copy the cached column and leave `u_full` alone.
function _eigenpairs_valueonly_consistent(new, ref, cache)
    all(eachindex(new)) do ik
        el, el_ref = new[ik], ref[ik]
        el.xk == el_ref.xk && el.rng == el_ref.rng && el.nband == el_ref.nband &&
            el.e_full == cache.e[:, _eigenpair_index(cache, el.xk)] &&
            all(iszero, el.u_full) &&
            maximum(abs, el.e_full - el_ref.e_full) < 1e-13
    end
end

function _eigenpairs_state_equal(a::ElectronState, b::ElectronState)
    a.xk == b.xk && a.e_full == b.e_full && a.u_full == b.u_full && a.nband == b.nband &&
        a.rng == b.rng && a.vdiag == b.vdiag &&
        no_offset_view(a.v) == no_offset_view(b.v) &&
        no_offset_view(a.rbar) == no_offset_view(b.rbar)
end

@testset "ElectronEigenpairs" begin
    model = _load_model_from_artifacts("pb"; load_epmat = false)
    kpts = GridKpoints(kpoints_grid((4, 4, 4)))

    # The eigenpairs come from the same `get_el_eigen!` call on the same H(k), so the cache must
    # reproduce `compute_electron_states` bit for bit. Both `fourier_mode`s, since the cache and
    # the states default to different ones.
    for fourier_mode in ("normal", "gridopt")
        eig = electron_eigenpairs(model, kpts; fourier_mode)
        states = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector"]; fourier_mode)
        @test eig.nw == model.nw
        @test eig.e isa Matrix{Float64}
        @test eig.u isa Array{ComplexF64, 3}
        @test size(eig.e) == (model.nw, kpts.n)
        @test size(eig.u) == (model.nw, model.nw, kpts.n)
        @test all(ik -> eig.e[:, ik] == states[ik].e_full, 1:kpts.n)
        @test all(ik -> eig.u[:, :, ik] == states[ik].u_full, 1:kpts.n)
        # Negative control for the two claims above: they must fail against a one-k offset, so a
        # cache that reorders its k points cannot leave them passing.
        @test !any(ik -> eig.u[:, :, ik] == states[mod1(ik + 1, kpts.n)].u_full, 1:kpts.n)

        # The bitwise claim is only interesting where the gauge is not unique, so pin the number of
        # k points of this grid that carry a degenerate multiplet.
        @test count(ik -> minimum(diff(eig.e[:, ik])) < electron_degen_cutoff, 1:kpts.n) == 22
    end

    @testset "lookup" begin
        # A cache over a strict subset of the grid, so that the omitted node is a genuine miss.
        sub = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
        eig = electron_eigenpairs(model, sub)
        @test all(ik -> _eigenpair_index(eig, sub.vectors[ik]) == ik, 1:sub.n)
        # A k point that reaches the lookup through grid arithmetic carries round-off, which must
        # still resolve (the tolerance is the `GridKpoints` constructor's `sqrt(eps(T))`).
        @test _eigenpair_index(eig, sub.vectors[5] .+ 1e-12) == 5
        # A node the cache does not hold, and two points that are not nodes at all. The latter are
        # the ones that must not silently alias: the lookup rounds onto the grid, so without the
        # accessor's own check they would return the index of a neighbouring cached k point (on
        # this grid, `ik = 1` and `ik = 17` respectively).
        @test_throws "is not one of its" _eigenpair_index(eig, kpts.vectors[kpts.n])
        @test_throws "is not on the" _eigenpair_index(eig, Vec3(0.05, 0.0, 0.0))
        @test_throws "is not on the" _eigenpair_index(eig, Vec3(0.25 + 1e-6, 0.0, 0.0))

        # A shifted grid: the check is against the cache's own `shift`, so Gamma -- a perfectly
        # legal k point -- is not a node of this cache and must be rejected rather than aliased.
        shifted = GridKpoints(kpoints_grid((4, 4, 4); shift = (0.125, 0.125, 0.125)))
        eig_shifted = electron_eigenpairs(model, shifted)
        @test eig_shifted.kpts.shift ≈ Vec3(0.125, 0.125, 0.125)
        @test all(ik -> _eigenpair_index(eig_shifted, shifted.vectors[ik]) == ik, 1:shifted.n)
        @test _eigenpair_index(eig_shifted, shifted.vectors[3] .+ 1e-12) == 3
        @test_throws "is not on the" _eigenpair_index(eig_shifted, Vec3(0.0, 0.0, 0.0))

        # A `Kpoints` input is validated against its own ngrid when the cache is built.
        off_grid = Kpoints{Float64}(2, [Vec3(0.0, 0.0, 0.0), Vec3(0.1, 0.0, 0.0)], [0.5, 0.5],
                                    kpts.ngrid)
        @test_throws "is not on the grid of size" electron_eigenpairs(model, off_grid)
    end

    @testset "consumed by compute_electron_states" begin
        # A cache must be inert: with it, every state a run produces is bit for bit what the same
        # run produces without it, because the eigenpairs are the same `get_el_eigen!` output on
        # the same H(k). Covered for both `compute_electron_states` methods (uniform window and
        # per-k band extent from a selection), every quantity list, and both `el_velocity_mode`s --
        # `:BerryConnection` is the one where "velocity" pulls in "position".
        model_bn = _load_model_from_artifacts("cubicBN"; load_epmat = false)
        window = (10.0, 25.0) .* unit_to_aru(:eV)
        kpts_bn = GridKpoints(kpoints_grid((4, 4, 4)))
        sel = ElectronPhonon.filter_electron_states((4, 4, 4), model_bn, window)
        @test sel.n > 0
        # A per-k band extent that actually varies, so the `sel` method is not just the uniform one.
        @test !allequal(sel.band_extent)

        # `fourier_mode` has to be the same on both arms: H(k) is not bitwise equal between the
        # Fourier modes, and inside a degenerate multiplet that difference is O(1) in `u`. The
        # device path ignores it (it has only the batched interpolator).
        arms = Tuple{AbstractBackend, String}[(CPUBackend(), "normal"),
                                              (CPUBackend(), "gridopt")]
        EIGENPAIRS_GPU_AVAILABLE && push!(arms, (gpu_backend(), "normal"))
        quantity_lists = (["eigenvalue"],
                          ["eigenvalue", "eigenvector"],
                          ["eigenvalue", "eigenvector", "velocity_diagonal"],
                          ["eigenvalue", "eigenvector", "velocity", "position"])
        for (backend, fourier_mode) in arms
            cache = electron_eigenpairs(model_bn, kpts_bn; backend, fourier_mode)
            for mode in (:Direct, :BerryConnection)
                model_bn.el_velocity_mode = mode
                for quantities in quantity_lists
                    ref = compute_electron_states(model_bn, kpts_bn, quantities, window; backend,
                                                  fourier_mode)
                    new = compute_electron_states(model_bn, kpts_bn, quantities, window; backend,
                                                  fourier_mode, eigenpairs = cache)
                    ref_sel = compute_electron_states(model_bn, sel, quantities; backend,
                                                      fourier_mode)
                    new_sel = compute_electron_states(model_bn, sel, quantities; backend,
                                                      fourier_mode, eigenpairs = cache)
                    if quantities == ["eigenvalue"]
                        @test _eigenpairs_valueonly_consistent(new, ref, cache)
                        @test _eigenpairs_valueonly_consistent(new_sel, ref_sel, cache)
                    else
                        @test all(_eigenpairs_state_equal.(new, ref))
                        @test all(_eigenpairs_state_equal.(new_sel, ref_sel))
                    end
                end
            end
        end
        model_bn.el_velocity_mode = :Direct

        # Teeth: a cache built on a different model is not silently accepted, and one that does not
        # cover a k point the run visits fails at that k point rather than recomputing it.
        pb_cache = electron_eigenpairs(model, kpts)
        @test_throws "Wannier functions" compute_electron_states(
            model_bn, kpts_bn, ["eigenvalue"], window; eigenpairs = pb_cache)
        sub = GridKpoints(Kpoints(kpts_bn.vectors[1:kpts_bn.n-1]; ngrid = kpts_bn.ngrid),
                          kpts_bn.ngrid)
        @test_throws "is not one of its" compute_electron_states(
            model_bn, kpts_bn, ["eigenvalue", "eigenvector"], window;
            eigenpairs = electron_eigenpairs(model_bn, sub))
    end

    @testset "GPU" begin
        if EIGENPAIRS_GPU_AVAILABLE
            # The device transfer must preserve the eltype: `CuArray(arr)` does, `cu(arr)` would
            # demote Float64 to Float32. Partial type, so the memory-type parameter stays free.
            @test to_device(gpu_backend(), zeros(ComplexF64, 2, 2)) isa CuArray{ComplexF64}

            eig_cpu = electron_eigenpairs(model, kpts)
            eig_gpu = electron_eigenpairs(model, kpts; backend = gpu_backend())
            # The arrays follow the backend that built them: the device cache stays on the device
            # (and is consumed there), the host one stays on the host.
            @test eig_cpu.e isa Matrix{Float64}
            @test eig_cpu.u isa Array{ComplexF64, 3}
            @test eig_gpu.e isa CuMatrix{Float64}
            @test eig_gpu.u isa CuArray{ComplexF64, 3}
            @test eig_gpu.kpts === eig_cpu.kpts  # only e/u move; kpts stays on the host
            e_gpu, u_gpu = Array(eig_gpu.e), Array(eig_gpu.u)
            # Eigenvalues only: the batched device eigensolve does not apply the degenerate-
            # multiplet gauge fix of the per-k CPU solve, so eigenvectors may legitimately differ
            # by a unitary rotation inside a multiplet.
            # This bound is also the only guard against a Float32 intermediate on the device: the
            # `copyto!` into the host arrays upcasts, so the eltype assertions above cannot see one.
            # Float32 floors at ~1e-7 relative, so do not loosen 1e-13 past ~1e-9.
            @test norm(e_gpu - eig_cpu.e) / norm(eig_cpu.e) < 1e-13
            unitarity = maximum(1:kpts.n) do ik
                u = @view u_gpu[:, :, ik]
                norm(u' * u - I)
            end
            @test unitarity < 1e-12

            # Both bases reconstruct the same H(k), but only to within the splitting of the groups
            # they are allowed to differ in, so the bound has to be taken per k point: at a k point
            # with no multiplet the two `u` agree to round-off, and only a k point carrying one may
            # differ, there by at most `electron_degen_cutoff`.
            hamiltonian_diff(ik) = norm(
                u_gpu[:, :, ik] * Diagonal(e_gpu[:, ik]) * u_gpu[:, :, ik]' -
                eig_cpu.u[:, :, ik] * Diagonal(eig_cpu.e[:, ik]) * eig_cpu.u[:, :, ik]')
            degenerate = [minimum(diff(eig_cpu.e[:, ik])) < electron_degen_cutoff
                          for ik in 1:kpts.n]
            @test maximum(hamiltonian_diff, (1:kpts.n)[.!degenerate]; init = 0.0) < 1e-13
            @test maximum(hamiltonian_diff, (1:kpts.n)[degenerate]; init = 0.0) <
                  electron_degen_cutoff
        else
            @info "CUDA not functional - skipping the GPU ElectronEigenpairs test"
        end
    end
end
