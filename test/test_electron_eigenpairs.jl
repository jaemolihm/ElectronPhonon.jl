using Test
using ElectronPhonon
using ElectronPhonon: Vec3, electron_degen_cutoff, electron_eigenpairs, _eigenpair_index,
    gpu_backend
using LinearAlgebra

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const EIGENPAIRS_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
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
        @test ElectronPhonon.nk(eig) == kpts.n
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
        @test_throws "cache's grid" _eigenpair_index(eig, Vec3(0.05, 0.0, 0.0))
        @test_throws "cache's grid" _eigenpair_index(eig, Vec3(0.25 + 1e-6, 0.0, 0.0))

        # A `Kpoints` input is validated against its own ngrid when the cache is built.
        off_grid = Kpoints{Float64}(2, [Vec3(0.0, 0.0, 0.0), Vec3(0.1, 0.0, 0.0)], [0.5, 0.5],
                                    kpts.ngrid)
        @test_throws "is not on the grid of size" electron_eigenpairs(model, off_grid)
    end

    @testset "GPU" begin
        if EIGENPAIRS_GPU_AVAILABLE
            eig_cpu = electron_eigenpairs(model, kpts)
            eig_gpu = electron_eigenpairs(model, kpts; backend = gpu_backend())
            @test eig_gpu.e isa Matrix{Float64}
            @test eig_gpu.u isa Array{ComplexF64, 3}
            # Eigenvalues only: the batched device eigensolve does not apply the degenerate-
            # multiplet gauge fix of the per-k CPU solve, so eigenvectors may legitimately differ
            # by a unitary rotation inside a multiplet.
            @test norm(eig_gpu.e - eig_cpu.e) / norm(eig_cpu.e) < 1e-13
            unitarity = maximum(1:kpts.n) do ik
                u = @view eig_gpu.u[:, :, ik]
                norm(u' * u - I)
            end
            @test unitarity < 1e-12

            # Both bases reconstruct the same H(k), but only to within the splitting of the groups
            # they are allowed to differ in, so the bound has to be taken per k point: at a k point
            # with no multiplet the two `u` agree to round-off, and only a k point carrying one may
            # differ, there by at most `electron_degen_cutoff`.
            hamiltonian_diff(ik) = norm(
                eig_gpu.u[:, :, ik] * Diagonal(eig_gpu.e[:, ik]) * eig_gpu.u[:, :, ik]' -
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
