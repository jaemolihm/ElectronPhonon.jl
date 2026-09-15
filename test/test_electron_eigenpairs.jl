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
        @test length(eig) == kpts.n
        @test size(eig.e) == (model.nw, kpts.n)
        @test size(eig.u) == (model.nw, model.nw, kpts.n)
        @test all(ik -> eig.e[:, ik] == states[ik].e_full, 1:kpts.n)
        @test all(ik -> eig.u[:, :, ik] == states[ik].u_full, 1:kpts.n)

        # The bitwise claim is only interesting where the gauge is not unique, so check that this
        # grid does contain degenerate multiplets (22 of the 64 k points, on this model).
        @test count(ik -> minimum(diff(eig.e[:, ik])) < electron_degen_cutoff, 1:kpts.n) > 0
    end

    @testset "lookup" begin
        # A cache over a strict subset of the grid, so that the omitted node is a genuine miss.
        # (On a full grid every node is present and any off-grid point aliases to a node, which is
        # why `electron_eigenpairs` validates on-grid-ness instead of relying on the lookup.)
        sub = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
        eig = electron_eigenpairs(model, sub)
        @test all(ik -> _eigenpair_index(eig, sub.vectors[ik]) == ik, 1:sub.n)
        @test_throws ArgumentError _eigenpair_index(eig, kpts.vectors[kpts.n])

        # Off-grid input is rejected at construction, not at lookup time.
        off_grid = Kpoints{Float64}(2, [Vec3(0.0, 0.0, 0.0), Vec3(0.1, 0.0, 0.0)], [0.5, 0.5],
                                    kpts.ngrid)
        @test_throws ArgumentError electron_eigenpairs(model, off_grid)
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
            # Both bases are still orthonormal, and both reconstruct the same H(k) to within the
            # splitting of the groups they differ in: the two `u` differ only inside groups whose
            # eigenvalues lie within `electron_degen_cutoff` of each other, so `u * diag(e) * u'`
            # can differ by that much and no more.
            unitarity = maximum(1:kpts.n) do ik
                u = @view eig_gpu.u[:, :, ik]
                norm(u' * u - I)
            end
            hamiltonian_diff = maximum(1:kpts.n) do ik
                h_gpu = eig_gpu.u[:, :, ik] * Diagonal(eig_gpu.e[:, ik]) * eig_gpu.u[:, :, ik]'
                h_cpu = eig_cpu.u[:, :, ik] * Diagonal(eig_cpu.e[:, ik]) * eig_cpu.u[:, :, ik]'
                norm(h_gpu - h_cpu)
            end
            @test unitarity < 1e-12
            @test hamiltonian_diff < electron_degen_cutoff
        else
            @info "CUDA not functional - skipping the GPU ElectronEigenpairs test"
        end
    end
end
