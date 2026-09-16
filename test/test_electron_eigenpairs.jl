using Test
using ElectronPhonon
using ElectronPhonon: Vec3, electron_degen_cutoff, electron_eigenpairs, gpu_backend, to_device
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
        @test eig.e_full isa Matrix{Float64}
        @test eig.u_full isa Array{ComplexF64, 3}
        @test size(eig.e_full) == (model.nw, kpts.n)
        @test size(eig.u_full) == (model.nw, model.nw, kpts.n)
        @test all(ik -> eig.e_full[:, ik] == states[ik].e_full, 1:kpts.n)
        @test all(ik -> eig.u_full[:, :, ik] == states[ik].u_full, 1:kpts.n)
        # Negative control for the two claims above: they must fail against a one-k offset, so a
        # cache that reorders its k points cannot leave them passing.
        @test !any(ik -> eig.u_full[:, :, ik] == states[mod1(ik + 1, kpts.n)].u_full, 1:kpts.n)

        # The bitwise claim is only interesting where the gauge is not unique, so pin the number of
        # k points of this grid that carry a degenerate multiplet.
        @test count(ik -> minimum(diff(eig.e_full[:, ik])) < electron_degen_cutoff, 1:kpts.n) == 22
    end

    @testset "lookup" begin
        # A cache over a strict subset of the grid, so that the omitted node is a genuine miss.
        sub = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
        eig = electron_eigenpairs(model, sub; fourier_mode = "gridopt")
        states = compute_electron_states(model, sub, ["eigenvalue", "eigenvector"];
                                         fourier_mode = "gridopt")
        # The cache's own contract: `xk_to_ik` on `eig.kpts` addresses the column of
        # `e_full`/`u_full` that holds *that* k point's eigenpair. The lookup's own behaviour --
        # the round-off tolerance, the two failure modes, shifted grids, the aliasing the check
        # closes -- is covered by "kpoints: xk_to_ik checked vs unsafe" in test_kpoints.jl.
        @test all(1:sub.n) do ik
            j = xk_to_ik(sub.vectors[ik], eig.kpts)
            eig.e_full[:, j] == states[ik].e_full && eig.u_full[:, :, j] == states[ik].u_full
        end
        # A node of the grid that this cache does not cover must not resolve to a neighbouring
        # column; `nothing` is what the caller then acts on.
        @test xk_to_ik(kpts.vectors[kpts.n], eig.kpts) === nothing

        # A `Kpoints` input is validated against its own ngrid when the cache is built, so a cache
        # whose grid does not contain its own points is unconstructible.
        off_grid = Kpoints{Float64}(2, [Vec3(0.0, 0.0, 0.0), Vec3(0.1, 0.0, 0.0)], [0.5, 0.5],
                                    kpts.ngrid)
        @test_throws "is not on the grid of size" electron_eigenpairs(model, off_grid)
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
            @test eig_cpu.e_full isa Matrix{Float64}
            @test eig_cpu.u_full isa Array{ComplexF64, 3}
            @test eig_gpu.e_full isa CuMatrix{Float64}
            @test eig_gpu.u_full isa CuArray{ComplexF64, 3}
            @test eig_gpu.kpts === eig_cpu.kpts  # only e_full/u_full move; kpts stays on the host
            e_gpu, u_gpu = Array(eig_gpu.e_full), Array(eig_gpu.u_full)
            # Eigenvalues only: the batched device eigensolve does not apply the degenerate-
            # multiplet gauge fix of the per-k CPU solve, so eigenvectors may legitimately differ
            # by a unitary rotation inside a multiplet.
            # This bound is also the only guard against a Float32 intermediate on the device: the
            # `copyto!` into the host arrays upcasts, so the eltype assertions above cannot see one.
            # Float32 floors at ~1e-7 relative, so do not loosen 1e-13 past ~1e-9.
            @test norm(e_gpu - eig_cpu.e_full) / norm(eig_cpu.e_full) < 1e-13
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
                eig_cpu.u_full[:, :, ik] * Diagonal(eig_cpu.e_full[:, ik]) *
                eig_cpu.u_full[:, :, ik]')
            degenerate = [minimum(diff(eig_cpu.e_full[:, ik])) < electron_degen_cutoff
                          for ik in 1:kpts.n]
            @test maximum(hamiltonian_diff, (1:kpts.n)[.!degenerate]; init = 0.0) < 1e-13
            @test maximum(hamiltonian_diff, (1:kpts.n)[degenerate]; init = 0.0) <
                  electron_degen_cutoff
        else
            @info "CUDA not functional - skipping the GPU ElectronEigenpairs test"
        end
    end
end
