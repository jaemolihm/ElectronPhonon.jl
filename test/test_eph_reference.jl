using Test
using ElectronPhonon
using ElectronPhonon: CPUBackend, gpu_backend, unit_to_aru, run_eph_over_k_and_kq, run_eph_over_k_and_q,
    run_eph_over_q_and_k

const EPH_REFERENCE_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

include("eph_reference_loop.jl")

# Every (loop order, backend, loop shape) of the current drivers against the reference double
# loop. Two windowed fixtures of the pb artifact model: nk = 12, E_F +- 0.2 eV (one in-window band
# per k point) and nk = 6, E_F - 0.5 / + 3 eV (0 to 3 bands per k point, empty windows, and windows
# ending at band nw with the box reaching past it). `|g|^2` is summed over degenerate multiplets,
# so the device eigensolves, which pick their own basis inside a multiplet, are compared on the
# same footing as the host ones.
@testset "e-ph loops against the reference double loop" begin
    model_el = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    model_ph = _load_model_from_artifacts("pb"; epmat_outer_momentum = "ph")
    eV = unit_to_aru(:eV)
    e_F = 11.68eV
    fixtures = (("nk = 12, +-0.2 eV", (12, 12, 12), (e_F - 0.2eV, e_F + 0.2eV), 216^2),
                ("nk = 6, -0.5 / +3 eV", (6, 6, 6), (e_F - 0.5eV, e_F + 3eV), 153^2))
    for (fixture, grid, window, npairs) in fixtures
        kpts = GridKpoints(kpoints_grid(grid))
        ref = eph_reference(model_el, kpts, kpts, window, window)
        @test length(ref.g2abs) == npairs
        # The comparison has teeth: the reference against itself with the pairs' values rotated by
        # one pair fails it.
        ks = collect(keys(ref.g2abs))
        rotated = (; g2abs = Dict(zip(ks, circshift([ref.g2abs[k] for k in ks], 1))), ref.ωq)
        @test compare_with_reference(ref, rotated).g2_reldev > 0.1

        common = (; window_k = window, window_kq = window, progress_print_step = 10^9,
                  verbosity = 0)
        # Each order requires its matching model layout: el for outer k, ph for outer q.
        outer_k = Any[("CPU", (; nchunks_threads = 4)),
            ("CPU, small tiles", (; backend = CPUBackend(), n_outer_batch = 5, n_inner_tile = 50))]
        outer_q = Any[("CPU", (; nchunks_threads = 4)),
            ("CPU, small tiles", (; backend = CPUBackend(), n_inner_tile = 50, n_outer_batch = 3)),
            ("CPU, precomputed k+q", (; precompute_el_kq = true, n_inner_tile = 50, nchunks_threads = 4))]
        # `run_eph_over_k_and_q` on the same commensurate grids: the pairs and values of the outer-k
        # arms above, with k + q solved per tile.
        outer_k_q = Any[("CPU", (; nchunks_threads = 4)),
            ("CPU, small tiles", (; backend = CPUBackend(), n_outer_batch = 5, n_inner_tile = 50))]
        if EPH_REFERENCE_GPU_AVAILABLE
            CUDA.allowscalar(false)
            push!(outer_k, ("GPU", (; backend = gpu_backend())),
                ("GPU, outer batch 1", (; backend = gpu_backend(), n_outer_batch = 1)))
            push!(outer_q, ("GPU", (; backend = gpu_backend())),
                ("GPU, precomputed k+q", (; backend = gpu_backend(), precompute_el_kq = true)))
            push!(outer_k_q, ("GPU", (; backend = gpu_backend(), n_inner_tile = 50)))
        end
        for (order, arms) in (("outer k", outer_k), ("outer k, inner q", outer_k_q), ("outer q", outer_q)),
                (name, kw_arm) in arms
            rec = _PairRecorder()
            (; model) = merge((; model = order == "outer q" ? model_ph : model_el), kw_arm)
            kw = Base.structdiff(kw_arm, NamedTuple{(:model,)})
            if order == "outer k"
                run_eph_over_k_and_kq(model, grid, grid; calculators = [rec],
                                      symmetry = nothing, common..., kw...)
            elseif order == "outer k, inner q"
                run_eph_over_k_and_q(model, grid, grid; calculators = [rec],
                                     symmetry = nothing, common..., kw...)
            else
                run_eph_over_q_and_k(model, grid, grid; calculators = [rec],
                                     symmetry = nothing, common..., kw...)
            end
            dev = compare_with_reference(ref, rec)
            @info "e-ph loop vs reference" fixture order name dev
            @test dev.nmissing == 0 && dev.nextra == 0
            # Measured 6e-15 to 3e-14 on every arm, CPU and GPU (A100).
            @test dev.g2_reldev < 1e-11
            @test dev.ω_dev < 1e-10
        end
    end
end

# `run_eph_over_k_and_q` off the commensurate grids: k on a band path on no grid with q on a grid,
# and k on a grid with q on a path (a q list with no grid). The pairs are keyed on coordinates
# rounded to 1e-6 (`keygrid`), which both sides compute as the same `x_k + x_q`.
@testset "run_eph_over_k_and_q against the reference double loop" begin
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    eV = unit_to_aru(:eV)
    e_F = 11.68eV
    window = (e_F - 0.5eV, e_F + 3eV)
    keygrid = (10^6, 10^6, 10^6)
    # The k path crosses the window (1, 1, 3, 2, 2, 2 in-window bands); the q path starts at Γ.
    kpath = Kpoints([Vec3(0.2513 + 0.0371j, 0.2487 + 0.0353j, 0.0129) for j in 0:5])
    qpath = Kpoints([Vec3(j / 24, 0, 0) for j in 0:5])
    grid4 = kpoints_grid((4, 4, 4))
    common = (; window_k = window, window_kq = window, progress_print_step = 10^9, verbosity = 0)
    for (name, kpts, qpts) in (("k path, q grid", kpath, grid4), ("k grid, q path", grid4, qpath))
        ref = eph_reference_k_and_q(model, kpts, qpts, window, window, keygrid)
        @test !isempty(ref.g2abs)
        arms = Any[("CPU", (; nchunks_threads = 4)), ("CPU, small tiles", (; n_inner_tile = 5))]
        EPH_REFERENCE_GPU_AVAILABLE && push!(arms, ("GPU", (; backend = gpu_backend())))
        for (arm, kw) in arms
            rec = _PairRecorder(keygrid)
            run_eph_over_k_and_q(model, kpts, qpts; calculators = [rec], symmetry = nothing, common..., kw...)
            dev = compare_with_reference(ref, rec)
            @info "run_eph_over_k_and_q vs reference" name arm dev
            @test dev.nmissing == 0 && dev.nextra == 0
            # Measured 3e-15 to 4e-15 (k grid), 8e-14 to 1.2e-13 (the k path), CPU and GPU (A6000).
            @test dev.g2_reldev < 1e-11
            @test dev.ω_dev < 1e-10
        end
    end
end
