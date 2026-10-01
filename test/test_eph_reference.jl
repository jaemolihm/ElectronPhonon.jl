using Test
using ElectronPhonon
using ElectronPhonon: CPUBackend, gpu_backend, unit_to_aru, run_eph_over_k_and_kq, run_eph_over_q_and_k

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
        outer_k = Any[("per-point", (; nchunks_threads = 4)),
            ("batched CPU", (; backend = CPUBackend(), batched = true, nk_outer_batch_max = 5,
                              nq_batch_max = 50))]
        outer_q = Any[("per-point", (; nchunks_threads = 4)),
            ("batched CPU", (; backend = CPUBackend(), batched = true, nk_batch_max = 50))]
        if EPH_REFERENCE_GPU_AVAILABLE
            CUDA.allowscalar(false)
            push!(outer_k, ("GPU", (; backend = gpu_backend())),
                ("GPU, outer batch 1", (; backend = gpu_backend(), nk_outer_batch_max = 1)))
            push!(outer_q, ("GPU", (; backend = gpu_backend())))
        end
        for (order, arms) in (("outer k", outer_k), ("outer q", outer_q)), (name, kw) in arms
            rec = _PairRecorder()
            if order == "outer k"
                run_eph_over_k_and_kq(model_el, grid, grid; calculators = [rec],
                                      symmetry = nothing, common..., kw...)
            else
                run_eph_over_q_and_k(model_ph, grid, grid; calculators = [rec],
                                     use_symmetry = false, common..., kw...)
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
