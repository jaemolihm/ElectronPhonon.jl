using Test
using ElectronPhonon
using ElectronPhonon: CPUBackend, gpu_backend, unit_to_aru, run_eph_over_k_and_kq, run_eph_over_q_and_k

# A disk epmat is a memory-mapped `op_r` of the same type as the in-memory one, so every path runs
# the same code on the same bits: compared with `==` against the in-memory epmat of the same model.

const DISK_EPMAT_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
DISK_EPMAT_GPU && CUDA.allowscalar(false)

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")
isdefined(@__MODULE__, :_PairRecorder) || include("eph_reference_loop.jl")

@testset "disk-backed epmat against the in-memory one" begin
    eV = unit_to_aru(:eV); e_F = 11.68eV
    window = (e_F - 0.5eV, e_F + 3eV)
    grid = (6, 6, 6)
    common = (; window_k = window, window_kq = window, symmetry = nothing, verbosity = 0,
              progress_print_step = 10^9, n_outer_batch = 5, n_inner_tile = 40)
    models = Dict(mom => _load_model_from_artifacts("pb"; epmat_outer_momentum = mom) for mom in ("el", "ph"))
    dir = mktempdir()
    disk_models = Dict(mom => _disk_epmat_model(models[mom], mkpath(joinpath(dir, mom))) for mom in ("el", "ph"))
    @test typeof(disk_models["el"].epmat) === typeof(models["el"].epmat)
    @test disk_models["el"].epmat.op_r == models["el"].epmat.op_r

    @testset "$order, $(nameof(typeof(backend)))" for (order, mom) in ((:k, "el"), (:q, "ph")),
            backend in (DISK_EPMAT_GPU ? Any[CPUBackend(), gpu_backend()] : Any[CPUBackend()])
        run(model) = (rec = _PairRecorder();
            order === :k ? run_eph_over_k_and_kq(model, grid, grid; calculators = [rec], backend, common...) :
                           run_eph_over_q_and_k(model, grid, grid; calculators = [rec], backend, common...);
            rec)
        ref = run(models[mom])
        disk = run(disk_models[mom])
        @test !isempty(ref.g2abs)
        @test keys(disk.g2abs) == keys(ref.g2abs)
        @test all(disk.g2abs[key] == ref.g2abs[key] && disk.ωq[key] == ref.ωq[key] for key in keys(ref.g2abs))
    end

    # The covariant derivative builds its position-weighted epmat from `op_r` on the host.
    @testset "covariant_derivative_of_g" begin
        run(model) = (rec = _DgRecorder();
            run_eph_over_k_and_kq(model, grid, grid; calculators = [rec],
                                  covariant_derivative_of_g = true, common...);
            rec)
        ref = run(models["el"])
        disk = run(disk_models["el"])
        @test !isempty(ref.sums)
        @test keys(disk.sums) == keys(ref.sums)
        @test all(disk.sums[key] == ref.sums[key] for key in keys(ref.sums))
    end

    # The per-k interpolators, among them "gridopt" of `run_coherence` and `get_eph_RR_to_Rq!`.
    @testset "per-k fourier_mode = $fourier_mode" for fourier_mode in ("normal", "gridopt", "batched")
        xks = [Vec3(0.1, 0.2, 0.3), Vec3(0.1, 0.2, -0.25), Vec3(0.5, 0.0, 0.125)]
        ops = map((models["ph"].epmat, disk_models["ph"].epmat)) do epmat
            itp = get_interpolator(epmat; fourier_mode)
            ElectronPhonon.register_kpoints!(itp, xks)
            [get_fourier!(zeros(ComplexF64, epmat.ndata), itp, xk) for xk in xks]
        end
        @test ops[2] == ops[1]
    end
end
