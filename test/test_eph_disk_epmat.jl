using Test
using ElectronPhonon
using ElectronPhonon: CPUBackend, gpu_backend, unit_to_aru, run_eph_over_k_and_kq, run_eph_over_q_and_k

# A disk-backed epmat streams through stage 1 in column chunks of the same GEMM. Against the in-memory
# epmat of the same model: both orders, each on its epmat layout, several chunks.

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
    common = (; window_k = window, window_kq = window, verbosity = 0, progress_print_step = 10^9,
              n_outer_batch = 5, n_inner_tile = 40)
    run(order, model, backend, epmat_chunk_bytes) = (rec = _PairRecorder();
        order === :k ?
            run_eph_over_k_and_kq(model, grid, grid; calculators = [rec], symmetry = nothing, backend,
                                  epmat_chunk_bytes, common...) :
            run_eph_over_q_and_k(model, grid, grid; calculators = [rec], symmetry = nothing, backend,
                                 epmat_chunk_bytes, common...);
        rec)
    layouts = ((:k, "el"), (:q, "ph"))
    arms = Any[(order, mom, CPUBackend()) for (order, mom) in layouts]
    DISK_EPMAT_GPU && append!(arms, [(order, mom, gpu_backend()) for (order, mom) in layouts])
    for (order, mom, backend) in arms
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = mom)
        e = model.epmat
        # Three columns per chunk: several chunks and a partial last one.
        chunk_bytes = 3 * sizeof(ComplexF64) * size(e.op_r, 1)
        @test cld(e.nr, 3) > 2 && mod(e.nr, 3) != 0
        disk_model = mktempdir() do dir
            dm = _disk_epmat_model(model, dir)
            (; ref = run(order, model, backend, chunk_bytes), disk = run(order, dm, backend, chunk_bytes))
        end
        (; ref, disk) = disk_model
        scale = maximum(maximum, values(ref.g2abs))
        dev = maximum(k -> maximum(abs, disk.g2abs[k] - ref.g2abs[k]), keys(ref.g2abs))
        @info "disk vs in-memory epmat" order mom backend = nameof(typeof(backend)) rel = dev / scale
        @test keys(disk.g2abs) == keys(ref.g2abs)
        # The column contraction sums its chunks one after the other (8e-16 to 1e-15 measured).
        @test dev <= 1e-12 * scale
    end
    # The covariant derivative builds its position-weighted epmat from the in-memory one.
    mktempdir() do dir
        dm = _disk_epmat_model(_load_model_from_artifacts("pb"; epmat_outer_momentum = "el"), dir)
        @test_throws "needs model.epmat in memory" run_eph_over_k_and_kq(dm, (3, 3, 3), (3, 3, 3);
            calculators = [_PairRecorder()], symmetry = nothing, covariant_derivative_of_g = true,
            verbosity = 0)
    end
end
