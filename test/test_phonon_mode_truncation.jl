using Test
using ElectronPhonon
using ElectronPhonon: OuterKEngine, stage1!, stage2!, compute_phonon_states_batched,
    eph_apply_rotations!, _run_options, _setup_states, OuterKLoop

# A run with a finite energy_conservation_tol on a GPU may keep only the lowest phonon modes
# (`nmodes_kept`, `allows_phonon_mode_truncation`). Pb has no mode that is high at every q, so the
# truncation is forced here at nmodes_kept = 2 of 3, and each piece is checked against the full run.

const MODE_TRUNCATION_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

@testset "phonon mode truncation (GPU)" begin
    if !MODE_TRUNCATION_GPU
        @info "CUDA not available/functional — skipping phonon mode truncation test"
    else
        backend = ElectronPhonon.gpu_backend()
        model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
        grid = (4, 4, 4)
        nkept = 2

        # The build keeps the leading modes of the full device solve, bitwise.
        qpts = kpoints_grid(grid)
        full = compute_phonon_states_batched(model, qpts, [:e, :u]; backend)
        part = compute_phonon_states_batched(model, qpts, [:e, :u]; backend, nmodes_kept = nkept)
        @test part.nmodes == nkept && size(part.u) == (model.nmodes, nkept, qpts.n)
        @test Array(part.e) == Array(full.e)[1:nkept, :]
        @test Array(part.u) == Array(full.u)[:, 1:nkept, :]
        # Only the device solve truncates.
        @test_throws ArgumentError compute_phonon_states_batched(model, qpts, [:e, :u]; nmodes_kept = nkept)

        # A rectangular phonon basis in both GPU rotation branches against the generic method.
        for (nw, ndisp) in ((3, 15), (8, 15))
            nbk, nbkq, nq = nw - 1, nw, 23
            g, ukqs = rand(ComplexF64, nw, nbk, ndisp, nq), rand(ComplexF64, nw, nbkq, nq)
            uphs = rand(ComplexF64, ndisp, 5, nq)
            ep_cpu = zeros(ComplexF64, nbkq, nbk, 5, nq)
            eph_apply_rotations!(ep_cpu, g, ukqs, uphs, zeros(ComplexF64, nbkq, nbk * ndisp, nq))
            ep_gpu = CUDA.zeros(ComplexF64, nbkq, nbk, 5, nq)
            eph_apply_rotations!(ep_gpu, CuArray(g), CuArray(ukqs), CuArray(uphs),
                                 CUDA.zeros(ComplexF64, nbkq, nbk * ndisp, nq))
            @test isapprox(Array(ep_gpu), ep_cpu; rtol = 1e-13)
        end

        # An engine on the truncated phonons gives the leading modes of the full engine's blocks.
        options = _run_options(model; inner_loop_kq = true, backend, symmetry = nothing, verbosity = 0)
        st = _setup_states(OuterKLoop(), model, grid, grid, options)
        phs_part = compute_phonon_states_batched(model, st.qpts, [:e, :u]; backend, nmodes_kept = nkept)
        common = (; st.kpts, st.kqpts, st.qpts, n_outer_batch = 5, n_inner_tile = 13, nchunks = 1,
                  covariant_derivative_of_g = false, eph_phonon_basis = :eigenmode)
        eng_full = OuterKEngine(model, backend, st.els_k, st.els_kq, st.phs, [:e, :u], [:e, :u]; common...)
        eng_part = OuterKEngine(model, backend, st.els_k, st.els_kq, phs_part, [:e, :u], [:e, :u]; common...)
        maxdev, scale, nwrong_shape = 0.0, 0.0, 0
        for batch in Iterators.partition(1:st.kpts.n, 5)
            stage1!(eng_full, batch); stage1!(eng_part, batch)
            for ik in batch, tile in Iterators.partition(1:st.kqpts.n, 13)
                ep_full = Array(stage2!(eng_full, ik, tile).ep)
                block = stage2!(eng_part, ik, tile)
                nwrong_shape += !(size(block.ep, 3) == nkept && block.phs.nmodes == nkept)
                maxdev = max(maxdev, maximum(abs, Array(block.ep) - ep_full[:, :, 1:nkept, :]))
                scale = max(scale, maximum(abs, ep_full))
            end
        end
        @test nwrong_shape == 0
        @test maxdev <= 1e-12 * scale
    end
end
