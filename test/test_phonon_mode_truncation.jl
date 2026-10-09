using Test
using ElectronPhonon
using ElectronPhonon: OuterKEngine, stage1!, stage2!, compute_phonon_states_batched, unit_to_aru,
    eph_apply_rotations!, _run_options, _setup_states, OuterKLoop, _kept_phonon_modes,
    _phonon_storage_plan, _phonon_state_bytes, _select_qs_per_outer_batch, AbstractCalculator,
    GPU_PHONONS_RESIDENT_MIN_TILE, _may_truncate_phonon_modes

# A run with a finite energy_conservation_tol on a GPU may keep only the lowest phonon modes
# (`nmodes_kept`, `allows_phonon_mode_truncation`), and build them per outer batch when even those do
# not fit (`phonon_storage`). Pb has no mode that is high at every q, so the policy is checked on
# synthetic numbers, and the truncation is forced on Pb at nmodes_kept = 2 of 3.

const MODE_TRUNCATION_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

@testset "phonon modes kept and their storage (policy)" begin
    # Two acoustic modes below Ω and a gapped optical branch above it at every q.
    Ω = 0.05
    ω = [0.0   0.01  0.02  0.015
         0.0   0.02  0.03  0.025
         0.10  0.11  0.12  0.105]
    @test _kept_phonon_modes(ω, Ω) == 2
    # A mode exactly at Ω can conserve energy, so it is kept.
    ω_at = copy(ω); ω_at[3, 2] = Ω
    @test _kept_phonon_modes(ω_at, Ω) == 3
    # ω exactly at -Ω still truncates; below it the ascending order no longer bounds the modes by
    # |ω| and every mode is kept.
    ω_neg = copy(ω); ω_neg[1, 3] = -Ω
    @test _kept_phonon_modes(ω_neg, Ω) == 2
    ω_neg[1, 3] = -Ω - 1e-3
    @test _kept_phonon_modes(ω_neg, Ω) == 3
    # At least one mode, even when none is below Ω.
    @test _kept_phonon_modes(fill(1.0, 3, 2), Ω) == 1

    # The storage: needed = table + committed + per_pair * min_tile at the 7/10 headroom.
    plan(; free, nkq = 10, n_inner_tile = nothing, phonon_storage = :auto, table_bytes = 100) =
        _phonon_storage_plan(; table_bytes, committed = 50, per_pair = 7, free, nkq, n_inner_tile,
                             phonon_storage)
    @test plan(; free = 250) == (; storage = :resident, fit = :fits, min_tile = 10, needed = 250)
    @test plan(; free = 249).storage == :per_batch && plan(; free = 249).fit == :narrow_tile
    @test plan(; free = 160).fit == :narrow_tile && plan(; free = 159).fit == :no_fit
    # The default tile is the preferred width, and a user tile replaces it.
    @test plan(; free = 10^6, nkq = 10^7).min_tile == GPU_PHONONS_RESIDENT_MIN_TILE
    @test plan(; free = 190, nkq = 10^7).storage == :per_batch
    @test plan(; free = 190, nkq = 10^7, n_inner_tile = 4) ==
          (; storage = :resident, fit = :fits, min_tile = 4, needed = 190)
    # The override sets the storage, not the fit.
    @test plan(; free = 10^6, phonon_storage = :per_batch) ==
          (; storage = :per_batch, fit = :fits, min_tile = 10, needed = 250)
    @test plan(; free = 0, phonon_storage = :resident).storage == :resident
    # Truncation decides the storage: all modes of all q do not fit, the two kept do.
    nq, nm = 1000, 3
    table(nk) = nq * _phonon_state_bytes(Float64, nk, (:e, :u); ndisp = nm)
    free = 50 + cld(7 * 10 * 10, 7) + table(2)
    @test plan(; free, table_bytes = table(nm)).storage == :per_batch
    @test plan(; free, table_bytes = table(_kept_phonon_modes(ω, Ω))).storage == :resident
end

# A calculator that reads `block.phs.u` lists `:u`; it records the blocks' eigenvectors and q.
mutable struct _PhononURecorder <: AbstractCalculator
    list_u::Bool
    us::Vector{Array{ComplexF64, 3}}
    iqs::Vector{Vector{Int}}
    nothing_u::Int
end
_PhononURecorder(list_u) = _PhononURecorder(list_u, [], [], 0)
ElectronPhonon.supports(::_PhononURecorder, ::Type{OuterKLoop}) = true
ElectronPhonon.required_ph_quantities(c::_PhononURecorder) = c.list_u ? [:u] : Symbol[]
ElectronPhonon.setup_calculator!(c::_PhononURecorder, backend, els_k, els_kq, phs; kwargs...) = c
ElectronPhonon.calculator_begin_batch!(::_PhononURecorder, ctx) = nothing
ElectronPhonon.calculator_end_batch!(::_PhononURecorder, ctx) = nothing
ElectronPhonon.postprocess_calculator!(c::_PhononURecorder; kwargs...) = c
function ElectronPhonon.run_calculator!(c::_PhononURecorder, block::ElectronPhonon.EPBlock{OuterKLoop}, ctx)
    if block.phs.u === nothing
        c.nothing_u += 1
    else
        push!(c.us, Array(block.phs.u)); push!(c.iqs, Array(block.iq))
    end
    c
end

# Does not allow the phonon modes to be truncated (the default).
struct _NoTruncation <: AbstractCalculator end

@testset "phonon eigenvectors and storage (GPU)" begin
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
        # Only the device solve truncates, and `phs_out` must match the request.
        @test_throws ArgumentError compute_phonon_states_batched(model, qpts, [:e, :u]; nmodes_kept = nkept)
        for phs_out in (ElectronPhonon.BatchedPhononState(backend, nkept, qpts.n, [:e]),
                        ElectronPhonon.BatchedPhononState(backend, nkept, qpts.n, [:e, :u]),
                        ElectronPhonon.BatchedPhononState(ElectronPhonon.CPUBackend(), nkept, qpts.n, [:e, :u];
                                                          ndisp = model.nmodes))
            @test_throws ArgumentError compute_phonon_states_batched(model, qpts, [:e, :u]; backend,
                nmodes_kept = nkept, phs_out)
        end

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

        # An engine on the truncated phonons gives the leading modes of the full engine's blocks; with
        # a finite tolerance, the same engine with the truncated phonons built per outer batch gives
        # the blocks of the resident one.
        tol = 1e-3
        options = _run_options(model; inner_loop_kq = true, backend, symmetry = nothing, verbosity = 0)
        st = _setup_states(OuterKLoop(), model, grid, grid, options)
        phs_part = compute_phonon_states_batched(model, st.qpts, [:e, :u]; backend, nmodes_kept = nkept)
        common = (; st.kpts, st.kqpts, st.qpts, n_outer_batch = 5, n_inner_tile = 13, nchunks = 1,
                  covariant_derivative_of_g = false, eph_phonon_basis = :eigenmode)
        eng_full = OuterKEngine(model, backend, st.els_k, st.els_kq, st.phs, [:e, :u], [:e, :u]; common...)
        eng_part = OuterKEngine(model, backend, st.els_k, st.els_kq, phs_part, [:e, :u], [:e, :u]; common...)
        eng_part_tol = OuterKEngine(model, backend, st.els_k, st.els_kq, phs_part, [:e, :u], [:e, :u];
            energy_conservation_tol = tol, common...)
        iqs_per_batch = _select_qs_per_outer_batch(backend, st.kpts, st.kqpts, st.qpts, st.els_k,
            st.els_kq, phs_part.e, 5, tol)
        phs_empty = ElectronPhonon.BatchedPhononState(backend, nkept, 0, [:e, :u]; ndisp = model.nmodes)
        eng_batch = OuterKEngine(model, backend, st.els_k, st.els_kq, phs_empty, [:e, :u], [:e, :u];
            energy_conservation_tol = tol, ω_all = phs_part.e, iqs_per_batch, common...)
        # The q sets are those of the engine's outer batches.
        @test cld(st.kpts.n, 5) != cld(st.kpts.n, 7)
        @test_throws ArgumentError OuterKEngine(model, backend, st.els_k, st.els_kq, phs_empty, [:e, :u],
            [:e, :u]; energy_conservation_tol = tol, ω_all = phs_part.e, iqs_per_batch, common...,
            n_outer_batch = 7)
        maxdev, maxdev_batch, scale, nwrong_shape, nblocks_tol, nmismatch = 0.0, 0.0, 0.0, 0, 0, 0
        for batch in Iterators.partition(1:st.kpts.n, 5)
            stage1!(eng_full, batch); stage1!(eng_part, batch)
            stage1!(eng_part_tol, batch); stage1!(eng_batch, batch)
            for ik in batch, tile in Iterators.partition(1:st.kqpts.n, 13)
                ep_full = Array(stage2!(eng_full, ik, tile).ep)
                block = stage2!(eng_part, ik, tile)
                nwrong_shape += !(size(block.ep, 3) == nkept && block.phs.nmodes == nkept)
                maxdev = max(maxdev, maximum(abs, Array(block.ep) - ep_full[:, :, 1:nkept, :]))
                scale = max(scale, maximum(abs, ep_full))
                block_tol, block_batch = stage2!(eng_part_tol, ik, tile), stage2!(eng_batch, ik, tile)
                nmismatch += (block_tol === nothing) != (block_batch === nothing)
                block_tol === nothing && continue
                nblocks_tol += 1
                nmismatch += Array(block_tol.iq) != Array(block_batch.iq)
                nmismatch += Array(block_tol.phs.e) != Array(block_batch.phs.e)
                maxdev_batch = max(maxdev_batch, maximum(abs, Array(block_tol.ep) - Array(block_batch.ep)))
            end
        end
        @test nwrong_shape == 0
        @test maxdev <= 1e-12 * scale
        @test nblocks_tol > 0 && nmismatch == 0
        @test maxdev_batch <= 1e-12 * scale

        # Every calculator must allow the truncation, which forced per-batch phonons need.
        eV = unit_to_aru(:eV); μ = 11.68eV
        boltzmann = BoltzmannCalculator{Float64}(;
            occ = ElectronOccupationParams(; Tlist = [300.0 * unit_to_aru(:K)], nlist = 4.0, μlist = μ,
                volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
            smearing_list = [SmearingType(:Gaussian, 0.01eV)])
        setup(calculators, phonon_storage) = _setup_states(OuterKLoop(), model, grid, grid,
            _run_options(model; inner_loop_kq = true, backend, symmetry = nothing, calculators,
                         energy_conservation_tol = tol, phonon_storage, verbosity = 0))
        st_batch = setup([boltzmann], :per_batch)
        @test st_batch.ω_all !== nothing && st_batch.phs.nq == 0 && st_batch.phs.nmodes == model.nmodes
        @test_throws ArgumentError setup([boltzmann, _NoTruncation()], :per_batch)
        # The eligibility of a run for the truncation.
        may_truncate(calculators; tol = tol, backend = backend) = _may_truncate_phonon_modes(OuterKLoop(),
            model, st.els_k, st.els_kq, _run_options(model; inner_loop_kq = true, backend, calculators,
                                                     energy_conservation_tol = tol))
        @test may_truncate([boltzmann])
        @test !may_truncate([boltzmann, _NoTruncation()])
        @test !may_truncate([boltzmann]; tol = Inf)
        @test !may_truncate([boltzmann]; backend = ElectronPhonon.CPUBackend())
        @test_throws ArgumentError _run_options(model; inner_loop_kq = true, phonon_storage = :host)

        # The blocks of a GPU run over a k+q grid carry `phs.u` only for a calculator listing `:u`, and
        # then those of the resident phonons at the block's q.
        for tol_run in (Inf, tol)
            rec = _PhononURecorder(true)
            out = ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid; calculators = [rec],
                symmetry = nothing, backend, energy_conservation_tol = tol_run, n_inner_tile = 13,
                progress_print_step = 10^9, verbosity = 0)
            u_res = Array(out.phs.u)
            @test rec.nothing_u == 0 && !isempty(rec.us)
            @test all(u == u_res[:, :, iq] for (u, iq) in zip(rec.us, rec.iqs))
            rec_no_u = _PhononURecorder(false)
            ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid; calculators = [rec_no_u],
                symmetry = nothing, backend, energy_conservation_tol = tol_run, n_inner_tile = 13,
                progress_print_step = 10^9, verbosity = 0)
            @test rec_no_u.nothing_u > 0 && isempty(rec_no_u.us)
        end
    end
end

# When even the kept modes of all q do not fit, the engine builds the phonons of each outer batch,
# for the q of its kept pairs only (forced here on Pb 6³): the BTE kernel is the one of the run with
# all phonons resident, and the driver returns no phonons. One-k batches whose k has no state within
# reach of the k+q window keep no pair.
@testset "phonons per outer batch (GPU)" begin
    if MODE_TRUNCATION_GPU
        model = _load_model_from_artifacts("pb")
        eV, K, meV = unit_to_aru(:eV), unit_to_aru(:K), unit_to_aru(:meV)
        μ = 11.68eV; window = (μ - 0.5eV, μ + 0.5eV); σ = 20meV
        runbte(phonon_storage; window_kq = window, n_outer_batch = 20) = (c = BoltzmannCalculator{Float64}(;
                occ = ElectronOccupationParams(; Tlist = [300.0K], nlist = 4.0, μlist = μ,
                    volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
                smearing_list = [SmearingType(:Gaussian, σ)], occupation_method = 5);
            out = ElectronPhonon.run_eph_over_k_and_kq(model, (6, 6, 6), (6, 6, 6); calculators = [c],
                symmetry = nothing, window_k = window, window_kq, n_outer_batch,
                energy_conservation_tol = 6σ, backend = ElectronPhonon.gpu_backend(),
                phonon_storage, progress_print_step = 10^9, verbosity = 0); (c, out))
        c_all, out_all = runbte(:resident)
        c_per_batch, out_per_batch = runbte(:per_batch)
        @test out_all.phs isa BatchedPhononState && out_per_batch.phs === nothing
        @test maximum(stack(c_all.Sₒ)) > 0
        @test isapprox(stack(c_per_batch.Sᵢ), stack(c_all.Sᵢ); rtol = 1e-12)
        @test isapprox(stack(c_per_batch.Sₒ), stack(c_all.Sₒ); rtol = 1e-12)

        window_kq = (μ - 0.1eV, μ + 0.5eV)
        backend = ElectronPhonon.gpu_backend()
        st = _setup_states(OuterKLoop(), model, (6, 6, 6), (6, 6, 6), _run_options(model;
            inner_loop_kq = true, backend, symmetry = nothing, window_k = window, window_kq,
            energy_conservation_tol = 6σ, phonon_storage = :per_batch, verbosity = 0))
        iqs_per_batch = _select_qs_per_outer_batch(backend, st.kpts, st.kqpts, st.qpts, st.els_k,
            st.els_kq, st.ω_all, 1, 6σ)
        @test any(isempty, iqs_per_batch) && !all(isempty, iqs_per_batch)
        c_all, _ = runbte(:resident; window_kq, n_outer_batch = 1)
        c_per_batch, _ = runbte(:per_batch; window_kq, n_outer_batch = 1)
        @test maximum(stack(c_all.Sₒ)) > 0
        @test isapprox(stack(c_per_batch.Sᵢ), stack(c_all.Sᵢ); rtol = 1e-12)
        @test isapprox(stack(c_per_batch.Sₒ), stack(c_all.Sₒ); rtol = 1e-12)
    end
end
