@testsnippet PhononModeTruncation begin
    using ElectronPhonon: OuterKEngine, stage1!, stage2!, compute_phonon_states_batched, unit_to_aru,
        eph_apply_rotations!, _run_options, _setup_states, OuterKLoop, _kept_phonon_modes,
        _phonon_storage_plan, _phonon_state_bytes, _select_qs_per_outer_batch, AbstractCalculator,
        GPU_PHONONS_RESIDENT_MIN_TILE, _may_truncate_phonon_modes

    const MODE_TRUNCATION_GPU = try
        @eval using CUDA
        CUDA.functional()
    catch
        false
    end

    # Does not allow the phonon modes to be truncated (the default).
    struct _NoTruncation <: AbstractCalculator end
end

# A run with a finite energy_conservation_tol on a GPU may keep only the lowest phonon modes
# (`nmodes_kept`, `allows_phonon_mode_truncation`), and build them per outer batch when even those do
# not fit. Pb has no mode that is high at every q, so the policy is checked on synthetic numbers, and
# the per-batch path is reached on Pb by a `max_device_bytes` below twice the phonon table of all modes,
# on a k+q grid that makes the table larger than the loop's fixed buffers.

@testitem "phonon modes kept and their storage (policy)" setup = [Common, PhononModeTruncation] begin
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
    plan(; budget, nkq = 10, n_inner_tile = nothing, table_bytes = 100) =
        _phonon_storage_plan(; table_bytes, committed = 50, per_pair = 7, budget, nkq, n_inner_tile)
    @test plan(; budget = 250) == (; per_batch = false, min_tile = 10, needed = 250)
    @test plan(; budget = 249).per_batch && plan(; budget = 249).needed == 250
    # The default tile is the preferred width, and a user tile replaces it.
    @test plan(; budget = 10^6, nkq = 10^7).min_tile == GPU_PHONONS_RESIDENT_MIN_TILE
    @test plan(; budget = 190, nkq = 10^7).per_batch
    @test plan(; budget = 190, nkq = 10^7, n_inner_tile = 4) ==
          (; per_batch = false, min_tile = 4, needed = 190)
    # Truncation decides the storage: all modes of all q do not fit, the two kept do.
    nq, nm = 1000, 3
    table(nk) = nq * _phonon_state_bytes(Float64, nk, (:e, :u); ndisp = nm)
    budget = 50 + cld(7 * 10 * 10, 7) + table(2)
    @test plan(; budget, table_bytes = table(nm)).per_batch
    @test plan(; budget, table_bytes = table(_kept_phonon_modes(ω, Ω))).per_batch == false
end

@testitem "phonon storage policy (GPU)" setup = [Common, PhononModeTruncation] begin
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
        phonon_batches = (; ω_kept = phs_part.e, iqs_per_batch)
        eng_batch = OuterKEngine(model, backend, st.els_k, st.els_kq, phs_empty, [:e, :u], [:e, :u];
            energy_conservation_tol = tol, phonon_batches, common...)
        # The q sets are those of the engine's outer batches.
        @test cld(st.kpts.n, 5) != cld(st.kpts.n, 7)
        @test_throws ArgumentError OuterKEngine(model, backend, st.els_k, st.els_kq, phs_empty, [:e, :u],
            [:e, :u]; energy_conservation_tol = tol, phonon_batches, common..., n_outer_batch = 7)
        maxdev, maxdev_batch, scale, nwrong_shape, nblocks_tol, nmismatch = 0.0, 0.0, 0.0, 0, 0, 0
        for batch in Iterators.partition(1:st.kpts.n, 5)
            stage1!(eng_full, batch); stage1!(eng_part, batch)
            stage1!(eng_part_tol, batch); stage1!(eng_batch, batch)
            for ik in batch, tile in Iterators.partition(1:st.kqpts.n, 13)
                ep_full = Array(stage2!(eng_full, ik, tile).ep)
                block = stage2!(eng_part, ik, tile)
                global nwrong_shape += !(size(block.ep, 3) == nkept && block.phs.nmodes == nkept)
                global maxdev = max(maxdev, maximum(abs, Array(block.ep) - ep_full[:, :, 1:nkept, :]))
                global scale = max(scale, maximum(abs, ep_full))
                block_tol, block_batch = stage2!(eng_part_tol, ik, tile), stage2!(eng_batch, ik, tile)
                global nmismatch += (block_tol === nothing) != (block_batch === nothing)
                block_tol === nothing && continue
                global nblocks_tol += 1
                global nmismatch += Array(block_tol.iq) != Array(block_batch.iq)
                global nmismatch += Array(block_tol.phs.e) != Array(block_batch.phs.e)
                global maxdev_batch = max(maxdev_batch, maximum(abs, Array(block_tol.ep) - Array(block_batch.ep)))
            end
        end
        @test nwrong_shape == 0
        @test maxdev <= 1e-12 * scale
        @test nblocks_tol > 0 && nmismatch == 0
        @test maxdev_batch <= 1e-12 * scale
        # A kept pair whose q the batch's phonons lack is an error, not an index 0 in the rotation.
        iqs_missing = [iqs[2:end] for iqs in iqs_per_batch]
        eng_missing = OuterKEngine(model, backend, st.els_k, st.els_kq, phs_empty, [:e, :u], [:e, :u];
            energy_conservation_tol = tol, phonon_batches = (; phonon_batches..., iqs_per_batch = iqs_missing),
            common...)
        @test !isempty(iqs_per_batch[1])
        @test_throws "not among its outer batch's phonons" begin
            stage1!(eng_missing, 1:5)
            for ik in 1:5, tile in Iterators.partition(1:st.kqpts.n, 13)
                stage2!(eng_missing, ik, tile)
            end
        end

        # Every calculator must allow the truncation, which per-batch phonons need.
        eV = unit_to_aru(:eV); μ = 11.68eV
        boltzmann = BoltzmannCalculator{Float64}(;
            occ = ElectronOccupationParams(; Tlist = [300.0 * unit_to_aru(:K)], nlist = 4.0, μlist = μ,
                volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
            smearing_list = [SmearingType(:Gaussian, 0.01eV)])
        # One k per outer batch on a fine k+q grid: the table of all modes outgrows the loop's buffers.
        grid_kq = (24, 24, 24)
        window = (μ - 0.5eV, μ + 0.5eV)
        run_options(max_device_bytes) = _run_options(model; inner_loop_kq = true, backend,
            symmetry = nothing, calculators = [boltzmann], window_k = window, window_kq = window,
            energy_conservation_tol = tol, n_outer_batch = 1, max_device_bytes, verbosity = 0)
        st_resident = _setup_states(OuterKLoop(), model, (6, 6, 6), grid_kq, run_options(nothing))
        @test st_resident.phonon_batches === nothing && st_resident.phs.nq == st_resident.qpts.n
        # A cap just under twice the table of all modes sends the run through the ω pass, and the
        # kept modes of all q do not fit with the loop's buffers.
        table_all = st_resident.qpts.n * _phonon_state_bytes(Float64, model.nmodes, run_options(nothing).ph_qty)
        st_batch = _setup_states(OuterKLoop(), model, (6, 6, 6), grid_kq, run_options(2 * table_all - 1))
        @test st_batch.phonon_batches !== nothing && st_batch.phs.nq == 0 && st_batch.phs.nmodes == model.nmodes
        # The eligibility of a run for the truncation.
        may_truncate(calculators; tol = tol, backend = backend) = _may_truncate_phonon_modes(OuterKLoop(),
            model, st.els_k, st.els_kq, _run_options(model; inner_loop_kq = true, backend, calculators,
                                                     energy_conservation_tol = tol))
        @test may_truncate([boltzmann])
        @test !may_truncate([boltzmann, _NoTruncation()])
        @test !may_truncate([boltzmann]; tol = Inf)
        @test !may_truncate([boltzmann]; backend = ElectronPhonon.CPUBackend())
    end
end

# When even the kept modes of all q do not fit, the engine builds the phonons of each outer batch,
# for the q of its kept pairs only (reached on Pb with a `max_device_bytes` just under twice the table
# of all modes, on a k+q grid whose table outgrows the loop's buffers): the BTE kernel is the one of
# the run with all phonons resident, and the driver returns no phonons. One-k batches whose k has no
# state within reach of the k+q window keep no pair.
@testitem "phonons per outer batch (GPU)" setup = [Common, PhononModeTruncation] begin
    if MODE_TRUNCATION_GPU
        model = _load_model_from_artifacts("pb")
        eV, K, meV = unit_to_aru(:eV), unit_to_aru(:K), unit_to_aru(:meV)
        μ = 11.68eV; window = (μ - 0.5eV, μ + 0.5eV); σ = 20meV
        backend = ElectronPhonon.gpu_backend()
        grid, grid_kq = (6, 6, 6), (24, 24, 24)
        newcalc() = BoltzmannCalculator{Float64}(;
            occ = ElectronOccupationParams(; Tlist = [300.0K], nlist = 4.0, μlist = μ,
                volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac),
            smearing_list = [SmearingType(:Gaussian, σ)], occupation_method = 5)
        options(; window_kq, n_outer_batch, max_device_bytes = nothing) = _run_options(model;
            inner_loop_kq = true, backend, symmetry = nothing, calculators = [newcalc()],
            window_k = window, window_kq, n_outer_batch, energy_conservation_tol = 6σ,
            max_device_bytes, verbosity = 0)
        # A cap just under twice the table of all modes sends the run through the ω pass.
        function cap_per_batch(; window_kq, n_outer_batch)
            opts = options(; window_kq, n_outer_batch)
            st = _setup_states(OuterKLoop(), model, grid, grid_kq, opts)
            2 * st.qpts.n * _phonon_state_bytes(Float64, model.nmodes, opts.ph_qty) - 1
        end
        runbte(max_device_bytes; window_kq = window, n_outer_batch = 3) = (c = newcalc();
            out = ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid_kq; calculators = [c],
                symmetry = nothing, window_k = window, window_kq, n_outer_batch,
                energy_conservation_tol = 6σ, backend, max_device_bytes, progress_print_step = 10^9,
                verbosity = 0); (c, out))
        cap = cap_per_batch(; window_kq = window, n_outer_batch = 3)
        st = _setup_states(OuterKLoop(), model, grid, grid_kq, options(; window_kq = window,
            n_outer_batch = 3, max_device_bytes = cap))
        @test st.phonon_batches !== nothing
        c_all, out_all = runbte(nothing)
        c_per_batch, out_per_batch = runbte(cap)
        @test out_all.phs === nothing && out_per_batch.phs === nothing
        @test maximum(stack(c_all.Sₒ)) > 0
        @test isapprox(stack(c_per_batch.Sᵢ), stack(c_all.Sᵢ); rtol = 1e-12)
        @test isapprox(stack(c_per_batch.Sₒ), stack(c_all.Sₒ); rtol = 1e-12)

        window_kq = (μ - 0.1eV, μ + 0.5eV)
        cap = cap_per_batch(; window_kq, n_outer_batch = 1)
        st = _setup_states(OuterKLoop(), model, grid, grid_kq, options(; window_kq, n_outer_batch = 1,
            max_device_bytes = cap))
        @test st.phonon_batches !== nothing
        iqs_per_batch = _select_qs_per_outer_batch(backend, st.kpts, st.kqpts, st.qpts, st.els_k,
            st.els_kq, st.phonon_batches.ω_kept, 1, 6σ)
        @test any(isempty, iqs_per_batch) && !all(isempty, iqs_per_batch)
        c_all, _ = runbte(nothing; window_kq, n_outer_batch = 1)
        c_per_batch, _ = runbte(cap; window_kq, n_outer_batch = 1)
        @test maximum(stack(c_all.Sₒ)) > 0
        @test isapprox(stack(c_per_batch.Sᵢ), stack(c_all.Sᵢ); rtol = 1e-12)
        @test isapprox(stack(c_per_batch.Sₒ), stack(c_all.Sₒ); rtol = 1e-12)
    end
end

# End to end with fewer kept modes than nmodes: cubicBN without the polar terms, whose optical
# branches (113-134 meV) lie above Ω = 80 meV of conduction-band window + tol = 30 meV at every q, so
# 3 of the 6 modes are kept. The table of all modes is far below the loop's buffers, so a cap just
# under those buffers plus twice the table opens the ω pass. The truncated run (capped) gives the BTE
# kernel of the untruncated one (same tol) and of tol = Inf, as does a run that builds the 3 kept modes
# per outer batch (one outer k per batch, a cap just under the kept table plus the loop's buffers).
@testitem "phonon mode truncation end to end (GPU)" setup = [Common, PhononModeTruncation] begin
    if MODE_TRUNCATION_GPU
        model = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum = "el")
        model.polar_phonon = model.polar_eph = ElectronPhonon.Polar(nothing)
        model.use_polar_dipole = false
        K, meV = unit_to_aru(:K), unit_to_aru(:meV)
        cbm = 15.338470435789237 * unit_to_aru(:eV)   # at X, on the grid
        σ = 5meV; window = (cbm - 5meV, cbm + 75meV); grid = (40, 40, 40)
        backend = ElectronPhonon.gpu_backend()
        newcalc() = BoltzmannCalculator{Float64}(; occ = ElectronOccupationParams(; Tlist = [300.0K],
                nlist = 1.0, μlist = cbm, volume = model.volume, nelec = 0, spin_degeneracy = 2,
                occ_type = :FermiDirac),
            smearing_list = [SmearingType(:Gaussian, σ)], occupation_method = 5)
        kwargs(tol, max_device_bytes; n_outer_batch = 256) = (; symmetry = nothing, window_k = window,
            window_kq = window, energy_conservation_tol = tol, backend, max_device_bytes, n_outer_batch,
            verbosity = 0)
        run_options(tol, max_device_bytes; n_outer_batch = 256) = _run_options(model;
            inner_loop_kq = true, calculators = [newcalc()],
            kwargs(tol, max_device_bytes; n_outer_batch)...)
        st = _setup_states(OuterKLoop(), model, grid, grid, run_options(6σ, nothing))
        @test st.phonon_batches === nothing && st.phs.nmodes == model.nmodes
        opts = run_options(6σ, nothing)
        phs_all = BatchedPhononState(backend, model.nmodes, 0, opts.ph_qty; ndisp = model.nmodes)
        committed = ElectronPhonon._plan_loop_without_phonon_table(OuterKLoop(), model, st.kpts,
            st.kqpts, st.qpts, st.els_k, st.els_kq, opts, phs_all).committed
        cap = committed + 2 * st.qpts.n * _phonon_state_bytes(Float64, model.nmodes, opts.ph_qty) - 1
        st = _setup_states(OuterKLoop(), model, grid, grid, run_options(6σ, cap))
        @test st.phs.nmodes == 3 < model.nmodes
        # One outer k per batch and a cap just under the kept table, the loop's buffers and the minimal
        # inner tile: the 3 kept modes are built per outer batch.
        opts1 = run_options(6σ, nothing; n_outer_batch = 1)
        phs_kept = BatchedPhononState(backend, 3, 0, opts1.ph_qty; ndisp = model.nmodes)
        loop1 = ElectronPhonon._plan_loop_without_phonon_table(OuterKLoop(), model, st.kpts, st.kqpts,
            st.qpts, st.els_k, st.els_kq, opts1, phs_kept)
        cap_pb = st.qpts.n * _phonon_state_bytes(Float64, 3, opts1.ph_qty; ndisp = model.nmodes) +
            loop1.committed + cld(loop1.bytes.per_pair * min(st.kqpts.n, GPU_PHONONS_RESIDENT_MIN_TILE) * 10, 7) - 1
        st_pb = _setup_states(OuterKLoop(), model, grid, grid, run_options(6σ, cap_pb; n_outer_batch = 1))
        @test st_pb.phs.nmodes == 3 && st_pb.phonon_batches !== nothing
        runbte(tol, max_device_bytes; n_outer_batch = 256) = (c = newcalc();
            ElectronPhonon.run_eph_over_k_and_kq(model, grid, grid; calculators = [c],
                progress_print_step = 10^9, kwargs(tol, max_device_bytes; n_outer_batch)...); c)
        c_trunc, c_full, c_inf = runbte(6σ, cap), runbte(6σ, nothing), runbte(Inf, nothing)
        c_trunc_pb = runbte(6σ, cap_pb; n_outer_batch = 1)
        Sᵢ_inf, Sₒ_inf = stack(c_inf.Sᵢ), stack(c_inf.Sₒ)
        @test count(!iszero, Sᵢ_inf) > 100 && maximum(Sₒ_inf) > 0
        for c in (c_trunc, c_trunc_pb, c_full)
            @test maximum(abs, stack(c.Sᵢ) - Sᵢ_inf) <= 1e-12 * maximum(abs, Sᵢ_inf)
            @test maximum(abs, stack(c.Sₒ) - Sₒ_inf) <= 1e-12 * maximum(abs, Sₒ_inf)
        end
        @test maximum(abs, stack(c_trunc.Sᵢ) - stack(c_full.Sᵢ)) <= 1e-12 * maximum(abs, Sᵢ_inf)
        @test maximum(abs, stack(c_trunc.Sₒ) - stack(c_full.Sₒ)) <= 1e-12 * maximum(abs, Sₒ_inf)
    end
end
