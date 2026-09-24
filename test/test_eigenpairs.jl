using Test
using ElectronPhonon
using ElectronPhonon: Vec3, electron_degen_cutoff, electron_eigenpairs, phonon_eigenpairs,
    gpu_backend, to_device, on_backend,
    AbstractBackend, CPUBackend, inside_window, state_xks
using LinearAlgebra

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const EIGENPAIRS_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# The `quantities == ["eigenvalue"]` branch is the one case that cannot be bitwise equal to its
# no-cache counterpart: without a cache it runs the value-only LAPACK driver, while a cache holds
# the eigenvalues of the full eigensolve, and the two agree only to round-off. What the cache branch
# must do exactly is copy the cached column and leave `u_full` alone.
function _eigenpairs_valueonly_consistent(new, ref, cache)
    e_cache = Array(cache.e_full)  # the cache may be device-resident; compare on the host
    all(eachindex(new)) do ik
        el, el_ref = new[ik], ref[ik]
        el.xk == el_ref.xk && el.rng == el_ref.rng && el.nband == el_ref.nband &&
            el.e_full == e_cache[:, xk_to_ik(el.xk, cache.kpts)] &&
            all(iszero, el.u_full) &&
            maximum(abs, el.e_full - el_ref.e_full) < 1e-13
    end
end

@testset "Eigenpairs" begin
    model = _load_model_from_artifacts("pb"; load_epmat = false)
    kpts = GridKpoints(kpoints_grid((4, 4, 4)))

    # The eigenpairs come from the same `get_el_eigen!` call on the same H(k), so the cache must
    # reproduce `compute_electron_states` bit for bit. Both `fourier_mode`s, since the cache and
    # the states default to different ones.
    for fourier_mode in ("normal", "gridopt")
        eig = electron_eigenpairs(model, kpts; fourier_mode)
        states = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector"]; fourier_mode)
        @test eig.nbasis == model.nw
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

    @testset "consumed by compute_electron_states" begin
        # A cache must be inert: with it, every state a run produces is bit for bit what the same
        # run produces without it, because the eigenpairs are the same `get_el_eigen!` output on
        # the same H(k). Covered for both `compute_electron_states` methods (uniform window and
        # per-k band extent from a selection), every quantity list, and both `el_velocity_mode`s --
        # `:BerryConnection` is the one where "velocity" pulls in "position".
        model_bn = _load_model_from_artifacts("cubicBN"; load_epmat = false)
        window = (10.0, 25.0) .* unit_to_aru(:eV)
        kpts_bn = GridKpoints(kpoints_grid((4, 4, 4)))
        sel = ElectronPhonon.filter_electron_states((4, 4, 4), model_bn, window)
        @test sel.n > 0
        # A per-k band extent that actually varies, so the `sel` method is not just the uniform one.
        @test !allequal(sel.band_extent)

        # `fourier_mode` has to be the same on both arms: H(k) is not bitwise equal between the
        # Fourier modes, and inside a degenerate multiplet that difference is O(1) in `u`. The
        # device path ignores it (it has only the batched interpolator).
        arms = Tuple{AbstractBackend, String}[(CPUBackend(), "normal"),
                                              (CPUBackend(), "gridopt")]
        EIGENPAIRS_GPU_AVAILABLE && push!(arms, (gpu_backend(), "normal"))
        quantity_lists = (["eigenvalue"],
                          ["eigenvalue", "eigenvector"],
                          ["eigenvalue", "eigenvector", "velocity_diagonal"],
                          ["eigenvalue", "eigenvector", "velocity", "position"])
        for (backend, fourier_mode) in arms
            cache = electron_eigenpairs(model_bn, kpts_bn; backend, fourier_mode)
            for mode in (:Direct, :BerryConnection)
                model_bn.el_velocity_mode = mode
                for quantities in quantity_lists
                    ref = compute_electron_states(model_bn, kpts_bn, quantities, window; backend,
                                                  fourier_mode)
                    new = compute_electron_states(model_bn, kpts_bn, quantities, window; backend,
                                                  fourier_mode, eigenpairs = cache)
                    ref_sel = compute_electron_states(model_bn, sel, quantities; backend,
                                                      fourier_mode)
                    new_sel = compute_electron_states(model_bn, sel, quantities; backend,
                                                      fourier_mode, eigenpairs = cache)
                    if quantities == ["eigenvalue"]
                        @test _eigenpairs_valueonly_consistent(new, ref, cache)
                        @test _eigenpairs_valueonly_consistent(new_sel, ref_sel, cache)
                    else
                        @test all(_electron_state_equal.(new, ref))
                        @test all(_electron_state_equal.(new_sel, ref_sel))
                    end
                end
            end
        end
        model_bn.el_velocity_mode = :Direct

        # Teeth: a cache built on a different model is not silently accepted, and one that does not
        # cover a k point the run visits fails at that k point rather than recomputing it.
        pb_cache = electron_eigenpairs(model, kpts)
        @test_throws "eigenpairs holds nbasis" compute_electron_states(
            model_bn, kpts_bn, ["eigenvalue"], window; eigenpairs = pb_cache)
        sub = GridKpoints(Kpoints(kpts_bn.vectors[1:kpts_bn.n-1]; ngrid = kpts_bn.ngrid),
                          kpts_bn.ngrid)
        # The error is raised at the lookup, not by the gather, so it names the k point -- and it
        # reaches here wrapped in a `TaskFailedException` by the threaded state loop, so match the
        # message rather than the exception type.
        @test_throws "does not cover" compute_electron_states(
            model_bn, kpts_bn, ["eigenvalue", "eigenvector"], window;
            eigenpairs = electron_eigenpairs(model_bn, sub))

        # A cache is resident on the backend that built it, so consuming it from the other side is
        # an error rather than a silent copy or a scalar-indexed crawl.
        if EIGENPAIRS_GPU_AVAILABLE
            host_cache = electron_eigenpairs(model_bn, kpts_bn)
            device_cache = electron_eigenpairs(model_bn, kpts_bn; backend = gpu_backend())
            @test_throws "resident on the host" compute_electron_states(
                model_bn, kpts_bn, ["eigenvalue", "eigenvector"], window;
                backend = gpu_backend(), eigenpairs = host_cache)
            @test_throws "resident on the device" compute_electron_states(
                model_bn, kpts_bn, ["eigenvalue", "eigenvector"], window;
                eigenpairs = device_cache)
        end
    end

    @testset "consumed by filter_electron_states" begin
        # The filter solves H(k) eigenvalue-only, so a cache -- which holds the full eigensolve's
        # eigenvalues -- cannot reproduce its energies bit for bit; on this fixture the two solves
        # differ by 8.9e-16 aru. What must be identical is the window DECISION, which is discrete.
        # The cache has to cover the INPUT grid: the filter needs the energies at every candidate
        # k point, not at the surviving ones.
        eV = unit_to_aru(:eV)
        ef = 11.594123 * eV
        window_pb = (ef - 0.2eV, ef + 0.2eV)
        grid = (12, 12, 12)
        kpts_f = GridKpoints(kpoints_grid(grid))

        selection_equal(a, b) = a.kpts.vectors == b.kpts.vectors && a.iks == b.iks &&
            a.ibands == b.ibands && a.band_extent == b.band_extent &&
            a.nstates_base == b.nstates_base

        ref = filter_electron_states(grid, model, window_pb; fourier_mode = "gridopt")
        @test 0 < ref.kpts.n < kpts_f.n        # a genuinely windowed selection,
        @test !allequal(ref.band_extent)       # whose in-window band range varies with k

        cache_host = electron_eigenpairs(model, kpts_f; fourier_mode = "gridopt")
        arms = Tuple{AbstractBackend, Eigenpairs}[(CPUBackend(), cache_host)]
        EIGENPAIRS_GPU_AVAILABLE && push!(arms,
            (gpu_backend(), electron_eigenpairs(model, kpts_f; backend = gpu_backend())))
        for (backend, cache) in arms
            new = filter_electron_states(grid, model, window_pb; backend,
                                         fourier_mode = "gridopt", eigenpairs = cache)
            @test selection_equal(new, ref)

            # What the cache buys: the window decision is taken on exactly the energies the states
            # of a run over the same cache carry, so the two cannot disagree.
            states = compute_electron_states(model, new, ["eigenvalue", "eigenvector"];
                                             backend, fourier_mode = "gridopt", eigenpairs = cache)
            e_cache = Array(cache.e_full)  # the cache may be device-resident; compare on the host
            @test all(1:new.kpts.n) do ik
                j = xk_to_ik(new.kpts.vectors[ik], cache.kpts)
                states[ik].e_full == e_cache[:, j] &&
                    new.band_extent[ik] == inside_window(e_cache[:, j], window_pb...)
            end

            # Teeth: the decision is read out of the cache at the looked-up k point. A cache whose
            # k columns are rotated by one is well-formed and covers the grid, so only a run that
            # reads it can notice.
            rot = circshift(1:kpts_f.n, 1)
            rotated = Eigenpairs(model.nw, kpts_f, cache.e_full[:, rot], cache.u_full[:, :, rot])
            @test !selection_equal(filter_electron_states(grid, model, window_pb; backend,
                fourier_mode = "gridopt", eigenpairs = rotated), ref)

            # A k point the cache does not hold is an error, not a silent recompute -- so a cache
            # over the *filtered* set, the natural mistake, is rejected. On the CPU path the throw
            # reaches here wrapped by the threaded loop, so match the message.
            @test_throws "does not cover" filter_electron_states(grid, model, window_pb; backend,
                fourier_mode = "gridopt", eigenpairs = electron_eigenpairs(model, ref.kpts; backend))
        end

        # The entry guards apply here too, on a path whose device eigensolve would otherwise take a
        # host cache without complaint.
        @test_throws "eigenpairs holds nbasis" filter_electron_states(grid, model, window_pb;
            eigenpairs = Eigenpairs(2, kpts_f, zeros(2, kpts_f.n),
                                    zeros(ComplexF64, 2, 2, kpts_f.n)))
        if EIGENPAIRS_GPU_AVAILABLE
            @test_throws "resident on the device" filter_electron_states(grid, model, window_pb;
                eigenpairs = electron_eigenpairs(model, kpts_f; backend = gpu_backend()))
        end

        # Under `mpi_comm` each rank filters its own slice of the grid, against the same
        # rank-replicated cache. COMM_SELF is one rank, so it checks the forward and the
        # gather/scatter; a genuine multi-rank slice check needs `mpiexec -n N`.
        MPI = ElectronPhonon.MPI
        MPI.Initialized() || MPI.Init()
        mpi = filter_electron_states(grid, model, window_pb; fourier_mode = "gridopt",
                                     mpi_comm = MPI.COMM_SELF, eigenpairs = cache_host)
        @test Set(zip(state_xks(mpi), mpi.ibands)) == Set(zip(state_xks(ref), ref.ibands))
        @test mpi.nstates_base == ref.nstates_base
    end

    @testset "consumed by compute_phonon_states" begin
        # A phonon cache is inert in exactly the same sense as an electron one: it holds the same
        # eigensolve output on the same dynamical matrix, so a with-cache run reproduces a
        # without-cache one bit for bit -- including `velocity_diagonal` and, on a polar model,
        # `eph_dipole_coeff`, both of which are derived from `ph.u` rather than stored in the cache.
        model_bn = _load_model_from_artifacts("cubicBN"; load_epmat = false)
        @test model_bn.polar_phonon.use   # so the dipole arm below is not vacuous
        full = ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"]
        quantity_lists = (["eigenvalue"], ["eigenvalue", "eigenvector"], full)

        # The device phonon path refuses polar models, so cubicBN is CPU-only there as it is in
        # production. One `fourier_mode` only: with a cache `dyn` is never built, so every claim
        # here is mode-independent, and the mode-crossed inertness of the cacheless path is the
        # subject of the A/B against the pre-cache code rather than of this testset.
        fourier_mode = "normal"
        arms = Tuple{Any, AbstractBackend}[(model, CPUBackend()), (model_bn, CPUBackend())]
        EIGENPAIRS_GPU_AVAILABLE && push!(arms, (model, gpu_backend()))
        for (m, backend) in arms
            solved = compute_phonon_states(m, kpts, full; fourier_mode, backend)
            cache = _phonon_eigenpairs(solved, kpts, backend)
            @test cache.nbasis == m.nmodes
            # The same cache with its q columns rotated by one: well-formed, on the right backend
            # and covering every q point, but holding the wrong eigenpair at each. This is what
            # gives every claim below teeth -- a run that ignored the cache would be unaffected.
            rot = circshift(1:kpts.n, 1)
            rotated = Eigenpairs(m.nmodes, kpts, cache.e_full[:, rot], cache.u_full[:, :, rot])

            for quantities in quantity_lists
                ref = compute_phonon_states(m, kpts, quantities; fourier_mode, backend)
                new = compute_phonon_states(m, kpts, quantities; fourier_mode, backend,
                                            eigenpairs = cache)
                bad = compute_phonon_states(m, kpts, quantities; fourier_mode, backend,
                                            eigenpairs = rotated)
                e_cache = Array(cache.e_full)   # the cache may be device-resident
                if quantities == ["eigenvalue"]
                    # The one list that cannot be bitwise: without a cache it runs the value-only
                    # solve, while the cache holds the full eigensolve's frequencies. What the
                    # cache branch must do exactly is copy the cached column and leave `u` alone.
                    # The two solves differ by 9.1e-17 relative on Pb and 1.8e-9 on the polar
                    # cubicBN, whose long-range term the value-only path treats differently.
                    @test all(iq -> new[iq].e == e_cache[:, iq], 1:kpts.n)
                    @test all(iq -> all(iszero, new[iq].u), 1:kpts.n)
                    @test all(iq -> new[iq].xq == ref[iq].xq, 1:kpts.n)
                    e_ref = reduce(hcat, [p.e for p in ref])
                    @test norm(reduce(hcat, [p.e for p in new]) - e_ref) / norm(e_ref) < 1e-8
                else
                    @test all(_phonon_state_equal.(new, ref))
                end
                @test !any(_phonon_state_equal.(bad, ref))
            end

            # A run whose q list is not the cache's own list in its own order. Every other arm
            # here looks the cache up at `1:nq`, where the device path's `view(u_full, :, :, iqs)`
            # is a trivial permutation; this one makes `iqs` a genuine reordering, so the indexed
            # view is materialized through its device index vector. Both arms run over the same
            # list, so the ordering `GridKpoints` itself chooses is irrelevant to the claim.
            rev = GridKpoints(Kpoints(reverse(kpts.vectors); ngrid = kpts.ngrid), kpts.ngrid)
            @test map(xq -> xk_to_ik(xq, kpts), rev.vectors) != 1:kpts.n
            @test all(_phonon_state_equal.(
                compute_phonon_states(m, rev, full; fourier_mode, backend, eigenpairs = cache),
                compute_phonon_states(m, rev, full; fourier_mode, backend)))

            # The tooth the plan that motivates this feature asks for, because "run without the
            # cache and check it differs" is vacuous if the eigensolver happens to reproduce the
            # cache. The perturbation must not be a gauge transformation: a phase on one column
            # leaves `vdiag = u' (dD/dk) u` and the dipole coefficients exactly invariant, so it
            # would prove nothing about them. A rotation mixing modes 1 and 2 is a real change of
            # basis -- and it has to be applied where those two modes are SPLIT, since inside a
            # degenerate pair the rotation is again only a gauge choice. Measured moves at the
            # most-split q point: `u` 0.64 (Pb) / 0.44 (cubicBN) of max|u|, `vdiag` 0.32 / 0.19 of
            # max|vdiag|, `eph_dipole_coeff` 3.0e-2 relative (cubicBN).
            iq_split = argmax([solved[iq].e[2] - solved[iq].e[1] for iq in 1:kpts.n])
            u_pert = Array(cache.u_full)
            u1, u2 = u_pert[:, 1, iq_split], u_pert[:, 2, iq_split]
            u_pert[:, 1, iq_split] .= cos(0.7) .* u1 .- sin(0.7) .* u2
            u_pert[:, 2, iq_split] .= sin(0.7) .* u1 .+ cos(0.7) .* u2
            perturbed = Eigenpairs(m.nmodes, kpts, cache.e_full, to_device(backend, u_pert))
            pert = compute_phonon_states(m, kpts, full; fourier_mode, backend,
                                         eigenpairs = perturbed)
            solved_u = maximum(iq -> maximum(abs, solved[iq].u), 1:kpts.n)
            solved_v = maximum(iq -> maximum(maximum.(abs, solved[iq].vdiag)), 1:kpts.n)
            @test maximum(iq -> maximum(abs, pert[iq].u - solved[iq].u), 1:kpts.n) > 0.1solved_u
            @test maximum(iq -> maximum(maximum.(abs, pert[iq].vdiag - solved[iq].vdiag)),
                          1:kpts.n) > 0.1solved_v
            @test all(iq -> pert[iq].e == solved[iq].e, 1:kpts.n)   # only `u` was perturbed
            if m.polar_phonon.use
                solved_d = maximum(iq -> maximum(abs, solved[iq].eph_dipole_coeff), 1:kpts.n)
                @test maximum(iq -> maximum(abs, pert[iq].eph_dipole_coeff -
                                  solved[iq].eph_dipole_coeff), 1:kpts.n) > 1e-3solved_d
            end

            # A cache that does not cover every q point a run visits is an error, not a silent
            # recompute, and it names the q point. On the CPU path the throw reaches here wrapped
            # by the threaded state loop, so match the message rather than the exception type.
            sub_q = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
            partial = _phonon_eigenpairs(compute_phonon_states(m, sub_q, full; fourier_mode,
                                                               backend), sub_q, backend)
            @test_throws "does not cover" compute_phonon_states(m, kpts, full; fourier_mode,
                                                                backend, eigenpairs = partial)
        end

        # An electron cache has `nbasis = nw`, a phonon one `nbasis = nmodes`, so passing the wrong
        # species is caught at the entry rather than broadcasting or mismatching deep inside.
        @test model.nw != model.nmodes
        @test_throws "eigenpairs holds nbasis" compute_phonon_states(
            model, kpts, ["eigenvalue"]; eigenpairs = electron_eigenpairs(model, kpts))

        # The same backend-residency rule the electron caches follow.
        if EIGENPAIRS_GPU_AVAILABLE
            host_cache = _phonon_eigenpairs(compute_phonon_states(model, kpts, full), kpts)
            device_cache = _phonon_eigenpairs(
                compute_phonon_states(model, kpts, full; backend = gpu_backend()), kpts,
                gpu_backend())
            @test_throws "resident on the host" compute_phonon_states(
                model, kpts, full; backend = gpu_backend(), eigenpairs = host_cache)
            @test_throws "resident on the device" compute_phonon_states(
                model, kpts, full; eigenpairs = device_cache)

            # CPU and device arms fed the SAME basis agree: the eigenvectors and frequencies are
            # bit-identical because both copy them out of the cache, and the only quantity either
            # backend still computes, `vdiag`, agrees to the rotation's round-off. Without a cache
            # the two bases are a multiplet rotation apart, which is what makes this worth stating.
            host_states = compute_phonon_states(model, kpts, full)
            dev_states = compute_phonon_states(model, kpts, full; backend = gpu_backend(),
                eigenpairs = _phonon_eigenpairs(host_states, kpts, gpu_backend()))
            @test all(iq -> dev_states[iq].e == host_states[iq].e &&
                            dev_states[iq].u == host_states[iq].u, 1:kpts.n)
            @test maximum(iq -> maximum(maximum.(abs, dev_states[iq].vdiag -
                                                      host_states[iq].vdiag)), 1:kpts.n) < 1e-17
        end
    end

    @testset "built by phonon_eigenpairs" begin
        # The builder runs the solve `compute_phonon_states` runs at the same q (per-q LAPACK on the
        # host, the batched eigensolve on the device), so the two agree bit for bit, and a cache
        # built here is inert in a run. cubicBN covers the polar (dipole) term on the host.
        model_bn = _load_model_from_artifacts("cubicBN"; load_epmat = false)
        arms = Tuple{Any, AbstractBackend, String}[(model, CPUBackend(), "normal"),
            (model, CPUBackend(), "gridopt"), (model_bn, CPUBackend(), "gridopt")]
        EIGENPAIRS_GPU_AVAILABLE && push!(arms, (model, gpu_backend(), "gridopt"))
        for (m, backend, fourier_mode) in arms
            cache = phonon_eigenpairs(m, kpts; fourier_mode, backend)
            ref = compute_phonon_states(m, kpts, ["eigenvalue", "eigenvector"]; fourier_mode,
                                        backend)
            @test cache.nbasis == m.nmodes
            @test on_backend(backend, cache.e_full) && on_backend(backend, cache.u_full)
            e, u = Array(cache.e_full), Array(cache.u_full)
            @test all(iq -> e[:, iq] == ref[iq].e && u[:, :, iq] == ref[iq].u, 1:kpts.n)
            # negative control: a one-q offset must fail the same comparison
            @test !any(iq -> u[:, :, iq] == ref[mod1(iq + 1, kpts.n)].u, 1:kpts.n)
        end
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
            # Placing a host cache on the device moves the arrays as they are.
            placed = to_device(gpu_backend(), eig_cpu)
            @test placed.e_full isa CuMatrix{Float64} && placed.u_full isa CuArray{ComplexF64, 3}
            @test Array(placed.e_full) == eig_cpu.e_full && Array(placed.u_full) == eig_cpu.u_full
            @test placed.kpts === eig_cpu.kpts && placed.nbasis == eig_cpu.nbasis
            @test to_device(CPUBackend(), eig_cpu) === eig_cpu
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
            @info "CUDA not functional - skipping the GPU Eigenpairs test"
        end
    end
end
