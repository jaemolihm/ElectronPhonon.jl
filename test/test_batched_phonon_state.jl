using Test
using ElectronPhonon
using ElectronPhonon: phonon_eigenpairs, gpu_backend, on_backend, to_device, Vec3, CPUBackend,
    copy_batched_phonon_states!

# CUDA is a weak dependency (not a test dependency), so load it defensively and skip the GPU
# tests when it is unavailable or non-functional (e.g. CPU-only CI).
const BATCHED_PHONON_GPU_AVAILABLE = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

@testset "BatchedPhononState" begin
    kpts = GridKpoints(kpoints_grid((4, 4, 4)))
    model_pb = _load_model_from_artifacts("pb"; load_epmat = false)
    # Polar. Its e-ph dipole coefficients are zero unless the e-ph data is loaded too.
    model_bn = _load_model_from_artifacts("cubicBN")

    # The per-point quantity strings of each batched quantity name.
    per_point_name = Dict(:e => "eigenvalue", :u => "eigenvector", :vdiag => "velocity_diagonal",
                          :eph_dipole_coeff => "eph_dipole_coeff")
    quantity_lists = ([:e], [:e, :u], [:e, :u, :vdiag],
        [:e, :u, :vdiag, :eph_dipole_coeff, :eph_r_coeff], [:u, :eph_dipole_coeff, :eph_r_coeff],
        [:e, :eph_dipole_coeff])

    @testset "host builder == compute_phonon_states" begin
        # The per-point builder is the oracle: both run the same per-q kernels, so every stored
        # quantity must equal the corresponding `PhononState` field bit for bit, with and without a
        # cache. A list without `:u` that needs the eigenmodes (`[:e, :eph_dipole_coeff]`) solves
        # into scratch, as the per-point builder solves into its state.
        for m in (model_pb, model_bn), fourier_mode in ("normal", "gridopt")
            cache = phonon_eigenpairs(m, kpts; fourier_mode)
            for quantities in quantity_lists, eigenpairs in (nothing, cache)
                ph_states = compute_phonon_states_batched(m, kpts, quantities; fourier_mode, eigenpairs)
                ref = compute_phonon_states(m, kpts,
                    [per_point_name[x] for x in quantities if haskey(per_point_name, x)];
                    fourier_mode, eigenpairs)
                @test ph_states.qpts === kpts && ph_states.nmodes == m.nmodes && ph_states.nq == kpts.n
                @test all(name -> (getfield(ph_states, name) !== nothing) == (name ∈ quantities),
                          (:e, :u, :vdiag, :eph_dipole_coeff, :eph_r_coeff))
                @test all(eachindex(ref)) do iq
                    ph = ref[iq]
                    (:e ∉ quantities || isequal(ph_states.e[:, iq], ph.e)) &&
                        (:u ∉ quantities || isequal(ph_states.u[:, :, iq], ph.u)) &&
                        (:vdiag ∉ quantities ||
                         isequal(ph_states.vdiag[:, :, iq], reinterpret(reshape, Float64, ph.vdiag))) &&
                        (:eph_dipole_coeff ∉ quantities ||
                         isequal(ph_states.eph_dipole_coeff[:, iq], ph.eph_dipole_coeff)) &&
                        (:eph_r_coeff ∉ quantities || isequal(ph_states.eph_r_coeff[:, :, iq], ph.eph_r_coeff))
                end
                # negative control: a one-q offset must fail the same comparison
                if :u ∈ quantities
                    @test !any(iq -> ph_states.u[:, :, iq] == ref[mod1(iq + 1, kpts.n)].u, 1:kpts.n)
                end
            end
        end
        # The polar model's dipole coefficients are not trivially zero, so the comparison above has
        # teeth there (`eph_r_coeff` is not implemented for 3D dipoles and stays zero).
        ph_states = compute_phonon_states_batched(model_bn, kpts, [:u, :eph_dipole_coeff])
        @test !all(iszero, ph_states.eph_dipole_coeff)
    end

    @testset "fields and argument checks" begin
        ph_states = compute_phonon_states_batched(model_pb, kpts, [:e, :u])
        @test ph_states.vdiag === nothing && ph_states.eph_dipole_coeff === nothing && ph_states.eph_r_coeff === nothing
        @test occursin("BatchedPhononState{Float64}(nmodes = 3, nq = 64", sprint(show, ph_states))
        @test compute_phonon_states_batched(model_pb, kpts, Symbol[]).e === nothing

        @test_throws "unknown phonon quantities [:velocity]" compute_phonon_states_batched(
            model_pb, kpts, [:velocity])
        @test_throws "duplicates" compute_phonon_states_batched(model_pb, kpts, [:e, :e])
        @test_throws "eigenpairs holds nbasis" compute_phonon_states_batched(model_pb, kpts,
            [:e]; eigenpairs = ElectronPhonon.electron_eigenpairs(model_pb, kpts))
        sub_q = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
        @test_throws "does not cover" compute_phonon_states_batched(model_pb, kpts,
            [:e, :u]; eigenpairs = phonon_eigenpairs(model_pb, sub_q))

        # The constructor checks every field against (nmodes, nq).
        @test_throws "e must be (nmodes, nq)" BatchedPhononState{Float64}(3, 64, kpts, ph_states.e[:, 1:2],
            ph_states.u, nothing, nothing, nothing)
        @test_throws "qpts holds 64 points" BatchedPhononState{Float64}(3, 2, kpts, nothing, nothing,
            nothing, nothing, nothing)
    end

    @testset "empty container and copy" begin
        ph_states = compute_phonon_states_batched(model_pb, kpts, [:e, :u, :vdiag])
        inds = [5, 2, 64, 17]
        dst = BatchedPhononState(CPUBackend(), 3, 6, [:e, :u, :vdiag])
        @test dst.qpts === nothing && dst.nq == 6 && dst.eph_dipole_coeff === nothing
        copy_batched_phonon_states!(dst, ph_states, inds)
        @test dst.e[:, 1:4] == ph_states.e[:, inds] && dst.u[:, :, 1:4] == ph_states.u[:, :, inds] &&
              dst.vdiag[:, :, 1:4] == ph_states.vdiag[:, :, inds]
        @test_throws "cannot copy 7 q points" copy_batched_phonon_states!(dst, ph_states, 1:7)
        @test_throws BoundsError copy_batched_phonon_states!(dst, ph_states, [65])
        @test_throws MethodError copy_batched_phonon_states!(BatchedPhononState(CPUBackend(), 3,
            6, [:e, :u]), ph_states, inds)
    end

    @testset "GPU" begin
        if BATCHED_PHONON_GPU_AVAILABLE
            CUDA.allowscalar(false)
            backend = gpu_backend()
            full = [:e, :u]

            # The device solve is chunked over q at the Fourier block width. Against one
            # eigensolve over the whole set with the same Fourier blocks (the unchunked build), it
            # is bitwise: the batched eigensolve is per matrix. The grid is sized for two chunks,
            # both above the eigensolver's own 65 536-matrix split.
            qgrid = GridKpoints(kpoints_grid((112, 112, 112)))
            ph_states = compute_phonon_states_batched(model_pb, qgrid, full; backend)
            itp = ElectronPhonon.get_interpolator(to_device(backend, model_pb.ph_dyn);
                fourier_mode = "batched", backend, nk_hint = qgrid.n)
            @test 65_536 < itp.batch_size < qgrid.n
            D = ElectronPhonon._fourier_hk_batched(itp, qgrid.vectors)
            msqrt = to_device(backend, sqrt.(model_pb.mass))
            D ./= reshape(msqrt, :, 1, 1)
            D ./= reshape(msqrt, 1, :, 1)
            Esq, U = ElectronPhonon.eigen_batched(D)
            @test on_backend(backend, ph_states.e) && on_backend(backend, ph_states.u)
            @test isequal(Array(ph_states.e), Array(sign.(Esq) .* sqrt.(abs.(Esq))))
            @test isequal(Array(ph_states.u), Array(U ./ reshape(msqrt, :, 1, 1)))
            ph_states = D = U = nothing

            # The cache arm is a gather: every q of a list in another order is the cache's
            # column at that q, bit for bit.
            cache = phonon_eigenpairs(model_pb, kpts; backend)
            rev = GridKpoints(Kpoints(reverse(kpts.vectors); ngrid = kpts.ngrid), kpts.ngrid)
            ph_states = compute_phonon_states_batched(model_pb, rev, full; backend, eigenpairs = cache)
            @test isequal(Array(ph_states.e), Array(cache.e_full)[:, end:-1:1])
            @test isequal(Array(ph_states.u), Array(cache.u_full)[:, :, end:-1:1])
            b_u = compute_phonon_states_batched(model_pb, rev, [:u]; backend, eigenpairs = cache)
            @test b_u.e === nothing && isequal(Array(b_u.u), Array(ph_states.u))
            sub_q = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
            @test_throws "does not cover" compute_phonon_states_batched(model_pb, kpts, full;
                backend, eigenpairs = phonon_eigenpairs(model_pb, sub_q; backend))

            # The value-only solve agrees with the full one to round-off.
            b_val = compute_phonon_states_batched(model_pb, kpts, [:e]; backend)
            @test maximum(abs, Array(b_val.e) - Array(cache.e_full)) < 1e-12

            q0 = GridKpoints(Kpoints(Vec3{Float64}[]; ngrid = kpts.ngrid), kpts.ngrid)
            @test size(compute_phonon_states_batched(model_pb, q0, full; backend).u) == (3, 3, 0)

            # The copy on the device, and a host container streamed into a device buffer.
            b_host = compute_phonon_states_batched(model_pb, kpts, full)
            inds = [5, 2, 64, 17]
            for src in (ph_states, b_host)
                tile = BatchedPhononState(backend, 3, 6, full)
                copy_batched_phonon_states!(tile, src, inds)
                @test on_backend(backend, tile.u)
                @test Array(tile.e)[:, 1:4] == Array(src.e)[:, inds]
                @test Array(tile.u)[:, :, 1:4] == Array(src.u)[:, :, inds]
            end

            # What the device does not compute is refused, not silently left zero.
            @test_throws "[:vdiag] are not supported" compute_phonon_states_batched(
                model_pb, kpts, [:e, :u, :vdiag]; backend)
            @test_throws "[:eph_dipole_coeff] are not supported" compute_phonon_states_batched(
                model_pb, kpts, [:u, :eph_dipole_coeff]; backend)
            @test_throws "does not support polar" compute_phonon_states_batched(model_bn, kpts,
                full; backend)
        end
    end
end
