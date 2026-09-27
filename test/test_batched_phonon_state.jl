using Test
using ElectronPhonon
using ElectronPhonon: phonon_eigenpairs, has_vdiag, has_dipole, HostBatchedPhononState,
    gpu_backend, on_backend, to_device, Vec3

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

    # Every quantity list a driver passes to `compute_phonon_states`.
    quantity_lists = (["eigenvalue"], ["eigenvalue", "eigenvector"],
        ["eigenvalue", "eigenvector", "velocity_diagonal"],
        ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"],
        ["eigenvector", "eph_dipole_coeff"], ["eigenvalue", "eigenvector", "eph_dipole_coeff"])

    @testset "host builder == compute_phonon_states" begin
        # The per-point builder is the oracle: both run the same per-q kernels, so every present
        # stack must equal the corresponding `PhononState` field bit for bit, with and without a
        # cache.
        for m in (model_pb, model_bn), fourier_mode in ("normal", "gridopt")
            cache = phonon_eigenpairs(m, kpts; fourier_mode)
            for quantities in quantity_lists, eigenpairs in (nothing, cache)
                b = compute_phonon_states_batched(m, kpts, quantities; fourier_mode, eigenpairs)
                ref = compute_phonon_states(m, kpts, quantities; fourier_mode, eigenpairs)
                @test b isa HostBatchedPhononState{Float64}
                @test b.qpts === kpts && b.nmodes == m.nmodes
                @test has_vdiag(b) == ("velocity_diagonal" ∈ quantities)
                @test has_dipole(b) == ("eph_dipole_coeff" ∈ quantities)
                @test all(eachindex(ref)) do iq
                    ph = ref[iq]
                    ph.xq == b.qpts.vectors[iq] && isequal(b.e[:, iq], ph.e) &&
                        (quantities == ["eigenvalue"] || isequal(b.u[:, :, iq], ph.u)) &&
                        (!has_vdiag(b) ||
                         isequal(b.vdiag[:, :, iq], reinterpret(reshape, Float64, ph.vdiag))) &&
                        (!has_dipole(b) || (isequal(b.eph_dipole_coeff[:, iq], ph.eph_dipole_coeff) &&
                                            isequal(b.eph_r_coeff[:, :, iq], ph.eph_r_coeff)))
                end
                # negative control: a one-q offset must fail the same comparison
                if quantities != ["eigenvalue"]
                    @test !any(iq -> b.u[:, :, iq] == ref[mod1(iq + 1, kpts.n)].u, 1:kpts.n)
                else
                    @test size(b.u) == (m.nmodes, m.nmodes, 0)  # value-only: no eigenvectors
                end
            end
        end
        # The polar model's dipole coefficients are not trivially zero, so the comparison above has
        # teeth there (`eph_r_coeff` is not implemented for 3D dipoles and stays zero).
        b = compute_phonon_states_batched(model_bn, kpts, ["eigenvector", "eph_dipole_coeff"])
        @test !all(iszero, b.eph_dipole_coeff)
    end

    @testset "absent quantities and argument checks" begin
        b = compute_phonon_states_batched(model_pb, kpts, ["eigenvalue", "eigenvector"])
        @test size(b.vdiag) == (3, model_pb.nmodes, 0)
        @test size(b.eph_dipole_coeff, 2) == 0 && size(b.eph_r_coeff, 3) == 0
        @test occursin("BatchedPhononState{Float64}(nmodes = 3, nq = 64", sprint(show, b))

        b0 = compute_phonon_states_batched(model_pb, kpts, String[])
        @test all(iszero, b0.e) && all(iszero, b0.u)

        @test_throws "not an allowed quantity" compute_phonon_states_batched(
            model_pb, kpts, ["velocity"])
        @test_throws "eigenpairs holds nbasis" compute_phonon_states_batched(model_pb, kpts,
            ["eigenvalue"]; eigenpairs = ElectronPhonon.electron_eigenpairs(model_pb, kpts))
        sub_q = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
        @test_throws "does not cover" compute_phonon_states_batched(model_pb, kpts,
            ["eigenvalue", "eigenvector"]; eigenpairs = phonon_eigenpairs(model_pb, sub_q))

        # The constructor checks every stack against (nmodes, qpts.n).
        (; e, u, vdiag, eph_dipole_coeff, eph_r_coeff) = b
        @test_throws "e must be of size (3, 64)" BatchedPhononState(3, kpts, e[:, 1:2], u, vdiag,
            eph_dipole_coeff, eph_r_coeff)
        @test_throws "vdiag must be of size (3, 3, 64) or empty" BatchedPhononState(3, kpts, e, u,
            zeros(3, 3, 2), eph_dipole_coeff, eph_r_coeff)
        @test_throws "both present or both empty" BatchedPhononState(3, kpts, e, u, vdiag,
            zeros(ComplexF64, 3, 64), eph_r_coeff)
    end

    @testset "GPU" begin
        if BATCHED_PHONON_GPU_AVAILABLE
            CUDA.allowscalar(false)
            backend = gpu_backend()
            full = ["eigenvalue", "eigenvector"]

            # The device solve is chunked over q at the Fourier block width. Against one
            # eigensolve over the whole set with the same Fourier blocks (the unchunked build), it
            # is bitwise: the batched eigensolve is per matrix. The grid is sized for two chunks,
            # both above the eigensolver's own 65 536-matrix split.
            qgrid = GridKpoints(kpoints_grid((112, 112, 112)))
            b = compute_phonon_states_batched(model_pb, qgrid, full; backend)
            itp = ElectronPhonon.get_interpolator(to_device(backend, model_pb.ph_dyn);
                fourier_mode = "batched", backend, nk_hint = qgrid.n)
            @test 65_536 < itp.batch_size < qgrid.n
            D = ElectronPhonon._fourier_hk_batched(itp, qgrid.vectors)
            msqrt = to_device(backend, sqrt.(model_pb.mass))
            D ./= reshape(msqrt, :, 1, 1)
            D ./= reshape(msqrt, 1, :, 1)
            Esq, U = ElectronPhonon.eigen_batched(D)
            @test on_backend(backend, b.e) && on_backend(backend, b.u)
            @test isequal(Array(b.e), Array(sign.(Esq) .* sqrt.(abs.(Esq))))
            @test isequal(Array(b.u), Array(U ./ reshape(msqrt, :, 1, 1)))
            b = D = U = nothing

            # The cache arm is a gather: every q of a list in another order is the cache's
            # column at that q, bit for bit.
            cache = phonon_eigenpairs(model_pb, kpts; backend)
            rev = GridKpoints(Kpoints(reverse(kpts.vectors); ngrid = kpts.ngrid), kpts.ngrid)
            b = compute_phonon_states_batched(model_pb, rev, full; backend, eigenpairs = cache)
            @test isequal(Array(b.e), Array(cache.e_full)[:, end:-1:1])
            @test isequal(Array(b.u), Array(cache.u_full)[:, :, end:-1:1])
            b_val = compute_phonon_states_batched(model_pb, rev, ["eigenvalue"]; backend,
                                                  eigenpairs = cache)
            @test isequal(Array(b_val.e), Array(b.e)) && size(b_val.u) == (3, 3, 0)
            sub_q = GridKpoints(Kpoints(kpts.vectors[1:kpts.n-1]; ngrid = kpts.ngrid), kpts.ngrid)
            @test_throws "does not cover" compute_phonon_states_batched(model_pb, kpts, full;
                backend, eigenpairs = phonon_eigenpairs(model_pb, sub_q; backend))

            # The value-only solve agrees with the full one to round-off and leaves `u` zero.
            b_val = compute_phonon_states_batched(model_pb, kpts, ["eigenvalue"]; backend)
            @test maximum(abs, Array(b_val.e) - Array(cache.e_full)) < 1e-12
            @test size(b_val.u) == (3, 3, 0) && on_backend(backend, b_val.u)

            q0 = GridKpoints(Kpoints(Vec3{Float64}[]; ngrid = kpts.ngrid), kpts.ngrid)
            @test size(compute_phonon_states_batched(model_pb, q0, full; backend).u) == (3, 3, 0)

            # What the device does not compute is refused, not silently left zero.
            @test_throws "\"velocity_diagonal\" is not supported" compute_phonon_states_batched(
                model_pb, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal"]; backend)
            @test_throws "\"eph_dipole_coeff\" is not supported" compute_phonon_states_batched(
                model_pb, kpts, ["eigenvector", "eph_dipole_coeff"]; backend)
            @test_throws "does not support polar" compute_phonon_states_batched(model_bn, kpts,
                full; backend)
        end
    end
end
