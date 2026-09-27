using Test
using ElectronPhonon
using ElectronPhonon: phonon_eigenpairs, has_vdiag, has_dipole, HostBatchedPhononState

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
                        isequal(b.u[:, :, iq], ph.u) &&
                        (!has_vdiag(b) ||
                         isequal(b.vdiag[:, :, iq], reinterpret(reshape, Float64, ph.vdiag))) &&
                        (!has_dipole(b) || (isequal(b.eph_dipole_coeff[:, iq], ph.eph_dipole_coeff) &&
                                            isequal(b.eph_r_coeff[:, :, iq], ph.eph_r_coeff)))
                end
                # negative control: a one-q offset must fail the same comparison
                if quantities != ["eigenvalue"]
                    @test !any(iq -> b.u[:, :, iq] == ref[mod1(iq + 1, kpts.n)].u, 1:kpts.n)
                else
                    @test all(iszero, b.u)
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
end
