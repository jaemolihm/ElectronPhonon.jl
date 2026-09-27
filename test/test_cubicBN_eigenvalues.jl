using Test
using ElectronPhonon
using LinearAlgebra

@testset "cubicBN eigenvalues" begin
    # Test routines that calculate eigenvalues
    model = _load_model_from_artifacts("cubicBN"; load_epmat = false)

    kpts = kpoints_grid((3, 3, 3))

    # electron
    el_states = compute_electron_states(model, kpts, ["eigenvalue"], fourier_mode="gridopt")
    el_e_ref = stack(el.e_full for el in el_states)

    for fourier_mode in ["normal", "gridopt", "batched", "batched-gridopt"]
        el_e = compute_eigenvalues_el(model, kpts; fourier_mode)
        @test el_e ≈ el_e_ref
    end

    # phonon
    ph_states = compute_phonon_states(model, kpts, ["eigenvalue"], fourier_mode="gridopt")
    ph_e_ref = stack(ph.e for ph in ph_states)

    # Gauge-invariant checks, since the modes' basis inside a degenerate multiplet is free:
    # `u Diag(ω|ω|) u'` (ω² rather than ω, whose √ amplifies round-off ~1e8x at the Γ acoustic
    # modes) and `Σ_ν 2ω_ν vdiag_ν = Σ_ν u_ν' ∂D u_ν` (a trace over each multiplet). The trace
    # vanishes by symmetry at some q, so it is compared against the per-mode scale: measured
    # 2e-19 against 1.3e-4.
    dyn_of(e, u) = u * Diagonal(e .* abs.(e)) * u'
    ph_full_ref = compute_phonon_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal"];
                                        fourier_mode = "gridopt")
    D_ref = [dyn_of(ph.e, ph.u) for ph in ph_full_ref]
    vtr_ref = [sum(2 .* ph.e .* ph.vdiag) for ph in ph_full_ref]
    vtr_atol = 1e-10 * maximum(maximum(norm, 2 .* ph.e .* ph.vdiag) for ph in ph_full_ref)

    for fourier_mode in ["normal", "gridopt", "batched", "batched-gridopt"]
        ph_e = compute_eigenvalues_ph(model, kpts; fourier_mode)
        @test ph_e ≈ ph_e_ref

        ph_e = stack(ph.e for ph in compute_phonon_states(model, kpts, ["eigenvalue"]; fourier_mode))
        @test ph_e ≈ ph_e_ref

        ph_full = compute_phonon_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal"];
                                        fourier_mode)
        @test stack(ph.e for ph in ph_full) ≈ ph_e_ref
        @test all(dyn_of(ph.e, ph.u) ≈ D for (ph, D) in zip(ph_full, D_ref))
        @test all(isapprox(sum(2 .* ph.e .* ph.vdiag), v; atol = vtr_atol)
                  for (ph, v) in zip(ph_full, vtr_ref))

        ph_ep = phonon_eigenpairs(model, kpts; fourier_mode)
        @test ph_ep.e_full ≈ ph_e_ref
        @test all(dyn_of(ph_ep.e_full[:, ik], ph_ep.u_full[:, :, ik]) ≈ D_ref[ik] for ik in axes(ph_ep.e_full, 2))
    end
end
