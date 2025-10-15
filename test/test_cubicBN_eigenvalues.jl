using Test
using ElectronPhonon

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

    for fourier_mode in ["normal", "gridopt", "batched", "batched-gridopt"]
        ph_e = compute_eigenvalues_ph(model, kpts; fourier_mode)
        @test ph_e ≈ ph_e_ref
    end
end
