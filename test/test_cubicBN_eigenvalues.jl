using Test
using ElectronPhonon

@testset "cubicBN eigenvalues" begin
    # Test routines that calculate eigenvalues
    model = _load_model_from_artifacts("cubicBN"; load_epmat = false)

    kpts = kpoints_grid((3, 3, 3))

    # electron
    el_states = compute_electron_states(model, kpts, ["eigenvalue"], fourier_mode="gridopt")
    el_e = compute_eigenvalues_el(model, kpts)
    @test hcat([el.e_full for el in el_states]...) ≈ el_e

    # phonon
    ph_states = compute_phonon_states(model, kpts, ["eigenvalue"], fourier_mode="gridopt")
    ph_e = compute_eigenvalues_ph(model, kpts)
    @test hcat([ph.e for ph in ph_states]...) ≈ ph_e
end
