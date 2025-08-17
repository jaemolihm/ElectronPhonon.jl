using Test
using ElectronPhonon

@testset "cubicBN eigenvalues" begin
    # Test routines that calculate eigenvalues
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp_el = joinpath(folder, "tmp_el")
    mkpath(folder_tmp_el)

    model = load_model_from_epw_new(folder, "temp", "bn"; load_epmat = false)

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
