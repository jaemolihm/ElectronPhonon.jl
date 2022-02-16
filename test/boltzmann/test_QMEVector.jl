using Test
using EPW
using HDF5
using LinearAlgebra
using EPW: QMEVector

@testset "QMEVector" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp = joinpath(folder, "tmp")
    mkpath(folder_tmp)

    model = load_model(folder)

    # Setup QMEVector
    fourier_mode = "gridopt"
    window = (15.0, 16.0) .* unit_to_aru(:eV)
    nk = 20
    symmetry = nothing
    qme_offdiag_cutoff = 1.0 .* unit_to_aru(:eV)

    @time kpts, iband_min, iband_max, nstates_base = filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window; symmetry, fourier_mode)
    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1
    @time el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"], window, nband, nband_ignore; fourier_mode);
    el, _ = EPW.electron_states_to_QMEStates(el_k_save, kpts, qme_offdiag_cutoff, nstates_base)

    # Test basic arithmetic operations
    x = QMEVector(el, rand(el.n))
    y = QMEVector(el, rand(el.n))
    z = 2 * x - y / 3
    @test z.state === el
    @test z.data ≈ @. 2 * x.data - y.data / 3

    # Test get_velocity_as_QMEVector
    v = EPW.get_velocity_as_QMEVector(el)
    @test v isa Vec3
    @test eltype(v) <: QMEVector
    for i in 1:3
        @test v[i].state === el
        @test v[i].data ≈ [v[i] for v in el.v]
    end
end
