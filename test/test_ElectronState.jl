using Test
using EPW
using OffsetArrays: no_offset_view

@testset "ElectronState" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp_el = joinpath(folder, "tmp_el")
    mkpath(folder_tmp_el)

    model = load_model(folder)
    window = (10.0, 25.0) .* unit_to_aru(:eV)

    @testset "basic" begin
        kpts = Kpoints([0.1, -0.4, 0.7])
        el = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "velocity", "position"], window)[1]
        @test el.rng_full == 5:6
        @test el.nband == length(el.rng_full)
        @test size(el.rbar) == (el.nband, el.nband)
        @test axes(el.rbar) == (el.rng_full, el.rng_full)
    end

    @testset "copyto!" begin
        kpts = Kpoints([0.1, -0.4, 0.7])

        # setup electron and phonon states
        el1 = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "velocity", "position"], window)[1]

        el2 = ElectronState(model.nw)
        copyto!(el2, el1)

        @test el1.nband_bound == el1.nband
        @test el2.nband_bound == 8 # model.nw

        @test el2.e_full ≈ el1.e_full
        @test el2.u_full ≈ el1.u_full
        @test el2.nband_ignore == el1.nband_ignore
        @test el2.nband == el1.nband
        @test el2.rng_full == el1.rng_full
        @test el2.rng == el1.rng

        @test el2.e ≈ el1.e
        @test el2.vdiag ≈ el1.vdiag
        @test el2.v ≈ el1.v
        @test el2.rbar ≈ el1.rbar

        @test el2.u ≈ el1.u
    end

    @testset "window" begin
        # Test that ElectronStates computed with and without windows are same within the window.

        quantities = ["eigenvalue", "eigenvector", "velocity", "position"]
        kpts = Kpoints([0., 0., 1/6])
        window = (8.0, 20.0) .* unit_to_aru(:eV)
        nband = 6

        for el_velocity_mode in [:BerryConnection, :Direct]
            @testset "$el_velocity_mode" begin
                model.el_velocity_mode = el_velocity_mode

                el1 = compute_electron_states(model, kpts, quantities, window)[1];
                el2 = compute_electron_states(model, kpts, quantities)[1];

                @test el1.e[el1.rng_full] ≈ el2.e[el1.rng_full]
                @test el1.vdiag[el1.rng_full] ≈ el2.vdiag[el1.rng_full]
                @test el1.v[el1.rng_full, el1.rng_full] ≈ el2.v[el1.rng_full, el1.rng_full]
                @test el1.rbar[el1.rng_full, el1.rng_full] ≈ el2.rbar[el1.rng_full, el1.rng_full]
            end
        end
    end
end