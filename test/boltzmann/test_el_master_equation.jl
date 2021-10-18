using Test
using EPW
using LinearAlgebra
using HDF5

@testset "Transport electron QME" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model_el = load_model(folder, epmat_outer_momentum="el")
    model_ph = load_model(folder, epmat_outer_momentum="ph")

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)

    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    @testset "electron doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (15.0, 16.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        # Disable symmetry because symmetry is not yet implemented in QME
        symmetry = nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = false,
            average_degeneracy = false,
            run_for_qme = true,
        )

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = 1.0e20 * model_el.volume / unit_to_aru(:cm)^3,
            volume = model_el.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        fid = h5open(filename, "r")
        el_i = load_BTData(fid["initialstate_electron"], EPW.QMEStates{Float64})
        el_f = load_BTData(fid["finalstate_electron"], EPW.QMEStates{Float64})
        ph = load_BTData(fid["phonon"], EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme = solve_electron_qme(el_i, el_f, S_out, transport_params, symmetry)
        _, mobility_qme_serta_SI = transport_print_mobility(out_qme.σ_serta, transport_params, do_print=false);

        # For electron-doped BN, there is only 1 band, so the result is identical to BTE
        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = false,
            energy_conservation = energy_conservation,
        )
        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

        output_serta = EPW.run_serta(filename_btedata, transport_params, symmetry, model_el.recip_lattice);

        # Test QME and BTE result is identical (because there is only one band)
        @test el_i.nband == 1
        @test all(el_i.ib1 .== 1)
        @test all(el_i.ib2 .== 1)
        @test μlist_qme ≈ transport_params.μlist
        @test mobility_qme_serta_SI ≈ output_serta.mobility_SI
    end

    @testset "hole doping" begin
        # Reference data created by Julia
        μlist_ref = [0.790917195763113, 0.7915721538246762, 0.7922179662218595]
        mobility_ref = reshape(hcat([
            [ 154.94103397017716      0.1326859955044396    0.1324776220752463;
                0.1326877605488312  154.94094743849948     -0.1326439790408618;
                0.13247694551742326  -0.13264006226947253 154.94099562350138],
            [  87.36380425317321      0.16932578335065523   0.16924768386860323;
                0.16932646614600216  87.36377170542119     -0.1693094755988512;
                0.1692474234844571   -0.1693079534381468   87.36378975901164],
            [  54.80031000518976      0.13764845554718236   0.13761236019306125;
                0.1376487790868413   54.80029498314644     -0.13764064298926706;
                0.13761223715127288  -0.13763991994381577  54.80030325365559]]...), 3, 3, 3)

        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        transport_params = ElectronTransportParams{Float64}(
            Tlist = Tlist,
            n = -1.0e21 * model_el.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2
        )

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        # Disable symmetry because symmetry is not yet implemented in QME
        symmetry = nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = false,
            average_degeneracy = false,
            run_for_qme = true,
        )

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        fid = h5open(filename, "r")
        el_i = load_BTData(fid["initialstate_electron"], EPW.QMEStates{Float64})
        el_f = load_BTData(fid["finalstate_electron"], EPW.QMEStates{Float64})
        ph = load_BTData(fid["phonon"], EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme = solve_electron_qme(el_i, el_f, S_out, transport_params, symmetry)
        _, mobility_qme_serta_SI = transport_print_mobility(out_qme.σ_serta, transport_params, do_print=false)

        @test transport_params.μlist ≈ μlist_ref
        @test mobility_qme_serta_SI ≈ mobility_ref
    end
end
