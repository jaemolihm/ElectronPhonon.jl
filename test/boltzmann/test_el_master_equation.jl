using Test
using EPW
using LinearAlgebra
using HDF5

@testset "Transport electron QME" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)
    model.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)

    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    @testset "electron doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window_k  = (15.0, 16.0) .* unit_to_aru(:eV)
        window_kq = (15.0, 16.2) .* unit_to_aru(:eV)

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = 1.0e20 * model.volume / unit_to_aru(:cm)^3,
            volume = model.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        for use_irr_k in [false, true]
            symmetry = use_irr_k ? model.el_sym.symmetry : nothing

            # Calculate matrix elements
            @time EPW.run_transport(
                model, nklist, nqlist,
                fourier_mode = "gridopt",
                folder = tmp_dir,
                window_k  = window_k,
                window_kq = window_kq,
                energy_conservation = energy_conservation,
                use_irr_k = use_irr_k,
                average_degeneracy = false,
                run_for_qme = true,
            )

            filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
            fid = h5open(filename, "r")
            el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
            el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
            ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
            close(fid)

            # Check that all bands are non-degenerate
            @test all(el_i.ib1 .== el_i.ib2)

            # Compute chemical potential
            bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
            μlist_qme = copy(transport_params.μlist)

            # Compute scattering matrix
            S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

            # Solve QME and compute mobility
            out_qme = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in; symmetry, filename)

            # For electron-doped BN, there is only 1 band, so the result is identical to BTE
            # Calculate matrix elements
            @time EPW.run_transport(
                model, nklist, nqlist,
                fourier_mode = "gridopt",
                folder = tmp_dir,
                window_k  = window_k,
                window_kq = window_kq,
                use_irr_k = use_irr_k,
                symmetry = symmetry, # Need to explicit set because we cannot use time-reversal in QME, but BTE uses it by default
                energy_conservation = energy_conservation,
            )
            filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

            output_serta = EPW.run_serta(filename_btedata, transport_params, symmetry, model.recip_lattice);

            # LBTE
            bte_scat_mat, el_i_bte, el_f_bte, ph_bte = EPW.compute_bte_scattering_matrix(filename_btedata, transport_params, model.recip_lattice);
            inv_τ = output_serta.inv_τ;
            output_lbte = EPW.solve_electron_bte(el_i_bte, el_f_bte, bte_scat_mat, inv_τ, transport_params, symmetry)

            # Test QME and BTE result is identical (because there is only one band)
            @test el_i.nband == 1
            @test all(el_i.ib1 .== 1)
            @test all(el_i.ib2 .== 1)
            @test μlist_qme ≈ transport_params.μlist
            @test out_qme.σ_serta ≈ output_serta.σ rtol=1e-6
            @test out_qme.σ_serta ≈ output_lbte.σ_serta rtol=1e-6
            @test out_qme.σ ≈ output_lbte.σ rtol=1e-6
        end
    end

    @testset "hole doping" begin
        # Reference data created by Julia
        μlist_ref = [0.7910477407091759, 0.7917002728887299, 0.7923440548581497]
        mobility_serta_ref = reshape(hcat([
            [ 157.63545651684313       -0.02433429293885603   0.16012775524237663;
               -0.024334665060661644  157.69221920282877     -0.1334744610703724;
                0.1601273403838144     -0.1334744779224233  157.30209187419118],
            [  89.8515084210407         0.0960584600863529    0.20301755040331973;
                0.09605831383231855    89.88461638762415     -0.1875630988052886;
                0.20301738715262915    -0.18756310552742428  89.65714860540842],
            [  56.64789168533079        0.09749668487108533   0.16545873468801564;
                0.09749661509615085    56.66899461653767     -0.1556390479622601;
                0.16545865675347082    -0.1556390511943863   56.52403385714702]]...), 3, 3, 3)
        mobility_iter_ref = reshape(hcat([
            [ 174.14790089293083    -1.073278780631703     -0.8877777888001277;
               -1.0800152562641083 174.1872321702434        0.8955979556976402;
               -0.923452993139115    0.933143027893385    173.80779732812354],
            [ 100.98573033633811    -0.5569397953785569    -0.4486531980807132;
              -0.5615840087531551  101.00731197191044       0.45146677361387494;
              -0.473881024691554     0.47824044518581615  100.7867707608588],
            [ 64.25091573221515     -0.3320817958752827    -0.2629730043766893;
              -0.33528847964380315  64.26420940446224       0.2641627087182129;
              -0.2806047534211816    0.2829273517117792    64.12378052578173]]...), 3, 3, 3)

        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model.volume,
            spin_degeneracy = 2
        )

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        # Disable symmetry because symmetry is not yet implemented in QME
        symmetry = nothing

        # Calculate matrix elements
        @time EPW.run_transport(
            model, nklist, nqlist,
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
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
        ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in; symmetry, filename)
        _, mobility_qme_serta_SI = transport_print_mobility(out_qme.σ_serta, transport_params, do_print=false)
        _, mobility_qme_iter_SI = transport_print_mobility(out_qme.σ, transport_params, do_print=false);

        @test transport_params.μlist ≈ μlist_ref
        @test mobility_qme_serta_SI ≈ mobility_serta_ref
        @test mobility_qme_iter_SI ≈ mobility_iter_ref
    end
end


@testset "Transport electron QME symmetry" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

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

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = 1.0e20 * model.volume / unit_to_aru(:cm)^3,
            volume = model.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        # Run without symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
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
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
        ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme_wo_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in; filename)

        # Run with symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = true,
            symmetry = model.el_sym.symmetry,
            average_degeneracy = false,
            run_for_qme = true,
        )

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        fid = h5open(filename, "r")
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
        ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme_w_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in,
                                                symmetry=model.el_sym.symmetry; filename)

        _, mobility_serta_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_serta_w_sym = transport_print_mobility(out_qme_w_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_iter_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ, transport_params, do_print=false)
        _, mobility_iter_w_sym = transport_print_mobility(out_qme_w_symmetry.σ, transport_params, do_print=false)

        @test all(isapprox.(mobility_serta_w_sym, mobility_serta_wo_sym, atol=1e-2))
        @test all(isapprox.(mobility_iter_w_sym, mobility_iter_wo_sym, atol=0.2))
    end

    @testset "hole doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model.volume,
            spin_degeneracy = 2
        )

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        # Run without symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
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
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
        ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme_wo_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in; filename)

        # Run with symmetry
        @time EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = true,
            symmetry = model.el_sym.symmetry,
            average_degeneracy = false,
            run_for_qme = true,
        )

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        fid = h5open(filename, "r")
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
        ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
        close(fid)

        # Compute chemical potential
        bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
        μlist_qme = copy(transport_params.μlist)

        # Compute scattering matrix
        S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

        # Solve QME and compute mobility
        out_qme_w_symmetry = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in,
                                                symmetry=model.el_sym.symmetry; filename)


        _, mobility_serta_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_serta_w_sym = transport_print_mobility(out_qme_w_symmetry.σ_serta, transport_params, do_print=false)
        _, mobility_iter_wo_sym = transport_print_mobility(out_qme_wo_symmetry.σ, transport_params, do_print=false)
        _, mobility_iter_w_sym = transport_print_mobility(out_qme_w_symmetry.σ, transport_params, do_print=false)

        # There is some error bewteen w/o and w/ symmetry, possibly because Wannier functions
        # break the symmetry. So we use a large error tolerance.
        @test all(isapprox.(mobility_serta_w_sym, mobility_serta_wo_sym, atol=5.0))
        @test all(isapprox.(mobility_iter_w_sym, mobility_iter_wo_sym, atol=5.0))

        # Transport distribution function
        @testset "TDF" begin
            elist = range(minimum(el_i.e1) - 5e-3, maximum(el_i.e1) + 5e-3, length=101)
            de = elist[2] - elist[1]
            smearing = 10 * unit_to_aru(:meV)

            Σ_tdf = EPW.compute_transport_distribution_function(out_qme_w_symmetry; elist, smearing, symmetry=model.el_sym.symmetry)

            @test size(Σ_tdf) == (length(elist), 3, 3, length(transport_params.Tlist))
            @test dropdims(sum(Σ_tdf, dims=1), dims=1) .* de ≈ out_qme_w_symmetry.σ
        end
    end
end


# Randomly perturb the gauge of eigenstates by setting the debugging flag
# DEBUG_random_gauge to true. And check that the result does not change.
@testset "Transport electron QME gauge" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))

    energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
    window = (10.5, 11.0) .* unit_to_aru(:eV)
    window_k  = window
    window_kq = window

    transport_params = ElectronTransportParams(
        Tlist = Tlist,
        n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
        smearing = smearing,
        nband_valence = 4,
        volume = model.volume,
        spin_degeneracy = 2
    )

    nklist = (12, 12, 12)
    nqlist = (12, 12, 12)

    for use_irr_k in [false, true]
        symmetry = use_irr_k ? model.el_sym.symmetry : nothing

        out_qme = Dict()
        for random_gauge in [false, true]
            # Calculate matrix elements
            @time EPW.run_transport(
                model, nklist, nqlist,
                fourier_mode = "gridopt",
                folder = tmp_dir,
                window_k  = window_k,
                window_kq = window_kq,
                energy_conservation = energy_conservation,
                use_irr_k = use_irr_k,
                average_degeneracy = false,
                run_for_qme = true,
                DEBUG_random_gauge = random_gauge,
            )

            filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
            fid = h5open(filename, "r")
            el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64})
            el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64})
            ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})
            close(fid)

            # Compute chemical potential
            bte_compute_μ!(transport_params, EPW.BTStates(el_i), do_print=false)
            μlist_qme = copy(transport_params.μlist)

            # Compute scattering matrix
            S_out, S_in = compute_qme_scattering_matrix(filename, transport_params, el_i, el_f, ph)

            # Solve QME and compute mobility
            out_qme[random_gauge] = solve_electron_qme(transport_params, el_i, el_f, S_out, S_in; filename, symmetry)
        end

        @test out_qme[true].σ_serta ≈ out_qme[false].σ_serta
        @test out_qme[true].σ ≈ out_qme[false].σ
    end
end