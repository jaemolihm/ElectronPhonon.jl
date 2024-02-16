using Test
using ElectronPhonon
using LinearAlgebra

@testset "Transport electron metal" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_Pb")

    model_el = load_model(folder, epmat_outer_momentum="el")
    model_el.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    # Reference data from EPW
    μlist_ref_epw = [11.449807639297191, 11.479920504373881, 11.513292262946232] .* unit_to_aru(:eV)
    σ_ref_epw_iter0 = reshape(hcat(0.966388E+05*I(3), 0.214829E+05*I(3), 0.843989E+04*I(3)), 3, 3, 3)
    σ_ref_epw_iter1 = reshape(hcat(0.107119E+06*I(3), 0.239672E+05*I(3), 0.943501E+04*I(3)), 3, 3, 3)
    σ_ref_epw_convd = reshape(hcat(0.110039E+06*I(3), 0.246179E+05*I(3), 0.968863E+04*I(3)), 3, 3, 3)

    # Parameters
    e_fermi = 11.594123 * unit_to_aru(:eV)
    window_k  = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi
    window_kq = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi
    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 50.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 10 * 50.0 * unit_to_aru(:meV))

    nklist = (10, 10, 10)
    nqlist = (10, 10, 10)

    # Calculate matrix elements
    @time output = ElectronPhonon.run_transport(
        model_el, nklist, nqlist,
        fourier_mode = "gridopt",
        folder = tmp_dir,
        window_k  = window_k,
        window_kq = window_kq,
        use_irr_k = true,
        energy_conservation = energy_conservation,
        average_degeneracy = true,
    )
    filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")
    @test output.qpts.n == 723

    transport_params = ElectronTransportParams(
        Tlist = Tlist,
        nlist = fill(4., length(Tlist)),
        smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
        nband_valence = 0,
        volume = model_el.volume,
        spin_degeneracy = 2
    )
    @test transport_params.type === :Metal

    # SERTA
    output_serta = ElectronPhonon.run_serta(filename_btedata, transport_params, model_el.symmetry, model_el.recip_lattice)
    @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=2e-6 * unit_to_aru(:eV)))
    @test_broken all(isapprox.(output_serta.σ_SI, σ_ref_epw_iter0, atol=0.2))

    # LBTE
    @time bte_scat_mat, el_i, el_f, ph = ElectronPhonon.compute_bte_scattering_matrix(filename_btedata, transport_params, model_el.recip_lattice)
    @test length(bte_scat_mat) == length(transport_params.Tlist)
    @test all(size.(bte_scat_mat) .== Ref((el_i.n, el_f.n)))

    inv_τ = output_serta.inv_τ
    @time output_lbte = ElectronPhonon.solve_electron_bte(el_i, el_f, bte_scat_mat, inv_τ, transport_params, model_el.symmetry)

    σ_SI_serta, _ = transport_print_mobility(output_lbte.σ_serta, transport_params, do_print=false)
    σ_SI_iter1, _ = transport_print_mobility(output_lbte.σ_iter[2,:,:,:], transport_params, do_print=false)
    σ_SI_convd, _ = transport_print_mobility(output_lbte.σ, transport_params, do_print=false)

    @test σ_SI_serta ≈ output_serta.σ_SI
    @test_broken all(isapprox.(σ_SI_serta, σ_ref_epw_iter0, atol=0.2))
    @test_broken all(isapprox.(σ_SI_iter1, σ_ref_epw_iter1, atol=0.2))
    @test_broken all(isapprox.(σ_SI_convd, σ_ref_epw_convd, atol=0.5))

    @testset "TDF" begin
        tdf_smearing = 10.0 * unit_to_aru(:meV)
        elist = range(minimum(el_i.e) - 3e-3, maximum(el_i.e) + 3e-3, length=1001)
        Σ_tdf = compute_transport_distribution_function(elist, tdf_smearing, el_i, output_serta.inv_τ, transport_params, model_el.symmetry)
        @test size(Σ_tdf) == (length(elist), 3, 3, length(transport_params.Tlist))
        @test dropdims(sum(Σ_tdf, dims=1), dims=1) .* (elist[2] - elist[1]) ≈ output_serta.σ
    end
end

# Test LBTE where window_k and window_kq are different
@testset "Transport electron metal window" begin
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_Pb")

    model = load_model(folder, epmat_outer_momentum="el")
    model.el_velocity_mode = :BerryConnection

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    # Parameters
    e_fermi = 11.594123 * unit_to_aru(:eV)
    window_k  = (-0.3, 0.3) .* unit_to_aru(:eV) .+ e_fermi
    window_kq = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi
    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 50.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 10 * 50.0 * unit_to_aru(:meV))

    transport_params = ElectronTransportParams(
        Tlist = Tlist,
        nlist = fill(4., length(Tlist)),
        smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
        nband_valence = 0,
        volume = model.volume,
        spin_degeneracy = 2
    )

    nklist = (10, 10, 10)
    nqlist = (10, 10, 10)

    # Check whether conductivity with and without symmetry are the same
    output_bte = Dict()
    for (use_irr_k, symmetry, key) in zip([false, true], [nothing, model.symmetry], ["nosym", "sym"])
        output = ElectronPhonon.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = use_irr_k,
            energy_conservation = energy_conservation,
            average_degeneracy = true,
        )
        filename_btedata = joinpath(tmp_dir, "btedata.rank0.h5")

        output_serta = ElectronPhonon.run_serta(filename_btedata, transport_params, symmetry, model.recip_lattice)
        bte_scat_mat, el_i, el_f, ph = ElectronPhonon.compute_bte_scattering_matrix(filename_btedata, transport_params, model.recip_lattice)
        inv_τ = output_serta.inv_τ
        output_bte[key] = ElectronPhonon.solve_electron_bte(el_i, el_f, bte_scat_mat, inv_τ, transport_params, symmetry)
    end

    @test all(isapprox.(output_bte["sym"].σ_serta, output_bte["nosym"].σ_serta, atol=1e-6))
    @test all(isapprox.(output_bte["sym"].σ, output_bte["nosym"].σ, atol=1e-6))
end
