using Test
using EPW
using LinearAlgebra

@testset "Transport electron CRTA" begin
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
        window = (15.0, 16.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window
        inv_τ_constant = 5.0 * unit_to_aru(:meV)

        nklist = (15, 15, 15)
        nqlist = (15, 15, 15)

        # Calculate matrix elements
        @time EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
        )

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            nlist = fill(1.0e20 * model.volume / unit_to_aru(:cm)^3, length(Tlist)),
            volume = model.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))
        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        # Constant RTA calculation using SERTA function by manually setting inv_τ to constant
        inv_τ = fill(inv_τ_constant, (btmodel.el_i.n, length(transport_params.Tlist)))
        σ_crta_by_serta = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model.recip_lattice)
        σ_crta_by_serta = symmetrize_array(σ_crta_by_serta, model.symmetry, order=2)

        # Constant RTA calculation
        transport_params.μlist .= NaN
        out_crta = run_transport_constant_relaxation_time(model, nklist, transport_params;
            inv_τ_constant, window=window_k, do_print=false);

        @test btmodel.el_i.nband == 1
        @test out_crta.σ_vdiag ≈ σ_crta_by_serta

        # Since all bands are nondegenerate (nband == 1), full_velocity is identical to vdiag
        @test out_crta.σ_intra_degen ≈ σ_crta_by_serta
        @test out_crta.σ_full ≈ σ_crta_by_serta
    end

    @testset "hole doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window_k  = (8.0, 20.0) .* unit_to_aru(:eV)
        window_kq = (10.0, 11.0) .* unit_to_aru(:eV)
        inv_τ_constant = 2000.0 * unit_to_aru(:meV)

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            nlist = fill(-1.0e21 * model.volume / unit_to_aru(:cm)^3, length(Tlist)),
            smearing = smearing,
            nband_valence = 4,
            volume = model.volume,
            spin_degeneracy = 2
        )

        symmetry = model.el_sym.symmetry

        # Constant RTA calculation
        out_crta = run_transport_constant_relaxation_time(model, nklist, transport_params;
            inv_τ_constant, symmetry, window=window_k, do_print=false);
        @test out_crta.σ_intra_degen ≈ cat(Ref(I(3)) .* [0.008557878321123907, 0.006098709949803376, 0.004730384577746854]..., dims=3)
        @test out_crta.σ_full ≈ cat(Ref(I(3)) .* [0.010403204637123844, 0.007943684266301154, 0.00657662658187597]..., dims=3)

        # Compare with BTE
        @time EPW.run_transport(
            model, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
            symmetry = symmetry,
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))
        transport_params.μlist .= NaN
        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        inv_τ = fill(inv_τ_constant, (btmodel.el_i.n, length(transport_params.Tlist)))
        σ_crta_by_serta = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model.recip_lattice)
        σ_crta_by_serta = symmetrize_array(σ_crta_by_serta, model.symmetry, order=2)

        @test btmodel.el_i.nband == 6
        @test out_crta.σ_vdiag ≈ σ_crta_by_serta

        # full_velocity should not be identical to vdiag because there are degenerate bands
        @test !(out_crta.σ_intra_degen ≈ σ_crta_by_serta)
        @test !(out_crta.σ_full ≈ σ_crta_by_serta)
        @test !(out_crta.σ_full ≈ out_crta.σ_intra_degen)

        # Compare with QME with degenerate bands only
        @time output = EPW.run_transport(
            model, nklist, nqlist,
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = true,
            run_for_qme = true,
            qme_offdiag_cutoff = EPW.electron_degen_cutoff,
        );

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        qme_model.transport_params.μlist .= NaN
        bte_compute_μ!(qme_model)

        # Set scattering matrix with constant relaxation time
        set_constant_qme_scattering_matrix!(qme_model, inv_τ_constant)

        # Solve QME and compute mobility
        out_qme = solve_electron_linear_conductivity(qme_model)
        @test out_crta.σ_intra_degen ≈ out_qme.σ_serta
        @test !(out_crta.σ_full ≈ out_qme.σ_serta)

        # Compare with full QME
        @time output = EPW.run_transport(
            model, nklist, nqlist,
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            energy_conservation = energy_conservation,
            use_irr_k = true,
            run_for_qme = true,
            qme_offdiag_cutoff = Inf,
        );

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5")
        qme_model = load_QMEModel(filename, transport_params)

        # Compute chemical potential
        transport_params.μlist .= NaN
        bte_compute_μ!(qme_model)

        # Set scattering matrix with constant relaxation time
        set_constant_qme_scattering_matrix!(qme_model, inv_τ_constant)

        # Solve QME and compute mobility
        out_qme = solve_electron_linear_conductivity(qme_model)
        @test out_qme.σ_serta ≈ out_crta.σ_full
        @test ! (out_qme.σ_serta ≈ out_crta.σ_intra_degen)

        # QME with qme_offdiag_cutoff set to include only degenerate bands. Should be equivalent
        # to the σ_intra_degen case of CRTA.
        out_qme_only_degen = solve_electron_linear_conductivity(qme_model, qme_offdiag_cutoff=EPW.electron_degen_cutoff)
        @test out_qme_only_degen.σ_serta ≈ out_crta.σ_intra_degen
        @test ! (out_qme_only_degen.σ_serta ≈ out_crta.σ_full)
    end
end
