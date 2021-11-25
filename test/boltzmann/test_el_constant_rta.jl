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
            n = 1.0e20 * model.volume / unit_to_aru(:cm)^3,
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
            n = -1.0e21 * model.volume / unit_to_aru(:cm)^3,
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
        @test out_crta.σ_full ≈ cat(Ref(I(3)) .* [0.010577697892226491, 0.008118177514709826, 0.006751119653947763]..., dims=3)

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

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5");
        fid = h5open(filename, "r");
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64});
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64});
        close(fid);

        # Compute chemical potential
        transport_params.μlist .= NaN
        bte_compute_μ!(transport_params, EPW.BTStates(el_i));

        # Set scattering matrix with constant relaxation time
        S_out = [I(el_i.n) * (-inv_τ_constant + 0.0im) for _ in 1:length(transport_params.Tlist)]

        # Solve QME and compute mobility
        out_qme = solve_electron_qme(transport_params, el_i, el_f, S_out; symmetry, filename);
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

        filename = joinpath(tmp_dir, "btedata_coherence.rank0.h5");
        fid = h5open(filename, "r");
        el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.QMEStates{Float64});
        el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.QMEStates{Float64});
        close(fid);

        # Compute chemical potential
        transport_params.μlist .= NaN
        bte_compute_μ!(transport_params, EPW.BTStates(el_i));

        # Set scattering matrix with constant relaxation time
        S_out = [I(el_i.n) * (-inv_τ_constant + 0.0im) for _ in 1:length(transport_params.Tlist)]

        # Solve QME and compute mobility
        out_qme = solve_electron_qme(transport_params, el_i, el_f, S_out; symmetry, filename);
        @test out_crta.σ_full ≈ out_qme.σ_serta
        @test !(out_crta.σ_intra_degen ≈ out_qme.σ_serta)

        # TODO: Add test with qme_offdiag_cutoff set at the level of solve_electron_qme
    end
end
