using Test
using EPW
using LinearAlgebra

@testset "Transport electron CRTA" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")

    model_el = load_model(folder, epmat_outer_momentum="el")
    model_ph = load_model(folder, epmat_outer_momentum="ph")
    model_el.el_velocity_mode = :BerryConnection
    model_ph.el_velocity_mode = :BerryConnection

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
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
        )

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = 1.0e20 * model_el.volume / unit_to_aru(:cm)^3,
            volume = model_el.volume,
            smearing = smearing,
            nband_valence = 4,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))
        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        # Constant RTA calculation using SERTA function by manually setting inv_τ to constant
        inv_τ = fill(inv_τ_constant, (btmodel.el_i.n, length(transport_params.Tlist)))
        σ_crta_by_serta = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σ_crta_by_serta = symmetrize_array(σ_crta_by_serta, model_el.symmetry, order=2)

        # Constant RTA calculation
        transport_params.μlist .= NaN
        out_crta = run_transport_constant_relaxation_time(model_el, nklist, transport_params; inv_τ_constant, window=window_k, use_irr_k=true, do_print=false);

        @test btmodel.el_i.nband == 1
        @test out_crta.σ_vdiag ≈ σ_crta_by_serta

        # Since all bands are nondegenerate (nband == 1), full_velocity is identical to vdiag
        @test out_crta.σ_intra_degen ≈ σ_crta_by_serta
        @test out_crta.σ_full ≈ σ_crta_by_serta
    end

    @testset "hole doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (8.0, 20.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window
        inv_τ_constant = 2000.0 * unit_to_aru(:meV)

        nklist = (12, 12, 12)
        nqlist = (12, 12, 12)

        # Calculate matrix elements
        @time EPW.run_transport(
            model_el, nklist, nqlist,
            fourier_mode = "gridopt",
            folder = tmp_dir,
            window_k  = window_k,
            window_kq = window_kq,
            use_irr_k = true,
            energy_conservation = energy_conservation,
        )

        transport_params = ElectronTransportParams(
            Tlist = Tlist,
            n = -1.0e21 * model_el.volume / unit_to_aru(:cm)^3,
            smearing = smearing,
            nband_valence = 4,
            volume = model_el.volume,
            spin_degeneracy = 2
        )

        btmodel = load_ElPhBTModel(joinpath(tmp_dir, "btedata.rank0.h5"))
        bte_compute_μ!(transport_params, btmodel.el_i, do_print=false)

        inv_τ = fill(inv_τ_constant, (btmodel.el_i.n, length(transport_params.Tlist)))
        σ_crta_by_serta = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σ_crta_by_serta = symmetrize_array(σ_crta_by_serta, model_el.symmetry, order=2)

        # Constant RTA calculation
        transport_params.μlist .= NaN
        out_crta = run_transport_constant_relaxation_time(model_el, nklist, transport_params; inv_τ_constant, window=window_k, use_irr_k=true, do_print=false);

        @test btmodel.el_i.nband == 6
        @test out_crta.σ_vdiag ≈ σ_crta_by_serta

        # full_velocity should not be identical to vdiag because there are degenerate bands
        @test !(out_crta.σ_intra_degen ≈ σ_crta_by_serta)
        @test !(out_crta.σ_full ≈ σ_crta_by_serta)
        @test !(out_crta.σ_full ≈ out_crta.σ_intra_degen)
        @test out_crta.σ_intra_degen ≈ cat(Ref(I(3)) .* [0.008557879364583812, 0.006098710261298616, 0.0047303846800104835]..., dims=3)
        @test out_crta.σ_full ≈ cat(Ref(I(3)) .* [0.009107071626286102, 0.006649677470582548, 0.005283351010710945]..., dims=3)
    end
end
