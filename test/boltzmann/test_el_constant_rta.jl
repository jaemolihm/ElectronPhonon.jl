using Test
using EPW
using LinearAlgebra

@testset "Transport electron CRTA" begin
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

        transport_params = ElectronTransportParams{Float64}(
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
        σ = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σ = symmetrize_array(σ, model_el.symmetry, order=2)
        _, mobility_SI = transport_print_mobility(σ, transport_params, do_print=false)

        # Constant RTA calculation
        transport_params.μlist .= NaN
        out_crta = run_transport_constant_relaxation_time(model_el, nklist, transport_params; inv_τ_constant, window=window_k, use_irr_k=true, do_print=false);

        @test btmodel.el_i.nband == 1
        @test out_crta.σ_vdiag ≈ σ
        @test out_crta.mobility_vdiag_SI ≈ mobility_SI

        # Since all bands are nondegenerate (nband == 1), full_velocity is identical to vdiag
        @test out_crta.σ_full_velocity ≈ σ
        @test out_crta.mobility_full_velocity_SI ≈ mobility_SI
    end

    @testset "hole doping" begin
        energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
        window = (10.5, 11.0) .* unit_to_aru(:eV)
        window_k  = window
        window_kq = window
        inv_τ_constant = 10.0 * unit_to_aru(:meV)

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

        transport_params = ElectronTransportParams{Float64}(
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
        σ = compute_conductivity_serta!(transport_params, inv_τ, btmodel.el_i, nklist, model_el.recip_lattice)
        σ = symmetrize_array(σ, model_el.symmetry, order=2)
        _, mobility_SI = transport_print_mobility(σ, transport_params, do_print=false)

        # Constant RTA calculation
        transport_params.μlist .= NaN
        out_crta = run_transport_constant_relaxation_time(model_el, nklist, transport_params; inv_τ_constant, window=window_k, use_irr_k=true, do_print=false);

        @test btmodel.el_i.nband == 3
        @test out_crta.σ_vdiag ≈ σ
        @test out_crta.mobility_vdiag_SI ≈ mobility_SI

        # full_velocity should not be identical to vdiag because there are degenerate bands
        @test !(out_crta.σ_full_velocity ≈ σ)
        @test !(out_crta.mobility_full_velocity_SI ≈ mobility_SI)
        @test out_crta.mobility_full_velocity_SI ≈ cat(Ref(I(3)) .* [491.4008316455876, 350.18972508269235, 271.5598769595231]..., dims=3)
    end
end
