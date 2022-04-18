using Test
using EPW
using LinearAlgebra

@testset "Transport electron AC CRTA" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    model = load_model(folder, epmat_outer_momentum="el", load_symmetry_operators=true)

    Tlist = [200.0, 300.0, 400.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 80.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 4 * 80.0 * EPW.unit_to_aru(:meV))
    window_k  = (8.0, 20.0) .* unit_to_aru(:eV)
    window_kq = (10.0, 11.0) .* unit_to_aru(:eV)
    inv_τ_constant = 20.0 * unit_to_aru(:meV)

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

    # AC conductivity with CRTA
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
    bte_compute_μ!(qme_model)

    # Set scattering matrix with constant relaxation time
    set_constant_qme_scattering_matrix!(qme_model, inv_τ_constant)

    # Solve QME and compute mobility
    out_qme_dc = solve_electron_linear_conductivity(qme_model, use_full_grid=true)

    ωlist = range(0, 5, length=50) .* inv_τ_constant
    σ_ac = zeros(Complex{Float64}, 3, 3, length(Tlist), length(ωlist))
    for (iω, ω) in enumerate(ωlist)
        out_qme_ac = solve_electron_linear_conductivity(qme_model, ω, use_full_grid=true)
        σ_ac[:, :, :, iω] .= out_qme_ac.σ_serta
    end

    coeff_drude = @. inv_τ_constant / (inv_τ_constant - im * ωlist)
    σ_ac_drude = cat([out_qme_dc.σ_serta .* c for c in coeff_drude]..., dims=4)
    @test σ_ac ≈ σ_ac_drude
end
