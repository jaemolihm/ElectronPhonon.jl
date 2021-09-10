using Test
using EPW
using HDF5

@testset "Transport electron metal" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_Pb")

    model_el = load_model(folder, epmat_outer_momentum="el")

    # temporary directory to store output data file
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    # Reference data from EPW
    μlist_ref_epw = [11.449807639297191, 11.479920504373881, 11.513292262946232] .* unit_to_aru(:eV)

    # Parameters
    e_fermi = 11.594123 * EPW.unit_to_aru(:eV)
    window_k  = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi
    window_kq = (-0.5, 0.5) .* unit_to_aru(:eV) .+ e_fermi
    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K)
    smearing = (:Gaussian, 50.0 * unit_to_aru(:meV))
    energy_conservation = (:Fixed, 10 * 50.0 * EPW.unit_to_aru(:meV))

    nklist = (10, 10, 10)
    nqlist = (10, 10, 10)

    # Calculate matrix elements
    @time output = EPW.run_transport(
        model_el, nklist, nqlist,
        fourier_mode = "gridopt",
        folder = tmp_dir,
        window_k  = window_k,
        window_kq = window_kq,
        use_irr_k = true,
        energy_conservation = energy_conservation,
        average_degeneracy = true,
    )
    @test output.qpts.n == 723

    transport_params = ElectronTransportParams{Float64}(
        Tlist = Tlist,
        n = 4,
        smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
        nband_valence = 0,
        volume = model_el.volume,
        spin_degeneracy = 2
    )

    # Read electron and phonon states (do not read scattering here)
    fid = h5open(joinpath(tmp_dir, "btedata.rank0.h5"), "r")
    el_i = load_BTData(fid["initialstate_electron"], EPW.BTStates{Float64})
    close(fid)

    # Compute chemical potential
    bte_compute_μ!(transport_params, el_i)

    @test all(isapprox.(transport_params.μlist, μlist_ref_epw, atol=2e-6 * unit_to_aru(:eV)))

    # TODO: Test Run serta
    # TODO: Test LBTE
    # TODO: Test TDF
end