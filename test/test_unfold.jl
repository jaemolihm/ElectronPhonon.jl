using Test
using EPW

@testset "unfold" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder)

    @testset "kpoints" begin
        # Unfolding of irreducible BZ to full BZ
        nk = 30
        kpts = GridKpoints(bzmesh_ir_wedge((nk, nk, nk), model.symmetry));
        kpts_unfold_ref = GridKpoints(generate_kvec_grid(nk, nk, nk));
        kpts_unfold = EPW.unfold_kpoints(kpts, model.symmetry);
        @test kpts_unfold == kpts_unfold_ref

        # Unfolding of irreducible BZ inside given energy window to full BZ
        nk = 53
        window = (0.7, 0.8)
        kpts = GridKpoints(filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window, nothing,
            symmetry=model.symmetry)[1])
        kpts_unfold_ref = GridKpoints(filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window,
            nothing, symmetry=nothing)[1])
        kpts_unfold = EPW.unfold_kpoints(kpts, model.symmetry);
        @test kpts_unfold == kpts_unfold_ref
    end

    @testset "QMEStates" begin
        # Unfolding of QMEStates
        fourier_mode = "gridopt"
        symmetry = model.symmetry
        nk = 20
        window = (0.7, 0.81)
        quantities = ["eigenvalue", "eigenvector", "velocity"]

        kpts, iband_min, iband_max, nstates_base = filter_kpoints((nk, nk, nk), model.nw, model.el_ham,
            window, nothing, symmetry=symmetry);
        nband = iband_max - iband_min + 1
        nband_ignore = iband_min - 1
        kpts = GridKpoints(kpts)
        el_k_save = compute_electron_states(model, kpts, quantities, window, nband, nband_ignore; fourier_mode);
        el, _ = EPW.electron_states_to_QMEStates(el_k_save, kpts, EPW.electron_degen_cutoff, nstates_base);

        kpts_unfold_ref = GridKpoints(filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window,
            nothing, symmetry=nothing)[1]);
        el_k_save_unfold = compute_electron_states(model, kpts_unfold_ref, quantities, window,
            nband, nband_ignore; fourier_mode);
        el_unfold_ref, _ = EPW.electron_states_to_QMEStates(el_k_save_unfold, kpts_unfold_ref,
            EPW.electron_degen_cutoff, nstates_base);

        el_unfold, isk_to_ik_isym = EPW.unfold_QMEStates(el, symmetry);

        # Test isk_to_ik_isym mapping is correct
        for isk in 1:el_unfold.kpts.n
            ik, isym = isk_to_ik_isym[isk]
            symop = symmetry[isym]
            sk = symop.is_tr ? -symop.S * el.kpts.vectors[ik] : symop.S * el.kpts.vectors[ik]
            @test EPW.normalize_kpoint_coordinate(sk) ≈ el_unfold_ref.kpts.vectors[isk]
        end

        # Test whether el_unfold is consistent with el_unfold_ref
        # Cannot directly compare v because of gauge dependence
        inds = sortperm(collect(zip(el_unfold.ik, el_unfold.ib2, el_unfold.ib1)))
        @test el_unfold.n == el_unfold_ref.n
        @test el_unfold.ik[inds] == el_unfold_ref.ik
        @test el_unfold.ib1[inds] == el_unfold_ref.ib1
        @test el_unfold.ib2[inds] == el_unfold_ref.ib2
        @test el_unfold.e1[inds] ≈ el_unfold_ref.e1
        @test el_unfold.e2[inds] ≈ el_unfold_ref.e2
        @test el_unfold.ib_rng == el_unfold_ref.ib_rng
        @test el_unfold.nstates_base ≈ el_unfold_ref.nstates_base
        @test el_unfold.kpts == el_unfold_ref.kpts

        # Test constant RTA conductivity is the same with el and el_unfold.
        transport_params = ElectronTransportParams{Float64}(
            Tlist = [300 * unit_to_aru(:K)],
            n = -1.0e16 * model.volume / unit_to_aru(:cm)^3,
            smearing = (:Gaussian, 50.0 * unit_to_aru(:meV)),
            volume = model.volume,
            nband_valence = 4,
            spin_degeneracy = 2
        )
        bte_compute_μ!(transport_params, EPW.BTStates(el); do_print=false)
        μ = transport_params.μlist[1]
        T = transport_params.Tlist[1]

        inv_τ_constant = 10 * unit_to_aru(:meV)
        δρ = @. -el.v * occ_fermion_derivative(el.e1 - μ, T) / inv_τ_constant;
        σ = symmetrize(EPW.occupation_to_conductivity(δρ, el, transport_params), symmetry);

        δρ_unfold = @. -el_unfold.v * occ_fermion_derivative(el_unfold.e1 - μ, T) / inv_τ_constant;
        σ_unfold = EPW.occupation_to_conductivity(δρ_unfold, el_unfold, transport_params);
        @test isapprox(σ_unfold, σ; rtol=1e-5)
    end
end
