using Test
using EPW

@testset "unfold" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, load_symmetry_operators=true)

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

        # Test folding of kpoints (inverse of unfolding)
        kpts_by_folding, ik_to_ikirr_isym = EPW.fold_kpoints(kpts_unfold, model.symmetry)
        @test kpts_by_folding.n == kpts.n
        for ik = 1:kpts_unfold.n
            ikirr, isym = ik_to_ikirr_isym[ik]
            symop = model.symmetry[isym]
            xk = kpts_unfold.vectors[ik]
            xkirr = kpts_by_folding.vectors[ikirr]
            skirr = symop.is_tr ? -symop.S * xkirr : symop.S * xkirr
            skirr = EPW.normalize_kpoint_coordinate(skirr)
            @test xk ≈ skirr
        end
    end

    @testset "ElectronStates" begin
        symmetry = model.el_sym.symmetry
        nk = 5
        kpts_full = GridKpoints(generate_kvec_grid(nk, nk, nk))
        kpts_irr, ik_to_ikirr_isym = EPW.fold_kpoints(kpts_full, symmetry)
        el_irr = compute_electron_states(model, kpts_irr, ["eigenvector", "velocity", "position"])
        el_full = compute_electron_states(model, kpts_full, ["eigenvector", "velocity", "position"])
        el_full_unfold = EPW.unfold_ElectronStates(model, el_irr, kpts_irr, kpts_full, ik_to_ikirr_isym,
            symmetry; quantities=["velocity_diagonal", "velocity", "position"])

        hk = zeros(ComplexF64, model.nw, model.nw)
        for ik = 1:kpts_full.n
            el1 = el_full_unfold[ik]
            el2 = el_full[ik]

            # Check eigenvectors in el is correct: U' * H(Sk) * U = Diagonal(e(Sk))
            get_fourier!(hk, model.el_ham, kpts_full.vectors[ik]);
            @test el1.u' * hk * el1.u ≈ Diagonal(el2.e) atol=1e-6

            # Check velocity matrix is correct by converting to Wannier basis
            v1 = el1.u * el1.v * el1.u'
            v2 = el2.u * el2.v * el2.u'
            @test v1 ≈ v2 atol=1e-4

            # Check vdiag = diag(v)
            @test diag(el1.v) ≈ el1.vdiag atol=1e-10

            # Check position matrix is correct by converting to Wannier basis
            r1 = el1.u * el1.rbar * el1.u'
            r2 = el2.u * el2.rbar * el2.u'
            @test r1 ≈ r2 atol=1e-4
        end
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

        kpts = GridKpoints(kpts)
        el_k_save = compute_electron_states(model, kpts, quantities, window; fourier_mode);
        el = EPW.electron_states_to_QMEStates(el_k_save, kpts, EPW.electron_degen_cutoff, nstates_base);

        kpts_unfold_ref = GridKpoints(filter_kpoints((nk, nk, nk), model.nw, model.el_ham, window,
            nothing, symmetry=nothing)[1]);
        el_k_save_unfold = compute_electron_states(model, kpts_unfold_ref, quantities, window; fourier_mode);
        el_unfold_ref = EPW.electron_states_to_QMEStates(el_k_save_unfold, kpts_unfold_ref,
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
            nlist = [-1.0e16 * model.volume / unit_to_aru(:cm)^3],
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
