using Test
using ElectronPhonon
const EP = ElectronPhonon
using ElectronPhonon: state_weights, state_xks, StateSelection

# Pure-BZ (e-ph-independent) quadrature check for the multigrid double-grid weights. Grounds the
# per-(k,band) weight correctness ([DECISION 1]) before trusting transport numbers: the per-state
# weights must integrate a smooth periodic function over the BZ, PER BAND, converging to the coarse
# spacing. `filter_electron_states_multigrid` now emits a `StateSelection`: all fine-grid states inside the
# narrow window1 at the fine weight, plus coarse-grid states inside the wide window2 (but not the
# narrow window) at the coarse weight, deduping the k-points on the fine grid.

isdefined(@__MODULE__, :_load_model_from_artifacts) ||
    include(joinpath(@__DIR__, "..", "common_models_from_artifacts.jl"))

@testset "Multigrid k-point weights (pure BZ quadrature)" begin
    model = _load_model_from_artifacts("pb")
    sym = model.symmetry
    eV = EP.unit_to_aru(:eV)
    e_fermi = 11.594123 * eV
    w1 = (e_fermi - 0.2eV, e_fermi + 0.2eV)   # narrow window: fine refinement
    w2 = (-Inf, Inf)                          # whole BZ: coarse grid tiles the full BZ, every band

    # A smooth, periodic test integrand of the crystal-coordinate k-vector (no e-ph, no energies).
    # ∫_BZ f dk = 1 exactly; any single-grid quadrature integrates it up to its own spacing error.
    f(xk) = 1 + 0.3 * cos(2π * xk[1]) * cos(2π * xk[2]) + 0.2 * cos(4π * xk[3])

    @testset "Dedup: coarse-region points absent from fine window1 set" begin
        # DECISION 1 developer-must-verify: every coarse node kept in the merge hashes to the fine
        # grid and is genuinely absent from the fine window1 set.
        nf, nc = (24, 24, 24), (12, 12, 12)
        kf = EP.filter_electron_states(nf, model.nw, model.el_ham, w1; symmetry=sym).kpts
        sel = EP.filter_electron_states_multigrid(nf, nc, w1, w2, model.nw, model.el_ham; symmetry=sym)
        @test sel isa StateSelection
        @test sel.kpts.ngrid == nf                  # fine ngrid stamped (DECISION 6 / q-lookup)
        # Every shared-grid point is a distinct fine-grid node (no duplicate keys).
        keys_seen = Set{NTuple{3,Int}}()
        for xk in sel.kpts.vectors
            key = Tuple(mod.(round.(Int, xk .* nf), nf))
            @test !(key in keys_seen)
            push!(keys_seen, key)
        end
        # Points beyond the fine window1 set are coarse-region nodes; each must hash to the fine grid.
        n_coarse_region = 0
        for xk in sel.kpts.vectors
            if EP.xk_to_ik(xk, kf) === nothing
                n_coarse_region += 1
                @test all(isapprox.(xk .* nf, round.(xk .* nf); atol=1e-8))
            end
        end
        @test n_coarse_region > 0                   # multigrid actually added coarse points
    end

    @testset "Per-band quadrature: Σ per-state weight over each band → 1" begin
        ref = 1.0  # ∫_BZ f dk and ∫_BZ 1 dk
        errs_w = Float64[]
        errs_f = Float64[]
        for (nf, nc) in ((12, 6), (24, 12), (48, 24))
            sel = EP.filter_electron_states_multigrid((nf, nf, nf), (nc, nc, nc), w1, w2,
                                              model.nw, model.el_ham; symmetry=sym)
            w = state_weights(sel)
            xks = state_xks(sel)
            coarse_tol = 4 / nc   # O(coarse-spacing) double-grid boundary quadrature error
            # Per BAND (window2 = whole BZ ⇒ every band is sampled over the full BZ): the per-state
            # weights of a single band must integrate 1 (and the smooth f) to the coarse tolerance.
            # This is what catches per-(k,band) over-selection — a merged single-window k-set could
            # not, since a fine node's wide-tail band would ride at the fine weight.
            for b in unique(sel.ibands)
                idx = findall(==(b), sel.ibands)
                sw  = sum(w[idx])
                sf  = sum(w[i] * f(xks[i]) for i in idx)
                @test sw ≈ ref atol = coarse_tol
                @test sf ≈ ref atol = coarse_tol
                push!(errs_w, abs(sw - ref)); push!(errs_f, abs(sf - ref))
            end
            # Coarse grid alone tiles the BZ exactly (sanity on the reference level).
            kc = EP.GridKpoints(EP.kpoints_grid((nc, nc, nc); symmetry=sym))
            @test sum(kc.weights) ≈ 1
        end
        # The per-band boundary quadrature error stays a small fraction of unity at every level.
        # The constant integrand (Σw → 1) is tighter; the oscillating f picks up a slightly larger
        # O(coarse-spacing) boundary-shell error at the coarsest level (nc = 6 ⇒ spacing ~0.17).
        @test maximum(errs_w) < 0.05
        @test maximum(errs_f) < 0.08
    end
end
