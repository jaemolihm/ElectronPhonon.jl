using Test
using ElectronPhonon
const EP = ElectronPhonon

# Pure-BZ (e-ph-independent) quadrature check for the multigrid k-sampling weights. Grounds the
# double-grid weight correctness before trusting transport numbers: the merged per-point weights
# must integrate a smooth periodic function over the BZ, converging to the coarse-grid tolerance.
# `filter_kpoints_multigrid` keeps all fine points inside window1 and adds coarse points inside
# window2 that do not coincide with a kept fine point (dedup on the fine grid).

isdefined(@__MODULE__, :_load_model_from_artifacts) ||
    include(joinpath(@__DIR__, "..", "common_models_from_artifacts.jl"))

@testset "Multigrid k-point weights (pure BZ quadrature)" begin
    model = _load_model_from_artifacts("pb")
    sym = model.symmetry
    eV = EP.unit_to_aru(:eV)
    e_fermi = 11.594123 * eV
    w1 = (e_fermi - 0.2eV, e_fermi + 0.2eV)   # narrow window: fine refinement
    w2 = (-Inf, Inf)                          # whole BZ: coarse grid tiles the full BZ

    # A smooth, periodic test integrand of the crystal-coordinate k-vector (no e-ph, no energies).
    # ∫_BZ f dk = 1 exactly; any single-grid quadrature integrates it up to its own spacing error.
    f(xk) = 1 + 0.3 * cos(2π * xk[1]) * cos(2π * xk[2]) + 0.2 * cos(4π * xk[3])
    quad(kpts) = sum(kpts.weights[i] * f(kpts.vectors[i]) for i in 1:kpts.n)

    @testset "Dedup: coarse-region points absent from fine window1 set" begin
        # DECISION 1 developer-must-verify: every coarse node kept in the merge hashes to the fine
        # grid and is genuinely absent from the fine window1 set.
        nf, nc = (24, 24, 24), (12, 12, 12)
        kf, = EP.filter_kpoints(nf, model.nw, model.el_ham, w1; symmetry=sym)
        kf = EP.GridKpoints(kf)
        kmg, = EP.filter_kpoints_multigrid(nf, nc, w1, w2, model.nw, model.el_ham; symmetry=sym)
        @test kmg.ngrid == nf                       # fine ngrid stamped (DECISION 6)
        # Every kept point is a distinct fine-grid node (no duplicate keys).
        keys_seen = Set{NTuple{3,Int}}()
        for xk in kmg.vectors
            key = Tuple(mod.(round.(Int, xk .* nf), nf))
            @test !(key in keys_seen)
            push!(keys_seen, key)
        end
        # Points beyond the fine window1 set are coarse-region nodes; each must hash to the fine grid.
        n_coarse_region = 0
        for xk in kmg.vectors
            if EP.xk_to_ik(xk, kf) === nothing
                n_coarse_region += 1
                # coincides with a fine-grid node
                @test all(isapprox.(xk .* nf, round.(xk .* nf); atol=1e-8))
            end
        end
        @test n_coarse_region > 0                   # multigrid actually added coarse points
    end

    @testset "Σ w → 1 and smooth-integrand quadrature converges" begin
        ref = 1.0  # ∫_BZ f dk and ∫_BZ 1 dk
        errs_w = Float64[]
        errs_f = Float64[]
        for (nf, nc) in ((12, 6), (24, 12), (48, 24))
            kmg, = EP.filter_kpoints_multigrid((nf, nf, nf), (nc, nc, nc), w1, w2,
                                               model.nw, model.el_ham; symmetry=sym)
            # coarse-grid alone tiles the BZ exactly
            kc = EP.GridKpoints(EP.kpoints_grid((nc, nc, nc); symmetry=sym))
            @test sum(kc.weights) ≈ 1
            @test quad(kc) ≈ ref rtol = 1 / nc      # coarse quadrature within its own spacing

            push!(errs_w, abs(sum(kmg.weights) - 1))
            push!(errs_f, abs(quad(kmg) - ref))
            # Multigrid weight sum / quadrature within the coarse spacing (boundary quadrature error).
            # This is the O(coarse-spacing) double-grid boundary error, not double counting.
            coarse_tol = 3 / nc
            @test sum(kmg.weights) ≈ 1 atol = coarse_tol
            @test quad(kmg) ≈ ref atol = coarse_tol
        end
        # The boundary quadrature error stays a small fraction of unity at every level (it is a
        # fixed shell effect, not necessarily monotone in a single smooth integrand).
        @test maximum(errs_w) < 0.05
        @test maximum(errs_f) < 0.05
    end
end
