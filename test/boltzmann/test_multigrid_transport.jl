using Test
using ElectronPhonon
const EP = ElectronPhonon
using LinearAlgebra

# Multigrid BTE transport: end-to-end validation on the Pb artifact model (portable fixture; the
# heavy production numbers are measured on Cu, see plans/multigrid_bte_ksampling.md). Three checks:
#   1. multigrid vs uniform reference — the double-grid σ discrepancy (reported, not pre-toleranced);
#   2. the fine refinement improves accuracy — multigrid beats the coarse grid alone;
#   3. CPU-vs-GPU equality on the multigrid — the grid change did not break backend agreement.
# All solved with interpolate=false (unfold-only δf feedback, exact on the shared multigrid spec).
#
# μ is FIXED here: the μ-solve reads nelec_below_window, which is only correct when the FULL grid is
# filtered (below-window k-points contribute). A pre-built windowed GridKpoints (the multigrid) has
# no below-window points, so an in-driver μ-solve would mis-count; fixing μ is the correct same-μ
# transport comparison and sidesteps that. (See the plan's μ-solve note.)

const _CUDA_OK = (get(ENV, "EP_TEST_CUDA", "1") == "1") && try
    @eval import CUDA
    CUDA.functional()
catch
    false
end

@testset "Multigrid BTE transport (Pb)" begin
    model = _load_model_from_artifacts("pb")
    model.el_velocity_mode = :BerryConnection
    eV = EP.unit_to_aru(:eV); K = EP.unit_to_aru(:K); meV = EP.unit_to_aru(:meV)
    μ = 11.594123 * eV                         # fixed chemical potential (see header)
    w_wide = (μ - 0.4eV, μ + 0.4eV)
    w_fine = (μ - 0.1eV, μ + 0.1eV)
    sym = model.symmetry

    occ() = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0, μlist = μ,
        volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac)
    mkcalc() = BoltzmannCalculator{Float64}(; occ = occ(),
        smearing_list = [SmearingType(:Gaussian, 100.0 * meV)], occupation_method = 5)
    σ_to_SI(σ) = σ .* EP.e2 / (EP.unit_to_aru(:A) / EP.unit_to_aru(:V) / EP.unit_to_aru(:cm))

    run_grid(kk, kq; use_gpu) = (c = mkcalc();
        EP.run_eph_over_k_and_kq(model, kk, kq; calculators = [c], symmetry = sym,
            el_kq_from_unfolding = false, window_k = w_wide, window_kq = w_wide,
            fourier_mode = "gridopt", use_gpu, progress_print_step = 10^9); c)
    solve(c) = EP.solve_electron_bte(c.el_i, c.el_f, c.Sᵢ, stack(c.Sₒ), occ(), sym; interpolate = false)
    reldiff(a, b) = norm(a - b) / norm(b)

    on_gpu = _CUDA_OK

    # Reference (uniform 12/±0.4), coarse-only (6/±0.4), multigrid (fine 12/±0.1 + coarse 6/±0.4).
    c_ref = run_grid((12, 12, 12), (12, 12, 12); use_gpu = on_gpu)
    c_coarse = run_grid((6, 6, 6), (6, 6, 6); use_gpu = on_gpu)
    kmg, nbw = EP.filter_kpoints_multigrid((12, 12, 12), (6, 6, 6), w_fine, w_wide,
                                           model.nw, model.el_ham; symmetry = sym, use_gpu = on_gpu)
    @test kmg.ngrid == (12, 12, 12)
    c_mg = run_grid(kmg, kmg; use_gpu = on_gpu)

    r_ref = solve(c_ref); r_coarse = solve(c_coarse); r_mg = solve(c_mg)
    @test all(isfinite, r_mg.σ) && all(isfinite, r_mg.σ_serta)

    # The multigrid shrinks the final-state count vs the uniform reference.
    @test c_mg.el_f.n < c_ref.el_f.n

    # Fine refinement improves accuracy: multigrid is closer to the uniform reference than the
    # coarse grid alone, for both SERTA and full BTE.
    @test reldiff(r_mg.σ_serta, r_ref.σ_serta) < reldiff(r_coarse.σ_serta, r_ref.σ_serta)
    @test reldiff(r_mg.σ, r_ref.σ)             < reldiff(r_coarse.σ, r_ref.σ)

    @info "multigrid vs uniform reference (Pb)" σ_SERTA_reldiff = reldiff(r_mg.σ_serta, r_ref.σ_serta) σ_BTE_reldiff = reldiff(r_mg.σ, r_ref.σ)

    # CPU-vs-GPU equality on the multigrid: same calculator, both backends fold the identical
    # bte_scattering_increments, so Sₒ/Sᵢ and σ agree to ~machine eps.
    if _CUDA_OK
        c_mg_cpu = run_grid(kmg, kmg; use_gpu = false)
        @test stack(c_mg_cpu.Sₒ) ≈ stack(c_mg.Sₒ) rtol = 1e-9
        @test stack(c_mg_cpu.Sᵢ) ≈ stack(c_mg.Sᵢ) rtol = 1e-9
        r_mg_cpu = solve(c_mg_cpu)
        @test r_mg_cpu.σ_serta ≈ r_mg.σ_serta rtol = 1e-9
        @test r_mg_cpu.σ       ≈ r_mg.σ       rtol = 1e-9
    else
        @info "CUDA not functional — skipping CPU-vs-GPU multigrid equality"
    end
end

# Chemical-potential solve on a pre-built WINDOWED multigrid. Auto-μ (nlist-based, no μlist) reads
# nelec_below_window; the windowed grid has no below-window k, so the driver's recompute under-counts
# and the μ bisection fails to bracket. `filter_kpoints_multigrid` returns the correct coarse-window
# count; passing it via `nelec_below_window_k/kq` fixes the solve.
@testset "Multigrid μ-solve: bracket bug + fix (Pb)" begin
    model = _load_model_from_artifacts("pb")
    model.el_velocity_mode = :BerryConnection
    eV = EP.unit_to_aru(:eV); K = EP.unit_to_aru(:K); meV = EP.unit_to_aru(:meV)
    e_fermi = 11.594123 * eV
    w_wide = (e_fermi - 0.4eV, e_fermi + 0.4eV)
    w_fine = (e_fermi - 0.1eV, e_fermi + 0.1eV)
    sym = model.symmetry
    on_gpu = _CUDA_OK

    # Auto-μ occupation: nlist set, μlist NOT set → the driver solves μ.
    occ_auto() = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0,
        volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac)
    mkcalc(o) = BoltzmannCalculator{Float64}(; occ = o,
        smearing_list = [SmearingType(:Gaussian, 100.0 * meV)], occupation_method = 5)
    run_grid(o, kk, kq; nbw = nothing) = EP.run_eph_over_k_and_kq(model, kk, kq;
        calculators = [mkcalc(o)], symmetry = sym, el_kq_from_unfolding = false,
        window_k = w_wide, window_kq = w_wide, fourier_mode = "gridopt", use_gpu = on_gpu,
        progress_print_step = 10^9, nelec_below_window_k = nbw, nelec_below_window_kq = nbw)

    # Uniform reference: the full grid is filtered, so the μ solve brackets.
    o_ref = occ_auto()
    run_grid(o_ref, (12, 12, 12), (12, 12, 12))
    μ_ref = o_ref.μlist[1]
    @test isfinite(μ_ref)

    kmg, nbw = EP.filter_kpoints_multigrid((12, 12, 12), (6, 6, 6), w_fine, w_wide,
                                           model.nw, model.el_ham; symmetry = sym, use_gpu = on_gpu)

    # BUG: without the below-window override, the windowed multigrid's μ solve fails to bracket
    # (Roots.bisection over [-Inf, Inf] cannot bracket the under-counted carrier target).
    @test_throws "the interval provided does not bracket a root" run_grid(occ_auto(), kmg, kmg)

    # FIX: pass the coarse-window nelec_below_window from filter_kpoints_multigrid.
    o_mg = occ_auto()
    run_grid(o_mg, kmg, kmg; nbw = nbw)
    @test isfinite(o_mg.μlist[1])
    @test abs(o_mg.μlist[1] - μ_ref) < 0.3eV        # sane on tiny Pb grids (Cu 100/50: Δμ≈2 meV)
end
