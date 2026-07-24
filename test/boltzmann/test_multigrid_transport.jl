using Test
using ElectronPhonon
const EP = ElectronPhonon
using ElectronPhonon: FilteredStates, state_xks
using LinearAlgebra

# Multigrid BTE transport: end-to-end validation on the Pb artifact model (portable fixture; the
# heavy production numbers are measured on Cu). The multigrid is a
# prebuilt `FilteredStates` (Generator 2, `filter_electron_states_multigrid`) with the double-grid per-(k,band)
# weights; the k+q selection is its explicit symmetry unfold (`unfold_band_states`). Checks:
#   1. multigrid vs uniform reference — the double-grid σ discrepancy (reported, not pre-toleranced);
#   2. the fine refinement improves accuracy — multigrid beats the coarse grid alone;
#   3. CPU-vs-GPU equality on the multigrid — the per-final-state weight is bit-identical on both;
#   4. over-selection regression — a wide-tail band at a fine-only node is absent from el_i/el_f;
#   5. auto-μ succeeds on the windowed multigrid selection (no bracket failure).
# All solved with interpolate=false (unfold-only δf feedback, exact on the shared multigrid spec).

isdefined(@__MODULE__, :_load_model_from_artifacts) ||
    include(joinpath(@__DIR__, "..", "common_models_from_artifacts.jl"))

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
    μ = 11.594123 * eV                         # fixed chemical potential
    w_wide = (μ - 0.4eV, μ + 0.4eV)
    w_fine = (μ - 0.1eV, μ + 0.1eV)
    sym = model.symmetry

    occ() = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0, μlist = μ,
        volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac)
    mkcalc() = BoltzmannCalculator{Float64}(; occ = occ(),
        smearing_list = [SmearingType(:Gaussian, 100.0 * meV)], occupation_method = 5)

    # Grid/tuple input runs Generator 1 internally (sugar); a FilteredStates passes through as-is.
    run_sel(kk, kq; use_gpu) = (c = mkcalc();
        EP.run_eph_over_k_and_kq(model, kk, kq; calculators = [c], symmetry = sym,
            el_kq_from_unfolding = false, window_k = w_wide, window_kq = w_wide,
            fourier_mode = "gridopt", use_gpu, progress_print_step = 10^9); c)
    solve(c) = EP.solve_electron_bte(c.el_i, c.el_f, c.Sᵢ, stack(c.Sₒ), occ(), sym; interpolate = false)
    reldiff(a, b) = norm(a - b) / norm(b)

    on_gpu = _CUDA_OK

    # Reference uniform 12/±0.4; coarse-only uniform 6/±0.4; multigrid fine 12/±0.1 + coarse 6/±0.4.
    c_ref = run_sel((12, 12, 12), (12, 12, 12); use_gpu = on_gpu)
    c_coarse = run_sel((6, 6, 6), (6, 6, 6); use_gpu = on_gpu)
    sel_k = EP.filter_electron_states_multigrid((12, 12, 12), (6, 6, 6), w_fine, w_wide,
                                        model.nw, model.el_ham; symmetry = sym, use_gpu = on_gpu)
    sel_kq = EP.unfold_band_states(sel_k, sym)     # explicit full-BZ k+q selection
    @test sel_k isa FilteredStates && sel_kq isa FilteredStates
    @test sel_k.kpts.ngrid == (12, 12, 12)
    c_mg = run_sel(sel_k, sel_kq; use_gpu = on_gpu)

    r_ref = solve(c_ref); r_coarse = solve(c_coarse); r_mg = solve(c_mg)
    @test all(isfinite, r_mg.σ) && all(isfinite, r_mg.σ_serta)

    # The multigrid shrinks the final-state count vs the uniform reference.
    @test c_mg.el_f.n < c_ref.el_f.n

    # Fine refinement improves accuracy: multigrid is closer to the uniform reference than the
    # coarse grid alone, for both SERTA and full BTE.
    @test reldiff(r_mg.σ_serta, r_ref.σ_serta) < reldiff(r_coarse.σ_serta, r_ref.σ_serta)
    @test reldiff(r_mg.σ, r_ref.σ)             < reldiff(r_coarse.σ, r_ref.σ)

    @info "multigrid vs uniform reference (Pb)" σ_SERTA_reldiff = reldiff(r_mg.σ_serta, r_ref.σ_serta) σ_BTE_reldiff = reldiff(r_mg.σ, r_ref.σ)

    @testset "over-selection regression: fine-only node has no wide-tail band" begin
        # A node on the fine grid but NOT on the coarse grid ("fine-only") must carry only bands in
        # the NARROW window — the old merged-grid + wide re-filter would have kept its wide-tail
        # bands at the fine weight (the confirmed bug). Verified on both el_i and el_f.
        oncoarse(xk) = all(isapprox.(xk .* 6, round.(xk .* 6); atol = 1e-8))
        for el in (c_mg.el_i, c_mg.el_f)
            xks = state_xks(el)
            n_fine_only = 0
            for i in 1:el.n
                oncoarse(xks[i]) && continue          # coincident node may carry wide-only bands
                n_fine_only += 1
                @test w_fine[1] - 1e-9 <= el.es[i] <= w_fine[2] + 1e-9
            end
            @test n_fine_only > 0                     # the check is non-vacuous
        end
    end

    # CPU-vs-GPU equality on the multigrid: same calculator, both backends fold the identical
    # bte_scattering_increments with the per-final-state weight, so Sₒ/Sᵢ and σ agree to ~machine eps.
    if _CUDA_OK
        c_mg_cpu = run_sel(sel_k, sel_kq; use_gpu = false)
        @test stack(c_mg_cpu.Sₒ) ≈ stack(c_mg.Sₒ) rtol = 1e-9
        @test stack(c_mg_cpu.Sᵢ) ≈ stack(c_mg.Sᵢ) rtol = 1e-9
        r_mg_cpu = solve(c_mg_cpu)
        @test r_mg_cpu.σ_serta ≈ r_mg.σ_serta rtol = 1e-9
        @test r_mg_cpu.σ       ≈ r_mg.σ       rtol = 1e-9
    else
        @info "CUDA not functional — skipping CPU-vs-GPU multigrid equality"
    end
end

# Chemical-potential solve on a prebuilt WINDOWED multigrid selection. Auto-μ (nlist-based, no μlist)
# reads `nstates_base`; the selection carries the correct coarse-window below-window count, so the μ
# bisection brackets normally — no override kwarg, no bracket failure.
@testset "Multigrid μ-solve succeeds on the selection (Pb)" begin
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
    run_grid(o, kk, kq) = EP.run_eph_over_k_and_kq(model, kk, kq;
        calculators = [mkcalc(o)], symmetry = sym, el_kq_from_unfolding = false,
        window_k = w_wide, window_kq = w_wide, fourier_mode = "gridopt", use_gpu = on_gpu,
        progress_print_step = 10^9)

    # Uniform reference: the full grid is filtered, so the μ solve brackets.
    o_ref = occ_auto()
    run_grid(o_ref, (12, 12, 12), (12, 12, 12))
    μ_ref = o_ref.μlist[1]
    @test isfinite(μ_ref)

    # Multigrid selection: `nstates_base` rides on the selection, so auto-μ SUCCEEDS (does not throw).
    sel_k = EP.filter_electron_states_multigrid((12, 12, 12), (6, 6, 6), w_fine, w_wide,
                                        model.nw, model.el_ham; symmetry = sym, use_gpu = on_gpu)
    sel_kq = EP.unfold_band_states(sel_k, sym)
    o_mg = occ_auto()
    run_grid(o_mg, sel_k, sel_kq)
    @test isfinite(o_mg.μlist[1])
    @test abs(o_mg.μlist[1] - μ_ref) < 0.3eV        # sane on tiny Pb grids (Cu 100/50: Δμ≈2 meV)
end

# Independent value-pinning of the auto-μ path. The transport tests above run at fixed μ and the
# multigrid μ test compares to a reference from the SAME new code path, so neither pins the μ VALUE
# that `bte_compute_μ!` produces. This guards it with hardcoded golden values: on a uniform windowed
# Pb selection, the below-window carrier count `nstates_base` and the solved metal μ are fixed
# regression targets (recompute + update deliberately if the μ convention ever changes).
@testset "auto-μ solve pinned values (Pb, uniform)" begin
    model = _load_model_from_artifacts("pb")
    eV = EP.unit_to_aru(:eV); K = EP.unit_to_aru(:K)
    e_fermi = 11.594123 * eV
    window = (e_fermi - 0.4eV, e_fermi + 0.4eV)

    sel = EP.filter_electron_states((12, 12, 12), model, window; symmetry = model.symmetry, fourier_mode = "gridopt")
    els = EP.compute_electron_states(model, sel, ["eigenvalue", "eigenvector", "velocity"]; fourier_mode = "gridopt")
    el  = EP.electron_states_to_BandStates(els, sel)

    # nstates_base = below-window fully-occupied carriers per cell (rides on the selection).
    @test el.nstates_base ≈ 1.898726851851852 rtol = 1e-10

    # Auto-μ (nlist set, μlist unset): metal branch, solved by bisection on the selection's states.
    occ = ElectronOccupationParams(; Tlist = [300.0 * K], nlist = 4.0, μlist = nothing,
        volume = model.volume, nelec = 0, spin_degeneracy = 2, occ_type = :FermiDirac)
    EP.bte_compute_μ!(occ, el; do_print = false)
    @test occ.type == :Metal
    @test occ.μlist[1] ≈ 0.8631722535557365 rtol = 1e-8      # 11.7441 eV
end
