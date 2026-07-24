# Multigrid (double-grid) BTE transport example — Cu.
#
# Instead of a single uniform k/k+q grid, sample a FINE grid in a narrow window near μ merged with
# a COARSE grid over a wider window. This shrinks both the initial (el_i) and final (el_f) state
# counts, so the dense scattering-in matrix Sᵢ (∝ nk⁶) shrinks by ~n_i-reduction × n_f-reduction —
# the OOM wall for uniform runs at large nk. Measured (Cu, fine 100³/±0.1 + coarse 50³/±0.4): σ
# within ~0.4% of the uniform 100³/±0.4 reference at ~8x smaller Sᵢ, vs ~5% for the coarse grid
# alone — the fine shell recovers near-μ accuracy cheaply.
#
# Flow (caller builds, cf. `bte_gpu_taas.jl`):
#   1. build the k `FilteredStates` with `filter_electron_states_multigrid` (fine + coarse spec); it carries
#      the per-(k,band) double-grid weights and `nstates_base` (the coarse full-grid below-window
#      carrier count), so the auto-μ solve brackets on the windowed selection with no override,
#   2. build the k+q `FilteredStates` as the explicit symmetry unfold of the k selection
#      (`unfold_band_states`) — the full-BZ final states,
#   3. pass BOTH selections to `run_eph_over_k_and_kq` (consumed as-is, WIDE window) and
#   4. solve with `solve_electron_bte(...; interpolate=false)` — el_f is the symmetry unfolding of
#      el_i on the shared multigrid spec, so the δf feedback map is an exact integer-grid-key
#      lookup (no linear interpolation, no uniform-grid assumption; benign band-edge misses are
#      treated as zero).

using ElectronPhonon
const EP = ElectronPhonon
using LinearAlgebra

# --- model ---
folder = "/mnt/home/jlihm/ceph/boltzmann/Cu/1_epw.projWF/"
model = EP.load_model_from_epw_new(folder, "temp", "Cu")
model.el_velocity_mode = :Direct

# --- transport setup ---
eV = unit_to_aru(:eV); meV = unit_to_aru(:meV); K = unit_to_aru(:K)
e_REF   = 17.3494 * eV
window_narrow = (-0.1, 0.1) .* eV .+ e_REF        # narrow window: dense refinement near μ
window_wide = (-0.4, 0.4) .* eV .+ e_REF        # wide window: coarse tail

# Auto-μ: nlist set, μlist left unset so the driver solves μ. The multigrid `FilteredStates` carries
# the below-window carrier count (`nstates_base`, from the coarse full-grid filter), so the μ solve
# brackets on the windowed selection without any override kwarg.
occupation_params() = ElectronOccupationParams(;
    Tlist = [300.0] .* K,
    nlist = 11.0,
    volume = model.volume,
    nelec = 0,
    spin_degeneracy = 2,
    occ_type = :FermiDirac,
)

# σ (atomic units) → (Ω·cm)⁻¹
σ_to_SI(σ) = σ .* EP.e2 / (unit_to_aru(:A) / unit_to_aru(:V) / unit_to_aru(:cm))

"""
    run_bte_multigrid(model, nk_narrow, nk_wide; η, window_narrow, window_wide, occ,
                      method, use_gpu, symmetry)

Run one multigrid BTE transport calculation and return the SERTA and full-BTE conductivity
tensors (SI, `(Ω·cm)⁻¹`, shape `(3,3,nT)`) plus the calculator. `nk_narrow` must be a multiple of
`nk_wide`. Builds the k `FilteredStates` (double grid) and the full-BZ k+q selection (its symmetry
unfold).
"""
function run_bte_multigrid(model, nk_narrow, nk_wide; η, window_narrow, window_wide, occ,
        method = 5, use_gpu = true, symmetry = model.symmetry)
    sel_k = EP.filter_electron_states_multigrid(
        (nk_narrow, nk_narrow, nk_narrow), (nk_wide, nk_wide, nk_wide),
        window_narrow, window_wide, model.nw, model.el_ham; symmetry, use_gpu)
    sel_kq = EP.unfold_band_states(sel_k, symmetry)   # explicit full-BZ k+q selection

    calc = BoltzmannCalculator{Float64}(; occ,
        smearing_list = [SmearingType(:Lorentzian, η) for _ in 1:length(occ)],
        occupation_method = method)
    EP.run_eph_over_k_and_kq(model, sel_k, sel_kq;
        calculators = [calc], symmetry, el_kq_from_unfolding = false,
        window_k = window_wide, window_kq = window_wide, use_gpu,
        nchunks_threads = Threads.nthreads(), progress_print_step = 200)

    # interpolate=false: exact unfold-only δf feedback on the shared multigrid spec.
    res = EP.solve_electron_bte(calc.el_i, calc.el_f, calc.Sᵢ, stack(calc.Sₒ), occ, symmetry;
                                interpolate = false)
    (; σ_serta_SI = σ_to_SI(res.σ_serta), σ_bte_SI = σ_to_SI(res.σ), res, calc, sel_k, sel_kq)
end

# --- run ---
nk_narrow   = 100
nk_wide = 50
η   = 5.0 * meV
occ = occupation_params()
out = run_bte_multigrid(model, nk_narrow, nk_wide; η, window_narrow, window_wide, occ,
                        method = 5, use_gpu = true)

println("\nmultigrid: fine $(nk_narrow)³/±0.1eV + coarse $(nk_wide)³/±0.4eV")
println("  el_i.n = $(out.calc.el_i.n)   el_f.n = $(out.calc.el_f.n)")
println("\nCu σ (multigrid BTE), trace/3, per temperature:")
for (iT, T) in enumerate(occ.Tlist)
    σs = sum(out.σ_serta_SI[a, a, iT] for a in 1:3) / 3
    σb = sum(out.σ_bte_SI[a, a, iT]   for a in 1:3) / 3
    println("  T = $(round(T/K, digits=0)) K :  σ_SERTA = $(round(σs, sigdigits=4))  ",
            "σ_BTE = $(round(σb, sigdigits=4))  (Ω·cm)⁻¹")
end
