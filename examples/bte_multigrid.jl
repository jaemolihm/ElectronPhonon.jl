# Multigrid (double-grid) BTE transport example — TaAs.
#
# Instead of a single uniform k/k+q grid, sample a FINE grid in a narrow window near μ merged with
# a COARSE grid over a wider window. This shrinks both the initial (el_i) and final (el_f) state
# counts, so the dense scattering-in matrix Sᵢ (∝ nk⁶) shrinks by ~n_i-reduction × n_f-reduction —
# the OOM wall for uniform runs at large nk.
#
# Flow (caller builds, cf. `bte_gpu_taas.jl`):
#   1. build ONE multigrid `GridKpoints` with `filter_kpoints_multigrid` (fine + coarse spec),
#   2. pass it as BOTH the k and k+q argument to `run_eph_over_k_and_kq` with the WIDE window
#      (the narrow window lives only inside the multigrid construction),
#   3. solve with `solve_electron_bte(...; interpolate=false)` — el_f is the exact symmetry
#      unfolding of el_i on the shared multigrid spec, so the δf feedback map is an exact
#      integer-grid-key lookup (no linear interpolation, no uniform-grid assumption).

using ElectronPhonon
const EP = ElectronPhonon
using LinearAlgebra

# --- model ---
folder = "/mnt/home/jlihm/ceph/downfolding/TaAs/2_epw.projWF"
model = EP.load_model_from_epw_new(folder, "temp", "TaAs")

# --- transport setup ---
eV = unit_to_aru(:eV); meV = unit_to_aru(:meV); K = unit_to_aru(:K)
e_REF   = 17.0662 * eV
window_fine = (-0.1, 0.1) .* eV .+ e_REF        # narrow window: dense refinement near μ
window_wide = (-0.4, 0.4) .* eV .+ e_REF        # wide window: coarse tail

# Auto-μ: nlist set, μlist left unset so the driver solves μ. A multigrid is a pre-built WINDOWED
# grid with no below-window k, so the below-window carrier count must be supplied from construction
# (nelec_below_window from filter_kpoints_multigrid, passed to run_eph_over_k_and_kq) or the μ solve
# cannot bracket.
occupation_params() = ElectronOccupationParams(;
    Tlist = collect(100.0:100.0:300.0) .* K,
    nlist = 16.0,
    volume = model.volume,
    nelec = 0,
    spin_degeneracy = 1,
    occ_type = :FermiDirac,
)

# σ (atomic units) → (Ω·cm)⁻¹  (matches EPW print_mobility / run_TaAs.jl)
σ_to_SI(σ) = σ .* EP.e2 / (unit_to_aru(:A) / unit_to_aru(:V) / unit_to_aru(:cm))

"""
    run_bte_multigrid(model, nk_fine, nk_coarse; η, window_fine, window_wide, occ,
                      method, use_gpu, symmetry)

Run one multigrid BTE transport calculation and return the SERTA and full-BTE conductivity
tensors (SI, `(Ω·cm)⁻¹`, shape `(3,3,nT)`) plus the calculator. `nk_fine` must be a multiple of
`nk_coarse`. Builds a single multigrid spec shared by the k and k+q arguments.
"""
function run_bte_multigrid(model, nk_fine, nk_coarse; η, window_fine, window_wide, occ,
        method = 5, use_gpu = true, symmetry = model.symmetry)
    kmg, nelec_below_window = EP.filter_kpoints_multigrid(
        (nk_fine, nk_fine, nk_fine), (nk_coarse, nk_coarse, nk_coarse),
        window_fine, window_wide, model.nw, model.el_ham; symmetry, use_gpu)

    calc = BoltzmannCalculator{Float64}(; occ,
        smearing_list = [SmearingType(:Gaussian, η) for _ in 1:length(occ)],
        occupation_method = method)
    EP.run_eph_over_k_and_kq(model, kmg, kmg;
        calculators = [calc], symmetry, el_kq_from_unfolding = false,
        window_k = window_wide, window_kq = window_wide, use_gpu,
        nelec_below_window_k = nelec_below_window, nelec_below_window_kq = nelec_below_window,
        nchunks_threads = Threads.nthreads(), progress_print_step = 200)

    # interpolate=false: exact unfold-only δf feedback on the shared multigrid spec.
    res = EP.solve_electron_bte(calc.el_i, calc.el_f, calc.Sᵢ, stack(calc.Sₒ), occ, symmetry;
                                interpolate = false)
    (; σ_serta_SI = σ_to_SI(res.σ_serta), σ_bte_SI = σ_to_SI(res.σ), res, calc, kmg)
end

# --- run ---
nk_fine   = 100
nk_coarse = 50
η   = 10.0 * meV
occ = occupation_params()
out = run_bte_multigrid(model, nk_fine, nk_coarse; η, window_fine, window_wide, occ,
                        method = 5, use_gpu = true)

println("\nmultigrid: fine $(nk_fine)³/±0.1eV + coarse $(nk_coarse)³/±0.4eV")
println("  el_i.n = $(out.calc.el_i.n)   el_f.n = $(out.calc.el_f.n)")
println("\nTaAs σ (multigrid BTE), trace/3, per temperature:")
for (iT, T) in enumerate(occ.Tlist)
    σs = sum(out.σ_serta_SI[a, a, iT] for a in 1:3) / 3
    σb = sum(out.σ_bte_SI[a, a, iT]   for a in 1:3) / 3
    println("  T = $(round(T/K, digits=0)) K :  σ_SERTA = $(round(σs, sigdigits=4))  ",
            "σ_BTE = $(round(σb, sigdigits=4))  (Ω·cm)⁻¹")
end
