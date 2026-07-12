# GPU Boltzmann transport (BTE) example — TaAs.
#
# Demonstrates the full flow with `BoltzmannCalculator`:
#   1. extract the scattering matrices Sₒ (SERTA lifetime) and Sᵢ (scattering-in) in ONE GPU pass
#      of `run_eph_over_k_and_kq` (use_gpu = true), then
#   2. solve the linearized BTE with `solve_electron_bte` (now BandStates-aware) to get σ.
#
# The same calculator runs on the CPU (use_gpu = false) for validation. Uses IBZ symmetry
# (`symmetry = model.symmetry`, `el_kq_from_unfolding = false`) — the outer k-grid is reduced to
# the irreducible BZ, the dominant transport speedup. FermiDirac occupation + Gaussian smearing.

using ElectronPhonon
const EP = ElectronPhonon
using LinearAlgebra

# --- model ---
# Point EP_TAAS_FOLDER at a TaAs EPW output folder (2_epw.projWF).
folder = get(ENV, "EP_TAAS_FOLDER", joinpath(homedir(), "ceph/downfolding/TaAs/2_epw.projWF"))
model = EP.load_model_from_epw_new(folder, "temp", "TaAs")

# --- transport setup ---
eV = unit_to_aru(:eV); meV = unit_to_aru(:meV); K = unit_to_aru(:K)
e_REF  = 17.0662 * eV
window = (-0.25, 0.25) .* eV .+ e_REF      # active energy window around the Fermi level
μ_fixed = 17.0633 * eV                     # fixed chemical potential (set explicitly to skip the μ solve)

occupation_params() = ElectronOccupationParams(;
    Tlist = collect(100.0:100.0:300.0) .* K,
    nlist = 16.0,                          # electrons per cell
    μlist = μ_fixed,
    volume = model.volume,
    nelec = 0,
    spin_degeneracy = 1,
    occ_type = :FermiDirac,
)

# σ (atomic units) → (Ω·cm)⁻¹  (matches EPW print_mobility / run_TaAs.jl)
σ_to_SI(σ) = σ .* EP.e2 / (unit_to_aru(:A) / unit_to_aru(:V) / unit_to_aru(:cm))

"""
    run_bte_gpu(model, nk; η, window, occ, method=5, use_gpu=true, symmetry=model.symmetry)

Run one BTE transport calculation on an `nk³` grid and return the SERTA and full-BTE
conductivity tensors (SI, `(Ω·cm)⁻¹`, shape `(3,3,nT)`) plus the calculator. `method` is the
occupation-factor convention 1..6 (see `bte_scattering_increments`).
"""
function run_bte_gpu(model, nk; η, window, occ, method = 5,
        use_gpu = true, symmetry = model.symmetry)
    calc = BoltzmannCalculator{Float64}(; occ, smearing = [(:Gaussian, η) for _ in 1:length(occ)],
        occupation_method = method)
    EP.run_eph_over_k_and_kq(model, (nk, nk, nk), (nk, nk, nk);
        calculators = [calc], symmetry, el_kq_from_unfolding = false,
        window_k = window, window_kq = window, use_gpu,
        nchunks_threads = Threads.nthreads(), progress_print_step = 200)
    res = EP.solve_electron_bte(calc.el_i, calc.el_f, calc.Sᵢ, stack(calc.Sₒ), occ, symmetry)
    (; σ_serta_SI = σ_to_SI(res.σ_serta), σ_bte_SI = σ_to_SI(res.σ), res, calc)
end

# --- run ---
nk = 16
η  = 10.0 * meV
occ = occupation_params()
out = run_bte_gpu(model, nk; η, window, occ, method = :Method5, use_gpu = true)

println("\nTaAs σ (GPU BTE), trace/3, per temperature:")
for (iT, T) in enumerate(occ.Tlist)
    σs = sum(out.σ_serta_SI[a, a, iT] for a in 1:3) / 3
    σb = sum(out.σ_bte_SI[a, a, iT]   for a in 1:3) / 3
    println("  T = $(round(T/K, digits=0)) K :  σ_SERTA = $(round(σs, sigdigits=4))  ",
            "σ_BTE = $(round(σb, sigdigits=4))  (Ω·cm)⁻¹")
end
