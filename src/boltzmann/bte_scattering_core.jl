# Shared BTE scattering physics — ONE implementation used by both the CPU calculator loop
# (`BoltzmannCalculator`'s `run_calculator!`) and the GPU device kernel (in the CUDA extension),
# so the two paths compute the same scattering (to round-off).
#
# Only the FermiDirac occupation + Gaussian smearing configuration is supported here (the
# configuration used for transport); the calculator
# asserts this at setup.

# --- Device-safe elementary occupation / smearing helpers ------------------------------------
# Specialized (no `occ_type`/smearing-type branches, no `throw` paths) so they compile inside a
# GPU kernel. Assume T > 0 (physical temperatures). Numerically identical to the FermiDirac /
# BoseEinstein / Gaussian branches of `occ_fermion`, `occ_fermion_derivative`, `occ_boson`,
# `delta_smeared` in src/common/utils.jl.
@inline _bte_occ_fd(e, T)       = 1 / (exp(e / T) + 1)
@inline _bte_occ_fd_deriv(e, T) = -1 / (2 + exp(e / T) + exp(-e / T)) / T
@inline _bte_occ_boson(e, T)    = e == 0 ? zero(e) : 1 / expm1(e / T)
@inline _bte_delta_gauss(Δe, η) = exp(-(Δe / η)^2) * inv_sqrtpi / η

"""
    bte_scattering_increments(method, ek, ekq, ωq, g2, wtq, μ, T, η) -> (sₒ, sᵢ)

Single-mode contribution to the BTE scattering-out (`sₒ`) and scattering-in (`sᵢ`) terms for one
`(electron k, electron k+q, phonon mode)` triple at one temperature/μ. The caller accumulates
`Sₒ[n] += sₒ` over `(m, ν, q)` and `Sᵢ[n, f] += sᵢ` over `ν` (see the CPU `run_calculator!`):

    sₒ = (δ₁·f₁ₒ + δ₂·f₂ₒ) · 2π·wtq·g2          # out-factors f1o,f2o per `method`
    sᵢ = (δ₂·f₁ᵢ + δ₁·f₂ᵢ) · 2π·wtq·g2          # in-factors  f1i,f2i per `method`

with `δ₁ = δ(ek−ekq+ωq)`, `δ₂ = δ(ek−ekq−ωq)` (Gaussian, width `η`). `method ∈ 1:6` selects the
occupation-factor convention (Method5 is the σ default). NOTE: the acoustic-phonon cutoff
(`ωq < omega_acoustic`) is NOT applied here — each caller skips sub-cutoff modes before calling
this (so a fresh caller must remember to). Vanishing-δ handling guards the `0·Inf → NaN` that
Methods 2–5 would otherwise hit when an out-of-window state has a vanishing `f`/`∂f` denominator
(→ `Inf` prefactor): both δ zero returns `(0, 0)` early (matching the `continue` skip in the CPU
reference), and each `δ·factor` product is zeroed per-δ so a single zero δ drops only its own term.

TODO: add citation to Lihm and Park (in preparation) for the occupation-factor conventions.
"""
@inline function bte_scattering_increments(method::Integer, ek, ekq, ωq, g2, wtq, μ, T, η)
    z = zero(ek)
    Δe1 = ek - ekq + ωq    # phonon absorption
    Δe2 = ek - ekq - ωq    # phonon emission
    δ1 = _bte_delta_gauss(Δe1, η)
    δ2 = _bte_delta_gauss(Δe2, η)
    (iszero(δ1) && iszero(δ2)) && return (z, z)   # fast path: no energy conservation at all

    nq   = _bte_occ_boson(ωq, T)
    f_k  = _bte_occ_fd(ek - μ, T)
    f_kq = _bte_occ_fd(ekq - μ, T)

    if method == 1
        f1o = nq + f_kq;          f2o = nq + 1 - f_kq
        f1i = nq + f_k;           f2i = nq + 1 - f_k
    elseif method == 2
        preo = f_kq * (1 - f_kq) / (f_k * (1 - f_k))
        f1o = preo * (nq + 1 - f_k);   f2o = preo * (nq + f_k)
        prei = f_k * (1 - f_k) / (f_kq * (1 - f_kq))
        f1i = prei * (nq + 1 - f_kq);  f2i = prei * (nq + f_kq)
    elseif method == 3
        preo = (1 - f_kq) / (1 - f_k); f1o = preo * nq;  f2o = preo * (nq + 1)
        prei = (1 - f_k) / (1 - f_kq); f1i = prei * nq;  f2i = prei * (nq + 1)
    elseif method == 4
        preo = f_kq / f_k;  f1o = preo * (nq + 1);  f2o = preo * nq
        prei = f_k / f_kq;  f1i = prei * (nq + 1);  f2i = prei * nq
    elseif method == 5
        df_k  = _bte_occ_fd_deriv(ek - μ, T)
        df_kq = _bte_occ_fd_deriv(ekq - μ, T)
        f1o = sqrt(nq * (nq + 1) * df_kq / df_k);  f2o = f1o
        f1i = sqrt(nq * (nq + 1) * df_k / df_kq);  f2i = f1i
    else  # method == 6
        nΔo = _bte_occ_boson(ekq - ek, T);  f1o =  (nΔo + f_kq);  f2o = -(nΔo + f_kq)
        nΔi = _bte_occ_boson(ek - ekq, T);  f1i =  (nΔi + f_k);   f2i = -(nΔi + f_k)
    end

    pref = 2 * oftype(ek, π) * wtq * g2
    # A vanishing δ contributes nothing, so zero its term *before* multiplying: the Method 2–5
    # occupation prefactors diverge (→ Inf) as an `f`/`∂f` denominator → 0 for a state far from μ,
    # and a bare `0 * Inf` would be NaN. Guarding each product per-δ keeps the surviving
    # (nonzero-δ) term when only one δ underflows — the both-zero case already returned above.
    #
    # TODO(physics): this guard is PARTIAL — verify it covers what production needs. It only fixes
    # `δ==0 × (finite Inf factor)`. A NaN can still form *inside* a factor (`0 * Inf` before the δ
    # multiply, e.g. Method5 `nq*(nq+1)*df_kq/df_k` when `nq` or a `df` underflows), and a
    # tiny-but-nonzero δ × Inf factor yields Inf (not caught). Both are unphysical corners: `df → 0`
    # (state far from μ) correlates with `δ → 0` (energy mismatch), so a realistic transport window
    # should preclude them. Confirm against the dev_BTE reference before relying on it near edges.
    sₒ = ((iszero(δ1) ? z : δ1 * f1o) + (iszero(δ2) ? z : δ2 * f2o)) * pref
    sᵢ = ((iszero(δ2) ? z : δ2 * f1i) + (iszero(δ1) ? z : δ1 * f2i)) * pref
    return (sₒ, sᵢ)
end
