# Shared BTE scattering physics — ONE implementation used by both the CPU calculator loop
# (`GPUBoltzmannCalculator`'s `run_calculator!`) and the GPU device kernel
# (`bte_window_scatter!` in the CUDA extension). Keeping the physics in a single device-safe
# `@inline` function guarantees the CPU and GPU paths compute bit-identical scattering and that
# there is exactly one place where the Migdal–Eliashberg-style occupation factors live.
#
# The CPU reference this reproduces is `BoltzmannCalculatorX2X::run_calculator!`
# (broadening/dev_BTE.jl). Only the FermiDirac occupation + Gaussian smearing configuration is
# supported here (the configuration used for transport); the calculator asserts this at setup.

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
`Sₒ[n] += sₒ` over `(m, ν, q)` and `Sᵢ[n, f] += sᵢ` over `ν` (see `dev_BTE.jl`):

    sₒ = (δ₁·f₁ₒ + δ₂·f₂ₒ) · 2π·wtq·g2          # out-factors f1o,f2o per `method`
    sᵢ = (δ₂·f₁ᵢ + δ₁·f₂ᵢ) · 2π·wtq·g2          # in-factors  f1i,f2i per `method`

with `δ₁ = δ(ek−ekq+ωq)`, `δ₂ = δ(ek−ekq−ωq)` (Gaussian, width `η`). `method ∈ 1:6` selects the
occupation-factor convention (Method5 is the σ default). NOTE: the acoustic-phonon cutoff
(`ωq < omega_acoustic`) is NOT applied here — each caller skips sub-cutoff modes before calling
this (so a fresh caller must remember to). When both δ underflow to ~0 we return
`(0, 0)` *before* forming the factors — this guards the `0·Inf → NaN` that Methods 2–5 would
otherwise hit when an out-of-window state has a vanishing `f`/`∂f` denominator (matches the
`(δ1 < eps && δ2 < eps) && continue` skip in the CPU reference).
"""
@inline function bte_scattering_increments(method::Integer, ek, ekq, ωq, g2, wtq, μ, T, η)
    z = zero(ek)
    Δe1 = ek - ekq + ωq    # phonon absorption
    Δe2 = ek - ekq - ωq    # phonon emission
    δ1 = _bte_delta_gauss(Δe1, η)
    δ2 = _bte_delta_gauss(Δe2, η)
    (δ1 < eps(δ1) && δ2 < eps(δ2)) && return (z, z)

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

    pref = 2 * oftype(z, π) * wtq * g2
    sₒ = (δ1 * f1o + δ2 * f2o) * pref
    sᵢ = (δ2 * f1i + δ1 * f2i) * pref
    return (sₒ, sᵢ)
end
