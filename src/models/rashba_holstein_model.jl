# Two-orbital spinful Holstein model on a square lattice with intrinsic (inversion-symmetric)
# and Rashba (inversion-breaking) spin-orbit coupling.
#
# Shares only the phonon and the on-site e-ph construction with `holstein_model`; it is a
# separate builder because the electron part (4×4 spinor Hamiltonian, symmetry group of the
# Hamiltonian smaller than the lattice's) has nothing in common with the scalar Holstein band.

public rashba_holstein_model

"""
    rashba_holstein_model(; t, M, λ_soc, α_R, ω₀, g₀, g₁ = 0.0, mass = 1.0, alat = 1.0,
                              ε₀ = 0.0, epmat_outer_momentum = "el", verbose = true) :: Model

Build a [`Model`](@ref) for a two-orbital, spin-1/2 tight-binding model on a square lattice
with spin-orbit coupling, coupled to one Einstein phonon by orbital-density (Holstein)
coupling. The phonon is spin-independent, so spin flips come only from the spin mixing of the
Bloch states (Elliott-Yafet), and the Rashba term adds a k-dependent spin splitting
(D'yakonov-Perel').

All quantities are in **Rydberg atomic units**: energies in Ry, `alat` in Bohr.

# Hamiltonian

Wannier basis `(1↑, 1↓, 2↑, 2↓)`, i.e. `τ ⊗ σ` with `τ` the orbital and `σ` the spin Pauli
matrices. With `kᵢ = k · aᵢ`,

```math
H(k) = ε(k) 𝟙 + M τ_z + λ_{soc} τ_x (\\sin k_x σ_y - \\sin k_y σ_x)
     + α_R (\\sin k_y σ_x - \\sin k_x σ_y),   ε(k) = ε₀ - 2t (\\cos k_x + \\cos k_y)
```

- `λ_soc`: intrinsic SOC, even under `P = τ_z` (e.g. s and p_z orbitals) and `T = -iσ_y K`;
  mixes spins between the orbitals and keeps every band doubly degenerate.
- `α_R`: Rashba SOC, breaks `P` and lifts the Kramers degeneracy.

The phonon and e-ph coupling are those of [`holstein_model`](@ref), with the scalar `g`
replaced by an on-site matrix in orbital space:

```math
Ĥ_{e-ph} = ∑_R (b̂_R + b̂†_R) Ψ†_R (g₀ 𝟙 + g₁ τ_z) Ψ_R,   ω_q = ω₀
```

so `g_{mn}(k, q) = ⟨m, k+q| g₀ 𝟙 + g₁ τ_z |n, k⟩`, with `g₁` half the difference of the two
orbitals' couplings. `ω₀ > 0`; `mass`, `alat`, `ε₀`, `epmat_outer_momentum` and `verbose` are
as in [`holstein_model`](@ref).

# Representation inside the `Model`
`nw = 4`, `nmodes = 1`, `dimension = 2`. Spin is part of the Wannier index, so downstream
calculations must use `spin_degeneracy = 1`. All four Wannier functions sit at the origin, so
position matrix elements vanish and the velocity is exactly `∂H/∂k`.

`model.symmetry` is the symmetry group of the Hamiltonian, not that of the lattice:
`symmetry_operations(model.structure)` is spglib's D4h × T for the padded square cell, and for
`α_R ≠ 0` `model.symmetry` keeps only its `C4v × T` subgroup (operations that preserve `z`).
Both orbitals are taken even under `C4v`. `el_sym` is `nothing`: the operations map k-points
and band energies, but Bloch states are never rotated as spinors, so spin-resolved quantities
must be computed on the full Brillouin zone.
"""
function rashba_holstein_model(;
        t,
        M,
        λ_soc,
        α_R,
        ω₀,
        g₀,
        g₁ = 0.0,
        mass :: Real = 1.0,
        alat :: Real = 1.0,
        ε₀ :: Real = 0.0,
        epmat_outer_momentum :: String = "el",
        verbose :: Bool = true,
    )

    alat > 0 || throw(ArgumentError("alat must be positive, got $alat"))
    mass > 0 || throw(ArgumentError("mass must be positive, got $mass"))
    ω₀ > 0 || throw(ArgumentError("ω₀ must be positive, got $ω₀"))
    epmat_outer_momentum ∈ ("el", "ph") || throw(ArgumentError(
        "epmat_outer_momentum must be \"el\" or \"ph\", got $epmat_outer_momentum"))

    FT = Float64
    t, M, λ_soc, α_R, ω₀, g₀, g₁ = FT(t), FT(M), FT(λ_soc), FT(α_R), FT(ω₀), FT(g₀), FT(g₁)
    mass, alat, ε₀ = FT(mass), FT(alat), FT(ε₀)

    if verbose
        println(@sprintf("Rashba Holstein model: square lattice, alat = %g bohr", alat))
        println(@sprintf("  t = %g Ry, ε₀ = %g Ry, M = %g Ry, λ_soc = %g Ry, α_R = %g Ry",
                         t, ε₀, M, λ_soc, α_R))
        println(@sprintf("  ω₀ = %g Ry, mass = %g, g₀ = %g Ry, g₁ = %g Ry", ω₀, mass, g₀, g₁))
    end

    dimension = 2
    nw = 4
    nmodes = 1

    # --- Crystal structure -------------------------------------------------------------
    vacuum = 100 * alat
    lattice = Mat3{FT}(Diagonal(Vec3(alat, alat, vacuum)))
    atom_pos = [zero(Vec3{FT})]  # alat units, Cartesian
    structure = Structure(alat, lattice, [mass], atom_pos, ["A"]; dimension)

    # Rashba breaks z → -z, which leaves the z-preserving C4v subgroup of D4h.
    lattice_symmetry = symmetry_operations(structure)
    symmetry = if iszero(α_R)
        lattice_symmetry
    else
        inds = [isym for (isym, symop) in enumerate(lattice_symmetry) if symop.Scart[3, 3] > 0]
        Symmetry(length(inds), lattice_symmetry.S[inds], lattice_symmetry.τ[inds],
            lattice_symmetry.Scart[inds], lattice_symmetry.τcart[inds],
            lattice_symmetry.is_inv[inds], lattice_symmetry.is_tr[inds])
    end

    # --- Real-space R vectors (sorted by reverse(R), as `WannierObject` requires) -------
    irvec_el = [Vec3(0, -1, 0), Vec3(-1, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    irvec_ph = [zero(Vec3{Int})]
    irvec_ep_e = [zero(Vec3{Int})]
    irvec_ep_p = [zero(Vec3{Int})]

    # --- Electron Hamiltonian ----------------------------------------------------------
    σ₀ = Matrix{Complex{FT}}(I, 2, 2)
    σx = Complex{FT}[0 1; 1 0]
    σy = Complex{FT}[0 -im; im 0]
    σz = Complex{FT}[1 0; 0 -1]
    τ₀, τx, τz = σ₀, σx, σz
    𝟙 = kron(τ₀, σ₀)

    # sin(k·a) = (e^{ik·a} - e^{-ik·a}) / 2i, so the matrix X multiplying sin(k·a) sits on
    # R = ±a with weight ±X/2i.
    sin_kx_coeff = λ_soc * kron(τx, σy) - α_R * kron(τ₀, σy)
    sin_ky_coeff = α_R * kron(τ₀, σx) - λ_soc * kron(τx, σx)
    ham = zeros(Complex{FT}, nw, nw, length(irvec_el))
    for (ir, R) in enumerate(irvec_el)
        ham[:, :, ir] .= if iszero(R)
            ε₀ * 𝟙 + M * kron(τz, σ₀)
        elseif R[2] == 0
            -t * 𝟙 + R[1] * sin_kx_coeff / 2im
        else
            -t * 𝟙 + R[2] * sin_ky_coeff / 2im
        end
    end
    el_ham = WannierObject(irvec_el, reshape(ham, nw^2, :))
    el_ham_R = wannier_object_multiply_R(el_ham, lattice)
    el_pos = WannierObject(irvec_el, zeros(Complex{FT}, nw^2 * 3, length(irvec_el)))
    el_vel = WannierObject(irvec_el, copy(el_ham_R.op_r))
    el_velocity_mode = :Direct

    # --- Phonon dynamical matrix (as in `holstein_model`) ------------------------------
    ph_dyn = WannierObject(irvec_ph, fill(Complex{FT}(mass * ω₀^2), nmodes^2, length(irvec_ph)))
    ph_dyn_R = wannier_object_multiply_R(ph_dyn, lattice)

    # --- Electron-phonon coupling ------------------------------------------------------
    # On-site vertex g₀ 𝟙 + g₁ τ_z, scaled by √(2ω₀ M) as in `holstein_model` so that
    # ep = u_{k+q}† (g₀ 𝟙 + g₁ τ_z) u_k.
    vertex = g₀ * 𝟙 + g₁ * kron(τz, σ₀)
    epmat = reshape(vertex * sqrt(2 * ω₀ * mass), nw^2 * nmodes, 1)
    ep = if epmat_outer_momentum == "el"
        WannierObject(irvec_ep_e, epmat; irvec_next = irvec_ep_p)
    else
        WannierObject(irvec_ep_p, epmat; irvec_next = irvec_ep_e)
    end

    Model(; structure, symmetry, nw, nmodes,
        wann_centers = fill(zero(Vec3{FT}), nw),
        mass = [mass],
        use_polar_dipole = false, polar_phonon = Polar(nothing), polar_eph = Polar(nothing),
        el_ham, el_ham_R, el_pos, el_vel, el_velocity_mode,
        ph_dyn, ph_dyn_R,
        epmat = ep, epmat_outer_momentum,
        el_sym = nothing,
    )
end
