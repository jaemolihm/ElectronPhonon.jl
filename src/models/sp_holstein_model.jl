# Simple-cubic s + p_x, p_y, p_z tight-binding model with one Einstein phonon and on-site
# orbital-density e-ph coupling.
#
# Shares the phonon and the on-site e-ph construction with `holstein_model` and
# `rashba_holstein_model` (the e-ph block is a copy of the latter's); it is a separate
# builder because the electron part (four orbitals on one atom, full O_h symmetry acting on the
# orbitals as D_S = blockdiag(1, Scart)) has nothing in common with the scalar Holstein band.

public sp_holstein_model

"""
    sp_holstein_model(; ε_s, ε_p, t_ss, V_sp, V_ppσ, V_ppπ, ω₀, g_s, g_p = g_s, mass = 1.0,
                        alat = 1.0, epmat_outer_momentum = "el", verbose = true) :: Model

Build a [`Model`](@ref) for a simple-cubic lattice with one atom at the origin carrying an `s`
and three `p` Wannier orbitals, with nearest-neighbour Slater-Koster hopping, coupled to one
Einstein phonon by orbital-density (Holstein) coupling.

All quantities are in **Rydberg atomic units**: energies in Ry, `alat` in Bohr.

# Hamiltonian

Wannier basis `(s, p_x, p_y, p_z)`, all at the origin. The hopping between the origin and the
neighbour at `R = ±aᵢ` is

```math
⟨s,0|H|s,R⟩ = t_{ss},\\quad ⟨p_j,0|H|p_j,R⟩ = V_{pp\\sigma}\\ (j = i)\\ \\text{or}\\ V_{pp\\pi}\\ (j ≠ i),\\quad
⟨s,0|H|p_i,R⟩ = R_i V_{sp} = -⟨p_i,0|H|s,R⟩,
```

with on-site energies `ε_s`, `ε_p`. With `kᵢ = k · aᵢ`,

```math
H_{ss} = ε_s + 2 t_{ss} ∑_i \\cos k_i,\\quad
H_{p_ip_i} = ε_p + 2 V_{pp\\sigma} \\cos k_i + 2 V_{pp\\pi} ∑_{j≠i} \\cos k_j,\\quad
H_{s p_i} = 2i V_{sp} \\sin k_i = H_{p_i s}^*.
```

The phonon and e-ph coupling are those of [`holstein_model`](@ref), with the scalar `g`
replaced by an on-site matrix in orbital space, an A1 (`O_h`-invariant) coupling:

```math
Ĥ_{e-ph} = ∑_R (b̂_R + b̂†_R) Ψ†_R \\,\\mathrm{diag}(g_s, g_p, g_p, g_p)\\, Ψ_R,   ω_q = ω₀
```

so `g_{mn}(k, q) = ⟨m, k+q| diag(g_s, g_p, g_p, g_p) |n, k⟩`. `ω₀ > 0`; `mass`, `alat`,
`epmat_outer_momentum` and `verbose` are as in [`holstein_model`](@ref).

# Representation inside the `Model`
`nw = 4`, `nmodes = 1`, `dimension = 3`. Spin is not included, so downstream calculations use
`spin_degeneracy = 2`. Position matrix elements are set to zero (including the O_h-allowed
on-site `⟨s|r|p⟩` dipole), so the velocity is `∂H/∂k`.

`model.symmetry` is the full `O_h × T` of the simple-cubic structure. `el_sym` is `nothing`; the
on-site operators `D_S = blockdiag(1, Scart)` are obtained with
[`fit_onsite_symmetry_operators`](@ref).
"""
function sp_holstein_model(;
        ε_s,
        ε_p,
        t_ss,
        V_sp,
        V_ppσ,
        V_ppπ,
        ω₀,
        g_s,
        g_p = g_s,
        mass :: Real = 1.0,
        alat :: Real = 1.0,
        epmat_outer_momentum :: String = "el",
        verbose :: Bool = true,
    )

    alat > 0 || throw(ArgumentError("alat must be positive, got $alat"))
    mass > 0 || throw(ArgumentError("mass must be positive, got $mass"))
    ω₀ > 0 || throw(ArgumentError("ω₀ must be positive, got $ω₀"))
    epmat_outer_momentum ∈ ("el", "ph") || throw(ArgumentError(
        "epmat_outer_momentum must be \"el\" or \"ph\", got $epmat_outer_momentum"))

    FT = Float64
    ε_s, ε_p, t_ss, V_sp, V_ppσ, V_ppπ = FT(ε_s), FT(ε_p), FT(t_ss), FT(V_sp), FT(V_ppσ), FT(V_ppπ)
    ω₀, g_s, g_p, mass, alat = FT(ω₀), FT(g_s), FT(g_p), FT(mass), FT(alat)

    if verbose
        println(@sprintf("sp Holstein model: simple cubic, alat = %g bohr", alat))
        println(@sprintf("  ε_s = %g Ry, ε_p = %g Ry, t_ss = %g Ry, V_sp = %g Ry, V_ppσ = %g Ry, V_ppπ = %g Ry",
                         ε_s, ε_p, t_ss, V_sp, V_ppσ, V_ppπ))
        println(@sprintf("  ω₀ = %g Ry, mass = %g, g_s = %g Ry, g_p = %g Ry", ω₀, mass, g_s, g_p))
    end

    nw = 4
    nmodes = 1

    # --- Crystal structure -------------------------------------------------------------
    lattice = Mat3{FT}(Diagonal(fill(alat, 3)))
    atom_pos = [zero(Vec3{FT})]  # alat units, Cartesian
    structure = Structure(alat, lattice, [mass], atom_pos, ["A"])
    symmetry = symmetry_operations(structure)

    # --- Real-space R vectors (sorted by reverse(R), as `WannierObject` requires) -------
    irvec_el = [Vec3(0, 0, 0)]
    for i in 1:3, s in (-1, 1)
        push!(irvec_el, Vec3(ntuple(j -> j == i ? s : 0, 3)))
    end
    sort!(irvec_el, by = reverse)
    irvec_ph = [zero(Vec3{Int})]
    irvec_ep_e = [zero(Vec3{Int})]
    irvec_ep_p = [zero(Vec3{Int})]

    # --- Electron Hamiltonian ----------------------------------------------------------
    ham = zeros(Complex{FT}, nw, nw, length(irvec_el))
    for (ir, R) in enumerate(irvec_el)
        if iszero(R)
            ham[1, 1, ir] = ε_s
            for j in 1:3
                ham[1 + j, 1 + j, ir] = ε_p
            end
        else
            i = findfirst(!iszero, R)
            ham[1, 1, ir] = t_ss
            for j in 1:3
                ham[1 + j, 1 + j, ir] = j == i ? V_ppσ : V_ppπ
            end
            # <s,0|H|p_i,R> = R_i V_sp and <p_i,0|H|s,R> = -R_i V_sp
            ham[1, 1 + i, ir] = R[i] * V_sp
            ham[1 + i, 1, ir] = -R[i] * V_sp
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
    # On-site vertex diag(g_s, g_p, g_p, g_p), scaled by √(2ω₀ M) as in `holstein_model`.
    vertex = Diagonal(Complex{FT}[g_s, g_p, g_p, g_p])
    epmat = reshape(Matrix(vertex) * sqrt(2 * ω₀ * mass), nw^2 * nmodes, 1)
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
