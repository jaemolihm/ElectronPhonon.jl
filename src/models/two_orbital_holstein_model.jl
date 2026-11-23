# Two-orbital (s, d_{x2-y2}) spinless Holstein-type model on a square lattice with a B1g
# bond-modulated e-ph vertex.
#
# Shares the phonon and the on-site e-ph construction with `holstein_model`; it is a separate
# builder because the electron part (a 2x2 Hamiltonian with orbital mixing) and the k- and
# q-dependent vertex have nothing in common with the scalar Holstein band. It is the minimal
# model with an interband (band-off-diagonal) e-ph vertex that is nontrivial in k and q, and it
# fills `Model.el_sym` with the Wannier representation of the point group.

using Printf

public two_orbital_holstein_model

"""
    two_orbital_holstein_model(; t, t_x, M, ω₀, g₀, ω₁ = 0.0, g_z = 0.0, g_x = 0.0,
                               mass = 1.0, alat = 1.0, ε₀ = 0.0, epmat_outer_momentum = "el",
                               verbose = true) :: Model

Build a [`Model`](@ref) for a spinless two-orbital tight-binding model on a square lattice,
coupled to one phonon per unit cell by an orbital-density coupling plus a B1g bond coupling.

All quantities are in **Rydberg atomic units**: energies in Ry, `alat` in Bohr.

# Hamiltonian

Wannier basis `(s, d_{x²-y²})` at the origin, `τ` the orbital Pauli matrices, `kᵢ = k · aᵢ`:

```math
H(k) = ε(k) 𝟙 + M τ_z + t_x γ(k) τ_x,
ε(k) = ε₀ - 2t (\\cos k_x + \\cos k_y),   γ(k) = 2 (\\cos k_x - \\cos k_y)
```

`γ` is B1g, so `V(Ck) = -V(k)` under a C4 rotation and the band energies
`ε ± √(M² + t_x² γ²)` are D4h-invariant. The band separation is `2√(M² + t_x² γ²)`, with
minimum `2|M|` on the diagonals `|k_x| = |k_y|`: `M` tunes the near-degeneracy and `t_x` the
hybridization.

# Phonon

One scalar (A1g) coordinate per cell with `ω_q² = ω₀² + ω₁² (2 - cos q_x - cos q_y)`. The
default `ω₁ = 0` is an Einstein phonon.

# Electron-phonon coupling

Rows are the orbitals at `k+q` and columns the orbitals at `k`. With `ω₁ = 0`

```math
g(k, q) = g₀ 𝟙 + g_z τ_z + g_x \\begin{pmatrix} 0 & γ(k) \\\\ γ(k+q) & 0 \\end{pmatrix}
```

is the Wannier-gauge vertex `ep / √(2ω)` that `epstate.g2` squares band by band. The `g_x`
term comes from the displacement modulating the `s(R)`-`d(R+δ)` bond amplitude with sign `+` on
`±a₁` and `-` on `±a₂`; it satisfies `g(k+q, -q)† = g(k, q)` and is real. The couplings are
constants of the Cartesian displacement coupling, stored as `g √(2 ω₀ mass)` as in
[`holstein_model`](@ref). For `ω₁ ≠ 0` the Wannier-gauge vertex is therefore the matrix above
times `√(ω₀ / ω_q)`.

`t_x = g_x = g_z = 0` leaves bands equal to orbitals (`ε(k) ± M`) and a diagonal vertex.

# Keyword arguments
- `t`, `ε₀`, `M`, `t_x` : hopping, band center, orbital splitting and B1g hybridization.
- `ω₀`, `ω₁` : phonon frequency at `q = 0` and nearest-neighbor dispersion. `ω₀` must be positive.
- `g₀`, `g_z`, `g_x` : on-site density, on-site orbital-difference and B1g bond couplings.
- `mass`, `alat`, `epmat_outer_momentum`, `verbose` : as in [`holstein_model`](@ref).

# Representation inside the `Model`
`nw = 2`, `nmodes = 1`, `dimension = 2`. The basis is spinless (spin is implicit). Position matrix elements vanish and the
velocity is exactly `∂H/∂k`.

`model.symmetry` is spglib's D4h × T for the padded square cell. `model.el_sym` holds the
unitary operations D4h (16) with the k-independent representation
`D_W(S) = diag(1, χ_B1g(S))`, where `χ_B1g = -1` for operations exchanging `x` and `y` (C4,
σ_d, C2″) and `+1` otherwise, so that `D_W(S) H(k) D_W(S)† = H(Sk)`. Antiunitary operations
are not included.
"""
function two_orbital_holstein_model(;
        t,
        t_x,
        M,
        ω₀,
        g₀,
        ω₁ = 0.0,
        g_z = 0.0,
        g_x = 0.0,
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
    t, t_x, M, ω₀, ω₁, g₀, g_z, g_x = FT.((t, t_x, M, ω₀, ω₁, g₀, g_z, g_x))
    mass, alat, ε₀ = FT(mass), FT(alat), FT(ε₀)

    if verbose
        println(@sprintf("Two-orbital Holstein model: alat = %g bohr", alat))
        println(@sprintf("  t = %g Ry, t_x = %g Ry, M = %g Ry, ε₀ = %g Ry", t, t_x, M, ε₀))
        println(@sprintf("  ω₀ = %g Ry, ω₁ = %g Ry, mass = %g", ω₀, ω₁, mass))
        println(@sprintf("  g₀ = %g Ry, g_z = %g Ry, g_x = %g Ry", g₀, g_z, g_x))
    end

    nw = 2
    nmodes = 1

    # --- Crystal structure -------------------------------------------------------------
    # A sheet: the third direction is stretched so that spglib returns the 2D group D4h × T.
    vacuum = 100 * alat
    lattice = Mat3{FT}(Diagonal(Vec3(alat, alat, vacuum)))
    atom_pos = [zero(Vec3{FT})]  # alat units, Cartesian
    structure = Structure(alat, lattice, [mass], atom_pos, ["A"]; dimension = 2)
    symmetry = symmetry_operations(structure)

    # --- Real-space R vectors (sorted by reverse(R), as `WannierObject` requires) -------
    irvec_origin = [zero(Vec3{Int})]
    irvec_nn = [Vec3(0, -1, 0), Vec3(-1, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    irvec_el = irvec_nn
    irvec_ph = iszero(ω₁) ? irvec_origin : irvec_nn
    irvec_ep_e = iszero(g_x) ? irvec_origin : irvec_nn  # electron R of the e-ph matrix
    irvec_ep_p = iszero(g_x) ? irvec_origin : irvec_nn  # phonon R of the e-ph matrix

    # Sign of the B1g form factor on the bond R: + along x, - along y.
    b1g_sign(R) = R[1] != 0 ? 1 : -1

    # --- Electron Hamiltonian ----------------------------------------------------------
    # H(R=0) = ε₀ 𝟙 + M τ_z, H(R = ±a_i) = -t 𝟙 ± t_x τ_x (+ for x, - for y), so that
    # H(k) = ∑_R e^{ik·R} H(R) is the Hamiltonian of the docstring.
    τx = Complex{FT}[0 1; 1 0]
    τz = Complex{FT}[1 0; 0 -1]
    𝟙 = Matrix{Complex{FT}}(I, nw, nw)
    ham = zeros(Complex{FT}, nw, nw, length(irvec_el))
    for (ir, R) in enumerate(irvec_el)
        ham[:, :, ir] .= iszero(R) ? ε₀ * 𝟙 + M * τz : -t * 𝟙 + b1g_sign(R) * t_x * τx
    end
    el_ham = WannierObject(irvec_el, reshape(ham, nw^2, :))
    el_ham_R = wannier_object_multiply_R(el_ham, lattice)
    el_pos = WannierObject(irvec_el, zeros(Complex{FT}, nw^2 * 3, length(irvec_el)))
    el_vel = WannierObject(irvec_el, copy(el_ham_R.op_r))
    el_velocity_mode = :Direct

    # --- Phonon dynamical matrix -------------------------------------------------------
    # D(R=0) = M (ω₀² + 2ω₁²), D(R = ±a_i) = -M ω₁²/2  ⟹  ω_q² = ω₀² + ω₁² (2 - cos q_x - cos q_y)
    dyn = zeros(Complex{FT}, nmodes^2, length(irvec_ph))
    for (ir, R) in enumerate(irvec_ph)
        dyn[1, ir] = iszero(R) ? mass * (ω₀^2 + 2 * ω₁^2) : -mass * ω₁^2 / 2
    end
    ph_dyn = WannierObject(irvec_ph, dyn)
    ph_dyn_R = wannier_object_multiply_R(ph_dyn, lattice)

    # --- Electron-phonon coupling ------------------------------------------------------
    # ep_ij(Rₑ, Rₚ), i the orbital at k+q and j the orbital at k, with
    # ep(k, q) = ∑ e^{i(k·Rₑ + q·Rₚ)} ep(Rₑ, Rₚ). The B1g term g_x γ(k) sits at (1, 2) on
    # Rₑ = δ, Rₚ = 0 and g_x γ(k+q) at (2, 1) on Rₑ = Rₚ = δ, δ ∈ {±a₁, ±a₂}. The scale
    # √(2 ω₀ mass) is that of `holstein_model`.
    scale = sqrt(2 * ω₀ * mass)
    function ep_matrix(Re, Rp)
        m = zeros(Complex{FT}, nw, nw)
        if iszero(Re) && iszero(Rp)
            m .= g₀ * 𝟙 + g_z * τz
        elseif iszero(Rp)
            m[1, 2] = b1g_sign(Re) * g_x
        elseif Re == Rp
            m[2, 1] = b1g_sign(Re) * g_x
        end
        scale * m
    end
    ep = if epmat_outer_momentum == "el"
        # op_r indexed as (i, j, imode, Rₚ, Rₑ)
        epmat = zeros(Complex{FT}, nw^2 * nmodes * length(irvec_ep_p), length(irvec_ep_e))
        for (ie, Re) in enumerate(irvec_ep_e), (ip, Rp) in enumerate(irvec_ep_p)
            epmat[(1:nw^2) .+ nw^2 * nmodes * (ip - 1), ie] .= vec(ep_matrix(Re, Rp))
        end
        WannierObject(irvec_ep_e, epmat; irvec_next = irvec_ep_p)
    else
        # op_r indexed as (i, j, imode, Rₑ, Rₚ)
        epmat = zeros(Complex{FT}, nw^2 * nmodes * length(irvec_ep_e), length(irvec_ep_p))
        for (ie, Re) in enumerate(irvec_ep_e), (ip, Rp) in enumerate(irvec_ep_p)
            epmat[(1:nw^2) .+ nw^2 * nmodes * (ie - 1), ip] .= vec(ep_matrix(Re, Rp))
        end
        WannierObject(irvec_ep_p, epmat; irvec_next = irvec_ep_e)
    end

    # --- Wannier representation of the unitary point-group operations ------------------
    # D_W(S) = diag(1, χ_B1g(S)) at R = 0 (both orbitals sit at the origin), with
    # χ_B1g = -1 iff S exchanges x and y.
    inds = [isym for isym in 1:symmetry.nsym if !symmetry.is_tr[isym]]
    symmetry_unitary = Symmetry(length(inds), symmetry.S[inds], symmetry.τ[inds],
        symmetry.Scart[inds], symmetry.τcart[inds], symmetry.is_inv[inds], symmetry.is_tr[inds])
    sym_operators = map(inds) do isym
        χ_B1g = iszero(symmetry.S[isym][1, 2]) ? 1 : -1
        WannierObject(irvec_origin, reshape(Complex{FT}[1, 0, 0, χ_B1g], nw^2, 1))
    end
    el_sym = SymmetryOperators(symmetry_unitary, sym_operators)

    Model(; structure, symmetry, nw, nmodes,
        wann_centers = fill(zero(Vec3{FT}), nw),
        mass = [mass],
        use_polar_dipole = false, polar_phonon = Polar(nothing), polar_eph = Polar(nothing),
        el_ham, el_ham_R, el_pos, el_vel, el_velocity_mode,
        ph_dyn, ph_dyn_R,
        epmat = ep, epmat_outer_momentum,
        el_sym,
    )
end
