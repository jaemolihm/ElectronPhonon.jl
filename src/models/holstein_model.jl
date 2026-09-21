# Holstein model: a single-orbital tight-binding band coupled to a dispersionless
# (Einstein) phonon with a momentum-independent coupling constant.
#
# EPSpectral.jl/src/holstein.jl has a `HolsteinLatticeModel` with the same physics, which is not
# reused here: it lives in a different package (EPSpectral depends on nothing from here), works in
# Hartree units, is 1d-only, and carries its own `μ` and `T`. Most of all it is not an
# ElectronPhonon `Model` — it feeds EPSpectral's own spectral-function solver directly, whereas the
# point of this builder is to produce a `Model` that the Wannier interpolation and e-ph drivers
# consume like any EPW-loaded one.

using Printf

public holstein_model

"""
    holstein_model(; t, ω₀, g = nothing, λ = nothing, mass = 1.0, alat = 1.0, ε₀ = 0.0,
                   dimension = 3, epmat_outer_momentum = "el", verbose = true) :: Model

Build a [`Model`](@ref) for the Holstein model on a `dimension`-dimensional (hyper)cubic
lattice with one orbital and one Einstein phonon per unit cell.

All quantities are in **Rydberg atomic units**: energies (`t`, `ω₀`, `g`, `ε₀`) in Ry, `alat`
in Bohr. `λ` is dimensionless.

# Hamiltonian

```math
Ĥ = ε₀ ∑_R ĉ†_R ĉ_R
  - t ∑_{⟨R,R'⟩} ĉ†_R ĉ_{R'}
  + ω₀ ∑_R b̂†_R b̂_R
  + g ∑_R ĉ†_R ĉ_R (b̂_R + b̂†_R)
```

where `R` runs over the sites of a simple cubic (`dimension = 3`), square (`dimension = 2`)
or linear (`dimension = 1`) lattice with lattice constant `alat`, and `⟨R,R'⟩` over
nearest-neighbor pairs. The electron hopping is *only* between nearest neighbors, the phonon
is strictly local (dispersionless), and the e-ph coupling is on-site (density-displacement).

In the Bloch representation this gives

```math
ε_k = ε₀ - 2t ∑_{i=1}^{dimension} cos(k · a_i),   ω_q = ω₀,   g_{mn,ν}(k, q) = g
```

so `epstate.g2` equals `|g|²` at every `(k, q)`. `g` is the coupling to the *displacement*
``b̂ + b̂†``, which is the convention of `ElectronPhonon`'s `g_{mn,ν}(k,q)` — the quantity whose
square the package stores as `g2 = |ep|² / (2ω)`.

# Coupling strength: `g` or `λ`

Pass **exactly one** of `g` and `λ`; the other is derived from

```math
g = \\sqrt{2 · dimension · λ · ω₀ · |t|},   λ = \\frac{g²}{2 · dimension · ω₀ · |t|} = \\frac{g²}{ω₀ · W/2}
```

with `W = 4 · dimension · |t|` the bandwidth, so `λ` is the usual dimensionless Holstein
coupling measured against the half-bandwidth. Deriving `g` from `λ` needs `t ≠ 0`.

# Keyword arguments
- `t` : nearest-neighbor hopping. The bandwidth is `4·dimension·|t|`, centered on `ε₀`.
- `ω₀` : Einstein phonon frequency. Must be positive.
- `g` / `λ` : e-ph coupling, identical for all `k` and `q`. Give one, not both.
- `mass = 1.0` : mass of the Einstein oscillator. It cancels exactly from `ω_q` and from
  every e-ph matrix element (the model is fixed by `ω₀` and `g` alone), so it is only useful
  for checking that mass independence.
- `alat = 1.0` : lattice constant in Bohr.
- `ε₀ = 0.0` : on-site energy, i.e. the center of the band. There is no chemical potential;
  set the occupation or `μ` in the downstream calculation as for any other `Model`.
- `dimension = 3` : spatial dimension of the *hopping*, 1, 2 or 3, recorded as
  `model.dimension`. The `Model` always lives on a 3d lattice: for `dimension < 3` the
  directions without hopping get a lattice constant of `100 * alat`, and `model.symmetry`
  holds only the operations that do not mix them with the hopping directions. `model.volume`
  is therefore `100^(3-dimension) * alat^3`, so per-volume quantities
  (`ElectronOccupationParams`, transport `volume`) must be normalized by the user to a
  length/area if a `dimension < 3` density is wanted.
- `epmat_outer_momentum = "el"` : `"el"` or `"ph"`, as in `load_model_from_epw_new`.
- `verbose = true` : print the parameters, including both `g` and `λ`.

# Representation inside the `Model`
`nw = 1`, `nmodes = 1`: the Einstein oscillator is a single *scalar* (totally symmetric)
coordinate rather than three Cartesian displacements of one atom. This is what makes a
constant `g` compatible with the full spglib symmetry group — a coupling to a Cartesian
displacement of an atom on a high-symmetry site is odd under the point group and could never
be `k`- and `q`-independent. Consequently `model.mass` has length `nmodes = 1` while
`model.structure.mass` (which is per-atom × 3, as for a real crystal) has length 3; only
`model.mass` is used by the phonon solver.

Polar (long-range) terms are absent, position matrix elements vanish (a single Wannier
function at the origin), and `el_sym` is `nothing`, matching `load_model_from_epw_new`.
"""
function holstein_model(;
        t,
        ω₀,
        g = nothing,
        λ = nothing,
        mass :: Real = 1.0,
        alat :: Real = 1.0,
        ε₀ :: Real = 0.0,
        dimension :: Integer = 3,
        epmat_outer_momentum :: String = "el",
        verbose :: Bool = true,
    )

    dimension ∈ (1, 2, 3) || throw(ArgumentError("dimension must be 1, 2 or 3, got $dimension"))
    alat > 0 || throw(ArgumentError("alat must be positive, got $alat"))
    mass > 0 || throw(ArgumentError("mass must be positive, got $mass"))
    ω₀ > 0 || throw(ArgumentError("ω₀ must be positive, got $ω₀"))
    epmat_outer_momentum ∈ ("el", "ph") || throw(ArgumentError(
        "epmat_outer_momentum must be \"el\" or \"ph\", got $epmat_outer_momentum"))
    (g === nothing) == (λ === nothing) && throw(ArgumentError(
        "pass exactly one of `g` and `λ`, got g = $g and λ = $λ"))

    FT = Float64
    t, ω₀, mass, alat, ε₀ = FT(t), FT(ω₀), FT(mass), FT(alat), FT(ε₀)

    # λ is measured against the half-bandwidth 2·dimension·|t|.
    half_bandwidth = 2 * dimension * abs(t)
    if g === nothing
        half_bandwidth > 0 || throw(ArgumentError("deriving g from λ needs t ≠ 0, got t = $t"))
        λ >= 0 || throw(ArgumentError("λ must be non-negative, got $λ"))
        λ = FT(λ)
        g = sqrt(λ * ω₀ * half_bandwidth)
    else
        g = FT(g)
        # A flat band (t = 0) has no bandwidth to measure the coupling against, so λ is infinite.
        # That is reported rather than rejected: g alone fixes the model.
        λ = g^2 / (ω₀ * half_bandwidth)
    end

    if verbose
        println(@sprintf("Holstein model: dimension = %d, alat = %g bohr", dimension, alat))
        println(@sprintf("  t = %g Ry, ε₀ = %g Ry, ω₀ = %g Ry, mass = %g", t, ε₀, ω₀, mass))
        println(@sprintf("  g = %g Ry, λ = %g", g, λ))
    end

    nw = 1
    nmodes = 1

    # --- Crystal structure -------------------------------------------------------------
    # Directions without hopping are stretched so that the cell is an isolated wire
    # (dimension=1) or sheet (dimension=2) and so that spglib returns the symmetry group of
    # the `dimension`-dimensional Hamiltonian instead of the cubic one.
    vacuum = 100 * alat
    lattice = Mat3{FT}(Diagonal(Vec3(
        alat,
        dimension >= 2 ? alat : vacuum,
        dimension >= 3 ? alat : vacuum,
    )))

    atom_pos = [zero(Vec3{FT})]  # alat units, Cartesian
    structure = Structure(alat, lattice, [mass], atom_pos, ["A"]; dimension)

    # --- Real-space R vectors ----------------------------------------------------------
    # Electrons: on-site plus the 2*dimension nearest neighbors, sorted by reverse(R) as
    # `WannierObject` requires. Phonon and e-ph: on-site only.
    irvec_el = if dimension == 1
        [Vec3(-1, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0)]
    elseif dimension == 2
        [Vec3(0, -1, 0), Vec3(-1, 0, 0), Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)]
    else
        [Vec3(0, 0, -1), Vec3(0, -1, 0), Vec3(-1, 0, 0), Vec3(0, 0, 0),
         Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
    end
    irvec_ph = [zero(Vec3{Int})]
    irvec_ep_e = [zero(Vec3{Int})]  # electron R of the e-ph matrix
    irvec_ep_p = [zero(Vec3{Int})]  # phonon R of the e-ph matrix

    # --- Electron Hamiltonian ----------------------------------------------------------
    # H(R=0) = ε₀, H(R = ±a_i) = -t  ⟹  ε_k = ε₀ - 2t ∑_i cos(k · a_i)
    ham = zeros(Complex{FT}, nw, nw, length(irvec_el))
    for (ir, R) in enumerate(irvec_el)
        ham[1, 1, ir] = iszero(R) ? ε₀ : -t
    end
    el_ham = WannierObject(irvec_el, reshape(ham, nw^2, :))
    el_ham_R = wannier_object_multiply_R(el_ham, lattice)

    # A single Wannier function at the origin: ⟨0|r|R⟩ = 0 for every R.
    el_pos = WannierObject(irvec_el, zeros(Complex{FT}, nw^2 * 3, length(irvec_el)))

    # The velocity matrix is dH/dk, which is exactly what `el_ham_R` interpolates. It is
    # copied rather than shared so that `update_op_r!` on one does not alias the other.
    el_vel = WannierObject(irvec_el, copy(el_ham_R.op_r))
    el_velocity_mode = :Direct

    # --- Phonon dynamical matrix -------------------------------------------------------
    # `get_ph_eigen!` divides by √(mass_i mass_j), so D(R=0) = M ω₀² gives ω_q = ω₀ for all q
    # and a phonon eigenvector u = 1/√M.
    dyn = fill(Complex{FT}(mass * ω₀^2), nmodes^2, length(irvec_ph))
    ph_dyn = WannierObject(irvec_ph, dyn)
    ph_dyn_R = wannier_object_multiply_R(ph_dyn, lattice)

    # --- Electron-phonon coupling ------------------------------------------------------
    # `epmat` is in the Cartesian-displacement basis: the loop builds
    # ep = epmat * u = epmat / √M, and g2 = |ep|² / (2ω₀). Storing g √(2ω₀ M) therefore
    # gives g2 = |g|² independent of k, q and M.
    # Only (Rₑ, Rₚ) = (0, 0) is nonzero, which makes g(k, q) a constant.
    # The coupling is on-site in both R indices, so a single R vector each; the e-ph R grids are
    # deliberately NOT the electron hopping grid `irvec_el`, which would carry 2·dimension columns
    # of exact zeros and suggest the coupling has range.
    epmat = fill(Complex{FT}(g * sqrt(2 * ω₀ * mass)), nw^2 * nmodes, 1)
    ep = if epmat_outer_momentum == "el"
        # op_r indexed as (iw, jw, imode, Rₚ, Rₑ)
        WannierObject(irvec_ep_e, epmat; irvec_next = irvec_ep_p)
    else
        # op_r indexed as (iw, jw, imode, Rₑ, Rₚ)
        WannierObject(irvec_ep_p, epmat; irvec_next = irvec_ep_e)
    end

    Model(; structure.alat, structure.lattice, structure.recip_lattice, structure.volume,
        dimension, nw, nmodes,
        wann_centers = [zero(Vec3{FT})],
        mass = [mass],
        structure.atom_pos, structure.atom_labels, structure, structure.symmetry,
        use_polar_dipole = false, polar_phonon = Polar(nothing), polar_eph = Polar(nothing),
        el_ham, el_ham_R, el_pos, el_vel, el_velocity_mode,
        ph_dyn, ph_dyn_R,
        epmat = ep, epmat_outer_momentum,
        el_sym = nothing,
    )
end
