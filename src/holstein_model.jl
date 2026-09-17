# Holstein model: a single-orbital tight-binding band coupled to a dispersionless
# (Einstein) phonon with a momentum-independent coupling constant.

public holstein_model

# Atomic mass used for the Einstein oscillator, in Rydberg atomic units (1 amu = 911.444).
# The value is arbitrary: `Model` stores the e-ph matrix in the Cartesian displacement basis,
# where the mass enters as epmat ∝ √M and the phonon eigenvector as 1/√M, so M cancels
# exactly from `ω_q` and from every e-ph matrix element. It is fixed here (rather than
# exposed) so that no parameter of `holstein_model` is physically inert.
const HOLSTEIN_MASS = 911.444

"""
    holstein_model(; t, ω₀, g, alat = 1.0, ε₀ = 0.0, dim = 3,
                   epmat_outer_momentum = "el") :: Model

Build a [`Model`](@ref) for the Holstein model on a `dim`-dimensional (hyper)cubic lattice
with one orbital and one Einstein phonon per unit cell.

All quantities are in **Rydberg atomic units**: energies (`t`, `ω₀`, `g`, `ε₀`) in Ry, `alat`
in Bohr.

# Hamiltonian

```math
Ĥ = ε₀ ∑_R ĉ†_R ĉ_R
  - t ∑_{⟨R,R'⟩} ĉ†_R ĉ_{R'}
  + ω₀ ∑_R b̂†_R b̂_R
  + g ∑_R ĉ†_R ĉ_R (b̂_R + b̂†_R)
```

where `R` runs over the sites of a simple cubic (`dim = 3`), square (`dim = 2`) or linear
(`dim = 1`) lattice with lattice constant `alat`, and `⟨R,R'⟩` over nearest-neighbor pairs.
The electron hopping is *only* between nearest neighbors, the phonon is strictly local
(dispersionless), and the e-ph coupling is on-site (density-displacement).

In the Bloch representation this gives

```math
ε_k = ε₀ - 2t ∑_{i=1}^{dim} cos(k · a_i),   ω_q = ω₀,   g_{mn,ν}(k, q) = g
```

so `epstate.g2` equals `|g|²` at every `(k, q)`. `g` is the coupling to the *displacement*
``b̂ + b̂†``, which is the convention of `ElectronPhonon`'s `g_{mn,ν}(k,q)` — the quantity whose
square the package stores as `g2 = |ep|² / (2ω)`.

# Keyword arguments
- `t` : nearest-neighbor hopping. The bandwidth is `4·dim·t`, centered on `ε₀`.
- `ω₀` : Einstein phonon frequency. Must be positive.
- `g` : e-ph coupling constant, identical for all `k` and `q`.
- `alat = 1.0` : lattice constant in Bohr.
- `ε₀ = 0.0` : on-site energy, i.e. the center of the band. There is no chemical potential;
  set the occupation or `μ` in the downstream calculation as for any other `Model`.
- `dim = 3` : spatial dimension of the *hopping*, 1, 2 or 3. The `Model` always lives on a
  3d lattice; for `dim < 3` the unconnected directions are given a lattice constant
  `100 * alat`, which both makes the cell an isolated wire/sheet and lowers the spglib
  symmetry group to one that the `dim`-dimensional Hamiltonian actually obeys. (A cubic cell
  would make spglib report operations mixing `x` and `z`, which is *not* a symmetry of e.g.
  the square-lattice band, and symmetry-reduced calculations would then be wrong.)
  `model.dim` records the choice; `model.volume` is `100^(3-dim) * alat^3`, so per-volume
  quantities (`ElectronOccupationParams`, transport `volume`) must be normalized by the user
  to a length/area if a `dim < 3` density is wanted.
- `epmat_outer_momentum = "el"` : `"el"` or `"ph"`, as in `load_model_from_epw_new`.

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
        g,
        alat :: Real = 1.0,
        ε₀ :: Real = 0.0,
        dim :: Integer = 3,
        epmat_outer_momentum :: String = "el",
    )

    dim ∈ (1, 2, 3) || throw(ArgumentError("dim must be 1, 2 or 3, got $dim"))
    alat > 0 || throw(ArgumentError("alat must be positive, got $alat"))
    ω₀ > 0 || throw(ArgumentError("ω₀ must be positive, got $ω₀"))
    epmat_outer_momentum ∈ ("el", "ph") || throw(ArgumentError(
        "epmat_outer_momentum must be \"el\" or \"ph\", got $epmat_outer_momentum"))

    FT = Float64
    t, ω₀, g, alat, ε₀ = FT(t), FT(ω₀), FT(g), FT(alat), FT(ε₀)
    nw = 1
    nmodes = 1
    mass = HOLSTEIN_MASS

    # --- Crystal structure -------------------------------------------------------------
    # Directions without hopping are stretched so that the cell is an isolated wire (dim=1)
    # or sheet (dim=2) and so that spglib returns the symmetry group of the `dim`-dimensional
    # Hamiltonian instead of the cubic one.
    vacuum = FT(100)
    lattice_diag = Vec3{FT}(ntuple(i -> i <= dim ? alat : vacuum * alat, 3))
    lattice = Mat3{FT}(Diagonal(lattice_diag))

    atom_pos = [zero(Vec3{FT})]  # alat units, Cartesian
    structure = Structure(alat, lattice, [mass], atom_pos, ["A"])

    # --- Real-space R vectors ----------------------------------------------------------
    # Electrons: on-site plus the 2*dim nearest neighbors. Phonon and e-ph: on-site only.
    irvec_el = [Vec3{Int}(ntuple(j -> j == i ? s : 0, 3)) for i in 1:dim for s in (-1, 1)]
    push!(irvec_el, zero(Vec3{Int}))
    sort!(irvec_el, by = x -> reverse(x))
    ir0_el = findfirst(isequal(zero(Vec3{Int})), irvec_el) :: Int

    irvec_ph = [zero(Vec3{Int})]
    irvec_ep = [zero(Vec3{Int})]

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

    # With vanishing position matrix elements the Berry-connection velocity reduces to
    # dH/dk, which is exactly what `el_ham_R` interpolates. `el_vel` is a copy of it (not the
    # same object, to avoid aliasing on `update_op_r!`) so that `el_velocity_mode = :Direct`
    # is also available and gives identical results.
    el_vel = WannierObject(irvec_el, copy(el_ham_R.op_r))
    el_velocity_mode = :BerryConnection

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
    epmat = zeros(Complex{FT}, nw, nw, nmodes, length(irvec_el), length(irvec_ep))
    epmat[1, 1, 1, ir0_el, 1] = g * sqrt(2 * ω₀ * mass)

    if epmat_outer_momentum == "el"
        # op_r indexed as (iw, jw, imode, Rₚ, Rₑ)
        data = reshape(permutedims(epmat, (1, 2, 3, 5, 4)),
                       nw^2 * nmodes * length(irvec_ep), length(irvec_el))
        ep = WannierObject(irvec_el, data; irvec_next = irvec_ep)
    else
        # op_r indexed as (iw, jw, imode, Rₑ, Rₚ)
        data = reshape(epmat, nw^2 * nmodes * length(irvec_el), length(irvec_ep))
        ep = WannierObject(irvec_ep, data; irvec_next = irvec_el)
    end

    Model(; structure.alat, structure.lattice, structure.recip_lattice, structure.volume,
        dim, nw, nmodes,
        wann_centers = [zero(Vec3{FT})],
        # `mass` is per phonon mode (length nmodes), not per atom: the scalar Einstein
        # coordinate has a single mass, while `structure.mass` follows the crystal
        # convention of 3 entries per atom.
        mass = [mass],
        structure.atom_pos, structure.atom_labels, structure, structure.symmetry,
        use_polar_dipole = false, polar_phonon = Polar(nothing), polar_eph = Polar(nothing),
        el_ham, el_ham_R, el_pos, el_vel, el_velocity_mode,
        ph_dyn, ph_dyn_R,
        epmat = ep, epmat_outer_momentum,
        el_sym = nothing,
    )
end
