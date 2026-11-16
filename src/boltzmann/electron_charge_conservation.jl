# Charge-conservation projection of the linearized BTE solve (`_solve_bte` in electron_lbte.jl): the
# projector onto the charge-conserving space, as a LinearMap, and the chemical-potential shift it removes.
# Included after electron_lbte.jl (uses `BTorBandStates`, `_linear_transport_mapping`).

"""
    ChargeProjection{FT} <: LinearMap{FT}

Projector `Π = I - B (UᵀB)⁻¹ Uᵀ` onto the charge-conserving space, on the BTE field stored as its flat
real components (`ncomp * n`, state-major), with `B[:, k] = f′ ⊗ dirs[:, k] / ‖f′‖` and
`U[:, k] = w ⊗ dirs[:, k] / ‖w‖`. `UᵀΠ = 0` (the charge `Σᵢ wᵢ dirsᵀ δfᵢ` is zero) and `Π B = 0` (the
uniform shift of μ is removed); `I` minus a rank-`m` term, `m = size(dirs, 2)`, and `Π = I` for `m = 0`.
Built by `charge_conservation_projection`, applied by `_solve_bte`; `chemical_potential_shift` gives the
coefficient of `f′` that `Π` removes.
"""
struct ChargeProjection{FT} <: LinearMap{FT}
    B::Matrix{FT}
    U::Matrix{FT}
    UtB::Matrix{FT}
    dirs::Matrix{FT}
    fprime_norm::FT
end

# No constraint direction (m = 0): Π = I.
ChargeProjection{FT}(ncomp::Integer, n::Integer) where {FT} =
    ChargeProjection{FT}(zeros(FT, ncomp * n, 0), zeros(FT, ncomp * n, 0), zeros(FT, 0, 0), zeros(FT, ncomp, 0), one(FT))

Base.size(P::ChargeProjection) = (size(P.B, 1), size(P.B, 1))

function LinearMaps._unsafe_mul!(y::AbstractVector, P::ChargeProjection, x::AbstractVector)
    # y = x - B (UᵀB)⁻¹ Uᵀ x; the correction is skipped when m = 0.
    coeff = size(P.B, 2) > 0 ? P.UtB \ (P.U' * x) : nothing
    y .= x
    coeff === nothing || mul!(y, P.B, coeff, -1, true)
    y
end

"""
    chemical_potential_shift(P::ChargeProjection, r) -> Vector

`δμ = dirs (UᵀB)⁻¹ Uᵀ r / ‖f′‖`, the per-component coefficient of `f′` that `P` removes from the flat
field `r`, i.e. `(I - P) r = f′ ⊗ δμ` on the constrained directions. Called by `_solve_bte` with the
residual `r = rhs - A δf` at the solution (`Uᵀ δf = 0`), where it is `δμ` of ``A δf + f′ δμ = rhs``.
Zero when `m = 0`.
"""
function chemical_potential_shift(P::ChargeProjection{FT}, r::AbstractVector) where {FT}
    size(P.B, 2) == 0 && return zeros(FT, size(P.dirs, 1))
    P.dirs * (P.UtB \ (P.U' * r)) ./ P.fprime_norm
end

"""
    charge_conservation_projection(el_i, symmetry, μ, T, mapping = _linear_transport_mapping; enforce = true)
        -> ChargeProjection

Charge-conservation projector of the BTE solve at `μ`, `T`, built by `solve_electron_bte` and
`solve_electron_bte_hall` at each temperature and passed to `_solve_bte` as `P_charge`.

The response conserves the particle number, `Σ_{k ∈ FBZ} wₖ δfₖ = 0` per component. With the
unfolding `δf_{Sk} = M(S) δf_k`, `M = mapping(is_tr, Scart)`, this reads on `el_i`
`Σᵢ wᵢ P δfᵢ = 0` with `P = (1/N_S) Σ_S M(S)`, the projector onto the directions `v` that every `M(S)`
leaves unchanged. Its range is the orthonormal columns of `dirs` (`ncomp × m`), computed as the common null
space of `M(S) - I` over the operations (tolerance 1e-6). `m = ncomp` for `symmetry === nothing`; `m = 0`
when the group has time reversal (`v -> -R v` for the default `_linear_transport_mapping`) or inversion,
where the constraint holds identically; in general `m` is the dimension of the invariant subspace (for the
linear mapping: 1 for Cn and Cnv, 2 for Cs, 3 for C1). With `enforce = false`, `m = 0` always.

Why it is needed: for the reciprocal occupation form (F5) pairwise detailed balance,
`Sᵢ[i, j] f′[j] = Sₒ[i, j] f′[i]` with `f′ = -∂f/∂ε`, makes `δf = f′ ⊗ v` a null vector of
`A = I - D⁻¹ Sᵢ U` for each `v` in the range of `P` (a uniform shift of μ). The right-hand side has a small
component along the left null vector (detailed balance is broken at the level of the g symmetry violation),
so without the constraint the system is inconsistent and GMRES and the fixed-point iteration fail.
`_solve_bte` solves ``A δf + B δμ = rhs`` with ``Uᵀ δf = 0`` (`B = f′ ⊗ dirs`, `U = w ⊗ dirs`) by
projection: `f′ᵢ δμ` is a uniform shift of the chemical potential, and `δμ` measures the inconsistency of
`rhs` along the left null vector (for F5 the detailed-balance violation; other forms add their own
non-conservation).
"""
function charge_conservation_projection(el_i::BTorBandStates{FT}, symmetry, μ, T,
        mapping = _linear_transport_mapping; enforce::Bool = true) where {FT}
    ncomp = size(mapping(false, one(Mat3{FT})), 1)
    enforce || return ChargeProjection{FT}(ncomp, el_i.n)  # No constraint: Π = I.

    if symmetry === nothing
        # No symmetry map: P = I, one constraint per component.
        dirs = Matrix{FT}(I, ncomp, ncomp)
    else
        # Range of P: common null space of mapping(is_tr, Scart) - I over the operations.
        stacked = reduce(vcat, [Matrix(mapping(is_tr, Scart)) - I for (is_tr, Scart) in zip(symmetry.is_tr, symmetry.Scart)])
        # (the stack has at least `ncomp` rows: the identity operation is always present)
        F = svd(stacked; full = true)
        dirs = FT.(F.V[:, F.S .< 1e-6])
    end
    m, n = size(dirs, 2), el_i.n
    m == 0 && return ChargeProjection{FT}(ncomp, n)
    # B and U as (ncomp n) × m matrices, columns normalized.
    fprime = @. -occ_fermion_derivative(el_i.es - μ, T)
    weights = bt_weights(el_i)
    nf, nw = norm(fprime), norm(weights)
    B = zeros(FT, ncomp * n, m)
    U = zeros(FT, ncomp * n, m)
    for k in 1:m, i in 1:n, c in 1:ncomp
        B[c + ncomp * (i - 1), k] = fprime[i] * dirs[c, k] / nf
        U[c + ncomp * (i - 1), k] = weights[i] * dirs[c, k] / nw
    end
    ChargeProjection{FT}(B, U, U' * B, dirs, nf)
end
