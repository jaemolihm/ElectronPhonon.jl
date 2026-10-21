# Linear-in-B Hall conductivity on the BandStates BTE path (`BoltzmannCalculator` +
# `solve_electron_bte`). This file takes `BandStates` only: it adds no `BTStates` method, because the
# legacy BTStates path is to be removed. The QME Hall solve is in `electron_transport_hall.jl`.

export solve_electron_bte_hall

"""
    solve_electron_bte_hall(out_bte, el_i, el_f, scat_mat, inv_τ_i, params, model,
                            symmetry = nothing; order = 1, max_iter = 100, rtol = 1e-10)

Linear Hall conductivity `σ_hall[a, b, c]` and Hall factor `r_hall[a, b, c]` of the BTE, for
`el_i` (IBZ if `symmetry` is given) and `el_f` on the same uniform Γ-centred grid. Arguments are
those of `solve_electron_bte`, whose output `out_bte` is passed in (it must have been computed with
the same `params.Tlist`, `params.μlist` and `symmetry`) plus the `model` (its `recip_lattice` and `dimension`). Returns
`(; σ_serta, σ, σ_hall_serta, σ_hall, r_hall_serta, r_hall)`, with `σ*` of size `(3, 3, nT)` and
`σ_hall*`, `r_hall*` of size `(3, 3, 3, nT)`. Host only (under MPI run it on the root with the
gathered `Sᵢ`, as `solve_electron_bte`). `order` is the order of the finite-difference gradient.

Convention (`e` and `ħ` left out): `j_a = σ_ab E_b + σ_hall[a, b, c] E_b B_c`. The physical Hall
tensor is `R = -σ⁻¹ σ_hall σ⁻¹`, `E_a = R_abc j_b B_c`, and `r_hall[a, b, c] = ∑_{de} σ⁻¹[a, d]
σ_hall[d, e, c] σ⁻¹[e, b] · nlist / volume`. For one parabolic band with constant `τ`,
`r_hall[1, 2, 3] = +1` and `r_hall[2, 1, 3] = -1`, for electrons and holes. With `d = model.dimension`
only the block `σ[1:d, 1:d]` is inverted; entries of `r_hall` with `a` or `b` > `d` are NaN.

The Lorentz force acts on the first-order field `δᴱf` (the `δf_i`, `δf_i_serta` of `out_bte`):
`δᴱᴮf = τ (v × ∇_k) δᴱf` for SERTA, and with `Sᵢ` for the full solution. The gradient is a
band-diagonal finite difference on the grid, from whole |b| shells.

Approximations:
- Band index matching: `δᴱf_n(k + b)` is read at the same band index `n`. This fails at band
  crossings and degeneracies, where a swapped index gives a spurious derivative of order `1/b`.
- `δᴱf = 0` outside the window, so the gradient at the window edge sees a step; widen the window to check.
  `window_kq` must contain `window_k`: a missing unfolded `el_i` state is an error.
- The finite-difference error is `O(b^(2 order))` only if `δᴱf ∝ -∂f/∂ε` is resolved by the grid.
- No Berry-curvature, orbital-moment or interband terms.
"""
function solve_electron_bte_hall(out_bte, el_i::BandStates{FT}, el_f::BandStates{FT}, scat_mat,
        inv_τ_i, params, model, symmetry = nothing; order = 1, max_iter = 100,
        rtol = 1e-10) where {FT}
    nT = length(params.Tlist)

    # Uniform Γ-centred grids only: a multigrid carries the coarse weights on its coarse states.
    (el_i.kpts isa GridKpoints && el_f.kpts isa GridKpoints) ||
        throw(ArgumentError("el_i and el_f must carry GridKpoints"))
    el_i.kpts.ngrid == el_f.kpts.ngrid ||
        throw(ArgumentError("el_i and el_f must be on the same grid, got $(el_i.kpts.ngrid) and $(el_f.kpts.ngrid)"))
    (all(iszero, el_i.kpts.shift) && all(iszero, el_f.kpts.shift)) ||
        throw(ArgumentError("shifted grids are not supported"))
    ngrid = el_i.kpts.ngrid
    all(≈(1 / prod(ngrid)), state_weights(el_f)) ||
        throw(ArgumentError("every el_f state must have the weight 1/prod(ngrid): multigrids are not supported"))
    # Every unfolded el_i state needs an el_f state (`vector_field_unfold_and_interpolate_map` would
    # drop it silently).
    for key in keys(last(unfold_data_map(el_i, symmetry)))
        state_index(el_f, Vec3{FT}(key[1], key[2], key[3]) ./ ngrid, key[4]) == 0 &&
            throw(ArgumentError("the unfolded el_i state at grid point $(Tuple(key)[1:3]), band $(key[4]) " *
                "is not in el_f: window_kq must contain window_k"))
    end

    # `out_bte` must belong to the same temperatures, chemical potentials and symmetry.
    (out_bte.params.Tlist == params.Tlist && out_bte.params.μlist == params.μlist) ||
        throw(ArgumentError("out_bte was computed with different Tlist or μlist than params"))
    same_symmetry = symmetry === out_bte.symmetry || (symmetry !== nothing &&
        out_bte.symmetry !== nothing && symmetry.S == out_bte.symmetry.S &&
        symmetry.is_tr == out_bte.symmetry.is_tr && symmetry.Scart ≈ out_bte.symmetry.Scart)
    same_symmetry || throw(ArgumentError("out_bte was computed with a different symmetry"))

    stencil = finite_difference_vectors(model.recip_lattice, ngrid; order, dimension = model.dimension)
    symmetry === nothing || _assert_stencil_closed(stencil, symmetry)
    grad = _finite_difference_gradient_matrices(el_i, el_f, stencil)

    map_1 = vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry; interpolate = false)
    map_2 = vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry,
        _hall_occupation_law; interpolate = false)

    σ_hall_serta = zeros(FT, 3, 3, 3, nT)
    σ_hall = zeros(FT, 3, 3, 3, nT)
    r_hall_serta = fill(FT(NaN), 3, 3, 3, nT)
    r_hall = fill(FT(NaN), 3, 3, 3, nT)
    inds = 1:model.dimension

    for iT in 1:nT
        inv_τ = inv_τ_i[:, iT]

        # SERTA: X = R, with δᴱf from SERTA.
        X_serta = _hall_source(grad, map_1 * out_bte.δf_i_serta[:, iT], el_i.vs, inv_τ)
        σ_hall_serta[:, :, :, iT] .= _hall_conductivity(X_serta, el_i, params, symmetry)

        # Full: X = R + (1 / τ⁻¹) Sᵢ (U X), with δᴱf from the full solution.
        source = _hall_source(grad, map_1 * out_bte.δf_i[:, iT], el_i.vs, inv_τ)
        sol = _solve_bte(source, scat_mat[iT], map_2, inv_τ; solver = :gmres, max_iter, rtol, iT)
        σ_hall[:, :, :, iT] .= _hall_conductivity(sol.δf, el_i, params, symmetry)

        # Hall factor, on the block `1:model.dimension`.
        density = params.nlist[iT] / params.volume
        r_hall_serta[inds, inds, :, iT] .= compute_hall_factor(out_bte.σ_serta[inds, inds, iT],
            σ_hall_serta[inds, inds, :, iT]) .* density
        r_hall[inds, inds, :, iT] .= compute_hall_factor(out_bte.σ[inds, inds, iT],
            σ_hall[inds, inds, :, iT]) .* density
    end
    (; out_bte.σ_serta, out_bte.σ, σ_hall_serta, σ_hall, r_hall_serta, r_hall)
end

# Source of the Hall equation on `el_i`, as the 9-vector `X[b + 3 (c - 1)]` of the matrix `X[b, c]`:
# `X[b, c] = (v[c1] (∂_c2 δᴱf_b) - v[c2] (∂_c1 δᴱf_b)) / τ⁻¹`, `(c, c1, c2)` cyclic, with `δᴱf` the
# first-order occupation on `el_f` and `grad` the three gradient matrices from `el_f` to `el_i`.
function _hall_source(grad, δᴱf::AbstractVector{Vec3{FT}}, vs, inv_τ) where {FT}
    ∂δᴱf = map(∂ -> ∂ * δᴱf, grad)
    X = Vector{SVector{9, FT}}(undef, length(vs))
    for i in eachindex(vs)
        X[i] = SVector{9, FT}(ntuple(9) do k
            b, c = mod1(k, 3), (k - 1) ÷ 3 + 1
            c1, c2 = mod1(c + 1, 3), mod1(c + 2, 3)
            (vs[i][c1] * ∂δᴱf[c2][i][b] - vs[i][c2] * ∂δᴱf[c1][i][b]) / inv_τ[i]
        end)
    end
    X
end

# `σ_hall[a, b, c] = (g_s / V) ∑_i w_i v_i[a] X_i[b, c]` over the states of `el_i`, symmetrized with the
# rank-3 law of a tensor with polar `a`, `b` and axial `c` (even under time reversal: `v` and `X` are odd).
function _hall_conductivity(X, el_i::BandStates{FT}, params, symmetry) where {FT}
    w = state_weights(el_i)
    σ_hall = zeros(FT, 3, 3, 3)
    for i in 1:el_i.n, k in 1:9, a in 1:3
        b, c = mod1(k, 3), (k - 1) ÷ 3 + 1
        σ_hall[a, b, c] += w[i] * el_i.vs[i][a] * X[i][k]
    end
    σ_hall .*= params.spin_degeneracy / params.volume
    symmetrize_array(σ_hall, symmetry; order = 3, axial = true)
end
