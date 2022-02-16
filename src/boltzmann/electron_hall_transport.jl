# Calculation of electron linear Hall mobility

# TODO: Beyond SERTA

export compute_linear_hall_conductivity

"""
    compute_linear_hall_conductivity(out_linear, qme_model::AbstractQMEModel)
# Inputs:
- `out_linear`: Output of linear electrical conductivity calculation.
- `qme_model::AbstractQMEModel`
"""
function compute_linear_hall_conductivity(out_linear, qme_model::AbstractQMEModel)
    # Function barrier because some fields of qme_model are not typed
    (; ∇, S_out) = qme_model
    compute_linear_hall_conductivity(out_linear, qme_model, ∇, S_out)
end

function compute_linear_hall_conductivity(out_linear, qme_model, ∇, S_out)
    transport_params = qme_model.transport_params
    nT = length(transport_params.Tlist)
    σ_hall = zeros(3, 3, 3, nT)
    r_hall = zeros(3, 3, 3, nT)

    v = get_velocity_as_QMEVector(qme_model.el)
    @views for iT in 1:nT
        δᴱρ_irr = QMEVector(qme_model.el_irr, out_linear.δρ_serta[:, iT])
        δᴱρ = unfold_QMEVector(δᴱρ_irr, qme_model, true, false)

        for c in 1:3
            c1, c2 = mod1(c + 1, 3), mod1(c + 2, 3)
            # Solve 0 = (S_out + S_in) * δᴱᴮρ - (v × ∇) δᴱρ
            v∇δᴱρ = v[c1] * (∇[c2] * δᴱρ) - v[c2] * (∇[c1] * δᴱρ);
            δᴱᴮρ = S_out[iT] \ v∇δᴱρ

            # Transpose because the index of occupation_to_conductivity is (current, E field),
            # but we want the index of σ_hall to be (E field, current).
            σ_hall[:, :, c, iT] = occupation_to_conductivity(δᴱᴮρ, transport_params)'
        end
        @views r_hall[:, :, :, iT] = compute_hall_factor(out_linear.σ_serta[:, :, iT],
            σ_hall[:, :, :, iT]) .* transport_params.n ./ transport_params.volume
    end
    (; σ_hall, r_hall)
end

"""
    compute_hall_factor(σ, σ_hall)
``r[a, b, c] = ∑_{d,e} inv(σ)[a, d] * σ_hall[d, e, c] * inv(σ)[e, b]``
Reference: Eq. (13) of S. Poncé et al, Phys. Rev. Research 3, 043022 (2021)
(Note that the carrier density is not multiplied in this function.)
"""
function compute_hall_factor(σ, σ_hall)
    r = zeros(3, 3, 3)
    σinv = inv(σ)
    for a in 1:3, b in 1:3, c in 1:3, d in 1:3, e in 1:3
        r[a, b, c] += σinv[a, d] * σ_hall[d, e, c] * σinv[e, b]
    end
    r
end