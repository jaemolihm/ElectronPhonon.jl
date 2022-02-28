# Calculation of electron linear Hall mobility

# TODO: Beyond SERTA

export compute_linear_hall_conductivity

"""
    compute_linear_hall_conductivity(out_linear, qme_model::AbstractQMEModel)
# Inputs:
- `out_linear`: Output of linear electrical conductivity calculation.
- `qme_model::AbstractQMEModel`
"""
function compute_linear_hall_conductivity(out_linear, qme_model::AbstractQMEModel; kwargs...)
    # Function barrier because some fields of qme_model are not typed
    (; ∇, S_out, S_in_irr) = qme_model
    compute_linear_hall_conductivity(out_linear, qme_model, ∇, S_out, S_in_irr; kwargs...)
end

function compute_linear_hall_conductivity(out_linear, qme_model::AbstractQMEModel{FT}, ∇,
        S_out, S_in_irr=nothing; max_iter=100, rtol=1e-6,) where FT
    transport_params = qme_model.transport_params
    nT = length(transport_params.Tlist)
    σ_hall_iter = fill(FT(NaN), (max_iter+1, 3, 3, 3, nT))
    σ_hall = fill(FT(NaN), (3, 3, 3, nT))
    σ_hall_serta = fill(FT(NaN), (3, 3, 3, nT))
    r_hall = fill(FT(NaN), (3, 3, 3, nT))
    r_hall_serta = fill(FT(NaN), (3, 3, 3, nT))

    v = get_velocity_as_QMEVector(qme_model.el)
    @views for iT in 1:nT
        δᴱρ_irr_serta = QMEVector(qme_model.el_irr, out_linear.δρ_serta[:, iT])
        δᴱρ_serta = unfold_QMEVector(δᴱρ_irr_serta, qme_model, true, false)
        if S_in_irr !== nothing
            δᴱρ_irr = QMEVector(qme_model.el_irr, out_linear.δρ[:, iT])
            δᴱρ = unfold_QMEVector(δᴱρ_irr, qme_model, true, false)
        end
        δᴱᴮρ_serta = similar(δᴱρ_serta)

        for c in 1:3
            c1, c2 = mod1(c + 1, 3), mod1(c + 2, 3)
            # SERTA: Solve δᴱᴮρ_serta = S_out⁻¹ (v × ∇) δᴱρ_serta
            v∇δᴱρ = v[c1] * (∇[c2] * δᴱρ_serta) - v[c2] * (∇[c1] * δᴱρ_serta);
            _solve_qme_direct!(δᴱᴮρ_serta, S_out[iT], v∇δᴱρ)

            # Transpose because the index of occupation_to_conductivity is (current, E field),
            # but we want the index of σ_hall to be (E field, current).
            σ_hall_serta[:, :, c, iT] .= occupation_to_conductivity(δᴱᴮρ_serta, transport_params)'

            if S_in_irr !== nothing
                # IBTE: Solve iteratively δᴱᴮρ = - S_out⁻¹ S_in δᴱᴮρ + δᴱᴮρ_serta
                # Initial guess: δᴱᴮρ_serta. Use IBTE solution from out_linear.
                v∇δᴱρ = v[c1] * (∇[c2] * δᴱρ) - v[c2] * (∇[c1] * δᴱρ);
                _solve_qme_direct!(δᴱᴮρ_serta, S_out[iT], v∇δᴱρ)
                σ_new = occupation_to_conductivity(δᴱᴮρ_serta, transport_params)'
                σ_hall_iter[1, :, :, c, iT] .= σ_new

                δᴱᴮρ = copy(δᴱᴮρ_serta)
                Sout⁻¹_Sin_δρ = similar(δᴱᴮρ)

                # Fixed point iteration
                for iter in 1:max_iter
                    σ_old = σ_new

                    # δᴱᴮρ(next) = - S_out⁻¹ S_in δᴱᴮρ(prev) + δᴱᴮρ_serta
                    Sin_δᴱᴮρ = EPW.multiply_S_in(δᴱᴮρ, S_in_irr[iT], qme_model);
                    EPW._solve_qme_direct!(Sout⁻¹_Sin_δρ, S_out[iT], Sin_δᴱᴮρ);
                    @. δᴱᴮρ.data = - Sout⁻¹_Sin_δρ.data + δᴱᴮρ_serta.data;

                    σ_new = occupation_to_conductivity(δᴱᴮρ, transport_params)'
                    σ_hall_iter[iter+1, :, :, c, iT] .= σ_new
                    # Check convergence
                    if norm(σ_new - σ_old) / norm(σ_new) < rtol
                        @info "iT=$iT, converged at iteration $iter"
                        break
                    elseif iter == max_iter
                        @info "iT=$iT, convergence not reached at maximum iteration $max_iter"
                    end
                end
                σ_hall[:, :, c, iT] .= σ_new
            end
        end
        @views r_hall_serta[:, :, :, iT] = compute_hall_factor(out_linear.σ_serta[:, :, iT],
            σ_hall_serta[:, :, :, iT]) .* transport_params.n ./ transport_params.volume
        if S_in_irr !== nothing
            @views r_hall[:, :, :, iT] = compute_hall_factor(out_linear.σ[:, :, iT],
                σ_hall[:, :, :, iT]) .* transport_params.n ./ transport_params.volume
        end
    end
    (; σ_hall_serta, r_hall_serta, σ_hall, r_hall, σ_hall_iter)
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

_solve_qme_direct!(δρ::QMEVector, S, δρ0::QMEVector) = _solve_qme_direct!(δρ.data, S, δρ0.data)