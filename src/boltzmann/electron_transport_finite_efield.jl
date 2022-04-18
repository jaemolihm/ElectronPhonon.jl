using IterativeSolvers

export solve_electron_finite_efield

function solve_electron_finite_efield(qme_model::AbstractQMEModel{FT}, out_linear, E; kwargs...) where FT
    # Function barrier because some fields of qme_model are not typed
    (; Sₒ, Sᵢ_irr) = qme_model
    E∇ = dropzeros!(sum(E .* qme_model.∇))
    solve_electron_finite_efield(qme_model, out_linear, E, E∇,
    Sₒ, Sᵢ_irr; kwargs...)
end

"""
Solve
```math
-i(e1 - e2) * δρ + (Sₒ + Sᵢ) * δρ + E∇ρ0 + (E ⋅ ∇) δρ = ∂ δρ / ∂t = 0
```
where
```math
E∇ρ0[i] = - E ⋅ v[i] * (df/dε)_{ε=e1[i]}                : if e1[i] == e2[i]
         = - E ⋅ v[i] * (f_mk - f_nk) / (e1[i] - e2[i])  : if e1[i] /= e2[i]
```
and ``i = (m, n, k)``, ``δρ[i] = δρ_{mn;k}``, ``e1[i], e2[i] = e_mk, e_nk``, and ``v[i] = v_{mn;k}``.
"""
function solve_electron_finite_efield(qme_model::AbstractQMEModel{FT}, out_linear, E, E∇,
    Sₒ, Sᵢ_irr=nothing; qme_offdiag_cutoff=Inf, rtol=1e-3, maxiter=100, verbose=false) where FT
    (; el, transport_params) = qme_model

    inds_exclude = @. (el.ib1 != el.ib2) & (abs(el.e1 - el.e2) > qme_offdiag_cutoff)
    δρ_serta = QMEVector(el, Complex{FT})
    δρ = QMEVector(el, Complex{FT})

    nT = length(Sₒ)
    current_serta = fill(FT(NaN), 3, nT)
    current = fill(FT(NaN), 3, nT)

    for iT = 1:nT
        # Add the scattering-out term and the bare Hamiltonian term into Sₒ and invert.
        Sₒ_iT = copy(Sₒ[iT])
        for i in 1:el.n
            (; e1, e2) = el[i]
            if abs(e1 - e2) >= electron_degen_cutoff
                Sₒ_iT[i, i] += -im * (e1 - e2)
            end
        end
        Sₒ⁻¹_iT = invert_scattering_out_matrix(Sₒ_iT, el)
        Sₒ⁻¹_iT[inds_exclude, :] .= 0
        Sₒ⁻¹_iT[:, inds_exclude] .= 0

        # SERTA
        # Compute δᴱρ_serta
        δᴱρ_irr_all = QMEVector(qme_model.el_irr, out_linear.δρ_serta[:, iT])
        δᴱρ_all = unfold_QMEVector(δᴱρ_irr_all, qme_model, true, false)
        δᴱρ_serta = QMEVector(δᴱρ_all.state, dot.(Ref(E), δᴱρ_all.data))

        # Solve (I + Sₒ⁻¹ * E∇) * δρ = δᴱρ_serta
        A = I(el.n) + Sₒ⁻¹_iT * E∇
        δρ_serta .= δᴱρ_serta
        IterativeSolvers.gmres!(δρ_serta.data, A, δᴱρ_serta.data; verbose, reltol=rtol, maxiter)
        current_serta[:, iT] .= vec(occupation_to_conductivity(δρ_serta, transport_params))

        if Sᵢ_irr !== nothing
            # IBTE: Solve (I + Sₒ⁻¹ * (Sᵢ + E∇)) * δρ = δᴱρ_serta
            scatmap = QMEScatteringMap(qme_model, qme_model.el, Sᵢ_irr[iT], Sₒ⁻¹_iT, E∇)
            δρ .= δᴱρ_serta
            _, history = IterativeSolvers.gmres!(δρ, scatmap, δᴱρ_serta; verbose, reltol=rtol, maxiter, log=true)
            @info history
            current[:, iT] .= vec(occupation_to_conductivity(δρ, transport_params))
        end
    end

    return (; current_serta, current, δρ_serta, δρ)
end
