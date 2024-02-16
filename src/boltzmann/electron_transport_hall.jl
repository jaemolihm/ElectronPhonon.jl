# Calculation of electron linear Hall mobility

using IterativeSolvers

export solve_electron_hall_conductivity

"""
    solve_electron_hall_conductivity(out_linear, qme_model::AbstractQMEModel)
Use GMRES itertive solver for IBTE.
# Inputs:
- `out_linear`: Output of linear electrical conductivity calculation.
- `qme_model::AbstractQMEModel`

FIXME: Currently assumes that only degenerate bands are included.
"""
function solve_electron_hall_conductivity(out_linear, qme_model::AbstractQMEModel; kwargs...)
    # Function barrier because some fields of qme_model are not typed
    (; ∇, Sₒ, Sᵢ_irr) = qme_model
    ∇ === nothing && error("qme_model.∇ must be set to compute Hall conductivity")
    solve_electron_hall_conductivity(out_linear, qme_model, ∇, Sₒ, Sᵢ_irr; kwargs...)
end

function solve_electron_hall_conductivity(out_linear, qme_model::AbstractQMEModel{FT}, ∇,
        Sₒ, Sᵢ_irr=nothing; maxiter=100, rtol=1e-3, atol=0, verbose=false) where FT
    transport_params = qme_model.transport_params
    nT = length(transport_params.Tlist)
    σ_hall = fill(FT(NaN), (3, 3, 3, nT))
    σ_hall_serta = fill(FT(NaN), (3, 3, 3, nT))
    r_hall = fill(FT(NaN), (3, 3, 3, nT))
    r_hall_serta = fill(FT(NaN), (3, 3, 3, nT))

    v = get_velocity_as_QMEVector(qme_model.el)
    δᴱρ = Tuple(QMEVector(qme_model.el, Complex{FT}) for i in 1:3)
    δᴱᴮρ_serta = similar(δᴱρ[1])
    δᴱᴮρ = similar(δᴱρ[1])

    @views for iT in 1:nT
        @info "iT = $iT"

        # SERTA: Solve δᴱᴮρ_serta = Sₒ⁻¹ (v × ∇) δᴱρ_serta
        # First take δᴱρ on el_irr as QMEVector{Vec3}, unfold them to el, and make QMEVector{Number}
        # Note that we use out_linear.δρ_serta, which is computed from SERTA.
        δᴱρ_all = unfold_QMEVector(out_linear.δρ_serta[iT], qme_model, true, false)
        for i in 1:3
            δᴱρ[i].data .= [x[i] for x in δᴱρ_all.data]
        end

        # FIXME: Add (e2 - e1) contribution
        Sₒ⁻¹ = invert_scattering_out_matrix(Sₒ[iT], qme_model.el)

        for b = 1:3, c = 1:3
            c1, c2 = mod1(c + 1, 3), mod1(c + 2, 3)
            v∇δᴱρ = v[c1] * (∇[c2] * δᴱρ[b]) - v[c2] * (∇[c1] * δᴱρ[b])
            mul!(δᴱᴮρ_serta.data, Sₒ⁻¹, v∇δᴱρ.data)
            σ_hall_serta[:, b, c, iT] .= real.(vec(occupation_to_conductivity(δᴱᴮρ_serta, transport_params)))
        end

        @views r_hall_serta[:, :, :, iT] = compute_hall_factor(out_linear.σ_serta[:, :, iT],
            σ_hall_serta[:, :, :, iT]) .* transport_params.nlist[iT] ./ transport_params.volume

        if Sᵢ_irr !== nothing
            # IBTE: Solve scatmap * δᴱᴮρ = (I + Sₒ⁻¹ Sᵢ) δᴱᴮρ = δᴱᴮρ_serta using GMRES
            # Equivalent to solving (Sₒ + Sᵢ)x = b with preconditioner P = Sₒ.
            # Note that we use out_linear.δρ, which is computed from IBTE.
            δᴱρ_all = unfold_QMEVector(out_linear.δρ[iT], qme_model, true, false)
            for i in 1:3
                δᴱρ[i].data .= [x[i] for x in δᴱρ_all.data]
            end

            # Define scattering map and GMRES iterable solver
            scatmap = QMEScatteringMap(qme_model, qme_model.el, Sᵢ_irr[iT], Sₒ⁻¹)
            g = IterativeSolvers.gmres_iterable!(δᴱᴮρ.data, scatmap, δᴱᴮρ_serta.data; maxiter)

            for b = 1:3, c = 1:3
                c1, c2 = mod1(c + 1, 3), mod1(c + 2, 3)
                v∇δᴱρ = v[c1] * (∇[c2] * δᴱρ[b]) - v[c2] * (∇[c1] * δᴱρ[b])

                # Compute the SERTA solution
                mul!(δᴱᴮρ_serta.data, Sₒ⁻¹, v∇δᴱρ.data)

                # Set and run GMRES solver. Initial guess is δᴱᴮρ = δᴱᴮρ_serta.
                reset_gmres_iterable!(g, δᴱᴮρ_serta.data, δᴱᴮρ_serta.data; reltol=rtol, abstol=atol)
                cnt = 0
                for (iteration, residual) in enumerate(g)
                    cnt += 1
                    verbose && @printf("%3d\t%1.2e\n", iteration, residual)
                end
                if IterativeSolvers.converged(g)
                    @info "b = $b, c = $c: converged in $cnt iterations"
                else
                    @info "b = $b, c = $c: convergence not reached in $cnt iterations"
                end

                # Converged result is stored at g.x === δᴱᴮρ.data.
                σ_hall[:, b, c, iT] .= real.(vec(occupation_to_conductivity(δᴱᴮρ, transport_params)))
            end
            @info "Total $(g.mv_products) matrix-vector products used"

            @views r_hall[:, :, :, iT] = compute_hall_factor(out_linear.σ[:, :, iT],
                σ_hall[:, :, :, iT]) .* transport_params.nlist[iT] ./ transport_params.volume
        end
    end
    (; σ_hall_serta, r_hall_serta, σ_hall, r_hall)
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
        r[a, b, c] += real(σinv[a, d] * σ_hall[d, e, c] * σinv[e, b])
    end
    r
end
