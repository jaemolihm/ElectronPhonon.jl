export solve_electron_linear_conductivity

"""
    function solve_electron_linear_conductivity(qme_model::AbstractQMEModel;
        max_iter=100, rtol=1e-10, qme_offdiag_cutoff=Inf)
Solve quantum master equation for electrons to compute linear electrical conductivity.
Linearized quantum master equation (stationary state case):
```math
0 = ∂ δρ / ∂t
  = -i(e1[i] - e2[i]) * δρ[i] + drive_efield[i] + ∑_j (Sₒ + Sᵢ)[i, j] * δρ[j],
```
where
```math
drive_efield[i] = - v[i] * (df/dε)_{ε=e1[i]}              : if e_mk  = e_nk
                = - v[i] * (f_mk - f_nk) / (e_mk - e_nk)  : if e_mk /= e_nk
```
and ``i = (m, n, k)``, ``δρ_i = δρ_{mn;k}``, ``e1[i], e2[i] = e_mk, e_nk``, and ``v[i] = v_{mn;k}``.
"""
function solve_electron_linear_conductivity(qme_model::AbstractQMEModel; kwargs...)
    # Function barrier because some fields of qme_model are not typed
    (; Sₒ_irr, Sᵢ_irr) = qme_model
    solve_electron_linear_conductivity(qme_model, Sₒ_irr, Sᵢ_irr; kwargs...)
end

function solve_electron_linear_conductivity(qme_model::AbstractQMEModel{FT}, Sₒ, Sᵢ=nothing;
    max_iter=100, rtol=1e-10, qme_offdiag_cutoff=Inf) where {FT}

    params = qme_model.transport_params
    el = qme_model.el_irr
    (; el_f, symmetry, filename) = qme_model

    if Sᵢ !== nothing
        ! isfile(filename) && error("filename = $filename is not a valid file.")
        el_f === nothing && throw(ArgumentError("If Sᵢ is used (exact LBTE), el_f must be provided."))
    end

    nT = length(params.Tlist)
    σ_serta = zeros(FT, 3, 3, nT)
    δρ_serta = zeros(Vec3{Complex{FT}}, el.n, nT)
    σ = fill(FT(NaN), 3, 3, nT)
    δρ = fill(Vec3(fill(Complex(FT(NaN)), 3)), el.n, nT)
    σ_iter = fill(FT(NaN), (max_iter+1, 3, 3, nT))

    drive_efield = zeros(Vec3{Complex{FT}}, el.n)
    δρ_iter = zeros(Vec3{Complex{FT}}, el.n)
    δρ_iter_tmp = zeros(Vec3{Complex{FT}}, el.n)

    inds_exclude = @. (el.ib1 != el.ib2) & (abs(el.e1 - el.e2) > qme_offdiag_cutoff)

    for (iT, (T, μ)) in enumerate(zip(params.Tlist, params.μlist))
        # Compute the E-field drive term
        for i in 1:el.n
            e1, e2 = el.e1[i], el.e2[i]
            if abs(e1 - e2) < EPW.electron_degen_cutoff
                drive_efield[i] = - el.v[i] * occ_fermion_derivative(e1 - μ, T)
            else
                drive_efield[i] = - el.v[i] * (occ_fermion(e1 - μ, T) - occ_fermion(e2 - μ, T)) / (e1 - e2)
            end
        end
        drive_efield[inds_exclude] .= Ref(zero(Vec3{Complex{FT}}))

        # Add the scattering-out term and the bare Hamiltonian term into S_serta
        Sₒ_iT = copy(Sₒ[iT])
        for i in 1:el.n
            (; e1, e2) = el[i]
            if abs(e1 - e2) >= EPW.electron_degen_cutoff
                Sₒ_iT[i, i] += -im * (e1 - e2)
            end
        end
        Sₒ⁻¹_iT = invert_scattering_out_matrix(Sₒ_iT, el)
        Sₒ⁻¹_iT[inds_exclude, :] .= 0
        Sₒ⁻¹_iT[:, inds_exclude] .= 0

        # QME-SERTA: Solve Sₒ * δρ + drive_efield = 0
        @views mul!(δρ_serta[:, iT], Sₒ⁻¹_iT, .-drive_efield)
        σ_serta[:, :, iT] .= symmetrize(occupation_to_conductivity(δρ_serta[:, iT], el, params), symmetry)

        # QME-exact: Iteratively solve (Sₒ + Sᵢ) * δρ + drive_efield = 0.
        if Sᵢ !== nothing
            # Initial guess: SERTA density matrix
            @views δρ_iter .= δρ_serta[:, iT]
            σ_new = symmetrize(occupation_to_conductivity(δρ_iter, el, params), symmetry)
            σ_iter[1, :, :, iT] .= σ_new

            scatmap = QMEScatteringMap(qme_model, el, Sᵢ[iT], Sₒ⁻¹_iT)

            # Fixed-point iteration
            for iter in 1:max_iter
                σ_old = σ_new

                # Compute δρ_iter_next = Sₒ⁻¹ * (-Sᵢ * δρ_iter_prev - drive_efield)
                #                      = - Sₒ⁻¹ Sᵢ * δρ_iter_prev + δρ_serta[:, iT]
                # Since scatmap = I + Sₒ⁻¹ Sᵢ, we have
                # δρ_iter_next = δρ_iter_prev - scatmap * δρ_iter_prev + δρ_serta
                δρ_iter_tmp .= Ref(zero(eltype(δρ_iter_tmp)))
                mul!(δρ_iter_tmp, scatmap, δρ_iter)
                @views @. δρ_iter += -δρ_iter_tmp + δρ_serta[:, iT]

                # NOTE: One cannot use IterativeSolvers.gmres! here because for symmetry,
                #       δρ for all 3 E-field directions must be computed at the same time,
                #       but IterativeSolvers.gmres! does not allow the vector to have
                #       non-Number elements.
                #       So, we use simple iterative scheme.

                σ_new = symmetrize(occupation_to_conductivity(δρ_iter, el, params), symmetry)
                σ_iter[iter+1, :, :, iT] .= σ_new

                # Check convergence
                if norm(σ_new - σ_old) / norm(σ_new) < rtol
                    @info "iT=$iT, converged at iteration $iter"
                    break
                elseif iter == max_iter
                    @info "iT=$iT, convergence not reached at maximum iteration $max_iter"
                end
            end
            σ[:, :, iT] .= σ_new
            δρ[:, iT] .= δρ_iter
        end
    end
    (; σ, σ_serta, δρ_serta, δρ, σ_iter, el, params)
end