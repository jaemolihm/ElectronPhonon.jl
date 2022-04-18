export solve_electron_linear_conductivity

"""
    function solve_electron_linear_conductivity(qme_model::AbstractQMEModel;
        maxiter=100, rtol=1e-10, qme_offdiag_cutoff=Inf)
Solve quantum master equation for electrons to compute linear electrical conductivity.

# Inputs
- `use_full_grid = false`: If `true`, solve the QME on the full grid. If `false`, solve the
QME on the irreducible grid and symmetrize the conductivity matrix.

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
function solve_electron_linear_conductivity(qme_model::AbstractQMEModel; use_full_grid=false, kwargs...)
    # Function barrier because some fields of qme_model are not typed
    Sₒ = use_full_grid ? qme_model.Sₒ : qme_model.Sₒ_irr
    solve_electron_linear_conductivity(qme_model, Sₒ, qme_model.Sᵢ_irr; use_full_grid, kwargs...)
end

function solve_electron_linear_conductivity(qme_model::AbstractQMEModel{FT}, Sₒ, Sᵢ=nothing;
    use_full_grid, maxiter=100, rtol=1e-10, qme_offdiag_cutoff=Inf, verbose=false) where {FT}

    params = qme_model.transport_params
    if use_full_grid
        el = qme_model.el
        symmetry = nothing
    else
        el = qme_model.el_irr
        symmetry = qme_model.symmetry
    end
    (; el_f, filename) = qme_model

    if Sᵢ !== nothing
        ! isfile(filename) && error("filename = $filename is not a valid file.")
        el_f === nothing && throw(ArgumentError("If Sᵢ is used (exact LBTE), el_f must be provided."))
    end

    nT = length(params.Tlist)
    σ_serta = zeros(FT, 3, 3, nT)
    δρ_serta = [QMEVector(el, Vec3{Complex{FT}}) for _ in 1:nT]
    σ = fill(FT(NaN), 3, 3, nT)
    δρ = [QMEVector(el, fill(Vec3(fill(Complex{FT}(NaN), 3)), el.n)) for _ in 1:nT]

    drive_efield = zeros(Vec3{Complex{FT}}, el.n)

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
        mul!(δρ_serta[iT].data, Sₒ⁻¹_iT, .-drive_efield)
        σ_serta[:, :, iT] .= symmetrize(occupation_to_conductivity(δρ_serta[iT], params), symmetry)

        # QME-exact: Solve (Sₒ + Sᵢ) * δρ + drive_efield = 0.
        # Here, we solve (I + Sₒ⁻¹ * Sᵢ) * δρ = -Sₒ⁻¹ * drive_efield = δρ_serta.
        if Sᵢ !== nothing
            scatmap = QMEScatteringMap{3}(qme_model, el, Sᵢ[iT], Sₒ⁻¹_iT)
            δρ[iT] .= δρ_serta[iT]
            _, history = IterativeSolvers.gmres!(
                reinterpret_to_numeric_vector(δρ[iT]),
                scatmap,
                reinterpret_to_numeric_vector(δρ_serta[iT]);
                verbose, reltol=rtol, maxiter, log=true
            )
            verbose && @info history
            σ[:, :, iT] .= symmetrize(occupation_to_conductivity(δρ[iT], params), symmetry)
        end
    end
    (; σ, σ_serta, δρ_serta, δρ, el, params)
end