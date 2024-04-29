using FiniteDifferences: central_fdm, grad, jacobian

export find_band_extrema
export find_effective_mass

"""
    find_band_extrema(model, iband; type, nk=50)
Find the extremum of the `iband`-th band.
* `type`: `:Minima` to find band minima, `:Maxima` to find band maxima.
"""
function find_band_extrema(model, iband, type; nk=50)
    @assert type ∈ [:Minima, :Maxima]

    kpts = kpoints_grid((nk, nk, nk); model.symmetry)
    es = compute_eigenvalues_el(model, kpts)[iband, :]
    if type === :Minima
        ik = argmin(es)
    else
        ik = argmax(es)
    end
    e = es[ik]
    xk = kpts.vectors[ik]

    # Find symmetry-equivalent k points
    xk_equiv = [xk]
    for (; is_tr, S) in model.symmetry
        sxk = is_tr ? -S*xk : S * xk

        found = false
        for xk_ in xk_equiv
            Δxk = normalize_kpoint_coordinate(sxk - xk_ .+ 0.5) .- 0.5
            if all(Δxk .< sqrt(eps(eltype(eltype(Δxk)))))
                found = true
                break
            end
        end
        found || push!(xk_equiv, sxk)
    end

    xk_equiv .= normalize_kpoint_coordinate(xk_equiv)

    Kpoints(xk_equiv), e
end


"""
    find_effective_mass(model, kpts, iband)
Compute the effective mass for each k point in `kpts` for the `iband`-th band.

In the Rydberg units, ``e(k + δk) = e0 + δk * m⁻¹ * δk``. So, ``m = 2(d²e / dδk²)⁻¹``.
"""
function find_effective_mass(model, kpts, iband)
    f(xk) = compute_eigenvalues_el(model, Kpoints(xk); fourier_mode="normal")[iband]
    map(kpts.vectors) do xk
        hessian = jacobian(central_fdm(5, 1), xk -> grad(central_fdm(5, 1, adapt=0), f, xk)[1], xk)[1]
        2 * inv(Mat3(hessian))
    end
end