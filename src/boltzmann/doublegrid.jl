"""
Routines for doublegrid calculation of sum over crystal momenta.
"""

using Interpolations: gradient
using TetrahedronIntegration

export compute_lifetime_serta_doublegrid_interpolation

const CUBE_VERTICES = (Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0), Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(0, 1, 1), Vec3(1, 1, 1))


"""
Compute the energy using `itp` on a subgrid points. Subgrid points divide the [-0.5, 0.5] / ngrid
box into `subgrid` cubes. When `points = :Corner (:Center)`, the subgrid points are the corners
(centers) of the cubes.
"""
function _compute_subgrid_energy(itp, state, subgrid, points)
    if points ∉ [:Corner, :Center]
        error("points must be :Corner or :Center")
    end
    nsize = points === :Corner ? subgrid .+ 1 : subgrid
    ngrid = state.ngrid
    e_subgrid = zeros(nsize..., state.n);
    for i in 1:state.n
        iband = state.iband[i]
        xk = state.xks[i]
        for ci in CartesianIndices(nsize)
            if points === :Corner
                δq = (((ci.I .- 1) ./ subgrid) .- 1/2) ./ ngrid
            else
                δq = (((ci.I .- 1/2) ./ subgrid) .- 1/2) ./ ngrid
            end
            e_subgrid[ci.I..., i] = itp[iband]((xk .+ δq).data...)
        end
    end
    e_subgrid
end

"""
Compute the band velocity using `itp` on a subgrid points. Subgrid points divide the [-0.5, 0.5] / ngrid
box into `subgrid` cubes. When `points = :Corner (:Center)`, the subgrid points are the corners
(centers) of the cubes.
"""
function _compute_subgrid_velocity(itp, state, subgrid, points)
    if points ∉ [:Corner, :Center]
        error("points must be :Corner or :Center")
    end
    nsize = points === :Corner ? subgrid .+ 1 : subgrid
    ngrid = state.ngrid
    FT = typeof(state).parameters[1]
    v_subgrid = zeros(Vec3{FT}, subgrid..., state.n);
    for i in 1:state.n
        iband = state.iband[i]
        xk = state.xks[i]
        for ci in CartesianIndices(nsize)
            if points === :Corner
                δq = (((ci.I .- 1) ./ subgrid) .- 1/2) ./ ngrid
            else
                δq = (((ci.I .- 1/2) ./ subgrid) .- 1/2) ./ ngrid
            end
            v_subgrid[ci.I..., i] = gradient(itp[iband], (xk .+ δq).data...)
        end
    end
    v_subgrid
end

"""
Test whether energy conservation (`e_i = e_f = e_el_f + sign_ph * e_ph_f`) can be satisfied given the
minimum and maximum for `e_el_f` and `e_ph_f`. `e_i` is a fixed number.
"""
@inline function _test_energy_conservation(e_i, e_el_f_min, e_el_f_max, e_ph_f_min, e_ph_f_max, sign_ph)
    if sign_ph == 1
        e_f_min = e_el_f_min + e_ph_f_min
        e_f_max = e_el_f_max + e_ph_f_max
    else
        e_f_min = e_el_f_min - e_ph_f_max
        e_f_max = e_el_f_max - e_ph_f_min
    end
    e_f_min ≤ e_i ≤ e_f_max
end

"""
    compute_lifetime_serta_doublegrid_interpolation(el_i, el_f, ph, scat_fid, params, subgrid, itp_el, itp_ph, method)
Compute SERTA inverse lifetime using the doublegrid method. Energies on the subgrid points
are evaluated using the interpolations `itp_el` and `itp_ph`.

Currently, only tetrahedron integration is implemented.

# Inputs
- `method`: Method for tetrahedron integration. `:Onepoint` or `:Ordinary`
"""
function compute_lifetime_serta_doublegrid_interpolation(el_i, el_f, ph, scat_fid, params, subgrid, itp_el, itp_ph, method)
    if params.smearing[1] !== :Tetrahedron
        error("Only Tetrahedron implemented. params.smearing=$(params.smearing)")
    end
    FT = Float64

    inv_τ = zeros(FT, el_i.n, length(params.Tlist))
    η = params.smearing[2]
    inv_η = 1 / η
    ngrid = el_f.ngrid

    nfinegrid = ngrid .* subgrid

    # Compute energy (and velocity for Onepoint) at the subgrid points
    if method === :Ordinary
        e_el_subgrid = _compute_subgrid_energy(itp_el, el_f, subgrid, :Corner)
        e_ph_subgrid = _compute_subgrid_energy(itp_ph, ph, subgrid, :Corner)

        # Minimum and maximum energy for the original cube
        e_el_grid_min = dropdims(minimum(e_el_subgrid, dims=(1, 2, 3)), dims=(1, 2, 3))
        e_el_grid_max = dropdims(maximum(e_el_subgrid, dims=(1, 2, 3)), dims=(1, 2, 3))
        e_ph_grid_min = dropdims(minimum(e_ph_subgrid, dims=(1, 2, 3)), dims=(1, 2, 3))
        e_ph_grid_max = dropdims(maximum(e_ph_subgrid, dims=(1, 2, 3)), dims=(1, 2, 3))

    elseif method === :Onepoint
        e_el_subgrid_center = _compute_subgrid_energy(itp_el, el_f, subgrid, :Center)
        e_ph_subgrid_center = _compute_subgrid_energy(itp_ph, ph, subgrid, :Center)
        v_el_subgrid_center = _compute_subgrid_velocity(itp_el, el_f, subgrid, :Center)
        v_ph_subgrid_center = _compute_subgrid_velocity(itp_ph, ph, subgrid, :Center)

        # Minimum and maximum energy for each subgrid cube
        e_el_subgrid_min = zero(e_el_subgrid_center)
        e_el_subgrid_max = zero(e_el_subgrid_center)
        e_ph_subgrid_min = zero(e_ph_subgrid_center)
        e_ph_subgrid_max = zero(e_ph_subgrid_center)
        for i in eachindex(e_el_subgrid_center)
            v_dot_L = sum(abs.(v_el_subgrid_center[i]) ./ nfinegrid) / 2
            e_el_subgrid_min[i] = e_el_subgrid_center[i] - v_dot_L
            e_el_subgrid_max[i] = e_el_subgrid_center[i] + v_dot_L
        end
        for i in eachindex(e_ph_subgrid_center)
            v_dot_L = sum(abs.(v_ph_subgrid_center[i]) ./ nfinegrid) / 2
            e_ph_subgrid_min[i] = e_ph_subgrid_center[i] - v_dot_L
            e_ph_subgrid_max[i] = e_ph_subgrid_center[i] + v_dot_L
        end

        # Minimum and maximum energy for the original cube
        e_el_grid_min = dropdims(minimum(e_el_subgrid_min, dims=(1, 2, 3)), dims=(1, 2, 3))
        e_el_grid_max = dropdims(maximum(e_el_subgrid_max, dims=(1, 2, 3)), dims=(1, 2, 3))
        e_ph_grid_min = dropdims(minimum(e_ph_subgrid_min, dims=(1, 2, 3)), dims=(1, 2, 3))
        e_ph_grid_max = dropdims(maximum(e_ph_subgrid_max, dims=(1, 2, 3)), dims=(1, 2, 3))
    else
        error("method must be :Ordinary or :Onepoint")
    end

    cnt_debug = 0
    δe_subgrid = zeros((subgrid .+ 1)...)
    for ik in 1:el_i.nk
        mod(ik, 100) == 0 && println("ik = $ik")
        scat = load_BTData(open_group(scat_fid, "ik$ik"), EPW.ElPhScatteringData{FT})

        for iscat in 1:scat.n
            # if mod(iscat, 200000) == 0
            #     println("$iscat at thread $(threadid())")
            # end
            ind_el_i = scat.ind_el_i[iscat]
            ind_el_f = scat.ind_el_f[iscat]
            ind_ph = scat.ind_ph[iscat]
            sign_ph = scat.sign_ph[iscat]
            g2 = scat.mel[iscat]
            e_i = el_i.e[ind_el_i]
            e_ph = ph.e[ind_ph]
            if e_ph < omega_acoustic
                continue
            end

            delta = zero(FT)
            e_ph_occupation = zero(FT)

            if ! _test_energy_conservation(e_i, e_el_grid_min[ind_el_f], e_el_grid_max[ind_el_f],
                                           e_ph_grid_min[ind_ph], e_ph_grid_max[ind_ph], sign_ph)
                continue
            end

            if method === :Ordinary
                @views δe_subgrid .= e_i .- e_el_subgrid[:, :, :, ind_el_f] .- sign_ph .* e_ph_subgrid[:, :, :, ind_ph]
            end

            # Divide [-0.5, 0.5] / ngrid into subgrid cubes and perform the integration.
            # e_ph_occupation: phonon energy to be used for computing occupations.
            # The electron energy for occupation is computed using energy conservation, because
            # the phonon velocity is typically much smaller than the electron velocity.
            for ci in CartesianIndices(subgrid)
                if method === :Ordinary
                    # Method 1: ordinary tetrahedron
                    let δe_subgrid = δe_subgrid, e_ph_subgrid = e_ph_subgrid
                        δe_cube = map(c -> δe_subgrid[(ci.I .+ c)...], CUBE_VERTICES)
                        if all(δe_cube .< 0) || all(δe_cube .> 0) # checking energy conservation
                            continue
                        end
                        cnt_debug += 1
                        delta = delta_parallelepiped_vertex(zero(FT), δe_cube...) / prod(subgrid)
                        e_ph_cube = map(c -> e_ph_subgrid[(ci.I .+ c)..., ind_ph], CUBE_VERTICES)
                        e_ph_occupation = sum(e_ph_cube) / length(e_ph_cube)
                    end
                elseif method === :Onepoint
                    # Method2 : onepoint
                    if ! _test_energy_conservation(e_i,
                            e_el_subgrid_min[ci, ind_el_f], e_el_subgrid_max[ci, ind_el_f],
                            e_ph_subgrid_min[ci, ind_ph],   e_ph_subgrid_max[ci, ind_ph], sign_ph)
                        continue
                    end

                    δe_0 = e_i - e_el_subgrid_center[ci, ind_el_f] - sign_ph * e_ph_subgrid_center[ci, ind_ph]
                    δe_gradient = - v_el_subgrid_center[ci, ind_el_f] - sign_ph * v_ph_subgrid_center[ci, ind_ph]
                    if abs(δe_0) > sum(abs.(δe_gradient) ./ nfinegrid) / 2 # checking energy conservation
                        continue
                    end
                    cnt_debug += 1
                    delta = delta_parallelepiped(zero(FT), δe_0, δe_gradient, 1 ./ nfinegrid) / prod(subgrid)
                    e_ph_occupation = e_ph_subgrid_center[ci, ind_ph]
                end

                if delta < eps(FT)
                    continue
                end

                coeff = g2 * delta * 2π * el_f.k_weight[ind_el_f]

                e_f_occupation = e_i - sign_ph .* e_ph_occupation
                for iT in 1:length(params.Tlist)
                    T = params.Tlist[iT]
                    μ = params.μlist[iT]
                    n_ph = occ_boson(e_ph_occupation, T)
                    f_kq = occ_fermion(e_f_occupation - μ, T)

                    fcoeff = sign_ph == 1 ? n_ph + 1 - f_kq : n_ph + f_kq

                    inv_τ[ind_el_i, iT] += coeff * fcoeff
                end
            end
        end
    end
    println("Number of nontrivial subgrid cubes: $cnt_debug")
    inv_τ
end