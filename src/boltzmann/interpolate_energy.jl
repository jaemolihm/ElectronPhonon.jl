"""
Functions for creating spline interpolations for the band energies.
"""

using Interpolations

export evaluate_itp_el, evaluate_itp_ph

function e_to_interpolation_iband(e, ngrid, kshift, deg)
    itp_iband = interpolate(e, BSpline(deg));
    etp_iband = extrapolate(itp_iband, Periodic());
    zero_to_one = [range(0., 1., length=N + 1)[1:end-1] .+ ks / N for (N, ks) in zip(ngrid, kshift)]
    stp_iband = scale(etp_iband, zero_to_one...)
    stp_iband
end

function e_to_interpolation_dict(e, iband_rng, ngrid, kshift, mode)
    itp_list = []
    @views for iband in iband_rng
        itp = e_to_interpolation_iband(e[:, :, :, iband], ngrid, kshift, mode)
        push!(itp_list, (iband, itp))
    end
    Dict(itp_list)
end

"""
    evaluate_itp_el(model, ngrid, window=(-Inf, Inf); kshift=(0, 0, 0), mode=Cubic(Periodic(OnCell())))
Evaluate interpolation for electron bands inside `window`. Returns `itp_dict`, where
`itp_dict[iband](k1, k2, k3)` is the energy of `iband`-th band at ``k=(k1, k2, k3)```.
"""
function evaluate_itp_el(model, ngrid, window=(-Inf, Inf); kshift=(0, 0, 0), mode=Cubic(Periodic(OnCell())))
    kpts = kpoints_grid(ngrid; shift=kshift)
    e = compute_eigenvalues_el(model, kpts)
    # kpts have k[3] as the fastest index. We need k[1] to be the fastest index.
    e = permutedims(reshape(e, (:, ngrid[3], ngrid[2], ngrid[1])), (4, 3, 2, 1))
    iband_min = findfirst(dropdims(maximum(e, dims=(1,2,3)), dims=(1,2,3)) .>= window[1])
    iband_max =  findlast(dropdims(minimum(e, dims=(1,2,3)), dims=(1,2,3)) .<= window[2])
    e_to_interpolation_dict(e, iband_min:iband_max, ngrid, kshift, mode)
end

"""
    evaluate_itp_ph(model, ngrid; kshift=(0, 0, 0), mode=Cubic(Periodic(OnCell())))
Evaluate interpolation for phonon bands inside `window`. Returns `itp_dict`, where
`itp_dict[iband](k1, k2, k3)` is the energy of `iband`-th band at ``k=(k1, k2, k3)```.
"""
function evaluate_itp_ph(model, ngrid; kshift=(0, 0, 0), mode=Cubic(Periodic(OnCell())))
    kpts = kpoints_grid(ngrid; shift=kshift)
    e = compute_eigenvalues_ph(model, kpts)
    # kpts have k[3] as the fastest index. We need k[1] to be the fastest index.
    e = permutedims(reshape(e, (:, ngrid[3], ngrid[2], ngrid[1])), (4, 3, 2, 1));
    e_to_interpolation_dict(e, 1:model.nmodes, ngrid, kshift, mode)
end
