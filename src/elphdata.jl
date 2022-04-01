
# For computing electron-phonon coupling at fine a k and q point

import Base.@kwdef
import EPW.WanToBloch: get_eph_Rq_to_kq!, get_eph_kR_to_kq!

export ElPhData
# export apply_gauge_matrix!
export epdata_set_g2!
export epdata_set_mmat!
export epdata_compute_eph_dipole!

# TODO: Rename to ElPhState?

# Energy and matrix elements at a single k and q point
@kwdef mutable struct ElPhData{T <: Real}
    nw::Int # Number of Wannier functions
    nmodes::Int # Number of modes
    nband::Int # Maximum number of bands inside the window
    wtk::T # Weight of the k point
    wtq::T # Weight of the q point

    # Electron states
    el_k::ElectronState{T} # electron state at k
    el_kq::ElectronState{T} # electron state at k+q

    # Phonon state
    ph::PhononState{T} # phonon state at q

    # U(k+q)' * U(k)
    mmat::Matrix{Complex{T}} = zeros(Complex{T}, nband, nband)

    # Electron-phonon coupling
    ep::Array{Complex{T}, 3} = zeros(Complex{T}, nband, nband, nmodes)
    g2::Array{T, 3} = zeros(T, nband, nband, nmodes)

    # Preallocated buffers
    buffer::Matrix{Complex{T}} = zeros(Complex{T}, nw, nw)
    buffer2::Array{T, 3} = zeros(T, nband, nband, nmodes)
end

function ElPhData{T}(nw, nmodes, nband=nw) where {T}
    @assert nw > 0
    @assert nmodes > 0
    @assert nband > 0

    ElPhData{T}(nw=nw, nmodes=nmodes, nband=nband, wtk=T(0), wtq=T(0),
        el_k=ElectronState{T}(nw, nband),
        el_kq=ElectronState{T}(nw, nband),
        ph=PhononState(nmodes, T),
    )
end

ElPhData(nw, nmodes, ::Type{FT}=Float64; nband=nw) where FT = ElPhData{FT}(nw, nmodes, nband)

# """
#     apply_gauge_matrix!(op_h, op_w, epdata, left, right, ndim=1)

# Compute op_h = uleft' * op_w * uright
# left, right are "k" or "k+q".

# ndim: Optional. Third dimension of op_h and op_w. Loop over i=1:ndim.
# """
# @timing "gauge" function apply_gauge_matrix!(op_h, op_w, epdata, left, right, ndim=1)
#     @warn "apply_gauge_matrix! is deprecated"
#     @assert size(op_h, 3) == ndim
#     @assert size(op_w, 3) == ndim
#     offset = epdata.nband_ignore

#     # TODO: Implement range
#     if left != "k" && left != "k+q"
#         error("left must be k or k+q, not $left")
#     end
#     if right != "k" && right != "k+q"
#         error("right must be k or k+q, not $right")
#     end
#     rngleft = (left == "k") ? epdata.el_k.rng : epdata.el_kq.rng
#     rngright = (right == "k") ? epdata.el_k.rng : epdata.el_kq.rng
#     uleft = (left == "k") ? epdata.uk_full : epdata.ukq_full
#     uright = (right == "k") ? epdata.uk_full : epdata.ukq_full
#     @views uleft_adj = uleft[:, rngleft .+ offset]'
#     @views uright = uright[:, rngright .+ offset]
#     @views tmp = epdata.buffer[:, rngright]

#     if length(size(op_w)) == 2
#         @views mul!(tmp, op_w, uright)
#         @views mul!(op_h[rngleft, rngright], uleft_adj, tmp)
#     elseif length(size(op_w)) == 3
#         @views @inbounds for i = 1:ndim
#             mul!(tmp, op_w[:,:,i], uright)
#             mul!(op_h[rngleft, rngright, i], uleft_adj, tmp)
#         end
#     end
# end

" Set epdata.g2[:, :, imode] = |epdata.ep[:, :, imode]|^2 / (2 omega)"
function epdata_set_g2!(epdata)
    rngk = epdata.el_k.rng
    rngkq = epdata.el_kq.rng
    for imode in 1:epdata.nmodes
        # The lower bound for phonon frequency is not set here. If ω is close to 0, g2 may
        # be very large. This should be handled when calculating physical quantities.
        ω = epdata.ph.e[imode]
        inv_2ω = 1 / (2 * ω)
        @views epdata.g2[rngkq, rngk, imode] .= (abs2.(epdata.ep[rngkq, rngk, imode]) .* inv_2ω)
    end
end

"Set mmat = ukq' * uk"
@timing "setmmat" function epdata_set_mmat!(epdata)
    rngk = epdata.el_k.rng
    rngkq = epdata.el_kq.rng
    epdata.mmat .= 0
    @views mul!(epdata.mmat[rngkq, rngk], epdata.el_kq.u', epdata.el_k.u)
end

# Define wrappers of WanToBloch functions

"""
    get_eph_Rq_to_kq!(epdata::ElPhData, epobj_eRpq, xk, fourier_mode="normal")
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
"""
function get_eph_Rq_to_kq!(epdata::ElPhData, epobj_eRpq, xk, fourier_mode="normal")
    @views ep_kq = epdata.ep[epdata.el_kq.rng, epdata.el_k.rng, :]
    get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, epdata.el_k.u, epdata.el_kq.u, fourier_mode)
end

"""
    get_eph_kR_to_kq!(epdata::ElPhData, epobj_ekpR, xq, fourier_mode="normal")
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
"""
function get_eph_kR_to_kq!(epdata::ElPhData, epobj_ekpR, xq, fourier_mode="normal")
    @views ep_kq = epdata.ep[epdata.el_kq.rng, epdata.el_k.rng, :]
    get_eph_kR_to_kq!(ep_kq, epobj_ekpR, xq, epdata.ph.u, epdata.el_kq.u, fourier_mode)
end

"""
    epdata_compute_eph_dipole!(epdata::ElPhData, polar::Polar, sign=1)
Compute electron-phonon coupling matrix elements using pre-computed `ph.eph_dipole_coeff` and `mmat`.
"""
function epdata_compute_eph_dipole!(epdata::ElPhData, sign=1)
    coeff = epdata.ph.eph_dipole_coeff
    @views @inbounds for imode = 1:epdata.nmodes
        epdata.ep[:, :, imode] .+= (sign * coeff[imode]) .* epdata.mmat
    end
end


"""
    epdata_g2_degenerate_average!(epdata::ElPhData)
Avearage g2 over degenerate bands of el_k and el_kq
"""
function epdata_g2_degenerate_average!(epdata::ElPhData{FT}) where {FT}
    el_k = epdata.el_k
    el_kq = epdata.el_kq
    g2_avg = epdata.buffer2

    # average over bands at k
    g2_avg .= 0
    @views for ib in el_k.rng
        ndegen = 0
        for jb in el_k.rng
            if abs(el_k.e[ib] - el_k.e[jb]) <= electron_degen_cutoff
                g2_avg[el_kq.rng, ib, :] .+= epdata.g2[el_kq.rng, jb, :]
                ndegen += 1
            end
        end
        g2_avg[:, ib, :] ./= ndegen
    end
    epdata.g2 .= g2_avg

    # average over bands at k+q
    g2_avg .= 0
    @views for ib in el_kq.rng
        ndegen = 0
        for jb in el_kq.rng
            if abs(el_kq.e[ib] - el_kq.e[jb]) <= electron_degen_cutoff
                g2_avg[ib, el_k.rng, :] .+= epdata.g2[jb, el_k.rng, :]
                ndegen += 1
            end
        end
        g2_avg[ib, :, :] ./= ndegen
    end
    epdata.g2 .= g2_avg
    nothing
end
