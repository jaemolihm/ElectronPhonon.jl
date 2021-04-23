
# For computing electron-phonon coupling at fine a k and q point

import Base.@kwdef
import EPW.WanToBloch: get_eph_Rq_to_kq!, get_eph_kR_to_kq!

export ElPhData
# export apply_gauge_matrix!
export epdata_set_g2!
export epdata_set_mmat!

# TODO: Rename to ElPhState?

# Energy and matrix elements at a single k and q point
@kwdef mutable struct ElPhData{T <: Real}
    nw::Int # Number of Wannier functions
    nmodes::Int # Number of modes
    nband::Int # Maximum number of bands inside the window
    wtk::T # Weight of the k point
    wtq::T # Weight of the q point
    mmat::Matrix{Complex{T}} # U(k+q)' * U(k)

    # Electron states
    el_k::ElectronState{T} # electron state at k
    el_kq::ElectronState{T} # electron state at k+q

    # Phonon state
    ph::PhononState{T} # phonon state at q

    # Electron-phonon coupling
    ep::Array{Complex{T}, 3}
    g2::Array{T, 3}

    # Preallocated buffer of size nw * nw.
    buffer::Matrix{Complex{T}}

    # Electron energy window.
    nband_ignore::Int
end

function ElPhData(T, nw, nmodes, nband=nw, nband_ignore=0)
    @assert nband > 0
    @assert nband_ignore >= 0
    @assert nband + nband_ignore <= nw

    ElPhData(nw=nw, nmodes=nmodes, nband=nband, wtk=T(0), wtq=T(0),
        el_k=ElectronState(T, nw, nband, nband_ignore),
        el_kq=ElectronState(T, nw, nband, nband_ignore),
        ph=PhononState(T, nmodes),
        mmat=Matrix{Complex{T}}(undef, nband, nband),
        ep=Array{Complex{T}, 3}(undef, nband, nband, nmodes),
        g2=Array{T, 3}(undef, nband, nband, nmodes),
        buffer=Matrix{Complex{T}}(undef, nw, nw),
        nband_ignore=nband_ignore,
    )
end

# """
#     apply_gauge_matrix!(op_h, op_w, epdata, left, right, ndim=1)

# Compute op_h = Adjoint(uleft) * op_w * uright
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
#     @views uleft_adj = Adjoint(uleft[:, rngleft .+ offset])
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

" Set epdata.g2[:, :, imode] = |epdata.ep[:, :, imode]|^2 / (2 omega)
g2 is set to 0.0 if omega < omega_acoustic."
@timing "setg2" function epdata_set_g2!(epdata)
    rngk = epdata.el_k.rng
    rngkq = epdata.el_kq.rng
    for imode in 1:epdata.nmodes
        omega = epdata.ph.e[imode]
        if (omega < omega_acoustic)
            epdata.g2[:, :, imode] .= 0
            continue
        end
        inv_2omega = 1 / (2 * omega)
        @views epdata.g2[rngkq, rngk, imode] .= (
            abs2.(epdata.ep[rngkq, rngk, imode]) .* inv_2omega)
    end
end

"Set mmat = ukq' * uk"
@timing "setmmat" function epdata_set_mmat!(epdata)
    rngk = epdata.el_k.rng
    rngkq = epdata.el_kq.rng
    uk = get_u(epdata.el_k)
    ukq = get_u(epdata.el_kq)
    epdata.mmat .= 0
    @views mul!(epdata.mmat[rngkq, rngk], Adjoint(ukq), uk)
end

# Define wrappers of WanToBloch functions

"""
    get_eph_Rq_to_kq!(epdata::ElPhData, epobj_eRpq, xk, fourier_mode="normal")
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
"""
function get_eph_Rq_to_kq!(epdata::ElPhData, epobj_eRpq, xk, fourier_mode="normal")
    uk = get_u(epdata.el_k)
    ukq = get_u(epdata.el_kq)
    @views ep_kq = epdata.ep[epdata.el_kq.rng, epdata.el_k.rng, :]
    get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, uk, ukq, fourier_mode)
end

"""
    get_eph_kR_to_kq!(epdata::ElPhData, epobj_ekpR, xq, fourier_mode="normal")
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
"""
function get_eph_kR_to_kq!(epdata::ElPhData, epobj_ekpR, xq, fourier_mode="normal")
    ukq = get_u(epdata.el_kq)
    @views ep_kq = epdata.ep[epdata.el_kq.rng, epdata.el_k.rng, :]
    get_eph_kR_to_kq!(ep_kq, epobj_ekpR, xq, epdata.ph.u, ukq, fourier_mode)
end
