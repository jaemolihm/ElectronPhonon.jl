
# For computing electron-phonon coupling at fine a k and q point

import Base.@kwdef
import EPW.WanToBloch: get_el_eigen!, get_el_velocity_diag!, get_eph_Rq_to_kq!

export ElPhData
export initialize_elphdata
export apply_gauge_matrix!
export epdata_set_g2!
export epdata_set_window!

export get_el_eigen!
export get_el_velocity_diag!

# Energy and matrix elements at a single k and q point
@kwdef mutable struct ElPhData{T <: Real}
    nw::Int
    nmodes::Int
    nband::Int # Maximum number of bands inside the window
    wtk::T # Weight of the k point
    wtq::T # Weight of the q point
    omega::Vector{T}
    ek_full::Vector{T}
    ekq_full::Vector{T}
    uk_full::Matrix{Complex{T}}
    ukq_full::Matrix{Complex{T}}
    vdiagk::Matrix{T} # Diagonal component of band velocity at k
    vdiagkq::Matrix{T} # Diagonal component of band velocity at k+q
    bmat::Matrix{Complex{T}} # U(k+q)' * U(k)

    # Electron-phonon coupling
    ep::Array{Complex{T}, 3}
    g2::Array{Complex{T}, 3}

    # Preallocated buffer of size nw * nw.
    buffer::Matrix{Complex{T}}

    # Electron energy window.
    # Applying to full array: arr_full[rng .+ iband_offset]
    # Applying to filtered array: arr[rng]
    iband_offset::Int
    nbandk::Int # Number of bands inside the energy window
    nbandkq::Int # Number of bands inside the energy window
    rngk::UnitRange{Int} # Index of bands inside the energy window
    rngkq::UnitRange{Int} # Index of bands inside the energy window
    ek::Vector{T}
    ekq::Vector{T}
end

function ElPhData(T, nw, nmodes, nband=nothing)
    @assert nband > 0
    @assert nband <= nw
    if nband === nothing
        nband = nw
    end

    ElPhData(nw=nw, nmodes=nmodes, nband=nband, wtk=T(0), wtq=T(0),
        omega=Vector{T}(undef, nmodes),
        ek_full=Vector{T}(undef, nw),
        ekq_full=Vector{T}(undef, nw),
        uk_full=Matrix{Complex{T}}(undef, nw, nw),
        ukq_full=Matrix{Complex{T}}(undef, nw, nw),
        vdiagk=zeros(T, (3, nband)),
        vdiagkq=zeros(T, (3, nband)),
        bmat=Matrix{Complex{T}}(undef, nband, nband),
        ep=Array{Complex{T}, 3}(undef, nband, nband, nmodes),
        g2=Array{Complex{T}, 3}(undef, nband, nband, nmodes),
        buffer=Matrix{Complex{T}}(undef, nw, nw),
        iband_offset=0,
        nbandk=nband,
        nbandkq=nband,
        rngk=1:nband,
        rngkq=1:nband,
        ek=Vector{T}(undef, nband),
        ekq=Vector{T}(undef, nband),
    )
end

"""
    apply_gauge_matrix!(op_h, op_w, epdata, left, right, ndim=1)

Compute op_h = Adjoint(uleft) * op_w * uright
left, right are "k" or "k+q".

ndim: Optional. Third dimension of op_h and op_w. Loop over i=1:ndim.
"""
@timing "gauge" function apply_gauge_matrix!(op_h, op_w, epdata, left, right, ndim=1)
    @warn "apply_gauge_matrix! is deprecated"
    @assert size(op_h, 3) == ndim
    @assert size(op_w, 3) == ndim
    offset = epdata.iband_offset

    # TODO: Implement range
    if left != "k" && left != "k+q"
        error("left must be k or k+q, not $left")
    end
    if right != "k" && right != "k+q"
        error("right must be k or k+q, not $right")
    end
    rngleft = (left == "k") ? epdata.rngk : epdata.rngkq
    rngright = (right == "k") ? epdata.rngk : epdata.rngkq
    uleft = (left == "k") ? epdata.uk_full : epdata.ukq_full
    uright = (right == "k") ? epdata.uk_full : epdata.ukq_full
    @views uleft_adj = Adjoint(uleft[:, rngleft .+ offset])
    @views uright = uright[:, rngright .+ offset]
    @views tmp = epdata.buffer[:, rngright]

    if length(size(op_w)) == 2
        @views mul!(tmp, op_w, uright)
        @views mul!(op_h[rngleft, rngright], uleft_adj, tmp)
    elseif length(size(op_w)) == 3
        @views @inbounds for i = 1:ndim
            mul!(tmp, op_w[:,:,i], uright)
            mul!(op_h[rngleft, rngright, i], uleft_adj, tmp)
        end
    end
end

" Set epdata.g2[:, :, imode] = |epdata.ep[:, :, imode]|^2 / (2 omega)
g2 is set to 0.0 if omega < omega_acoustic."
@timing "setg2" function epdata_set_g2!(epdata)
    rngk = epdata.rngk
    rngkq = epdata.rngkq
    for imode in 1:epdata.nmodes
        omega = epdata.omega[imode]
        if (omega < omega_acoustic)
            epdata.g2[:, :, imode] .= 0
            continue
        end
        inv_2omega = 1 / (2 * omega)
        @views epdata.g2[rngkq, rngk, imode] .= (
            abs2.(epdata.ep[rngkq, rngk, imode]) .* inv_2omega)
    end
end

"Set bmat = ukq' * uk"
@timing "setg2" function epdata_set_bmat!(epdata)
    offset = epdata.iband_offset
    epdata.bmat .= 0
    for ibkq in epdata.rngkq
        @views for ibk in epdata.rngk
            epdata.bmat[ibkq, ibk] = dot(epdata.ukq_full[:, ibkq + offset],
                epdata.uk_full[:, ibk + offset])
        end
    end
end

function epdata_set_window!(epdata, ktype, window=(-Inf,Inf))
    offset = epdata.iband_offset
    if ktype == "k"
        ibs = EPW.inside_window(epdata.ek_full, window...)
    elseif ktype == "k+q"
        ibs = EPW.inside_window(epdata.ekq_full, window...)
    else
        error("ktype must be k or k+q, not $ktype")
    end
    # If no bands are selected, return true.
    if isempty(ibs)
        return true
    end
    if ktype == "k"
        epdata.rngk = (ibs[1]:ibs[end]) .- offset
        epdata.nbandk = length(epdata.rngk)
        @views epdata.ek[epdata.rngk] .= epdata.ek_full[epdata.rngk .+ offset]
    elseif ktype == "k+q"
        epdata.rngkq = (ibs[1]:ibs[end]) .- offset
        epdata.nbandkq = length(epdata.rngkq)
        @views epdata.ekq[epdata.rngkq] .= epdata.ekq_full[epdata.rngkq .+ offset]
    end
    return false
end

# Define wrappers of WanToBloch functions

"""
    get_el_eigen!(epdata::ElPhData, ktype::String, el_ham, xk, fourier_mode="normal")

Compute electron eigenenergy and eigenvector and save them in epdata.
ktype: "k" or "k+q"
"""
function get_el_eigen!(epdata::ElPhData, ktype::String, el_ham, xk, fourier_mode="normal")
    if ktype == "k"
        values = epdata.ek_full
        vectors = epdata.uk_full
    elseif ktype == "k+q"
        values = epdata.ekq_full
        vectors = epdata.ukq_full
    else
        error("ktype must be k or k+q, not $ktype")
    end
    get_el_eigen!(values, vectors, epdata.nw, el_ham, xk, fourier_mode)
end

"""
    get_el_velocity!(epdata::ElPhData, ktype::String, el_ham_R, xk, fourier_mode="normal")
Compute electron band velocity, only the band-diagonal part.

Compute electron eigenenergy and eigenvector and save them in epdata.
ktype: "k" or "k+q"
"""
function get_el_velocity_diag!(epdata::ElPhData, ktype::String, el_ham_R, xk, fourier_mode="normal")
    offset = epdata.iband_offset
    if ktype == "k"
        @views uk = epdata.uk_full[:, epdata.rngk .+ offset]
        @views velocity_diag = epdata.vdiagk[:, epdata.rngk]
    elseif ktype == "k+q"
        @views uk = epdata.ukq_full[:, epdata.rngkq .+ offset]
        @views velocity_diag = epdata.vdiagkq[:, epdata.rngkq]
    else
        error("ktype must be k or k+q, not $ktype")
    end
    get_el_velocity_diag!(velocity_diag, epdata.nw, el_ham_R, xk, uk, fourier_mode)
end


"""
    get_eph_Rq_to_kq!(epdata::ElPhData, epobj_eRpq, xk, fourier_mode="normal")
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
"""
function get_eph_Rq_to_kq!(epdata::ElPhData, epobj_eRpq, xk, fourier_mode="normal")
    offset = epdata.iband_offset
    @views uk = epdata.uk_full[:, epdata.rngk .+ offset]
    @views ukq = epdata.ukq_full[:, epdata.rngkq .+ offset]
    @views ep_kq = epdata.ep[epdata.rngkq, epdata.rngk, :]
    get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, uk, ukq, fourier_mode)
end
