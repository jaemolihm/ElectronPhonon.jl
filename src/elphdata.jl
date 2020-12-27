
# For computing electron-phonon coupling at fine a k and q point

import Base.@kwdef

export ElPhData
export initialize_elphdata
export apply_gauge_matrix!
export set_g2!

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
    ep::Array{Complex{T}, 3}
    g2::Array{Complex{T}, 3}

    # Preallocated buffer of size nw * nw.
    buffer::Matrix{Complex{T}}

    # TODO: Implement window
    # rngbandk, rngbandkq::UnitRange{Int}
end

function ElPhData(T, nw, nmodes, nband=nothing)
    if nband === nothing
        nband = nw
    end

    ElPhData(nw=nw, nmodes=nmodes, nband=nband, wtk=T(0), wtq=T(0),
        omega=Vector{T}(undef, nmodes),
        ek_full=Vector{T}(undef, nw),
        ekq_full=Vector{T}(undef, nw),
        uk_full=Matrix{Complex{T}}(undef, nw, nw),
        ukq_full=Matrix{Complex{T}}(undef, nw, nw),
        ep=Array{Complex{T}, 3}(undef, nband, nband, nmodes),
        g2=Array{Complex{T}, 3}(undef, nband, nband, nmodes),
        buffer=Matrix{Complex{T}}(undef, nw, nw)
    )
end

function apply_gauge_matrix!(op_h, op_w, epdata, left, right, ndim=1)
    """
    Compute op_h = Adjoint(uleft) * op_w * uright
    left, right are k or k+q.

    Optional input ndim: third dimension of op_h and op_w. Loop over i=1:ndim.
    """
    @assert size(op_h, 3) == ndim
    @assert size(op_w, 3) == ndim

    # TODO: Implement range
    if left != "k" && left != "k+q"
        error("left must be k or k+q, not $left")
    end
    if right != "k" && right != "k+q"
        error("right must be k or k+q, not $right")
    end
    uleft = left == "k" ? epdata.uk_full : epdata.ukq_full
    uright = right == "k" ? epdata.uk_full : epdata.ukq_full
    uleft_adj = Adjoint(uleft)
    tmp = epdata.buffer

    if length(size(op_w)) == 2
        mul!(tmp, uleft_adj, op_w)
        mul!(op_h, tmp, uright)
    elseif length(size(op_w)) == 3
        @views @inbounds for i = 1:ndim
            mul!(tmp, uleft_adj, op_w[:,:,i])
            mul!(op_h[:,:,i], tmp, uright)
        end
    end
end

" Set epdata.g2[:, :, imode] = |epdata.ep[:, :, imode]|^2 / (2 omega)
g2 is set to 0.0 if omega < omega_acoustic."
function set_g2!(epdata)
    for imode in 1:epdata.nmodes
        omega = epdata.omega[imode]
        if (omega < omega_acoustic)
            epdata.g2[:, :, imode] .= 0
            continue
        end
        inv_2omega = 1 / (2 * omega)
        @views epdata.g2[:, :, imode] .= abs2.(epdata.ep[:, :, imode]) .* inv_2omega
    end
end
