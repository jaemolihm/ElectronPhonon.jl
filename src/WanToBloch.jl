"""
Module for transformation from Wannier basis to Bloch eigenstate basis.
Use thread-save preallocated buffers.
"""
module WanToBloch

using LinearAlgebra
using EPW: @timing
using EPW: AbstractWannierObject, WannierObject
using EPW: get_fourier!, update_op_r!, reset_gridopts_in_obj!
using EPW: solve_eigen_el!, solve_eigen_el_valueonly!, solve_eigen_ph!, solve_eigen_ph_valueonly!
using EPW: dynmat_dipole!

export get_el_eigen!
export get_el_eigen_valueonly!
export get_el_velocity_diag!
export get_el_velocity!
export get_ph_eigen!
export get_ph_eigen_valueonly!
export get_ph_velocity_diag!
export get_eph_RR_to_Rq!
export get_eph_Rq_to_kq!
export get_eph_RR_to_kR!
export get_eph_kR_to_kq!

# TODO: Allow the type to change.
# Preallocated buffers for temporary arrays. Access via _get_buffer. When multiple arrays
# are needed, use _buffer1 for the largest array, _buffer2 for the next largest, and so on.
const _buffer_nothreads1 = [Vector{ComplexF64}(undef, 0)]
const _buffer1 = [Vector{ComplexF64}(undef, 0)]
const _buffer2 = [Vector{ComplexF64}(undef, 0)]

function __init__()
    Threads.resize_nthreads!(_buffer1)
    Threads.resize_nthreads!(_buffer2)
end

"""
    _get_buffer(buffer::Vector{Vector{T, N}}, dims::NTuple{N, Int}) where {T, N}
Get preallocated buffer as a ReshapedArray in a thread-safe way.
Resize buffer if the allocated size is smaller than the requested size"""
function _get_buffer(buffer::Vector{Vector{T}}, dims::NTuple{N, Int}) where {T, N}
    tid = Threads.threadid()
    if length(buffer[tid]) < prod(dims)
        resize!(buffer[tid], prod(dims))
    end
    Base.ReshapedArray(buffer[tid], dims, ())
end

# =============================================================================
#  Electrons

"""
    get_el_eigen!(values, vectors, nw, el_ham, xk, fourier_mode="normal")
Compute electron eigenenergy and eigenvector.
"""
@timing "w2b_el_eig" function get_el_eigen!(values, vectors, nw, el_ham, xk, fourier_mode="normal")
    @assert size(values) == (nw,)
    @assert size(vectors) == (nw, nw)

    hk = _get_buffer(_buffer1, (nw, nw))
    get_fourier!(hk, el_ham, xk, mode=fourier_mode)
    solve_eigen_el!(values, vectors, hk)
    nothing
end

"""
    get_el_eigen_valueonly!(values, nw, el_ham, xk, fourier_mode="normal")
"""
@timing "w2b_el_eigval" function get_el_eigen_valueonly!(values, nw, el_ham, xk, fourier_mode="normal")
    # FIXME: Names get_el_eigen_valueonly! and solve_eigen_el_valueonly! are confusing.
    @assert size(values) == (nw,)

    hk = _get_buffer(_buffer1, (nw, nw))
    get_fourier!(hk, el_ham, xk, mode=fourier_mode)
    solve_eigen_el_valueonly!(values, hk)
    nothing
end

"""
    get_el_velocity_diag_berry_connection!(velocity_diag, nw, el_ham_R, xk, uk, fourier_mode="normal")
Compute the diagoanl part of electron band velocity using the Berry connection formula. See
docstring for get_el_velocity_berry_connection! for details.
For the diagonal part, the Berry connection contribution is zero.

velocity_diag: nband-dimensional vector.
uk: nw * nband matrix containing nband eigenvectors of H(k).
"""
@timing "w2b_el_vel" function get_el_velocity_diag_berry_connection!(velocity_diag, nw, el_ham_R, xk, uk, fourier_mode="normal")
    @assert size(uk, 1) == nw
    nband = size(uk, 2)
    @assert size(velocity_diag) == (3, nband)

    vk = _get_buffer(_buffer1, (nw, nw, 3))
    tmp = _get_buffer(_buffer2, (nw, nband))

    get_fourier!(vk, el_ham_R, xk, mode=fourier_mode)

    # velocity_diag[idir, iband] = uk'[iband, :] * vk[:, :, idir] * uk[:, iband]
    @views @inbounds for idir = 1:3
        mul!(tmp, vk[:, :, idir], uk)
        @views @inbounds for iband in 1:nband
            velocity_diag[idir, iband] = real(dot(uk[:, iband], tmp[:, iband]))
        end
    end
    nothing
end

"""
    get_el_velocity_berry_connection!(velocity, nw, el_ham_R, el_pos, ek, xk, uk, fourier_mode="normal")
Compute electron band velocity using the Berry connection formula:
``v_{m,n} = (U^\\dagger dH^{(W)}(k) / dk U)_{m,n} + i * (e_m - e_n) * (U^\\dagger A U)_{m,n}``
(Eq. (31) of X. Wang et al, PRB 74 195118 (2006)).

- `velocity``: (3, `nband`, `nband`) matrix
- `uk`: `nw` * `nband` matrix containing nband eigenvectors of ``H(k)``.
"""
@timing "w2b_el_vel" function get_el_velocity_berry_connection!(velocity, nw, el_ham_R, el_pos, ek, xk, uk, fourier_mode="normal")
    @assert size(uk, 1) == nw
    nband = size(uk, 2)
    @assert size(velocity) == (3, nband, nband)

    vk = _get_buffer(_buffer1, (nw, nw, 3))
    tmp = _get_buffer(_buffer2, (nw, nband))

    get_fourier!(vk, el_ham_R, xk, mode=fourier_mode)

    # velocity[idir, :, :] = uk' * vk[:, :, idir] * uk
    @views @inbounds for idir = 1:3
        mul!(tmp, vk[:, :, idir], uk)
        mul!(velocity[idir, :, :], uk', tmp)
    end

    # Add the Berry connection contribution
    # vk = A^{(W)} (Berry connection in Wannier basis)
    # tmp = vk * uk
    get_fourier!(vk, el_pos, xk, mode=fourier_mode)
    @views @inbounds for idir = 1:3
        mul!(tmp, vk[:, :, idir], uk)
        for jb = 1:nband, ib = 1:nband
            velocity[idir, ib, jb] += im * (ek[ib] - ek[jb]) * dot(uk[:, ib], tmp[:, jb])
        end
    end

    nothing
end

"""
    get_el_velocity_direct!(velocity, nw, el_vel, xk, uk, fourier_mode="normal")
Compute electron band velocity by direct Wannier interpolation of ``dH/dk`` matrix elements.
- `el_vel`: a `WannierObject`, containing matrix elements of ``dH/dk`` in real-space Wannier basis.
- `velocity``: (3, `nband`, `nband`) matrix
- `uk`: `nw` * `nband` matrix containing nband eigenvectors of ``H(k)``.

TODO: Can we reduce code duplication with get_el_velocity_berry_connection?
"""
@timing "w2b_el_vel" function get_el_velocity_direct!(velocity, nw, el_vel, xk, uk, fourier_mode="normal")
    @assert size(uk, 1) == nw
    nband = size(uk, 2)
    @assert size(velocity) == (3, nband, nband)

    vk = _get_buffer(_buffer1, (nw, nw, 3))
    tmp = _get_buffer(_buffer2, (nw, nband))

    get_fourier!(vk, el_vel, xk, mode=fourier_mode)

    # velocity[idir, :, :] = uk' * vk[:, :, idir] * uk
    @views @inbounds for idir = 1:3
        mul!(tmp, vk[:, :, idir], uk)
        mul!(velocity[idir, :, :], uk', tmp)
    end
    nothing
end


# =============================================================================
#  Phonons
# FIXME: fourier_mode vs mode, keyword or positional

"""
    get_ph_eigen!(values, vectors, ph_dyn, mass, xq, polar=nothing; fourier_mode="normal")
Compute electron eigenenergy and eigenvector.
"""
@timing "w2b_ph_eig" function get_ph_eigen!(values, vectors, ph_dyn, mass, xq, polar=nothing; fourier_mode="normal")
    nmodes = length(values)
    @assert size(vectors) == (nmodes, nmodes)
    @assert size(mass) == (nmodes,)
    @assert ph_dyn.ndata == nmodes^2

    # dynq = _get_buffer(_buffer_ph_eigen, (nmodes, nmodes))
    # Use vectors as a temporary storage for the dynamical matrix
    dynq = vectors

    get_fourier!(dynq, ph_dyn, xq, mode=fourier_mode)
    if ! isnothing(polar)
        dynmat_dipole!(dynq, xq, polar, 1)
    end
    @inbounds for j=1:nmodes, i=1:nmodes
        dynq[i, j] /= sqrt(mass[i])
        dynq[i, j] /= sqrt(mass[j])
    end
    solve_eigen_ph!(values, vectors, dynq, mass)
    nothing
end

"""
    get_el_eigen_valueonly!(values, nw, el_ham, xk, fourier_mode="normal")
"""
@timing "w2b_ph_eigval" function get_ph_eigen_valueonly!(values, ph_dyn, mass, xq, polar=nothing, fourier_mode="normal")
    nmodes = length(values)
    @assert size(mass) == (nmodes,)
    @assert ph_dyn.ndata == nmodes^2

    dynq = _get_buffer(_buffer1, (nmodes, nmodes))

    get_fourier!(dynq, ph_dyn, xq, mode=fourier_mode)
    if ! isnothing(polar)
        dynmat_dipole!(dynq, xq, polar, 1)
    end
    @inbounds for j=1:nmodes, i=1:nmodes
        dynq[i, j] /= sqrt(mass[i])
        dynq[i, j] /= sqrt(mass[j])
    end
    solve_eigen_ph_valueonly!(values, dynq)
    nothing
end


"""
    get_ph_velocity_diag!(vel_diag, nw, el_ham_R, xk, uk, fourier_mode="normal")
Compute electron band velocity, only the band-diagonal part.
# Outputs
- `vel_diag`: (3, nmodes) array, contains diagonal band velocity.
# Inputs
- `uk`: nmodes * nmodes matrix containing phonon eigenvectors.
"""
@timing "w2b_ph_vel" function get_ph_velocity_diag!(vel_diag, ph_dyn_R, xk, uk, fourier_mode="normal")
    # FIXME: Polar is not implemented.
    nmodes = size(uk, 1)
    @assert size(uk) == (nmodes, nmodes)
    @assert size(vel_diag) == (3, nmodes)

    vk = _get_buffer(_buffer1, (nmodes, nmodes, 3))
    tmp = _get_buffer(_buffer2, (nmodes, nmodes))

    get_fourier!(vk, ph_dyn_R, xk, mode=fourier_mode)

    # vel_diag[idir, i] = uk'[i, :] * vk[:, :, idir] * uk[:, i]
    @views @inbounds for idir = 1:3
        # The sqrt(mass) factors are already included in the eigenvectors
        mul!(tmp, vk[:, :, idir], uk)
        for i in 1:nmodes
            vel_diag[idir, i] = real(dot(uk[:, i], tmp[:, i]))
        end
    end
    nothing
end

# =============================================================================
#  Electron-phonon coupling

"""
`get_eph_RR_to_Rq!(epobj_eRpq::WannierObject{T}, epmat::AbstractWannierObject{T},
xq, u_ph, fourier_mode="normal") where {T}`

Compute electron-phonon coupling matrix in electron Wannier, phonon Bloch basis.
Multithreading is not supported because of large buffer array size.

# Arguments
- `epobj_eRpq`: Output. E-ph matrix in electron Wannier, phonon Bloch basis.
    Must be initialized before calling. Only the op_r field is modified.
- `epmat`: Input. E-ph matrix in electron Wannier, phonon Wannier basis.
- `xq`: Input. q point vector.
- `u_ph`: Input. nmodes * nmodes matrix containing phonon eigenvectors.
"""
@timing "w2b_eph_RRtoRq" function get_eph_RR_to_Rq!(epobj_eRpq::WannierObject{T},
        epmat::AbstractWannierObject{T}, xq, u_ph, fourier_mode="normal") where {T}
    nr_el = epobj_eRpq.nr
    nmodes = size(u_ph, 1)
    nbasis = div(epobj_eRpq.ndata, nmodes) # Number of electron basis squared.
    @assert size(u_ph) == (nmodes, nmodes)
    @assert Threads.threadid() == 1
    @assert epobj_eRpq.ndata == nbasis * nmodes
    @assert epmat.ndata == nbasis * nmodes * nr_el

    ep_Rq = _get_buffer(_buffer_nothreads1, (nbasis, nmodes, nr_el))
    ep_Rq_tmp = _get_buffer(_buffer1, (nbasis, nmodes))

    get_fourier!(ep_Rq, epmat, xq, mode=fourier_mode)

    # Transform from phonon Cartesian to eigenmode basis, one ir_el at a time.
    for ir in 1:nr_el
        ep_Rq_tmp .= 0
        @views mul!(ep_Rq_tmp, ep_Rq[:, :, ir], u_ph)
        ep_Rq[:, :, ir] .= ep_Rq_tmp
    end
    update_op_r!(epobj_eRpq, ep_Rq)
    nothing
end

"""
    get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, uk, ukq, fourier_mode="normal")
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.

# Arguments
- `ep_kq`: Output. E-ph matrix in electron and phonon Bloch basis.
- `epobj_eRpq`: Input. AbstractWannierObject. E-ph matrix in electron Wannier,
    phonon Bloch basis.
- `xk`: Input. k point vector.
- `uk`, `ukq`: Input. Electron eigenstate at k and k+q, respectively.
"""
@timing "w2b_eph_Rqtokq" function get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, uk, ukq, fourier_mode="normal")
    nbandkq, nbandk, nmodes = size(ep_kq)
    @assert size(uk, 2) == nbandk
    @assert size(ukq, 2) == nbandkq
    @assert epobj_eRpq.ndata == size(ukq, 1) * size(uk, 1) * nmodes

    ep_kq_wan = _get_buffer(_buffer1, (size(ukq, 1), size(uk, 1), nmodes))
    tmp = _get_buffer(_buffer2, (size(ukq, 1), nbandk))

    get_fourier!(ep_kq_wan, epobj_eRpq, xk, mode=fourier_mode)

    # Rotate e-ph matrix from electron Wannier to eigenstate basis
    # ep_kq[ibkq, ibk, imode] = ukq'[ibkq, :] * ep_kq_wan[:, :, imode] * uk[:, ibk]
    @views @inbounds for imode = 1:nmodes
        mul!(tmp, ep_kq_wan[:, :, imode], uk)
        mul!(ep_kq[:, :, imode], ukq', tmp)
    end
    nothing
end

"""
`get_eph_RR_to_kR!(epobj_eRpq::WannierObject{T}, epmat::AbstractWannierObject{T},
    xk, uk, fourier_mode="normal") where {T}`

Compute electron-phonon coupling matrix in electron Bloch, phonon Wannier basis.
Multithreading is not supported because of large buffer array size.

# Arguments
- `epobj_ekpR`: Output. E-ph matrix in electron Wannier, phonon Bloch basis.
    Must be initialized before calling. Only the op_r field is modified.
- `epmat`: Input. E-ph matrix in electron Wannier, phonon Wannier basis.
- `xk`: Input. k point vector.
- `uk`: Input. nw * nw matrix containing electron eigenvectors at k.
"""
@timing "w2b_eph_RRtokR" function get_eph_RR_to_kR!(epobj_ekpR::WannierObject{T},
        epmat::AbstractWannierObject{T}, xk, uk, fourier_mode="normal") where {T}
    """
    size(uk) = (nw, nband)
    size(epobj_ekpR.op_r) = (nw * nband_bound * nmodes, nr_ep)
    size(epmat.op_r) = (nw^2 * nmodes * nr_ep, nr_el)
    """
    nr_ep = length(epmat.irvec_next)
    nw, nband = size(uk)
    nmodes = div(epmat.ndata, nw^2 * nr_ep)
    ndata = nw * nband * nmodes
    @assert epobj_ekpR.nr == nr_ep
    @assert size(epobj_ekpR.op_r, 1) >= ndata
    @assert Threads.threadid() == 1

    ep_kR = _get_buffer(_buffer_nothreads1, (nw, nw, nmodes, nr_ep))
    ep_kR2 = _get_buffer(_buffer1, (nw, nband, nmodes))
    ep_kR_tmp = _get_buffer(_buffer2, (nw, nband))

    get_fourier!(ep_kR, epmat, xk, mode=fourier_mode)

    # Transform from electron Wannier to eigenmode basis, one ir_el and modes at a time.
    for ir in 1:nr_ep
        @views @inbounds for imode in 1:nmodes
            mul!(ep_kR_tmp, ep_kR[:, :, imode, ir], uk)
            ep_kR2[:, :, imode] .= ep_kR_tmp
        end
        epobj_ekpR.op_r[1:ndata, ir] .= Base.ReshapedArray(ep_kR2, (ndata,), ())
    end
    epobj_ekpR.ndata = ndata
    reset_gridopts_in_obj!(epobj_ekpR)
    nothing
end

"""
    get_eph_kR_to_kq!(ep_kq, epobj_ekpR, xk, u_ph, ukq, fourier_mode="normal")
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
The electron state at k should be already in the eigenstate basis in epobj_ekpR.

# Arguments
- `ep_kq`: Output. E-ph matrix in electron and phonon Bloch basis.
- `epobj_ekpR`: Input. AbstractWannierObject. E-ph matrix in electron Wannier,
    phonon Bloch basis.
- `xq`: Input. q point vector.
- `u_ph`: Input. nmodes * nmodes matrix containing phonon eigenvectors.
- `ukq`: Input. Electron eigenstate at k+q.
- `rngk`, `rngkq`: Input. Range of electron states inside the window.
"""
@timing "w2b_eph_kRtokq" function get_eph_kR_to_kq!(ep_kq, epobj_ekpR, xq, u_ph, ukq,
        fourier_mode="normal")
    """
    size(ep_kq) = (nbandkq, nbandk, nmodes)
    size(epobj_ekpR.op_r) = (nw * nband_bound * nmodes, nr_ep)
    epobj_ekpR.ndata = nw * nbandk * nmodes
    size(ukq) = (nw, nbandkq)
    size(u_ph) = (nmodes, nmodes)
    """
    nbandkq, nbandk, nmodes = size(ep_kq)
    nw = size(ukq, 1)
    @assert size(u_ph) == (nmodes, nmodes)
    @assert size(ukq, 2) == nbandkq
    @assert epobj_ekpR.ndata == nw * nbandk * nmodes

    ep_kq_wan = _get_buffer(_buffer1, (nw, nbandk, nmodes))
    tmp = _get_buffer(_buffer2, (nbandkq, nbandk))

    get_fourier!(ep_kq_wan, epobj_ekpR, xq, mode=fourier_mode)

    # Transform from phonon Cartesian to eigenmode basis and from electron Wannier at k+q
    # to eigenstate basis. The electron at k is already in eigenstate basis.
    # ep_kq[ibkq, :, imode] = ukq'[ibkq, iw] * ep_kq_wan[iw, :, jmode] * u_ph[jmode, imode]
    ep_kq .= 0
    @timing "rotate" begin
        for jmode = 1:nmodes
            @views mul!(tmp, ukq', ep_kq_wan[:, :, jmode])
            @views @inbounds for imode in 1:nmodes
                ep_kq[:, :, imode] .+= tmp .* u_ph[jmode, imode]
            end
            # The tullio version is slightly (~10%) faster. Not added now to avoid adding
            # Tullio as dependency, but may added later.
            # @tullio threads=false ep_kq[ib, jb, imode] += tmp[ib, jb] * u_ph[$jmode, imode]
        end
    end
    nothing
end

end
