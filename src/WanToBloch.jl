"""
Module for transformation from Wannier basis to Bloch eigenstate basis.
Use thread-save preallocated buffers.
"""
module WanToBloch

using LinearAlgebra
using EPW: AbstractWannierObject, WannierObject
using EPW: get_fourier!, update_op_r!
using EPW: solve_eigen_el!, solve_eigen_el_valueonly!, solve_eigen_ph!
using EPW: dynmat_dipole!

export get_el_eigen!
export get_el_eigen_valueonly!
export get_el_velocity_diag!
export get_ph_eigen!
export get_eph_RR_to_Rq!
export get_eph_Rq_to_kq!

# TODO: Allow the type to change.
# Preallocated buffers
const _buffer_el_eigen = [Array{ComplexF64, 2}(undef, 0, 0)]
const _buffer_el_velocity = [Array{ComplexF64, 3}(undef, 0, 0, 0)]
const _buffer_el_velocity_tmp = [Array{ComplexF64, 2}(undef, 0, 0)]
const _buffer_ph_eigen = [Array{ComplexF64, 2}(undef, 0, 0)]
const _buffer_nothreads_eph_RR_to_Rq = [Array{ComplexF64, 3}(undef, 0, 0, 0)]
const _buffer_nothreads_eph_RR_to_Rq_tmp = [Array{ComplexF64, 2}(undef, 0, 0)]
const _buffer_eph_Rq_to_kq = [Array{ComplexF64, 3}(undef, 0, 0, 0)]
const _buffer_eph_Rq_to_kq_tmp = [Array{ComplexF64, 2}(undef, 0, 0)]

function __init__()
    Threads.resize_nthreads!(_buffer_el_eigen)
    Threads.resize_nthreads!(_buffer_el_velocity)
    Threads.resize_nthreads!(_buffer_el_velocity_tmp)
    Threads.resize_nthreads!(_buffer_ph_eigen)
    Threads.resize_nthreads!(_buffer_eph_Rq_to_kq)
    Threads.resize_nthreads!(_buffer_eph_Rq_to_kq_tmp)
end

"""
    _get_buffer(buffer::Vector{Array{T, N}}, size_needed::NTuple{N, Int}) where {T, N}
Get preallocated buffer in a thread-safe way.
Resize buffer if the size is differet from the needed size"""
function _get_buffer(buffer::Vector{Array{T, N}}, size_needed::NTuple{N, Int}) where {T, N}
    tid = Threads.threadid()
    if size(buffer[tid]) != size_needed
        buffer[tid] = zeros(T, size_needed)
    end
    buffer[tid]
end

# =============================================================================
#  Electrons

"""
    get_el_eigen!(values, vectors, nw, el_ham, xk, fourier_mode="normal")
Compute electron eigenenergy and eigenvector.
"""
function get_el_eigen!(values, vectors, nw, el_ham, xk, fourier_mode="normal")
    @assert size(values) == (nw,)
    @assert size(vectors) == (nw, nw)

    hk = _get_buffer(_buffer_el_eigen, (nw, nw))

    get_fourier!(hk, el_ham, xk, mode=fourier_mode)
    values .= solve_eigen_el!(vectors, hk)
    nothing
end

"""
    get_el_eigen_valueonly!(values, nw, el_ham, xk, fourier_mode="normal")
"""
function get_el_eigen_valueonly!(values, nw, el_ham, xk, fourier_mode="normal")
    @assert size(values) == (nw,)

    hk = _get_buffer(_buffer_el_eigen, (nw, nw))

    get_fourier!(hk, el_ham, xk, mode=fourier_mode)
    values .= solve_eigen_el_valueonly!(hk)
    nothing
end

"""
    get_el_velocity!(velocity_diag, nw, el_ham_R, xk, uk, fourier_mode="normal")
Compute electron band velocity, only the band-diagonal part.

velocity_diag: nband-dimensional vector.
uk: nw * nband matrix containing nband eigenvectors of H(k).
"""
function get_el_velocity_diag!(velocity_diag, nw, el_ham_R, xk, uk, fourier_mode="normal")
    @assert size(uk, 1) == nw
    nband = size(uk, 2)
    @assert size(velocity_diag) == (3, nband)

    vk = _get_buffer(_buffer_el_velocity, (nw, nw, 3))
    tmp_full = _get_buffer(_buffer_el_velocity_tmp, (nw, nw))
    tmp = view(tmp_full, :, 1:nband)

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

# =============================================================================
#  Phonons

"""
    get_ph_eigen!(values, vectors, ph_dyn, mass, xq, polar=nothing; fourier_mode="normal")
Compute electron eigenenergy and eigenvector.
"""
function get_ph_eigen!(values, vectors, ph_dyn, mass, xq, polar=nothing; fourier_mode="normal")
    nmodes = length(values)
    @assert size(vectors) == (nmodes, nmodes)
    @assert size(mass) == (nmodes,)
    @assert ph_dyn.ndata == nmodes^2

    dynq = _get_buffer(_buffer_ph_eigen, (nmodes, nmodes))

    get_fourier!(dynq, ph_dyn, xq, mode=fourier_mode)
    if polar !== nothing
        dynmat_dipole!(dynq, xq, polar, 1)
    end
    dynq[:, :] ./= sqrt.(mass)
    dynq[:, :] ./= sqrt.(mass)'
    values .= solve_eigen_ph!(vectors, dynq, mass)
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
function get_eph_RR_to_Rq!(epobj_eRpq::WannierObject{T},
        epmat::AbstractWannierObject{T}, xq, u_ph, fourier_mode="normal") where {T}
    nr_el = epobj_eRpq.nr
    nmodes = size(u_ph, 1)
    nbasis = div(epobj_eRpq.ndata, nmodes) # Number of electron basis squared.
    @assert size(u_ph) == (nmodes, nmodes)
    @assert Threads.threadid() == 1
    @assert epobj_eRpq.ndata == nbasis * nmodes
    @assert epmat.ndata == nbasis * nmodes * nr_el

    ep_Rq = _get_buffer(_buffer_nothreads_eph_RR_to_Rq, (nbasis, nmodes, nr_el))
    ep_Rq_tmp = _get_buffer(_buffer_nothreads_eph_RR_to_Rq_tmp, (nbasis, nmodes))

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
function get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, uk, ukq, fourier_mode="normal")
    nbandkq, nbandk, nmodes = size(ep_kq)
    @assert size(uk, 2) == nbandk
    @assert size(ukq, 2) == nbandkq
    @assert epobj_eRpq.ndata == size(ukq, 1) * size(uk, 1) * nmodes

    ep_kq_wan = _get_buffer(_buffer_eph_Rq_to_kq, (size(ukq, 1), size(uk, 1), nmodes))
    tmp_full = _get_buffer(_buffer_eph_Rq_to_kq_tmp, (size(ukq, 1), size(uk, 1)))
    tmp = view(tmp_full, :, 1:nbandk)

    get_fourier!(ep_kq_wan, epobj_eRpq, xk, mode=fourier_mode)

    # Rotate e-ph matrix from electron Wannier to eigenstate basis
    # ep_kq[ibkq, ibk, imode] = ukq'[ibkq, :] * ep_kq_wan[:, :, imode] * uk[:, ibk]
    ukq_adj = Adjoint(ukq)
    @views @inbounds for imode = 1:nmodes
        mul!(tmp, ep_kq_wan[:, :, imode], uk)
        mul!(ep_kq[:, :, imode], ukq_adj, tmp)
    end
    nothing
end

end
