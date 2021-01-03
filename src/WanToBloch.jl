"""
Module for transformation from Wannier basis to Bloch eigenstate basis.
Use thread-save preallocated buffers.
"""
module WanToBloch


using LinearAlgebra
using EPW: AbstractWannierObject, WannierObject
using EPW: get_fourier!, update_op_r!
using EPW: solve_eigen_el!, solve_eigen_el_valueonly!

export get_el_eigen!
export get_el_eigen_valueonly!
export get_el_velocity_diag!
export get_eph_RR_to_Rq!

# Type for preallocated array. Did this because resize! only works for vectors.
# TODO: Is there a better way?
struct BufferArray{T, N}
    arr::Array{T, N}
end
BufferArray(T, N) = BufferArray(Array{T, N}(undef, (0 for _ in 1:N)...))

# TODO: Allow the type to change.
# Preallocated buffers
const _buffer_el_eigen = [BufferArray(ComplexF64, 2)]
const _buffer_el_velocity = [BufferArray(ComplexF64, 3)]
const _buffer_el_velocity_tmp = [BufferArray(ComplexF64, 2)]
const _buffer_nothreads_eph_RR_to_Rq = [BufferArray(ComplexF64, 3)]
const _buffer_nothreads_eph_RR_to_Rq_tmp = [BufferArray(ComplexF64, 2)]

function __init__()
    Threads.resize_nthreads!(_buffer_el_eigen)
    Threads.resize_nthreads!(_buffer_el_velocity)
    Threads.resize_nthreads!(_buffer_el_velocity_tmp)
end

function _get_buffer(buffer::Vector{BufferArray{T, N}}, size_needed::NTuple{N, Int}) where {T, N}
    tid = Threads.threadid()
    if size(buffer[tid].arr) != size_needed
        buffer[tid] = BufferArray(zeros(T, size_needed))
    end
    buffer[tid].arr
end

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


"""
`get_eph_RR_to_Rq!(epobj_eRpq::WannierObject{T}, epmat::AbstractWannierObject{T},
xq, u_ph, nmodes, nr_el, fourier_mode="normal") where {T}`

Compute electron-phonon coupling matrix in electron Wannier, phonon Bloch basis.
Multithreading is not supported because of large buffer array size.

# Arguments
- `epobj_eRpq`: Output. E-ph matrix in electron Wannier, phonon Bloch basis.
    Must be initialized before calling. Only the op_r field is modified.
- `epmat`: Input. E-ph matrix in electron Wannier, phonon Wannier basis.
- `xq`: Input. q point vector.
- `nmodes`: Input. Number of phonon modes.
- `nr_el`: Input. Number of R points for electron Wannier basis.
- `u_ph`: Input. nmodes * nmodes matrix containing phonon eigenvectors.
"""
function get_eph_RR_to_Rq!(epobj_eRpq::WannierObject{T},
        epmat::AbstractWannierObject{T}, xq, u_ph, nmodes, nr_el, fourier_mode="normal") where {T}
    @assert size(u_ph) == (nmodes, nmodes)
    @assert mod(epmat.ndata, nmodes * nr_el) == 0
    @assert Threads.threadid() == 1
    nbasis = div(epmat.ndata, nmodes * nr_el) # Number of electron basis squared.

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

end
