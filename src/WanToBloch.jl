"""
Module for transformation from Wannier basis to Bloch eigenstate basis.
Use thread-save preallocated buffers.
"""
module WanToBloch

using LinearAlgebra
using ElectronPhonon: @timing
using ElectronPhonon: AbstractWannierObject, WannierObject, AbstractWannierInterpolator
using ElectronPhonon: get_fourier!, update_op_r!
using ElectronPhonon: solve_eigen_el!, solve_eigen_el_valueonly!, solve_eigen_ph!, solve_eigen_ph_valueonly!
using ElectronPhonon: dynmat_dipole!

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
export get_symmetry_representation_wannier!

# TODO: Allow the type to change.
# Preallocated buffers for temporary arrays. Access via _get_buffer. When multiple arrays
# are needed, use _buffer1 for the largest array, _buffer2 for the next largest, and so on.
const _buffer_nothreads1 = [Vector{ComplexF64}(undef, 0)]
const _buffer1 = [Vector{ComplexF64}(undef, 0)]
const _buffer2 = [Vector{ComplexF64}(undef, 0)]
const _buffer3 = [Vector{ComplexF64}(undef, 0)]

function __init__()
    Threads.resize_nthreads!(_buffer1)
    Threads.resize_nthreads!(_buffer2)
    Threads.resize_nthreads!(_buffer3)
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
    Base.ReshapedArray(view(buffer[tid], 1:prod(dims)), dims, ())
end

function _reshape_workspace(workspace::AbstractVector{T}, dims::NTuple{N, Int}) where {T, N}
    n = prod(dims)
    if length(workspace) < n
        resize!(workspace, n)
    end
    Base.ReshapedArray(view(workspace, 1:n), dims, ())
end

# =============================================================================
#  Electrons

"""
    get_el_eigen!(values, vectors, nw, ham, xk)
Compute electron eigenenergy and eigenvector.
"""
@timing "w2b_el_eig" function get_el_eigen!(values, vectors, nw, ham, xk)
    @assert size(values) == (nw,)
    @assert size(vectors) == (nw, nw)

    hk = _get_buffer(_buffer1, (nw, nw))
    # hk = zeros(eltype(ham), (nw, nw))
    # hk = _reshape_workspace(workspaces[1], (nw, nw))
    get_fourier!(hk, ham, xk)
    solve_eigen_el!(values, vectors, hk)
    nothing
end

"""
    get_el_eigen_valueonly!(values, nw, ham, xk)
"""
@timing "w2b_el_eigval" function get_el_eigen_valueonly!(values, nw, ham, xk)
    # FIXME: Names get_el_eigen_valueonly! and solve_eigen_el_valueonly! are confusing.
    @assert size(values) == (nw,)

    hk = _get_buffer(_buffer1, (nw, nw))
    get_fourier!(hk, ham, xk)
    solve_eigen_el_valueonly!(values, hk)
    nothing
end

"""
    get_el_velocity_diag_berry_connection!(velocity_diag, nw, ham_R, xk, uk)
Compute the diagoanl part of electron band velocity using the Berry connection formula. See
docstring for get_el_velocity_berry_connection! for details.
For the diagonal part, the Berry connection contribution is zero.

velocity_diag: nband-dimensional vector.
uk: nw * nband matrix containing nband eigenvectors of H(k).
"""
@timing "w2b_el_vel" function get_el_velocity_diag_berry_connection!(velocity_diag, nw, ham_R, xk, uk)
    @assert size(uk, 1) == nw
    nband = size(uk, 2)
    @assert size(velocity_diag) == (3, nband)

    vk = _get_buffer(_buffer1, (nw, nw, 3))
    tmp = _get_buffer(_buffer2, (nw, nband))

    get_fourier!(vk, ham_R, xk)

    # velocity_diag[idir, iband] = uk'[iband, :] * vk[:, :, idir] * uk[:, iband]
    @views @inbounds for idir = 1:3
        mul!(tmp, vk[:, :, idir], uk)
        for iband in 1:nband
            velocity_diag[idir, iband] = real(dot(uk[:, iband], tmp[:, iband]))
        end
    end
    nothing
end

"""
    get_el_velocity_berry_connection!(velocity, nw, ham_R, ek, xk, uk, rbar)
Compute electron band velocity using the Berry connection formula:
``v_{m,n} = (U^\\dagger dH^{(W)}(k) / dk U)_{m,n} + i * (e_m - e_n) * rbar_{m,n}``,
where ``rbar = U^\\dagger A U``
(Eq. (31) of X. Wang et al, PRB 74 195118 (2006)).

- `velocity``: (3, `nband`, `nband`) matrix
- `uk`: `nw` * `nband` matrix containing nband eigenvectors of ``H(k)``.
"""
@timing "w2b_el_vel" function get_el_velocity_berry_connection!(velocity, nw, ham_R, ek, xk, uk, rbar)
    @assert size(uk, 1) == nw
    nband = size(uk, 2)
    @assert size(velocity) == (3, nband, nband)

    vk = _get_buffer(_buffer1, (nw, nw, 3))
    tmp = _get_buffer(_buffer2, (nw, nband))

    get_fourier!(vk, ham_R, xk)

    # velocity[idir, :, :] = uk' * vk[:, :, idir] * uk
    @views @inbounds for idir = 1:3
        mul!(tmp, vk[:, :, idir], uk)
        mul!(velocity[idir, :, :], uk', tmp)
    end

    # Add the Berry connection contribution
    @inbounds for jb = 1:nband, ib = 1:nband
        velocity[:, ib, jb] .+= im .* (ek[ib] - ek[jb]) .* rbar[ib, jb]
    end

    nothing
end

"""
    get_el_velocity_direct!(velocity, nw, vel, xk, uk)
Compute electron band velocity by direct Wannier interpolation of ``dH/dk`` matrix elements.
- `vel`: WannierInterpolator for the velocity matrix (``dH/dk``).
- `velocity``: (3, `nband`, `nband`) matrix
- `uk`: `nw` * `nband` matrix containing nband eigenvectors of ``H(k)``.

TODO: Can we reduce code duplication with get_el_velocity_berry_connection?
"""
@timing "w2b_el_vel" function get_el_velocity_direct!(velocity, nw, vel, xk, uk)
    @assert size(uk, 1) == nw
    nband = size(uk, 2)
    @assert size(velocity) == (3, nband, nband)

    vk = _get_buffer(_buffer1, (nw, nw, 3))
    tmp = _get_buffer(_buffer2, (nw, nband))

    get_fourier!(vk, vel, xk)

    # velocity[idir, :, :] = uk' * vk[:, :, idir] * uk
    @views @inbounds for idir = 1:3
        mul!(tmp, vk[:, :, idir], uk)
        mul!(velocity[idir, :, :], uk', tmp)
    end
    nothing
end

"""Compute the symmetry representation in the Bloch Wannier basis."""
function get_symmetry_representation_wannier!(sym_W, el_sym_op, xk, is_tr)
    @assert length(sym_W) == el_sym_op.ndata
    # For time reversal, the complex conjugation part acts on the Fourier factor so one needs -xk
    (is_tr ? get_fourier!(sym_W, el_sym_op, -xk)
           : get_fourier!(sym_W, el_sym_op, xk))
end

"""Compute the symmetry representation in the eigenstate basis."""
function get_symmetry_representation_eigen!(sym_H, el_sym_op, xk, uk, usk, is_tr)
    nw, nband_k = size(uk)
    nband_sk = size(usk, 2)
    @assert size(sym_H) == (nband_sk, nband_k)

    sym_W = _get_buffer(_buffer1, (nw, nw))
    tmp = _get_buffer(_buffer2, (nw, nband_k))
    u_tmp = _get_buffer(_buffer3, (nw, nband_k))

    # Compute matrix in Wannier basis
    get_symmetry_representation_wannier!(sym_W, el_sym_op, xk, is_tr)

    # Apply gauge to transform to the eigenstate basis
    if is_tr
        # Due to complex conjugation in time-reversal operation, one needs to use conj(uk).
        u_tmp .= conj.(uk)
    else
        u_tmp .= uk
    end
    mul!(tmp, sym_W, u_tmp)
    mul!(sym_H, usk', tmp)
    sym_H
end

# =============================================================================
#  Phonons

"""
    get_ph_eigen!(values, vectors, xq, dyn, mass, polar)
Compute electron eigenenergy and eigenvector.
"""
@timing "w2b_ph_eig" function get_ph_eigen!(values, vectors, xq, dyn, mass, polar)
    nmodes = length(values)
    @assert size(vectors) == (nmodes, nmodes)
    @assert size(mass) == (nmodes,)
    @assert dyn.ndata == nmodes^2

    # dynq = _get_buffer(_buffer_ph_eigen, (nmodes, nmodes))
    # Use vectors as a temporary storage for the dynamical matrix
    dynq = vectors

    get_fourier!(dynq, dyn, xq)
    if ! isnothing(polar)
        dynmat_dipole!(dynq, xq, polar, 1)
    end
    @inbounds for j=1:nmodes, i=1:nmodes
        dynq[i, j] /= sqrt(mass[i])
        dynq[i, j] /= sqrt(mass[j])
    end
    solve_eigen_ph!(values, vectors, dynq, mass)
    values, vectors
end

"""
    get_el_eigen_valueonly!(values, xq, dyn, mass, polar)
"""
@timing "w2b_ph_eigval" function get_ph_eigen_valueonly!(values, xq, dyn, mass, polar)
    nmodes = length(values)
    @assert size(mass) == (nmodes,)
    @assert dyn.ndata == nmodes^2

    dynq = _get_buffer(_buffer1, (nmodes, nmodes))

    get_fourier!(dynq, dyn, xq)
    if ! isnothing(polar)
        dynmat_dipole!(dynq, xq, polar, 1)
    end
    @inbounds for j=1:nmodes, i=1:nmodes
        dynq[i, j] /= sqrt(mass[i])
        dynq[i, j] /= sqrt(mass[j])
    end
    solve_eigen_ph_valueonly!(values, dynq)
    values
end


"""
    get_ph_velocity_diag!(vel_diag, dyn_R, xk, uk)
Compute phonon band velocity, only the band-diagonal part.
# Outputs
- `vel_diag`: (3, nmodes) array, contains diagonal band velocity.
# Inputs
- `uk`: nmodes * nmodes matrix containing phonon eigenvectors.
"""
@timing "w2b_ph_vel" function get_ph_velocity_diag!(vel_diag, dyn_R, xk, uk)
    # FIXME: Polar is not implemented.
    nmodes = size(uk, 1)
    @assert size(uk) == (nmodes, nmodes)
    @assert size(vel_diag) == (3, nmodes)

    vk = _get_buffer(_buffer1, (nmodes, nmodes, 3))
    tmp = _get_buffer(_buffer2, (nmodes, nmodes))

    get_fourier!(vk, dyn_R, xk)

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
`get_eph_RR_to_Rq!(epobj_eRpq::WannierObject, epmat::AbstractWannierInterpolator, xq, u_ph)`

Compute electron-phonon coupling matrix in electron Wannier, phonon Bloch basis.
Multithreading is not supported because of large buffer array size.

# Arguments
- `epobj_eRpq`: Output. E-ph matrix in electron Wannier, phonon Bloch basis.
    Must be initialized before calling. Only the op_r field is modified.
- `epmat`: Input. E-ph matrix in electron Wannier, phonon Wannier basis.
- `xq`: Input. q point vector.
- `u_ph`: Input. nmodes * nmodes matrix containing phonon eigenvectors.
"""
@timing "w2b_eph_RRtoRq" function get_eph_RR_to_Rq!(epobj_eRpq::WannierObject,
        epmat::AbstractWannierInterpolator, xq, u_ph)
    nr_el = epobj_eRpq.nr
    nmodes = size(u_ph, 1)
    nbasis = div(epobj_eRpq.ndata, nmodes) # Number of electron basis squared.
    @assert size(u_ph) == (nmodes, nmodes)
    @assert Threads.threadid() == 1
    @assert epobj_eRpq.ndata == nbasis * nmodes
    @assert epmat.ndata == nbasis * nmodes * nr_el

    ep_Rq = _get_buffer(_buffer_nothreads1, (nbasis, nmodes, nr_el))
    ep_Rq_tmp = _get_buffer(_buffer1, (nbasis, nmodes))

    get_fourier!(ep_Rq, epmat, xq)

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
    get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, uk, ukq)
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.

# Arguments
- `ep_kq`: Output. E-ph matrix in electron and phonon Bloch basis.
- `epobj_eRpq`: Input. AbstractWannierObject. E-ph matrix in electron Wannier,
    phonon Bloch basis.
- `xk`: Input. k point vector.
- `uk`, `ukq`: Input. Electron eigenstate at k and k+q, respectively.
"""
@timing "w2b_eph_Rqtokq" function get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, uk, ukq)
    nbandkq, nbandk, nmodes = size(ep_kq)
    @assert size(uk, 2) == nbandk
    @assert size(ukq, 2) == nbandkq
    @assert epobj_eRpq.ndata == size(ukq, 1) * size(uk, 1) * nmodes

    ep_kq_wan = _get_buffer(_buffer1, (size(ukq, 1), size(uk, 1), nmodes))
    tmp = _get_buffer(_buffer2, (size(ukq, 1), nbandk))

    get_fourier!(ep_kq_wan, epobj_eRpq, xk)

    # Rotate e-ph matrix from electron Wannier to eigenstate basis
    # ep_kq[ibkq, ibk, imode] = ukq'[ibkq, :] * ep_kq_wan[:, :, imode] * uk[:, ibk]
    @views @inbounds for imode = 1:nmodes
        mul!(tmp, ep_kq_wan[:, :, imode], uk)
        mul!(ep_kq[:, :, imode], ukq', tmp)
    end
    nothing
end

"""
    get_eph_RR_to_kR!(epobj_eRpq::WannierObject{T}, epmat, xk, uk) where {T}

Compute electron-phonon coupling matrix in electron Bloch, phonon Wannier basis.
Multithreading is not supported because of large buffer array size.

# Arguments
- `epobj_ekpR`: Output. E-ph matrix in electron Wannier, phonon Bloch basis.
    Must be initialized before calling. Only the op_r field is modified.
- `epmat`: Input. E-ph matrix in electron Wannier, phonon Wannier basis.
- `xk`: Input. k point vector.
- `uk`: Input. nw * nw matrix containing electron eigenvectors at k.
"""
@timing "w2b_eph_RRtokR" function get_eph_RR_to_kR!(epobj_ekpR::WannierObject{T}, epmat, xk, uk) where {T}
    """
    size(uk) = (nw, nband)
    size(epobj_ekpR.op_r) = (nw * nband_bound * nmodes, nr_ep)
    size(epmat.op_r) = (nw^2 * nmodes * nr_ep, nr_el)
    """
    nr_ep = length(epmat.parent.irvec_next)
    nw, nband = size(uk)
    nmodes = div(epmat.ndata, nw^2 * nr_ep)
    ndata = nw * nband * nmodes
    @assert epobj_ekpR.nr == nr_ep
    @assert size(epobj_ekpR.op_r, 1) >= ndata

    ep_kR = _get_buffer(_buffer_nothreads1, (nw, nw, nmodes, nr_ep))
    get_fourier!(ep_kR, epmat, xk)

    # Transform from electron Wannier to eigenmode basis, one ir_el and modes at a time.
    @views for ir in 1:nr_ep
        ep_kR2 = Base.ReshapedArray(epobj_ekpR.op_r[1:ndata, ir], (nw, nband, nmodes), ())
        ep_kR_tmp = _get_buffer(_buffer2, (nw, nband))
        @inbounds for imode in 1:nmodes
            mul!(ep_kR_tmp, ep_kR[:, :, imode, ir], uk)
            ep_kR2[:, :, imode] .= ep_kR_tmp
        end
    end
    epobj_ekpR.ndata = ndata
    nothing
end

"""
    get_eph_kR_to_kq!(ep_kq, epobj_ekpR, xk, u_ph, ukq)
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
The electron state at k should be already in the eigenstate basis in epobj_ekpR.

# Arguments
- `ep_kq`: Output. E-ph matrix in electron and phonon Bloch basis.
- `epobj_ekpR`: Input. WannierInterpolator. E-ph matrix in electron Wannier,
    phonon Bloch basis.
- `xq`: Input. q point vector.
- `u_ph`: Input. nmodes * nmodes matrix containing phonon eigenvectors.
- `ukq`: Input. Electron eigenstate at k+q.
- `rngk`, `rngkq`: Input. Range of electron states inside the window.
"""
@timing "w2b_eph_kRtokq" function get_eph_kR_to_kq!(ep_kq, epobj_ekpR, xq, u_ph, ukq)
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
    @assert epobj_ekpR.parent.ndata == nw * nbandk * nmodes

    ep_kq_wan = _get_buffer(_buffer1, (nw, nbandk, nmodes))
    tmp = _get_buffer(_buffer2, (nbandkq, nbandk))

    get_fourier!(ep_kq_wan, epobj_ekpR, xq)

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
