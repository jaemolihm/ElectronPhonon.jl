using ElectronPhonon.AllocatedLAPACK: HermitianEigenWsSYEV, syev!

export eigvals_batched
export eigen_batched
export get_el_eigen_batched
export get_el_eigen_valueonly_batched

# Batched Wannier -> Bloch diagonalization over a whole k-grid.
#
# These complement the per-k `get_el_eigen!` / `get_el_eigen_valueonly!` (wannier_to_bloch.jl):
# `get_fourier_batched!` interpolates `H(k)` for all k at once (one GEMM chain), then a
# batched Hermitian eigensolve diagonalizes the stack. Everything runs on the backend of
# `ham.op_r` (CPU or GPU); the CUDA extension provides the batched `heevjBatched!` methods.
#
# Naming mirrors the per-k routines:
#   get_el_eigen_batched           <-> get_el_eigen!            (eigenvalues + eigenvectors)
#   get_el_eigen_valueonly_batched <-> get_el_eigen_valueonly!  (eigenvalues only)

# =============================================================================
#  Batched Hermitian eigensolves (CPU methods; CUDA extension adds CuArray methods)

"""
    eigvals_batched(Hk) -> W

Eigenvalues of a stack of Hermitian matrices `Hk` of size `(nw, nw, nk)`, returned as
`(nw, nk)`. CPU method loops over LAPACK `syev!`; the CUDA extension provides a batched
`heevjBatched!` method for `CuArray`s.
"""
function eigvals_batched(Hk::AbstractArray{Complex{T},3}) where {T}
    nw, n2, nk = size(Hk)
    @assert nw == n2
    W = Matrix{T}(undef, nw, nk)
    ws = HermitianEigenWsSYEV{Complex{T},T}()
    A = Matrix{Complex{T}}(undef, nw, nw)
    @views for ik in 1:nk
        A .= Hk[:, :, ik]
        W[:, ik] .= syev!(ws, 'N', 'U', A)[1]
    end
    W
end

"""
    eigen_batched(Hk) -> (W, V)

Eigenvalues `W` `(nw, nk)` and eigenvectors `V` `(nw, nw, nk)` of a stack of Hermitian
matrices `Hk` of size `(nw, nw, nk)`. CPU method loops over LAPACK `syev!`; the CUDA
extension provides a batched `heevjBatched!` method for `CuArray`s.

Note: unlike the per-k `get_el_eigen!`, no EPW degeneracy gauge-fixing is applied, so for
degenerate bands the eigenvectors may differ from `get_el_eigen!` by a gauge (the
eigenvalues, and the eigen-decomposition, are unaffected).
"""
function eigen_batched(Hk::AbstractArray{Complex{T},3}) where {T}
    nw, n2, nk = size(Hk)
    @assert nw == n2
    W = Matrix{T}(undef, nw, nk)
    V = Array{Complex{T},3}(undef, nw, nw, nk)
    ws = HermitianEigenWsSYEV{Complex{T},T}()
    A = Matrix{Complex{T}}(undef, nw, nw)
    @views for ik in 1:nk
        A .= Hk[:, :, ik]
        W[:, ik] .= syev!(ws, 'V', 'U', A)[1]   # A is overwritten with the eigenvectors
        V[:, :, ik] .= A
    end
    W, V
end

# =============================================================================
#  Band-eigenvalue drivers over a k-grid

# Interpolate H(k) for all k into an (ndata, nk) array on `ham.op_r`'s backend.
function _fourier_hk_batched(ham::WannierObject{T}, xk_list; batch_size::Int) where {T}
    nw = isqrt(ham.ndata)
    nw^2 == ham.ndata || throw(ArgumentError(
        "ndata=$(ham.ndata) is not a perfect square; expected nw^2 for a Hamiltonian"))
    nk = length(xk_list)
    itp = BatchedWannierInterpolator(ham; batch_size)
    Hk = similar(ham.op_r, Complex{T}, ham.ndata, nk)
    get_fourier_batched!(Hk, itp, xk_list)
    reshape(Hk, nw, nw, nk)
end

"""
    get_el_eigen_valueonly_batched(ham::WannierObject, xk_list; batch_size=length(xk_list)) -> W

Electron band eigenvalues `(nw, nk)` at every k-point in `xk_list`. Batched counterpart of
[`get_el_eigen_valueonly!`](@ref). Runs on the backend of `ham.op_r`; the result is on that
same backend. `batch_size` controls the Fourier chunking (default: a single batch).
"""
function get_el_eigen_valueonly_batched(ham::WannierObject, xk_list; batch_size::Int=length(xk_list))
    eigvals_batched(_fourier_hk_batched(ham, xk_list; batch_size))
end

"""
    get_el_eigen_batched(ham::WannierObject, xk_list; batch_size=length(xk_list)) -> (W, V)

Electron band eigenvalues `(nw, nk)` and eigenvectors `(nw, nw, nk)` at every k-point in
`xk_list`. Batched counterpart of [`get_el_eigen!`](@ref). Runs on the backend of
`ham.op_r`; the results are on that same backend. See [`eigen_batched`](@ref) for the
eigenvector gauge caveat.
"""
function get_el_eigen_batched(ham::WannierObject, xk_list; batch_size::Int=length(xk_list))
    eigen_batched(_fourier_hk_batched(ham, xk_list; batch_size))
end
