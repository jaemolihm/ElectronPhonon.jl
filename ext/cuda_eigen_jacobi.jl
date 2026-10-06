# One-thread-per-matrix batched Hermitian eigensolver: cyclic complex Jacobi on small matrices held
# in `MMatrix` (thread-local memory). `jacobi_eigen!` is plain Julia and also runs on the CPU, which
# is how the tests compare it with LAPACK without a GPU.
#
# Cyclic Jacobi is the textbook method (Golub & Van Loan, "Matrix Computations", Sec. 8.5).
# cuSOLVER's `syevjBatched` is a Jacobi solver too, but runs one thread block per matrix; for
# nw <= JACOBI_NW_MAX one thread per matrix is faster (A100, H100, A6000) and as accurate as LAPACK.
# `MMatrix` with plain loops, not `SMatrix` (compile time blows up with nw) or a `@generated`
# straight-line kernel (a few % faster for much more code).

using StaticArrays: MMatrix, MVector

"""
    jacobi_eigen!(A::MMatrix{N,N,ComplexF64}, V; maxsweep = 20) -> (d, perm, nsweep, converged)

Diagonalize the Hermitian `A` (both triangles filled) in place by cyclic complex Jacobi rotations.
`V` is `nothing` (eigenvalues only) or an identity `MMatrix` that accumulates the eigenvectors.
Returns the eigenvalues `d` in ascending order, the permutation `perm` such that eigenvector `j` is
column `perm[j]` of `V`, the number of sweeps, and whether it converged: the off-diagonal Frobenius
norm is at most `eps` times that of `A`. `converged` is false for NaN or Inf input and when
`maxsweep` sweeps did not converge. The rotations depend on `A` only, so `d` is the same bit for
bit with and without `V`.
"""
@inline function jacobi_eigen!(A::MMatrix{N,N,ComplexF64}, V; maxsweep::Int = 20) where {N}
    # off: squared Frobenius norm of the strict upper triangle (half that of the off-diagonal)
    off = 0.0
    nrm2 = 0.0
    @inbounds for j in 1:N
        for i in 1:j-1
            off += abs2(A[i, j])
        end
        nrm2 += abs2(A[j, j])
    end
    tol2 = eps(Float64)^2 * (nrm2 + 2off)
    skip2 = tol2 / (N * N)            # rotations below this cannot block convergence
    nsweep = 0
    # `!(2off <= tol2 < Inf)` so a NaN, or an Inf that makes `tol2` infinite, keeps iterating to
    # maxsweep instead of exiting as converged
    @inbounds while !(2off <= tol2 < Inf) && nsweep < maxsweep
        nsweep += 1
        for q in 2:N, p in 1:q-1
            apq = A[p, q]
            g2 = abs2(apq)
            g2 <= skip2 && continue
            g = sqrt(g2)
            app = real(A[p, p])
            aqq = real(A[q, q])
            θ = (aqq - app) / (2g)
            t = copysign(1.0, θ) / (abs(θ) + hypot(1.0, θ))
            c = 1 / sqrt(1 + t * t)
            se = t * c * (apq / g)
            sec = conj(se)
            # J = identity except J[p,p] = J[q,q] = c, J[p,q] = s e, J[q,p] = -s ē
            # A ← A J and V ← V J (columns p, q)
            for k in 1:N
                akp = A[k, p]; akq = A[k, q]
                A[k, p] = c * akp - sec * akq
                A[k, q] = se * akp + c * akq
            end
            if V !== nothing
                for k in 1:N
                    vkp = V[k, p]; vkq = V[k, q]
                    V[k, p] = c * vkp - sec * vkq
                    V[k, q] = se * vkp + c * vkq
                end
            end
            # A ← Jᴴ A (rows p, q)
            for k in 1:N
                apk = A[p, k]; aqk = A[q, k]
                A[p, k] = c * apk - se * aqk
                A[q, k] = sec * apk + c * aqk
            end
            A[p, q] = 0
            A[q, p] = 0
            A[p, p] = app - t * g
            A[q, q] = aqq + t * g
        end
        off = 0.0
        for q in 2:N, p in 1:q-1
            off += abs2(A[p, q])
        end
    end
    converged = 2off <= tol2 < Inf
    # insertion sort of the diagonal, ascending
    d = MVector{N,Float64}(undef)
    perm = MVector{N,Int}(undef)
    @inbounds for i in 1:N
        x = real(A[i, i]); j = i - 1
        while j >= 1 && d[j] > x
            d[j+1] = d[j]; perm[j+1] = perm[j]
            j -= 1
        end
        d[j+1] = x; perm[j+1] = i
    end
    d, perm, nsweep, converged
end

# Hermitian MMatrix from the upper triangle of H[:, :, b] (real diagonal), as LAPACK's uplo = 'U'
@inline function load_hermitian(H, b, ::Val{N}) where {N}
    A = MMatrix{N,N,ComplexF64}(undef)
    @inbounds for j in 1:N, i in 1:N
        A[i, j] = i < j ? H[i, j, b] : i > j ? conj(H[j, i, b]) : complex(real(H[i, i, b]))
    end
    A
end

@inline function identity_mmatrix(::Val{N}) where {N}
    V = MMatrix{N,N,ComplexF64}(undef)
    @inbounds for j in 1:N, i in 1:N
        V[i, j] = ifelse(i == j, one(ComplexF64), zero(ComplexF64))
    end
    V
end

# Thread b solves H[:, :, b]: E[:, b] (ascending) and, unless `U === nothing`, U[:, :, b]. A matrix
# that did not converge gets NaN in all of E[:, b] and U[:, :, b]. `U` may alias `H`: the thread
# reads all of H[:, :, b] into `A` before it stores anything, and no other thread touches slice b.
function jacobi_eigen_kernel!(E, U, H, ::Val{N}) where {N}
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > size(H, 3) && return
    A = load_hermitian(H, b, Val(N))
    V = U === nothing ? nothing : identity_mmatrix(Val(N))
    d, perm, _, converged = jacobi_eigen!(A, V)
    @inbounds for j in 1:N
        E[j, b] = converged ? d[j] : NaN
        if U !== nothing
            for i in 1:N
                U[i, j, b] = converged ? V[i, perm[j]] : complex(NaN, NaN)
            end
        end
    end
    return
end

"""
    jacobi_eigen_batched!(E, U, H) -> E

Eigenvalues `E (nw, nb)` (ascending) and, unless `U === nothing`, eigenvectors `U (nw, nw, nb)` of
the Hermitian device matrices `H (nw, nw, nb)` (upper triangle read), one thread per matrix. `U`
may be `H` itself. A matrix that does not converge (NaN or Inf input) gets NaN eigenvalues.
"""
function jacobi_eigen_batched!(E, U, H)
    nw, n2, nb = size(H)
    nw == n2 || throw(DimensionMismatch("H must be square in its first two dimensions, got $(size(H))"))
    nb == 0 && return E
    threads = 128
    # `Val(nw)` is the function barrier: one kernel per nw (and per eigenvalues-only/full).
    @cuda threads = threads blocks = cld(nb, threads) jacobi_eigen_kernel!(E, U, H, Val(nw))
    E
end
