"""
Wrapper of a very small part of LAPACK routines. Use allocated workspaces for repeated calls.
Also, make syev! type stable.

TODO: Add tests
TODO: Add zheevd, zheevr, zheevx
TODO: Add utility function to check runtime of different diagonalization routines
"""

module AllocatedLAPACK

using Base.Threads
using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare
using LinearAlgebra.LAPACK: @blasfunc, chkstride1, liblapack, chklapackerror

mutable struct WorkspaceZHEEV{T}
    n::Int
    work::Vector{Complex{T}}
    lwork::BlasInt
    rwork::Vector{T}
end
WorkspaceZHEEV{T}() where {T} = WorkspaceZHEEV{T}(0, Vector{Complex{T}}(), BlasInt(-1), Vector{T}())

const _buffer_zheev = [WorkspaceZHEEV{Float64}()]
const _buffer_cheev = [WorkspaceZHEEV{Float32}()]

function __init__()
    Threads.resize_nthreads!(_buffer_zheev)
    Threads.resize_nthreads!(_buffer_cheev)
end

# Hermitian eigensolvers
for (syev, elty, relty, workspaces) in
    ((:zheev_,:ComplexF64,:Float64,:_buffer_zheev),
     (:cheev_,:ComplexF32,:Float32,:_buffer_cheev))
    @eval begin
        # SUBROUTINE ZHEEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, RWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LWORK, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   RWORK( * ), W( * )
        #       COMPLEX*16         A( LDA, * ), WORK( * )
        function epw_syev!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty}, W::AbstractVector{$relty})
            n = checksquare(A)
            workspace = $workspaces[threadid()]
            if n > workspace.n
                # Need to regenerate workspace
                workspace = _epw_syev_alloc(jobz, uplo, A)
                $workspaces[threadid()] = workspace
            end
            _epw_syev!(jobz, uplo, A, W, workspace)
        end

        function _epw_syev!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty},
                W::AbstractVector{$relty}, workspace::WorkspaceZHEEV{$relty})
            chkstride1(A)
            n = checksquare(A)
            work  = workspace.work
            lwork = workspace.lwork
            rwork = workspace.rwork
            info  = Ref{BlasInt}()
            ccall((@blasfunc($syev), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                    Clong, Clong),
                    jobz, uplo, n, A, stride(A,2), W, work, lwork, rwork, info,
                    1, 1)
            chklapackerror(info[])
            W, A
        end

        function _epw_syev_alloc(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)
            W     = similar(A, $relty, n)
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            rwork = Vector{$relty}(undef, max(1, 3n-2))
            info  = Ref{BlasInt}()
            ccall((@blasfunc($syev), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                    Clong, Clong),
                    jobz, uplo, n, A, stride(A,2), W, work, lwork, rwork, info,
                    1, 1)
            chklapackerror(info[])
            lwork = BlasInt(real(work[1]))
            resize!(work, lwork)
            WorkspaceZHEEV{$relty}(n, work, lwork, rwork)
        end
    end
end

end