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

# Adapted from https://github.com/DynareJulia/FastLapackInterface.jl/blob/main/src/eigen.jl
# We change the code to use syev instead of syevr

struct WorkspaceSizeError <: Exception
    nws::Int
    n::Int
end
Base.showerror(io::IO, e::WorkspaceSizeError) = print(io, "Workspace has the wrong size: expected $(e.n), got $(e.nws).\nUse resize!(ws, A).")


struct HermitianEigenWsSYEV{T,RT<:AbstractFloat}
    W::Vector{RT}
    work::Vector{T}
    rwork::Vector{RT}
end
function Base.getproperty(ws::HermitianEigenWsSYEV, name::Symbol)
    if name === :n
        length(getfield(ws, :W))
    else
        getfield(ws, name)
    end
end


HermitianEigenWsSYEV{T,RT}() where {T,RT} = HermitianEigenWsSYEV{T,RT}(
    Vector{T}(undef, 0), Vector{RT}(undef, 0), Vector{RT}(undef, 0))


for (syev, elty, relty) in ((:zheev_, :ComplexF64, :Float64),
                             (:cheev_, :ComplexF32, :Float32),)
    @eval begin
        function Base.resize!(ws::HermitianEigenWsSYEV, A::AbstractMatrix{$elty}; work=true)
            if work
                chkstride1(A)
                n = checksquare(A)
                resize!(ws.W, n)
                resize!(ws.rwork, max(1, 3n-2))

                isempty(ws.work) && resize!(ws.work, 1)

                info  = Ref{BlasInt}()
                ccall((@blasfunc($syev), liblapack), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                        Clong, Clong),
                        'V', 'U', n, A, stride(A,2),
                        ws.W, ws.work, BlasInt(-1), ws.rwork, info,
                        1, 1)
                chklapackerror(info[])
                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end
        function HermitianEigenWsSYEV(A::AbstractMatrix{$elty})
            return resize!(HermitianEigenWsSYEV{$elty,$relty}(), A; work=true)
        end


        # SUBROUTINE ZHEEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, RWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LWORK, N
        # *     ..
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   RWORK( * ), W( * )
        #       COMPLEX*16         A( LDA, * ), WORK( * )
        function syev!(ws::HermitianEigenWsSYEV, jobz::AbstractChar, uplo::AbstractChar,
                        A::AbstractMatrix{$elty}; resize=true)

            chkstride1(A)
            n = checksquare(A)
            nws = ws.n
            if nws != n
                if resize
                    resize!(ws, A, work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end

            m = Ref{BlasInt}()
            info = Ref{BlasInt}()
            ccall((@blasfunc($syev), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                    Clong, Clong),
                    jobz, uplo, n, A, stride(A,2),
                    ws.W, ws.work, length(ws.work), ws.rwork, info,
                    1, 1)
            chklapackerror(info[])
            return view(ws.W, 1:n), A
        end
    end
end

"""
    syev!(ws, jobz, uplo, A; resize=true) -> (ws.W, ws.Z)

Finds the eigenvalues (`jobz = N`) or eigenvalues and eigenvectors
(`jobz = V`) of a symmetric matrix `A` using a preallocated [`HermitianEigenWsSYEV`](@ref).
If the workspace is not appropriate for `A` and `resize==true` it will be automatically
resized.
If `uplo = U`, the upper triangle of `A` is used. If `uplo = L`, the lower triangle of `A` is used.

The eigenvalues are returned as `ws.W` and the eigenvectors in `A`.
"""
syev!(ws::HermitianEigenWsSYEV, jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix; resize=true)




# =========================================================================================
# Old code

# const _buffer_zheev = [HermitianEigenWsSYEV{ComplexF64,Float64}()]
# const _buffer_cheev = [HermitianEigenWsSYEV{ComplexF32,Float32}()]

# function __init__()
#     Threads.resize_nthreads!(_buffer_zheev)
#     Threads.resize_nthreads!(_buffer_cheev)
# end

# # Hermitian eigensolvers
# for (syev, elty, relty, workspaces) in
#     ((:zheev_,:ComplexF64,:Float64,:_buffer_zheev),
#      (:cheev_,:ComplexF32,:Float32,:_buffer_cheev))
#     @eval begin
#         # SUBROUTINE ZHEEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, RWORK, INFO )
#         # *     .. Scalar Arguments ..
#         #       CHARACTER          JOBZ, UPLO
#         #       INTEGER            INFO, LDA, LWORK, N
#         # *     ..
#         # *     .. Array Arguments ..
#         #       DOUBLE PRECISION   RWORK( * ), W( * )
#         #       COMPLEX*16         A( LDA, * ), WORK( * )
#         function epw_syev!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
#             n = checksquare(A)
#             workspace = $workspaces[threadid()]
#             if n > workspace.n
#                 # Need to regenerate workspace
#                 workspace = _epw_syev_alloc(jobz, uplo, A)
#                 $workspaces[threadid()] = workspace
#             end
#             _epw_syev!(jobz, uplo, A, workspace.W, workspace)
#         end

#         function epw_syev!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty}, W::AbstractVector{$relty})
#             n = checksquare(A)
#             ws = $workspaces[threadid()]
#             if n > ws.n
#                 # Need to regenerate workspace
#                 ws = _epw_syev_alloc(jobz, uplo, A)
#                 $workspaces[threadid()] = ws
#             end
#             _epw_syev!(jobz, uplo, A, W, ws)
#         end

#         function _epw_syev!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty},
#                 W::AbstractVector{$relty}, ws::HermitianEigenWsSYEV{$elty,$relty})
#             chkstride1(A)
#             n = checksquare(A)
#             work  = ws.work
#             rwork = ws.rwork
#             info  = Ref{BlasInt}()
#             ccall((@blasfunc($syev), liblapack), Cvoid,
#                     (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
#                     Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
#                     Clong, Clong),
#                     jobz, uplo, n, A, stride(A,2),
#                     W, work, length(ws.work), rwork, info,
#                     1, 1)
#             chklapackerror(info[])
#             W, A
#         end

#         function _epw_syev_alloc(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
#             chkstride1(A)
#             n = checksquare(A)
#             W     = similar(A, $relty, n)
#             work  = Vector{$elty}(undef, 1)
#             lwork = BlasInt(-1)
#             rwork = Vector{$relty}(undef, max(1, 3n-2))
#             info  = Ref{BlasInt}()
#             ccall((@blasfunc($syev), liblapack), Cvoid,
#                     (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
#                     Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
#                     Clong, Clong),
#                     jobz, uplo, n, A, stride(A,2), W, work, lwork, rwork, info,
#                     1, 1)
#             chklapackerror(info[])
#             resize!(work, BlasInt(real(work[1])))
#             HermitianEigenWsSYEV{$elty,$relty}(W, work, rwork)
#         end
#     end
# end

end
