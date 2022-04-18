# Define QME scattering operators as a LinearMap.

using LinearMaps

"""
Multiplies I + Sₒ⁻¹ (Sᵢ + A), acts on el (can be qme_model.el or qme_model.el_irr)
The elements of the QMEVector to be multiplied has type `SVector{N}` for N > 1 and `Real` for
N = 1. The `N > 1` case is used to use iterative solvers which support only `Real`-valued
AbstractVectors.
"""
struct QMEScatteringMap{N, MT, FT, SₒType, AType} <: LinearMap{Complex{FT}}
    qme_model::MT
    el::QMEStates{FT}
    Sᵢ_irr::Matrix{Complex{FT}} # Scattering-in matrix (for the irreducible grid)
    S₀⁻¹::SₒType # Inverse scattering-out matrix (for the full grid)
    A::AType # Additional matrix to multiply
    function QMEScatteringMap{N}(qme_model::MT, el::QMEStates{FT}, Sᵢ_irr::Matrix{Complex{FT}},
        S₀⁻¹::SₒType, A::AType=nothing) where {N, MT, FT, SₒType, AType}
        new{N, MT, FT, SₒType, AType}(qme_model, el, Sᵢ_irr, S₀⁻¹, A)
    end
end
QMEScatteringMap(args...) = QMEScatteringMap{1}(args...)

Base.size(A::QMEScatteringMap{N}) where N = (N * A.el.n, N * A.el.n)

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::QMEScatteringMap{N}, x::AbstractVector{T}) where {T, N}
    if N == 1
        mul!(QMEVector(A.el, y), A, QMEVector(A.el, x))
    else
        mul!(QMEVector(A.el, reinterpret(SVector{N, T}, y)), A, QMEVector(A.el, reinterpret(SVector{N, T}, x)))
    end
end

function LinearAlgebra.mul!(y::QMEVector, A::QMEScatteringMap{N}, x::QMEVector{T}) where {T, N}
    # y = (I + Sₒ⁻¹ (Sᵢ + A)) * x
    Sᵢx = multiply_Sᵢ(x, A.Sᵢ_irr, A.qme_model)
    A.A !== nothing && mul!(Sᵢx.data, A.A, x.data, true, true)
    mul!(y.data, A.S₀⁻¹, Sᵢx.data)
    y .+= x
    y
end