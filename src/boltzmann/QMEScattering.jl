# Define QME scattering operators as a LinearMap.

using LinearMaps

"""Multiplies I + Sₒ⁻¹ (Sᵢ + A), acts on el (can be qme_model.el or qme_model.el_irr)"""
struct QMEScatteringMap{MT, FT, SₒType, AType} <: LinearMap{Complex{FT}}
    qme_model::MT
    el::QMEStates{FT}
    Sᵢ_irr::Matrix{Complex{FT}} # Scattering-in matrix (for the irreducible grid)
    S₀⁻¹::SₒType # Inverse scattering-out matrix (for the full grid)
    A::AType # Additional matrix to multiply
end
QMEScatteringMap(qme_model, el, Sᵢ_irr, S₀⁻¹) = QMEScatteringMap(qme_model, el, Sᵢ_irr, S₀⁻¹, nothing)

Base.size(A::QMEScatteringMap) = (A.el.n, A.el.n)

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::QMEScatteringMap, x::AbstractVector)
    mul!(QMEVector(A.el, y), A, QMEVector(A.el, x))
end

function LinearAlgebra.mul!(y::QMEVector, A::QMEScatteringMap, x::QMEVector)
    # y = (I + Sₒ⁻¹ (Sᵢ + A)) * x
    Sᵢx = multiply_Sᵢ(x, A.Sᵢ_irr, A.qme_model)
    A.A !== nothing && mul!(Sᵢx.data, A.A, x.data, true, true)
    mul!(y.data, A.S₀⁻¹, Sᵢx.data)
    y .+= x
    y
end