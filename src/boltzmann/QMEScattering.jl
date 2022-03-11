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
    # y = (I + Sₒ⁻¹ (Sᵢ + A)) * x
    x_QME = QMEVector(A.el, x)
    Sᵢx = multiply_Sᵢ(x_QME, A.Sᵢ_irr, A.qme_model)
    A.A !== nothing && mul!(Sᵢx.data, A.A, x)
    mul!(y, A.S₀⁻¹, Sᵢx.data)
    y .+= x
    y
end
