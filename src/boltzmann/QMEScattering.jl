# Define QME scattering operators as a LinearMap.

using LinearMaps

"""Applies I + Sₒ⁻¹ Sᵢ, acts on el (can be qme_model.el or qme_model.el_irr)"""
struct QMEScatteringMap{MT, FT, SₒType} <: LinearMap{Complex{FT}}
    qme_model::MT
    el::QMEStates{FT}
    Sᵢ_irr::Matrix{Complex{FT}} # Scattering-in matrix (for the irreducible grid)
    S₀⁻¹::SₒType # Inverse scattering-out matrix (for the full grid)
end
Base.size(A::QMEScatteringMap) = (A.el.n, A.el.n)

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::QMEScatteringMap, x::AbstractVector)
    # y = (I + Sₒ⁻¹ Sᵢ) * x
    x_QME = QMEVector(A.el, x)
    Sᵢx = multiply_Sᵢ(x_QME, A.Sᵢ_irr, A.qme_model)
    mul!(y, A.S₀⁻¹, Sᵢx.data)
    y .+= x
    y
end
