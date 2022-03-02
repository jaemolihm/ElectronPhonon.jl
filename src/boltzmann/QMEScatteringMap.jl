# Define QME scattering operators as a LinearMap.

using LinearMaps

"""Applies I + S_out⁻¹ S_in, acts on qme_model.el"""
struct QMEScatteringMap{MT, FT, SₒType} <: LinearMap{Complex{FT}}
    qme_model::MT
    Sᵢ_irr::Matrix{Complex{FT}} # Scattering-in matrix (for the irreducible grid)
    Sₒ::SₒType # Scattering-out matrix (for the full grid)
end
Base.size(A::QMEScatteringMap) = (A.qme_model.el.n, A.qme_model.el.n)

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::QMEScatteringMap, x::AbstractVector)
    # y = (I + Sₒ⁻¹ Sᵢ) * x
    x_QME = QMEVector(A.qme_model.el, x)
    Sᵢx = EPW.multiply_S_in(x_QME, A.Sᵢ_irr, A.qme_model)
    EPW._solve_qme_direct!(y, A.Sₒ, Sᵢx.data)
    y .+= x
    y
end
