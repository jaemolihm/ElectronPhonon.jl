abstract type AbstractQMEModel{FT} end

Base.@kwdef mutable struct QMEIrreducibleKModel{FT} <: AbstractQMEModel{FT}
    # Mandatory fields

    # Symmetry operation used to map the irreducible k grid to the full k grid.
    symmetry::Symmetry{FT}
    # Mapping from index of full k grid to the index of irreducible k grid and the symmetry
    # operation between the two.
    ik_to_ikirr_isym::Vector{Tuple{Int, Int}}
    # Electron states in the irreducible k grid.
    el_irr::QMEStates{FT}
    # Electron states in the full k grid.
    el::QMEStates{FT}
    # Transport parameters
    transport_params::ElectronTransportParams{FT}

    # Optional fields

    # Covariant derivative operator (in Cartesian coordinates) that acts on el.
    ∇ = nothing
    # Scattering-out matrix
    S_out = nothing
end

"""
    unfold_QMEVector(f_irr::QMEVector, model::QMEIrreducibleKModel, trodd, invodd)
Unfold QMEVector defined on `model.el_irr` to `model.el`` using `model.symmetry`.
TODO: Generalize ``symop.Scart * x[i]`` to work with any datatype (scalar, vector, tensor).
"""
function unfold_QMEVector(f_irr::QMEVector{ElType, FT}, model::QMEIrreducibleKModel, trodd, invodd) where {ElType, FT}
    @assert f_irr.state === model.el_irr
    indmap_irr = EPW.states_index_map(model.el_irr)
    f = QMEVector(model.el, ElType)
    for i in 1:model.el.n
        (; ik, ib1, ib2) = model.el[i]

        ik_irr, isym = model.ik_to_ikirr_isym[ik]
        symop = model.symmetry[isym]
        i_irr = indmap_irr[EPW.CI(ib1, ib2, ik_irr)]

        f[i] = symop.Scart * f_irr[i_irr]
        if trodd && symop.is_tr
            f[i] *= -1
        end
        if invodd && symop.is_inv
            f[i] *= -1
        end
    end
    f
end


"""
QME model defined on a full grid without any symmetry.
`model.el_irr` returns `model.el`.
"""
Base.@kwdef mutable struct QMEModel{FT} <: AbstractQMEModel{FT}
    # Mandatory fields

    # Electron states in the full k grid.
    el::QMEStates{FT}
    # Transport parameters
    transport_params::ElectronTransportParams{FT}

    # Optional fields

    # Covariant derivative operator (in Cartesian coordinates) that acts on el.
    ∇ = nothing
    # Scattering-out matrix
    S_out = nothing
end

function Base.getproperty(obj::QMEModel, sym::Symbol)
    if sym === :el_irr
        getfield(obj, :el)
    else
        getfield(obj, sym)
    end
end

"""
    unfold_QMEVector(f_irr::QMEVector, model::QMEModel, trodd, invodd)
Since QMEModel does not use symmetry, unfolding is a do-nothing operation.
"""
function unfold_QMEVector(f_irr::QMEVector, model::QMEModel, trodd, invodd)
    QMEVector(f_irr.state, copy(f_irr.data))
end