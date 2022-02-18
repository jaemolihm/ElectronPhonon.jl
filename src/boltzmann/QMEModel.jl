export load_QMEModel
export bte_compute_μ!
export solve_electron_qme
export compute_qme_scattering_matrix!
export set_constant_qme_scattering_matrix!

abstract type AbstractQMEModel{FT} end

Base.@kwdef mutable struct QMEIrreducibleKModel{FT} <: AbstractQMEModel{FT}
    # === Mandatory fields ===

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
    # Electron final states
    el_f::QMEStates{FT}
    # Phonon states
    ph::BTStates{Float64}
    # File containing the data
    filename::String

    # === Optional fields ===

    # Covariant derivative operator (in Cartesian coordinates) that acts on el.
    ∇ = nothing
    # Scattering-out matrix for the irreducible k grid
    S_out_irr = nothing
    # Scattering-out matrix for the full k grid
    S_out = nothing
    # Scattering-in matrix for the irreducible k grid (Scattering-in matrix for full k grid
    # is not stored to reduce memory usage.)
    # TODO: Document way to apply S_in to objects on the full k grid.
    S_in_irr = nothing
end

"""
QME model defined on a full grid without any symmetry.
`model.X_irr` refers to `model.X` for `X = el, S_out`.
"""
Base.@kwdef mutable struct QMEModel{FT} <: AbstractQMEModel{FT}
    # === Mandatory fields ===

    # Electron states in the full k grid.
    el::QMEStates{FT}
    # Transport parameters
    transport_params::ElectronTransportParams{FT}
    # Electron final states
    el_f::QMEStates{FT}
    # Phonon states
    ph::BTStates{Float64}
    # File containing the data
    filename::String

    # === Optional fields ===

    # Covariant derivative operator (in Cartesian coordinates) that acts on el.
    ∇ = nothing
    # Scattering-out matrix
    S_out = nothing
    # Scattering-in matrix
    S_in = nothing
end

function Base.getproperty(obj::QMEModel, name::Symbol)
    if name === :el_irr
        getfield(obj, :el)
    elseif name === :S_out_irr
        getfield(obj, :S_out)
    elseif name === :S_in_irr
        getfield(obj, :S_in)
    elseif name === :symmetry
        nothing
    else
        getfield(obj, name)
    end
end

function Base.setproperty!(obj::QMEModel, name::Symbol, x)
    if name === :S_out_irr
        setfield!(obj, :S_out, x)
    elseif name === :S_in_irr
        setfield!(obj, :S_in, x)
    else
        setfield!(obj, name, x)
    end
end

"""
    load_QMEModel(filename, symmetry, transport_params) => qme_model::AbstractQMEModel
Read a QMEModel or QMEIrreducibleKModel from a hdf5 file containing the information.
# TODO: Write symmetry to file, automatically detect usage of symmetry.
"""
function load_QMEModel(filename, transport_params, ::Type{FT}=Float64) where FT
    fid = h5open(filename, "r")
    if haskey(fid, "symmetry")
        symmetry = load_BTData(fid["symmetry"], Symmetry{FT})
        el_i_irr = load_BTData(fid["initialstate_electron"], EPW.QMEStates{FT})
        el_i = load_BTData(fid["initialstate_electron_unfolded"], EPW.QMEStates{FT})
        ik_to_ikirr_isym = EPW._data_hdf5_to_julia(read(fid, "ik_to_ikirr_isym"), Vector{Tuple{Int, Int}})
        el_f = load_BTData(fid["finalstate_electron"], EPW.QMEStates{FT})
        ph = load_BTData(fid["phonon"], EPW.BTStates{FT})
        ∇ = EPW.load_covariant_derivative_matrix(fid["covariant_derivative"])

        qme_model = EPW.QMEIrreducibleKModel(; symmetry, ik_to_ikirr_isym,
            el_irr=el_i_irr, el=el_i, ∇=Vec3(∇), transport_params, el_f, ph, filename)
    else
        el_i = load_BTData(fid["initialstate_electron"], EPW.QMEStates{FT})
        el_f = load_BTData(fid["finalstate_electron"], EPW.QMEStates{FT})
        ph = load_BTData(fid["phonon"], EPW.BTStates{FT})
        ∇ = EPW.load_covariant_derivative_matrix(fid["covariant_derivative"])

        qme_model = EPW.QMEModel(; el=el_i, ∇=Vec3(∇), transport_params, el_f, ph, filename)
    end
    close(fid)
    qme_model
end

# Wrappers for transport-related functions

function bte_compute_μ!(model::AbstractQMEModel)
    bte_compute_μ!(model.transport_params, EPW.BTStates(model.el_irr))
end

function compute_qme_scattering_matrix!(model::AbstractQMEModel; compute_S_in=true)
    (; filename, transport_params, el_irr, el_f, ph) = model
    model.S_out_irr, model.S_in_irr = compute_qme_scattering_matrix(filename, transport_params,
                                                                    el_irr, el_f, ph; compute_S_in)
    unfold_scattering_out_matrix!(model)
end

function set_constant_qme_scattering_matrix!(model::AbstractQMEModel, inv_τ_constant)
    model.S_out_irr = [I(model.el_irr.n) * (-inv_τ_constant + 0.0im) for _ in model.transport_params.Tlist]
    unfold_scattering_out_matrix!(model)
end

function solve_electron_qme(model::AbstractQMEModel; kwargs...)
    (; transport_params, S_out_irr, S_in_irr, symmetry, el_f, filename) = model
    el_i_irr = model.el_irr
    solve_electron_qme(transport_params, el_i_irr, el_f, S_out_irr, S_in_irr; filename, symmetry, kwargs...)
end