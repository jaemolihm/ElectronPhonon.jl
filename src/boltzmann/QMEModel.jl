export load_QMEModel
export bte_compute_μ!
export compute_qme_scattering_matrix!
export set_constant_qme_scattering_matrix!

using SparseArrays

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
    Sₒ_irr = nothing
    # Scattering-out matrix for the full k grid
    Sₒ = nothing
    # Scattering-in matrix for the irreducible k grid (Scattering-in matrix for full k grid
    # is not stored to reduce memory usage.)
    # To multiply Sᵢ to a QMEVector, use `multiply_Sᵢ(x::QMEVector, Sᵢ_irr, qme_model::AbstractQMEModel)`.
    Sᵢ_irr = nothing
    # Map from el_irr to el_f, assuming time-reversal odd, vector elements.
    map_i_to_f_vector::SparseMatrixCSC{Mat3{Complex{FT}}, Int}
    # Map from el_irr to el_f, assuming time-reversal odd, vector elements for symmetries
    # involving time reversal operation.
    map_i_to_f_vector_tr::SparseMatrixCSC{Mat3{Complex{FT}}, Int}
    # Map from el to el_f by each symmetry operator. Needed in multiply_Sᵢ.
    # (storage size ~ N_sym^2 * N_kirr * N_band)
    el_to_el_f_sym_maps::Union{Nothing, Vector{SparseMatrixCSC{Complex{FT}, Int}}} = nothing

    # === Buffers ===
    # Costs memory ~ N_k_irr * N_sym^2.
    _buffer_el::Vector{Complex{FT}} = zeros(Complex{eltype(el.e1)}, el.n)
    _buffer_el_f_sym::Matrix{Complex{FT}} = zeros(Complex{eltype(el.e1)}, el_f.n, symmetry.nsym)
    _buffer_el_irr_sym::Matrix{Complex{FT}} = zeros(Complex{eltype(el.e1)}, el_irr.n, symmetry.nsym)
end

"""
QME model defined on a full grid without any symmetry.
`model.X_irr` refers to `model.X` for `X = el, Sₒ`.
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
    Sₒ = nothing
    # Scattering-in matrix
    # To multiply Sᵢ to a QMEVector, use `multiply_Sᵢ(x::QMEVector, Sᵢ_irr, qme_model::AbstractQMEModel)`.
    Sᵢ = nothing
    # Map from el_irr to el_f, assuming time-reversal odd, vector elements.
    map_i_to_f::SparseMatrixCSC{Complex{FT}, Int}
end

function Base.getproperty(obj::QMEModel, name::Symbol)
    if name === :el_irr
        getfield(obj, :el)
    elseif name === :Sₒ_irr
        getfield(obj, :Sₒ)
    elseif name === :Sᵢ_irr
        getfield(obj, :Sᵢ)
    elseif name === :symmetry
        nothing
    elseif name === :map_i_to_f_vector
        getfield(obj, :map_i_to_f)
    else
        getfield(obj, name)
    end
end

function Base.setproperty!(obj::QMEModel, name::Symbol, x)
    if name === :Sₒ_irr
        setfield!(obj, :Sₒ, x)
    elseif name === :Sᵢ_irr
        setfield!(obj, :Sᵢ, x)
    else
        setfield!(obj, name, x)
    end
end

"""
    load_QMEModel(filename, symmetry, transport_params) => qme_model::AbstractQMEModel
Read a QMEModel or QMEIrreducibleKModel from a hdf5 file containing the information.
# TODO: Write symmetry to file, automatically detect usage of symmetry.
"""
function load_QMEModel(filename, transport_params, ::Type{FT}=Float64; derivative_order=1) where FT
    fid = h5open(filename, "r")
    el_f = load_BTData(fid["finalstate_electron"], QMEStates{FT})
    ph = load_BTData(fid["phonon"], BTStates{FT})

    if haskey(fid, "covariant_derivative_order$derivative_order")
        ∇ = Vec3(load_covariant_derivative_matrix(fid["covariant_derivative_order$derivative_order"]))
    else
        if derivative_order != 1
            # derivative_order was explicitly set but was not found in the file.
            error("Covariant derivative at order $derivative_order not found")
        end
        @info "Covariant derivative not found"
        ∇ = nothing
    end

    if haskey(fid, "symmetry")
        symmetry = load_BTData(fid["symmetry"], Symmetry{FT})
        el_i_irr = load_BTData(fid["initialstate_electron"], QMEStates{FT})
        el_i = load_BTData(fid["initialstate_electron_unfolded"], QMEStates{FT})
        ik_to_ikirr_isym = _data_hdf5_to_julia(read(fid, "ik_to_ikirr_isym"), Vector{Tuple{Int, Int}})
        map_i_to_f_vector, map_i_to_f_vector_tr = _qme_linear_response_unfold_map(el_i_irr, el_f, filename)
        qme_model = QMEIrreducibleKModel(; symmetry, ik_to_ikirr_isym, el_irr=el_i_irr,
                                           el=el_i, ∇, transport_params, el_f, ph,
                                           filename, map_i_to_f_vector, map_i_to_f_vector_tr)
        qme_model.el_to_el_f_sym_maps = _el_to_el_f_symmetry_maps(qme_model)
    else
        el_i = load_BTData(fid["initialstate_electron"], QMEStates{FT})
        map_i_to_f = _qme_linear_response_unfold_map_nosym(el_i, el_f, filename)
        qme_model = QMEModel(; el=el_i, ∇, transport_params, el_f, ph, filename, map_i_to_f)
    end
    close(fid)
    qme_model
end

"""
    multiply_Sᵢ(x::QMEVector, Sᵢ_irr, qme_model)
Multiply `Sᵢ` to a QMEVector `x` defined on the irreducible or full grid.
For the irredubiel grid, requires O(N_k^2 / N_sym) operations.
For the full grid case, requires O(N_k^2) operations but O(N_k^2 / N_sym) storage
(i.e. Sᵢ is stored only for the irreducible BZ, not the full BZ).
For each `k`, ``Sx_{m,n,k} = ∑_{k'} Sᵢ_irr_{m,n,kirr <- m',n',k'} x'(S^-1)_{m',n',k'}``
where ``k = S * k_irr` and `x'(S) = rotate_QMEVector_to_el_f(x, qme_model, isym)`.
"""
@timing "Sᵢ" function multiply_Sᵢ(x::QMEVector{<:Number}, Sᵢ_irr, qme_model::AbstractQMEModel)
    if qme_model isa QMEModel && x.state === qme_model.el_irr
        QMEVector(x.state, Sᵢ_irr * (qme_model.map_i_to_f * x.data))
    elseif x.state === qme_model.el
        # This block is called only when qme_model is a QMEIrreducibleKModel.
        Sin_x = similar(x)
        (; el, el_irr, symmetry, ik_to_ikirr_isym, el_to_el_f_sym_maps) = qme_model

        x_f = qme_model._buffer_el_f_sym
        Sx_irr = qme_model._buffer_el_irr_sym
        @views for (isym, symop) in enumerate(symmetry)
            isym_inv = findfirst(s -> s ≈ inv(symop), symmetry)
            if symop.is_tr
                mul!(x_f[:, isym], el_to_el_f_sym_maps[isym_inv], conj.(x.data))
            else
                mul!(x_f[:, isym], el_to_el_f_sym_maps[isym_inv], x.data)
            end
        end
        mul!(Sx_irr, Sᵢ_irr, x_f)
        for i = 1:el.n
            (; ib1, ib2, ik) = el[i]
            ikirr, isym = ik_to_ikirr_isym[ik]
            ind_irr = get_1d_index(el_irr, ib1, ib2, ikirr)
            if symmetry[isym].is_tr
                Sin_x[i] = conj(Sx_irr[ind_irr, isym])
            else
                Sin_x[i] = Sx_irr[ind_irr, isym]
            end
        end
        Sin_x
    elseif qme_model isa QMEIrreducibleKModel && x.state === qme_model.el_irr
        # To use this, one needs to define unfolding (map_i_to_f) of a scalar-element QMEVector.
        # But usually the elements of x is a vector (e.g. δᴱρ has three components for the
        # three E field directions), so this will function not be used anyway.
        error("multiply_Sᵢ for QMEVector with scalar elements not implemented")
    else
        error("x.state must be qme_model.el or qme_model.el_irr.")
    end
end

function multiply_Sᵢ(x::QMEVector{Vec3{FT}}, Sᵢ_irr, qme_model::AbstractQMEModel) where FT
    if x.state === qme_model.el_irr
        # We need `map_i_to_f` because `Sᵢ` maps states in `el_f` to `el_i` (i.e. has size
        # `(el_i.n, el_f.n)`), while `δρ` is for states in `el_i`. `el_i` and `el_f` can
        # differ due to use of irreducible grids, different windows, different grids, etc.
        # So, we need to first map `δρ` to states `el_f` using `map_i_to_f`.

        # Can be optimized if Sᵢ_irr * qme_model.map_i_to_f_vector is computed and stored,
        # because we are doing (Nk_irr, Nk) * (Nk, Nk_irr) * (Nk_irr,) multiplication.
        if qme_model isa QMEModel
            QMEVector(x.state, Sᵢ_irr * (qme_model.map_i_to_f_vector * x.data))
        elseif qme_model isa QMEIrreducibleKModel
            # Here, x must be a time-reversal odd and inversion even vector
            # because qme_model.map_i_to_f_vector is calculated with such assumptions.
            ( QMEVector(x.state, Sᵢ_irr * (qme_model.map_i_to_f_vector * x.data))
            + QMEVector(x.state, Sᵢ_irr * (qme_model.map_i_to_f_vector_tr * conj.(x.data))) )
        else
            throw(ArgumentError("invalid qme_model type $(typeof(qme_model))"))
        end
    elseif x.state === qme_model.el
        # @warn "Can be very inefficient compared to QMEVector{ComplexF64}."
        Sx = zeros(FT, 3, size(x)...)
        @views for i in 1:3
            x_i = QMEVector(x.state, [v[i] for v in x.data])
            Sx[i, :] .= multiply_Sᵢ(x_i, Sᵢ_irr, qme_model).data
        end
        QMEVector(x.state, Vec3.(eachcol(Sx)))
    else
        error("x.state must be qme_model.el or qme_model.el_irr.")
    end
end

# Wrappers for transport-related functions

function bte_compute_μ!(model::AbstractQMEModel; kwargs...)
    bte_compute_μ!(model.transport_params, BTStates(model.el_irr); kwargs...)
end

function compute_qme_scattering_matrix!(model::AbstractQMEModel; kwargs...)
    (; filename, transport_params, el_irr, el_f, ph) = model
    model.Sₒ_irr, model.Sᵢ_irr = compute_qme_scattering_matrix(filename, transport_params,
                                    el_irr, el_f, ph; kwargs...)
    unfold_scattering_out_matrix!(model)
    model
end

function set_constant_qme_scattering_matrix!(model::AbstractQMEModel, inv_τ_constant)
    model.Sₒ_irr = [I(model.el_irr.n) * (-inv_τ_constant + 0.0im) for _ in model.transport_params.Tlist]
    unfold_scattering_out_matrix!(model)
    model
end
