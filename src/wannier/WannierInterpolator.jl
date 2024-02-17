export get_interpolator
export get_fourier!

abstract type AbstractWannierInterpolator{T} end
Base.eltype(::Type{<:AbstractWannierInterpolator{T}}) where {T} = Complex{T}

struct NormalWannierInterpolator{T, WT <: AbstractWannierObject} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated
    parent::WT

    # Buffers for normal Fourier transform
    rdotk::Vector{T}
    phase::Vector{Complex{T}}

    function NormalWannierInterpolator(parent::WT) where {WT <: AbstractWannierObject{T}} where {T}
        nr = length(parent.irvec)
        new{T, WT}(parent, zeros(T, nr), zeros(Complex{T}, nr))
    end
end


mutable struct GridoptWannierInterpolator{T, WT <: AbstractWannierObject} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated
    const parent::WT

    # For gridopt Fourier transform
    const gridopt::GridOpt{T}

    # Uheck if `gridopt` is up-to-date with `parent`.
    # If the ids do not match, reset `gridopt`.
    _id::Int

    function GridoptWannierInterpolator(parent::WT) where {WT <: AbstractWannierObject{T}} where {T}
        new{T, WT}(parent, GridOpt(T, parent.irvec, parent.ndata), parent._id)
    end
end


function Base.getproperty(obj::AbstractWannierInterpolator, name::Symbol)
    if name === :nr || name === :ndata
        getfield(obj.parent, name)
    else
        getfield(obj, name)
    end
end


function get_interpolator(obj::AbstractWannierObject; fourier_mode="normal")
    if fourier_mode === "normal"
        NormalWannierInterpolator(obj)
    elseif fourier_mode === "gridopt"
        GridoptWannierInterpolator(obj)
    else
        throw(ArgumentError("Wrong fourier_mode $fourier_mode"))
    end
end


@timing "get_fourier" function get_fourier!(op_k, obj::NormalWannierInterpolator{T, WT}, xk) where {T, WT}
    (; parent, phase) = obj
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == parent.ndata
    op_k_1d = _reshape(op_k, (length(op_k),))

    phase .= cispi.(2 .* dot.(parent.irvec, Ref(xk)))

    if WT <: DiskWannierObject
        op_k_1d .= 0
        for ir in 1:parent.nr
            op_k_1d .+= phase[ir] .* read_op_r(parent, ir)
        end
    else
        @views mul!(op_k_1d, parent.op_r[1:parent.ndata, :], phase)
    end

    op_k
end


@timing "get_fourier" function get_fourier!(op_k, obj::GridoptWannierInterpolator{T}, xk) where {T}
    (; parent, gridopt) = obj
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == parent.ndata
    ndata = parent.ndata
    op_k_1d = _reshape(op_k, (length(op_k),))

    if obj._id != parent._id
        reset_gridopt!(obj.gridopt)
        obj._id = parent._id
    end

    if ! isapprox(xk[1], gridopt.k1, atol=sqrt(eps(T))/100)
        gridopt_set23!(gridopt, parent, xk[1], ndata)
    end
    if ! isapprox(xk[2], gridopt.k2, atol=sqrt(eps(T))/100)
        gridopt_set3!(gridopt, xk[2], ndata)
    end

    gridopt_get3!(op_k_1d, gridopt, xk[3], ndata)

    op_k
end
