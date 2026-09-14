export SmearingType

# A device-safe, isbits smeared delta function usable inside GPU kernels.
#
# Design note (why not reuse `delta_smeared` in common/utils.jl): `delta_smeared` branches on a
# `Symbol` smearing type at runtime and has a `throw` path for unknown types, neither of which
# compiles inside a CUDA kernel. `SmearingType` instead stores the method as an `Int` and
# pre-computes the normalization `fac` at construction, so the callable has no `Symbol`/`throw`
# branches and is `isbitstype` (required to live in a `CuArray` and be evaluated on the device).
# It intentionally supports only the analytic Gaussian/Lorentzian deltas the transport scatter needs.

"""
    SmearingType(method_name::Symbol, η::Real)

Callable, `isbitstype` smeared delta function `δ_η(Δe)` for use in both host and GPU-kernel code.
`method_name` is `:Gaussian` or `:Lorentzian`; `η` is the smearing width. Evaluate it as a function:

    δ = SmearingType(:Gaussian, 0.01)
    δ(Δe)   # == exp(-(Δe/η)^2) / (sqrt(π) η)

Formulas (both normalized to unit integral):
- `:Gaussian`   - `δ_η(Δe) = exp(-(Δe/η)^2) / (sqrt(π) η)`
- `:Lorentzian` - `δ_η(Δe) = (η/π) / (Δe^2 + η^2)`

The normalization constant is pre-computed into `fac` at construction so evaluation is branch-light.
See the design note above for why this is separate from `delta_smeared`.
"""
struct SmearingType{FT <: Real}
    method::Int  # Smearing method: 1=Gaussian, 2=Lorentzian
    η::FT        # Smearing width
    fac::FT      # Pre-computed constant factor for optimization

    # Inner constructor to validate the smearing method and pre-compute the constant factor.
    function SmearingType(method_name::Symbol, η::FT) where {FT <: Real}
        if method_name == :Gaussian
            method = 1
            fac = FT(1.0 / sqrt(π)) / η
        elseif method_name == :Lorentzian
            method = 2
            fac = η / FT(π)
        else
            throw(ArgumentError("Unknown smearing method: $method_name (expected :Gaussian or :Lorentzian)"))
        end
        new{FT}(method, η, fac)
    end
end


@inline function (smearing::SmearingType)(Δe)
    (; method, η, fac) = smearing
    if method == 1
        fac * exp(-(Δe / η)^2)
    else  # method == 2
        fac / (Δe^2 + η^2)
    end
end
