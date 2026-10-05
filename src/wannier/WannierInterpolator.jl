using ElectronPhonon.AllocatedLAPACK: HermitianEigenWsSYEV

export get_interpolator
export get_interpolator_channel
export get_fourier!

abstract type AbstractWannierInterpolator{T} end
Base.eltype(::Type{<:AbstractWannierInterpolator{T}}) where {T} = Complex{T}

struct NormalWannierInterpolator{T, WT <: AbstractWannierObject} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated
    parent::WT

    # Buffers for normal Fourier transform
    rdotk::Vector{T}
    phase::Vector{Complex{T}}

    # Output buffer
    out::Vector{Complex{T}}

    # Buffer for intermediate calculations
    buffer::Vector{Complex{T}}

    # Buffer for diagonalization
    ws::HermitianEigenWsSYEV{Complex{T},T}

    function NormalWannierInterpolator(parent::WT) where {WT <: AbstractWannierObject{T}} where {T}
        nr = length(parent.irvec)
        ws = HermitianEigenWsSYEV{Complex{T},T}()
        new{T, WT}(parent, zeros(T, nr), zeros(Complex{T}, nr), zeros(Complex{T}, parent.ndata), Complex{T}[], ws)
    end
end


mutable struct GridoptWannierInterpolator{T, WT <: AbstractWannierObject} <: AbstractWannierInterpolator{T}
    # Parent WannierObject to be interpolated
    const parent::WT

    # For gridopt Fourier transform
    const gridopt::GridOpt{T}

    # Output buffer
    out::Vector{Complex{T}}

    # Buffer for intermediate calculations
    buffer::Vector{Complex{T}}

    # Buffer for diagonalization
    ws::HermitianEigenWsSYEV{Complex{T},T}

    # Uheck if `gridopt` is up-to-date with `parent`.
    # If the ids do not match, reset `gridopt`.
    _id::Int

    function GridoptWannierInterpolator(parent::WT, threads = false) where {WT <: AbstractWannierObject{T}} where {T}
        gridopt = GridOpt(T, parent.irvec, parent.ndata, threads)
        ws = HermitianEigenWsSYEV{Complex{T},T}()
        new{T, WT}(parent, gridopt, zeros(Complex{T}, parent.ndata), Complex{T}[], ws, parent._id)
    end
end


@inline function Base.getproperty(obj::AbstractWannierInterpolator, name::Symbol)
    if name === :nr
        getfield(obj.parent, name)
    else
        getfield(obj, name)
    end
end


"""
    get_interpolator(obj::AbstractWannierObject; fourier_mode="normal", batch_size=nothing,
                     nk_hint=typemax(Int), threads=false, backend=CPUBackend())
Return a interpolator for the given object.
For a multithreaded use, one must use `get_interpolator_channel` instead.

# Keyword Arguments
- `fourier_mode`: Interpolation mode - "normal", "batched", "gridopt", or "batched-gridopt".
- `batch_size`: Block width for the "batched" and "batched-gridopt" modes. `nothing` takes
  [`ElectronPhonon._default_batch_size`](@ref) for `backend`.
- `nk_hint`: Largest k-list the caller will hand this interpolator, if known. Only narrows a
  budgeted default — `batch_size = min(default, nk_hint)` — so unlike a plain `batch_size = kpts.n`
  it stays bounded however large the grid, and it never widens the block past the byte budget.
- `threads`: Enable threading for "gridopt" and "batched-gridopt" modes (default: false)
- `backend`: Where the "batched" mode's buffers live; must match `obj.op_r` (default: `CPUBackend()`)
"""
function get_interpolator(obj::AbstractWannierObject; fourier_mode="normal", batch_size=nothing,
                          nk_hint=typemax(Int), threads=false, backend=CPUBackend())
    if fourier_mode === "normal"
        NormalWannierInterpolator(obj)
    elseif fourier_mode === "batched"
        bs = something(batch_size, _default_batch_size(backend, length(obj.irvec), obj.ndata; nk_hint))
        BatchedWannierInterpolator(obj; batch_size = bs, backend)
    elseif fourier_mode === "gridopt"
        GridoptWannierInterpolator(obj, threads)
    elseif fourier_mode === "batched-gridopt"
        # Host-only mode: its buffers are plain `Matrix`, so it takes the CPU default and cannot
        # honour another backend. Say so rather than silently running on the host.
        backend isa CPUBackend || throw(ArgumentError(
            "fourier_mode=\"batched-gridopt\" is host-only and cannot run on a " *
            "$(nameof(typeof(backend))); use \"batched\""))
        bs = something(batch_size, _default_batch_size(backend, length(obj.irvec), obj.ndata))
        BatchedGridoptWannierInterpolator(obj; batch_size = bs, threads)
    else
        throw(ArgumentError("Wrong fourier_mode $fourier_mode"))
    end
end

"""
    get_interpolator_channel(obj::AbstractWannierObject{T}; fourier_mode, batch_size = nothing,
                             nbuffers = nthreads(), backend = CPUBackend())
Return a `Channel` of `nbuffers` interpolators for multithreading.

Host-only: the channel exists to give one interpolator per CPU thread, and every call site is a
CPU-threaded e-ph loop. A non-`CPUBackend` would put `nbuffers` copies of a device-budgeted scratch
buffer on the card at once, so it is rejected rather than silently multiplied.
"""
function get_interpolator_channel(obj::AbstractWannierObject{T}; fourier_mode, batch_size = nothing,
        nbuffers = nthreads(), backend = CPUBackend()) where {T}
    backend isa CPUBackend || throw(ArgumentError(
        "get_interpolator_channel is host-only and cannot run on a $(nameof(typeof(backend))); " *
        "build the interpolator directly with get_interpolator"))
    bs = something(batch_size, _default_batch_size(backend, length(obj.irvec), obj.ndata))
    itp_channel = Channel{AbstractWannierInterpolator{T}}(nbuffers)
    Folds.foreach(1:nbuffers) do _
        put!(itp_channel, get_interpolator(obj; fourier_mode, batch_size = bs, backend))
    end
    itp_channel
end


function register_kpoints!(obj::AbstractWannierInterpolator, xk_list)
    # Null operation. Used only for bached interpolators.
end

function skip_registered_kpoint!(obj::AbstractWannierInterpolator)
    # Null operation. Used only for bached interpolators.
end


@timing "get_fourier" function get_fourier!(op_k, obj::NormalWannierInterpolator{T}, xk) where {T}
    (; parent, phase) = obj
    @assert eltype(op_k) == Complex{T}
    @assert length(op_k) == parent.ndata
    op_k_1d = _reshape(op_k, (length(op_k),))

    phase .= cispi.(2 .* dot.(parent.irvec, Ref(xk)))

    @views mul!(op_k_1d, parent.op_r[1:parent.ndata, :], phase)

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
