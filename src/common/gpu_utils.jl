# Backend/device primitives. Each has a generic (CPU / host) method here and a `CuArray` method in
# the CUDA extension (`ext/ElectronPhononCUDAExt.jl`); the batched Wannier→Bloch drivers and the
# device-resident calculators dispatch on the backend of their arrays. Nothing here is exported.

"""
    to_device(obj)
Move an object (e.g. a Wannier object / interpolator) to a compute device such as a GPU.
Methods are provided by package extensions (e.g. `ElectronPhononCUDAExt` for CUDA); the base
package defines no method, so calling this without the relevant extension loaded raises a
`MethodError`. Not exported (the name is generic); use `ElectronPhonon.to_device`.
"""
function to_device end

"""
    device_array_prototype(::Type{T}) -> a length-0 device array of eltype `T`

A minimal device array used purely as a backend tag / `similar` prototype: it carries the device
array type (so callers can `similar(proto, eltype, dims)` and dispatch `device_free_bytes`) without
allocating real storage. The base package defines no method; the CUDA extension returns a 0-length
`CuArray{T}`. Called only on the GPU path (`use_gpu = true`), where the relevant extension is
guaranteed loaded (calling it without the extension raises a `MethodError`, like `to_device`).

Returns a length-0 array INSTANCE (0 bytes of device buffer) rather than a bare `Type`: the
device-allocation idiom used throughout the GPU path is `similar(proto, eltype, dims)`, and that
3-arg form with a differing eltype (e.g. `Int` imap arrays from an `FT` prototype) has no
`similar(::Type, ::Type, dims)` method for `CuArray`. Dispatching on an instance keeps `src`
backend-generic — it never needs to name `CuArray{Int}(undef, …)` or a backend-specific
constructor. See the PR #6 discussion on `BoltzmannCalculator` residency.
"""
function device_array_prototype end

"""
    device_free_bytes(proto) -> Int

Free memory (bytes) on the backend `proto` lives on, used to decide whether a large buffer fits
on the device. Generic fallback returns `typemax(Int)` (host memory: assume it always fits — the
caller's host allocation is governed by RAM, not this check). The CUDA extension returns
`CUDA.free_memory()` for a `CuArray` proto. Used by `BoltzmannCalculator`'s `setup_calculator!` to
choose between full- and block-device-resident Sᵢ.
"""
device_free_bytes(proto) = typemax(Int)

"""
    device_synchronize(proto)

Block until queued device work on `proto`'s backend completes. Generic fallback is a no-op
(host work is synchronous); the CUDA extension calls `CUDA.synchronize()`. Used to bound the
host look-ahead in the GPU e-ph loop so per-tile scratch does not pile up in the memory pool.
"""
device_synchronize(proto) = nothing

@inline _batched_op(t::Char, X) = t == 'N' ? X : (t == 'T' ? transpose(X) : adjoint(X))

"""
    batched_gemm!(transA, transB, A, B, C)

`C[:,:,b] = op(transA, A[:,:,b]) * op(transB, B[:,:,b])` for every batch `b` (α=1, β=0),
where `op('N',X)=X`, `op('T',X)=transpose(X)`, `op('C',X)=adjoint(X)`. The CPU method loops
over `mul!`; the CUDA extension uses `CUBLAS.gemm_strided_batched!`.
"""
function batched_gemm!(transA::Char, transB::Char,
                       A::AbstractArray{T,3}, B::AbstractArray{T,3}, C::AbstractArray{T,3}) where {T}
    @assert size(A, 3) == size(B, 3) == size(C, 3)
    @views for b in axes(C, 3)
        mul!(C[:, :, b], _batched_op(transA, A[:, :, b]), _batched_op(transB, B[:, :, b]))
    end
    C
end
