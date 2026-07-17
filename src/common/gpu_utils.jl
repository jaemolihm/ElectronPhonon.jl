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
    device_free_bytes(gpu_array) -> Int

Free memory (bytes) on the backend `gpu_array` lives on, used to decide whether a large buffer fits
on the device. Generic fallback returns `typemax(Int)` (host memory: assume it always fits — the
caller's host allocation is governed by RAM, not this check). The CUDA extension returns
`CUDA.free_memory()` for a `CuArray` gpu_array.
"""
device_free_bytes(gpu_array) = typemax(Int)

"""
    device_synchronize(gpu_array)

Block until queued device work on `gpu_array`'s backend completes. Generic fallback is a no-op
(host work is synchronous); the CUDA extension calls `CUDA.synchronize()`. Used to bound the
host look-ahead in the GPU e-ph loop so per-tile scratch does not pile up in the memory pool.
"""
device_synchronize(gpu_array) = nothing

# Backend objects: one resolution point per driver entry (`backend = use_gpu ? GPUBackend(proto)
# : CPUBackend()`), then carried in `LoopContext` (see calculator/AbstractCalculator.jl). Below the
# driver entry, calculators allocate device/host buffers via `alloc(backend, T, dims...)` and query
# `free_bytes(backend)` / `synchronize(backend)`, so `use_gpu`/`gpu_array` never thread through the
# calculator interface. No extension code is needed for the types themselves: `GPUBackend` carries a
# device-array prototype, and `free_bytes`/`synchronize` route through the `device_*` primitives
# above whose `CuArray` methods already live in the extension.
abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct GPUBackend{AT <: AbstractArray} <: AbstractBackend
    proto :: AT     # allocation prototype (a device array, e.g. `to_device(model.epmat).op_r`)
end

alloc(::CPUBackend, ::Type{T}, dims...) where {T} = Array{T}(undef, dims...)
alloc(b::GPUBackend, ::Type{T}, dims...) where {T} = similar(b.proto, T, dims...)
free_bytes(::CPUBackend)  = typemax(Int)
free_bytes(b::GPUBackend) = device_free_bytes(b.proto)
synchronize(::CPUBackend)  = nothing
synchronize(b::GPUBackend) = device_synchronize(b.proto)

# Backend-routed move-to-device: the CPU backend is an identity (host object stays host); the GPU
# backend forwards to the extension's converting `to_device(obj)`. Loop bodies below the driver
# entry call this 2-arg form so no `use_gpu`/backend `Bool` threads through them — the backend
# object alone decides. (The base has no 1-arg `to_device` method; only the extension defines one,
# so `to_device(::CPUBackend, obj)` is what lets the base package load and run on the host.)
to_device(::CPUBackend, obj) = obj
to_device(b::GPUBackend, obj) = to_device(obj)

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
