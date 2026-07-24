# Architecture/backend primitives, in the spirit of DFTK's `architecture.jl`: a backend object
# (`CPUBackend()` / `GPUBackend(proto)`) selects device placement, and `to_device` / `free_bytes` /
# `synchronize` dispatch on it. The `GPUBackend` methods live in the CUDA extension
# (`ext/ElectronPhononCUDAExt.jl`); the base package defines only the CPU methods, so it loads and
# runs on CPU-only machines. Nothing here is exported (use `ElectronPhonon.<name>`).

# Backend objects: one resolution point per driver entry (`backend = use_gpu ? gpu_backend() :
# CPUBackend()`), then carried in `LoopContext` (see calculator/AbstractCalculator.jl). Below the
# driver entry, code allocates buffers via `alloc(backend, T, dims...)`, moves data with
# `to_device(backend, x)`, and queries `free_bytes(backend)` / `synchronize(backend)`, so `use_gpu`
# never threads through the interfaces. `GPUBackend` carries a device-array prototype that `alloc`
# uses as a `similar` template; `gpu_backend()` (extension) builds one with an empty prototype so a
# backend can be constructed before any array is moved.
abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct GPUBackend{AT <: AbstractArray} <: AbstractBackend
    proto :: AT     # allocation prototype (a device array); `alloc` uses `similar(proto, T, dims...)`
end

"""
    gpu_backend() -> GPUBackend

Construct a GPU backend carrying a device-array prototype. Provided by a package extension (e.g.
`ElectronPhononCUDAExt` for CUDA); calling it without the relevant extension loaded raises a
`MethodError`. Not exported; use `ElectronPhonon.gpu_backend`.
"""
function gpu_backend end

"""
    to_device(backend, x)

Move `x` (a host array or `WannierObject`) onto `backend`'s device. `CPUBackend` is the identity;
the CUDA extension converts to a `CuArray`-backed object for a `GPUBackend`. The backend always
says where "device" is (mirrors DFTK's `to_device(architecture, x)`); there is deliberately no 1-arg
form. Not exported; use `ElectronPhonon.to_device`.
"""
to_device(::CPUBackend, x) = x

alloc(::CPUBackend, ::Type{T}, dims...) where {T} = Array{T}(undef, dims...)
alloc(b::GPUBackend, ::Type{T}, dims...) where {T} = similar(b.proto, T, dims...)

"""
    free_bytes(backend) -> Int

Free device memory (bytes) on `backend`, used to decide whether a large buffer fits. `CPUBackend`
returns `typemax(Int)` (host allocation is governed by RAM, not this check); the CUDA extension
returns `CUDA.free_memory()` for a `GPUBackend`.
"""
free_bytes(::CPUBackend) = typemax(Int)

"""
    synchronize(backend)

Block until queued device work on `backend` completes. No-op on `CPUBackend` (host work is
synchronous); the CUDA extension calls `CUDA.synchronize()`. Used to bound the host look-ahead in
the GPU e-ph loop so per-tile scratch does not pile up in the memory pool.
"""
synchronize(::CPUBackend) = nothing

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
