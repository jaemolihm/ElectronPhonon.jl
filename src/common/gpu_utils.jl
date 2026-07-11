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
    device_free_bytes(proto) -> Int

Free memory (bytes) on the backend `proto` lives on, used to decide whether a large buffer fits
on the device. Generic fallback returns `typemax(Int)` (host memory: assume it always fits — the
caller's host allocation is governed by RAM, not this check). The CUDA extension returns
`CUDA.available_memory()` for a `CuArray` proto.

TODO: no in-repo caller yet — this exists for the deferred device memory-estimate helper
(see README_GPU.md). Wire it up when that helper is written, or remove it.
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
