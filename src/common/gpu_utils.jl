# Architecture/backend primitives, in the spirit of DFTK's `architecture.jl`: a backend object
# (`CPUBackend()` / `GPUBackend(proto)`) selects device placement, and `to_device` / `free_bytes` /
# `synchronize` dispatch on it. The `GPUBackend` methods live in the CUDA extension
# (`ext/ElectronPhononCUDAExt.jl`); the base package defines only the CPU methods, so it loads and
# runs on CPU-only machines. Nothing here is exported (use `ElectronPhonon.<name>`).

# Backend objects: the user passes one to a driver entry (`backend = CPUBackend()` or
# `backend = gpu_backend()`), which carries it in `LoopContext` (see calculator/AbstractCalculator.jl).
# Everywhere below, code allocates buffers via `alloc(backend, T, dims...)`, moves data with
# `to_device(backend, x)`, and queries `free_bytes(backend)` / `synchronize(backend)`, so the backend
# object is the only thing that says where "device" is. `GPUBackend` carries a device-array prototype
# that `alloc` uses as a `similar` template; `gpu_backend()` (extension) builds one with an empty
# prototype so a backend can be constructed before any array is moved. Note the backend does NOT say
# which loop SHAPE runs — that is the drivers' separate `batched` keyword, which defaults from the
# backend but can be set independently (batched-on-`CPUBackend` is a validation configuration).
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
    is_host(backend) -> Bool

Whether `backend` places its arrays in host memory: `true` for `CPUBackend`, `false` for every
other backend. The question a consumer asks when the host route is not array-generic -- e.g. a
precompute that only exists in RAM -- and the device route is the generic one.

Named after the host side because that side is the closed one: `CPUBackend` is a set EP always
knows by name, while device backends are open-ended, so the predicate is `true` on a known list and
`false` by fallback. A future host-side backend (threaded / MPI / pinned-host) then opts in by
adding a method, instead of having to remember to override an inherited `true`. Spelled `is_` and
not `on_` because `on_backend` below already asks about *array* residency. Not exported; use
`ElectronPhonon.is_host`.
"""
is_host(::CPUBackend) = true
is_host(::AbstractBackend) = false

"""
    backend_from(use_gpu::Bool) -> AbstractBackend

`gpu_backend()` if `use_gpu`, `CPUBackend()` otherwise — the one place a `Bool` becomes a backend.
A driver script's `use_gpu` switch stops here: below it the backend object is the single authority
on residency, so a selection, an `Eigenpairs` cache and a calculator cannot disagree about it.
Throws an `ArgumentError` naming the extension when `use_gpu` is asked for and none is loaded,
which `gpu_backend()`'s bare `MethodError` does not say. Not exported; use
`ElectronPhonon.backend_from`.
"""
function backend_from(use_gpu::Bool)
    use_gpu || return CPUBackend()
    hasmethod(gpu_backend, Tuple{}) || throw(ArgumentError(
        "backend_from(true) needs a GPU extension loaded (`using CUDA`); gpu_backend() has no method"))
    gpu_backend()
end

"""
    to_device(backend, x)

Move `x` (a host array or `WannierObject`) onto `backend`'s device. `CPUBackend` is the identity;
the CUDA extension converts to a `CuArray`-backed object for a `GPUBackend`. The backend always
says where "device" is (mirrors DFTK's `to_device(architecture, x)`); there is deliberately no 1-arg
form. Not exported; use `ElectronPhonon.to_device`.

Placement, not a copy: because `CPUBackend` is the identity, the result may *be* `x`, so never
write to a buffer obtained this way unless you own `x` -- the write would go through to the
caller's array, and a read-only alias is invisible to a bitwise comparison. For a buffer that will
be written use [`to_device_copy`](@ref), or `alloc` / `alloc_zeros` plus a copy.
"""
to_device(::CPUBackend, x) = x

"""
    on_backend(backend, x::AbstractArray) -> Bool

Whether `x` is already resident on `backend`: host memory for `CPUBackend`, an array of
`backend.proto`'s type for a `GPUBackend`. The residency test that `to_device` is the conversion
for. Not exported; use `ElectronPhonon.on_backend`.

Host residency is read off `Base.BroadcastStyle` rather than `x isa Array`, so a `reinterpret` or
`view` wrapper over host memory counts as resident — it is, and `to_device(::CPUBackend, ·)` is the
identity on it. Every device array type carries its own array style (`CuArrayStyle` for CUDA.jl),
which is what separates the two cases without naming a device type here.
"""
on_backend(::CPUBackend, x::AbstractArray) =
    Base.BroadcastStyle(typeof(x)) isa Base.Broadcast.DefaultArrayStyle
on_backend(b::GPUBackend, x::AbstractArray) = x isa Base.typename(typeof(b.proto)).wrapper

"""
    check_on_backend(backend, x::AbstractArray, name = "array")

Throw an `ArgumentError` unless `x` is resident on `backend`, calling it `name` in the message. For
a caller-supplied array that has to live on the same side as the run's own buffers: a mismatch is
then reported here instead of surfacing deeper in as a mixed host/device operation, a per-element
transfer, or -- on a path that happens to take `Array(x)` anyway -- no error at all.
"""
function check_on_backend(backend::AbstractBackend, x::AbstractArray, name = "array")
    # Name the side `x` is actually on with the same test, so a host wrapper (a `view`, a
    # `reinterpret`) is not reported as a device array.
    on_backend(backend, x) || throw(ArgumentError(
        "$name is resident on the $(on_backend(CPUBackend(), x) ? "host" : "device") " *
        "(::$(typeof(x))), but the run uses $(nameof(typeof(backend))); build it with the run's " *
        "backend"))
    nothing
end

"""
    alloc(backend, ::Type{T}, dims...) -> AbstractArray{T}

Uninitialised array of `T` on `backend`: `Array{T}(undef, dims...)` for `CPUBackend`,
`similar(backend.proto, T, dims...)` for a `GPUBackend`. The contents are `undef`, so this is the
allocation for a buffer that is fully overwritten before it is read; a reduction target -- a buffer
accumulated INTO -- needs [`alloc_zeros`](@ref) instead. Not exported; use `ElectronPhonon.alloc`.
"""
alloc(::CPUBackend, ::Type{T}, dims...) where {T} = Array{T}(undef, dims...)
alloc(b::GPUBackend, ::Type{T}, dims...) where {T} = similar(b.proto, T, dims...)

"""
    alloc_zeros(backend, ::Type{T}, dims...) -> AbstractArray{T}

Zero-filled array of `T` on `backend`. `alloc` hands back `undef` memory, so a buffer that is
accumulated INTO rather than fully overwritten — a reduction target — has to be zeroed
first, and this is that allocation in one call. Not exported; use
`ElectronPhonon.alloc_zeros`.
"""
alloc_zeros(backend, ::Type{T}, dims...) where {T} = fill!(alloc(backend, T, dims...), zero(T))

"""
    to_device_copy(backend, A::AbstractArray) -> AbstractArray

Copy of `A` on `backend`, always a DISTINCT array; element type and size follow `A`. `to_device` is
the identity on `CPUBackend`, so it returns `A` itself there and a buffer taken from it would be
written through to the caller's array. This is the form to use when the result will be mutated.
Not exported; use `ElectronPhonon.to_device_copy`.
"""
to_device_copy(backend, A::AbstractArray) = copyto!(alloc(backend, eltype(A), size(A)), A)

"""
    free_bytes(backend) -> Int

Free device memory (bytes) on `backend`, used to decide whether a large buffer fits. `CPUBackend`
returns `typemax(Int)` (host allocation is governed by RAM, not this check); the CUDA extension
returns `CUDA.free_memory()` for a `GPUBackend`.

On `CPUBackend` that value is a sentinel rather than a measurement: use it in a comparison
(`nbytes <= free_bytes(backend)`, `x <= 0.7 * free_bytes(backend)`), never as an arithmetic
operand -- `free_bytes(backend) - nbytes` is a nonsense number one overflow away from a wrong
branch.
"""
free_bytes(::CPUBackend) = typemax(Int)

"""
    reclaim_device_memory(backend)

Return `backend`'s cached-but-unused device memory to the driver, so that the next `free_bytes`
reports what a large allocation can actually get. No-op on `CPUBackend`; the CUDA extension calls
`CUDA.reclaim()`.

CUDA.jl parks freed device memory in a stream-ordered pool instead of returning it, so `free_bytes`
reads far below the truth once an earlier calculation has run, and a buffer that would fit is
rejected. `GC.gc(true)` does not help — a dead device array goes back to the pool, not to the
driver. Trimming the pool is what moves `free_bytes`.
"""
reclaim_device_memory(::CPUBackend) = nothing

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
