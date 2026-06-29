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

"""
    eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                        nbandkq, nbandk, nm, nqc, ni_stride; i0 = 0)

Device-resident scatter for a calculator that keeps `g2`/`ωq` on the device (no per-chunk
host streaming). For every `(m, n, ν, j)` entry of `g2vals` `(nbandkq, nbandk, nm, nqc)`, look
up the state indices `i = imap_i_col[n]` and `f = imap_f[m, ikqs[j]]`; if both are in-window
(`> 0`), write the value into the mode-fastest linear slot
`lin = ν + nm·(i−i0−1) + nm·ni_stride·(f−1)` of the flat `g2_out` / `ωq_out`. For the full
device-resident buffer pass `ni_stride = n_i`, `i0 = 0`. For a *block* buffer holding only one
outer-k tile, pass the buffer's i-extent as `ni_stride` and the tile's global-i offset as `i0`
(so global state `i` lands at local row `i − i0`)
(`ω = ωq[ν, j]`). The target `lin` indices are unique across the run (distinct k → distinct i,
distinct k+q → distinct f), so the writes never collide (no atomics needed). Generic
(CPU/fallback) method; the CUDA extension provides a one-kernel `CuArray` method.

A helper for downstream device-resident calculators: from their `run_calculator_batched!` hook
they call this to scatter each e-ph chunk's `g2`/`ωq` into their own window-mapped device
accumulators. The library itself stays agnostic to any particular calculator.

TODO: the non-collision invariant (unique `lin` indices across the run) has no in-repo test —
correctness currently rides on the downstream calculator's tests. Add a small scatter round-trip
test that checks the CPU and CUDA methods agree and that no two writes collide.
"""
function eph_window_scatter!(g2_out, ωq_out, g2vals, imap_i_col, imap_f, ikqs, ωq,
                             nbandkq::Int, nbandk::Int, nm::Int, nqc::Int, ni_stride::Int;
                             i0::Int = 0)
    @inbounds for j in 1:nqc, ν in 1:nm, n in 1:nbandk, m in 1:nbandkq
        i = imap_i_col[n]
        f = imap_f[m, ikqs[j]]
        if i > 0 && f > 0
            # Block-resident buffer: write the global outer state `i` at local row `i − i0`
            # (i0 = the tile's i offset, 0 for the full-resident buffer) with i-stride `ni_stride`
            # (= the buffer's i-extent; = total n_i for the full buffer).
            lin = ν + nm * (i - i0 - 1) + nm * ni_stride * (f - 1)
            g2_out[lin] = g2vals[m, n, ν, j]
            ωq_out[lin] = ωq[ν, j]
        end
    end
    nothing
end
