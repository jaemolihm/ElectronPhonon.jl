# Shared helpers of the batched state containers.

# The index side of `copy_batched_electron_states!` / `copy_batched_phonon_states!`: `inds` on the
# backend of `array_on_backend`. A range, or an index array already on that backend, is returned as
# it is; a host vector is checked against `1:npoints` on the host and uploaded once.
function _copy_indices_on_backend(array_on_backend, inds, npoints)
    inds_on_host = on_backend(CPUBackend(), inds)
    inds_on_host && checkbounds(Base.OneTo(npoints), inds)
    if inds_on_host && !(inds isa AbstractUnitRange) && !on_backend(CPUBackend(), array_on_backend)
        copyto!(similar(array_on_backend, Int, length(inds)), Vector{Int}(inds))   # a host view has no bulk upload
    else
        inds
    end
end

# Copy `src[..., inds]` into `dst[..., 1:length(inds)]`, `inds` as `_copy_indices_on_backend`
# returns it; nothing to do for a quantity the destination does not hold.
_copy_last_axis!(::Nothing, src, inds) = nothing
function _copy_last_axis!(dst::AbstractArray{T, N}, src::AbstractArray{T, N}, inds) where {T, N}
    dst_head = selectdim(dst, N, 1:length(inds))
    colons = ntuple(_ -> Colon(), N - 1)
    if !on_backend(CPUBackend(), src)
        # GPU ← GPU. The indices were checked on the host (`_copy_indices_on_backend`); a device
        # check would cost a reduction kernel and a device-to-host read per array.
        @inbounds dst_head .= view(src, colons..., inds)
    elseif on_backend(CPUBackend(), dst)
        # CPU ← CPU
        dst_head .= view(src, colons..., inds)
    else
        # GPU ← CPU: not one broadcast; gather on the host, then one contiguous upload.
        copyto!(dst_head, src[colons..., inds])
    end
    dst
end
