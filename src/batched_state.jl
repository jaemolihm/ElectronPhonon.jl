# The gather shared by `gather_batched_electron_states!` and `gather_batched_phonon_states!`.

# `inds` on the backend of `x`: a range or an index array already on that backend as it is, a host
# vector checked against `1:n` on the host and uploaded once.
function _gather_index(x, inds, n)
    host_inds = on_backend(CPUBackend(), inds)
    host_inds && checkbounds(Base.OneTo(n), inds)
    host_inds && !(inds isa AbstractUnitRange) && !on_backend(CPUBackend(), x) ?
        copyto!(similar(x, Int, length(inds)), inds) : inds
end

# Copy `src[..., inds]` into `dst[..., 1:length(inds)]`. A host source and a device destination are
# not one broadcast: gather on the host, then one contiguous upload.
_gather_last!(::Nothing, ::Nothing, inds) = nothing
function _gather_last!(dst::AbstractArray{T, N}, src::AbstractArray{T, N}, inds) where {T, N}
    d = selectdim(dst, N, 1:length(inds))
    colons = ntuple(_ -> Colon(), N - 1)
    if !on_backend(CPUBackend(), src)
        # The indices were checked on the host (`_gather_index`); a device check would cost a
        # reduction kernel and a device-to-host read per array.
        @inbounds d .= view(src, colons..., inds)
    elseif on_backend(CPUBackend(), dst)
        d .= view(src, colons..., inds)
    else
        copyto!(d, src[colons..., inds])
    end
    dst
end
