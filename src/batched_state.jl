# The batched state containers' shared machinery. `BatchedPhononState` and `BatchedElectronState`
# hold their quantities in a `qty` NamedTuple with the point index on the last axis, read them as
# properties, and are gathered into tiles by `stage!`.

abstract type AbstractBatchedState end

# A quantity name must not shadow a plain field, or `getproperty` could not tell them apart.
function _check_quantity_names(::Type{S}, names) where {S}
    for name in names
        name ∈ fieldnames(S) && throw(ArgumentError(
            "quantity name :$name is a field of $(nameof(S))"))
    end
end

# A plain field by `getfield`, any other name looked up in `qty`.
function Base.getproperty(b::AbstractBatchedState, name::Symbol)
    hasfield(typeof(b), name) && return getfield(b, name)
    qty = getfield(b, :qty)
    haskey(qty, name) || throw(ArgumentError("quantity :$name was not requested from this " *
        "$(nameof(typeof(b))); it holds $(keys(qty))"))
    getfield(qty, name)
end

Base.propertynames(b::AbstractBatchedState) =
    (fieldnames(typeof(b))..., keys(getfield(b, :qty))...)

"""
    quantity_arrays(b) -> Tuple

Every array of `b` with one entry per point on its last axis: the quantities, and for electrons
`iband_offset` and `nband`. Allocation of a tile, the gather `stage!` and byte counts map over it.
"""
function quantity_arrays end

"""
    alloc_tile(b, backend, width) -> typeof(b)-like

An uninitialised buffer of `width` points with the quantities, box width and element types of `b`,
on `backend`, and no point set. Filled by [`stage!`](@ref) or by a builder.
"""
function alloc_tile end

"""
    stage!(tile, b, inds)

Copy the points `inds` of `b` into the first `length(inds)` points of `tile`, for every array of
[`quantity_arrays`](@ref), so a tile carries the window metadata with the states. Entries past
`length(inds)` are left as they were.

`inds` is a range or a host vector, bounds-checked here, or an index array already on `b`'s
device, used as it is: its caller has checked it, since a device check costs a reduction and a
device-to-host read. A host index vector for a device `b` is uploaded once per call, so a hot loop
keeps its own device index buffer instead. A host `b` into a device `tile` is gathered on the host
and uploaded in one copy per array.
"""
function stage!(tile::AbstractBatchedState, b::AbstractBatchedState, inds)
    keys(tile.qty) == keys(b.qty) || throw(ArgumentError(
        "tile holds $(keys(tile.qty)), the source $(keys(b.qty))"))
    _check_box(tile, b)
    length(inds) <= tile.n || throw(ArgumentError(
        "$(length(inds)) points do not fit a tile of width $(tile.n)"))
    host_inds = on_backend(CPUBackend(), inds)
    host_inds && checkbounds(Base.OneTo(b.n), inds)
    x = first(quantity_arrays(b))
    src_inds = host_inds && !(inds isa AbstractUnitRange) && !on_backend(CPUBackend(), x) ?
        copyto!(similar(x, Int, length(inds)), inds) : inds
    foreach((dst, src) -> _gather_last!(dst, src, src_inds), quantity_arrays(tile),
            quantity_arrays(b))
    tile
end

_check_box(tile, b) = throw(ArgumentError(
    "cannot stage a $(nameof(typeof(b))) into a $(nameof(typeof(tile)))"))

# Gather `src[..., inds]` into `dst[..., 1:length(inds)]`, `inds` a range or on `src`'s backend.
# A host source and a device destination are not one broadcast: gather on the host, then one
# contiguous upload.
function _gather_last!(dst::AbstractArray{T, N}, src::AbstractArray{T, N}, inds) where {T, N}
    d = selectdim(dst, N, 1:length(inds))
    colons = ntuple(_ -> Colon(), N - 1)
    if !on_backend(CPUBackend(), src)
        # The indices were checked on the host (`stage!`); a device check would cost a reduction
        # kernel and a device-to-host read per array.
        @inbounds d .= view(src, colons..., inds)
    elseif on_backend(CPUBackend(), dst)
        d .= view(src, colons..., inds)
    else
        copyto!(d, src[colons..., inds])
    end
    dst
end
