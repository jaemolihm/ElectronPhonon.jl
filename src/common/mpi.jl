
# Adapted from DFTK.jl/src/common/mpi.jl

# Convenience functions for working with MPI
using MPI
using OffsetArrays

export mpi_world_comm
export mpi_initialized
export mpi_nprocs
export mpi_isroot
export mpi_myrank
export mpi_sum
export mpi_sum!
export mpi_gather
export mpi_allgather
export mpi_scatter
export mpi_bcast
export mpi_bcast!



"""
Initialize MPI. Must be called before doing any non-trivial MPI work
(even in the single-process case). Unlike the MPI.Init() function,
this can be called multiple times.
"""
function mpi_ensure_initialized()
    # MPI Thread level 3 means that the environment is multithreaded, but that only
    # one thread will call MPI at once
    # see https://www.open-mpi.org/doc/current/man3/MPI_Init_thread.3.php#toc7
    # TODO look more closely at interaction between MPI and threads
    MPI.Initialized() || MPI.Init_thread(MPI.THREAD_MULTIPLE)
end

mpi_world_comm() = MPI.COMM_WORLD
mpi_initialized() = MPI.Initialized()
mpi_finalize() = MPI.Finalize()

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD) = (mpi_ensure_initialized(); MPI.Comm_size(comm))
mpi_isroot(comm=MPI.COMM_WORLD) = (mpi_ensure_initialized(); MPI.Comm_rank(comm) == 0)
mpi_myrank(comm=MPI.COMM_WORLD) = (mpi_ensure_initialized(); MPI.Comm_rank(comm))
mpi_nprocs(comm::Nothing) = 1
mpi_isroot(comm::Nothing) = true
mpi_myrank(comm::Nothing) = 0
const MPI_ROOT = 0

"""
    _sync_device(x)

Block until the device work that filled `x` has completed, so a GPU-aware MPI implementation reads
a buffer that is actually written. A no-op for a host array or a scalar, which is every `x` the
wrappers below see today; the CUDA extension adds the `CuArray` method. Mirrors DFTK's `sync`.

Dispatches on the buffer and not on a backend, so it cannot reuse `synchronize(::AbstractBackend)`:
an MPI wrapper is handed an array by a caller that holds no backend object. The device method
dispatches on the concrete array type in the extension rather than on
`GPUArraysCore.AbstractGPUArray`, which would be a dependency for the residency test alone: the
`synchronize` call itself is per-vendor either way (DFTK does take that dependency and still
implements `synchronize_device` in its CUDA and AMDGPU extensions).
"""
_sync_device(x) = nothing

mpi_sum( arr, comm::MPI.Comm)  = (_sync_device(arr); MPI.Allreduce( arr, +, comm))
mpi_sum!(arr, comm::MPI.Comm)  = (_sync_device(arr); MPI.Allreduce!(arr, +, comm))
mpi_min( arr, comm::MPI.Comm)  = (_sync_device(arr); MPI.Allreduce( arr, min, comm))
mpi_min!(arr, comm::MPI.Comm)  = (_sync_device(arr); MPI.Allreduce!(arr, min, comm))
mpi_max( arr, comm::MPI.Comm)  = (_sync_device(arr); MPI.Allreduce( arr, max, comm))
mpi_max!(arr, comm::MPI.Comm)  = (_sync_device(arr); MPI.Allreduce!(arr, max, comm))
mpi_mean(arr, comm::MPI.Comm)  = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

mpi_reduce(arr, op, comm::MPI.Comm) = (_sync_device(arr); MPI.Allreduce(arr, op, comm))

mpi_bcast!(buf, root::Integer, comm::MPI.Comm) = (_sync_device(buf); MPI.Bcast!(buf, root, comm))
mpi_bcast( obj, root::Integer, comm::MPI.Comm) = (_sync_device(obj); MPI.bcast( obj, root, comm))
mpi_bcast!(buf, comm::MPI.Comm) = mpi_bcast!(buf, MPI_ROOT, comm)
mpi_bcast( obj, comm::MPI.Comm) = mpi_bcast( obj, MPI_ROOT, comm)

# Do nothing if comm is nothing.
mpi_min(  arr, comm::Nothing) = arr
mpi_max(  arr, comm::Nothing) = arr
mpi_sum(  arr, comm::Nothing) = arr
mpi_mean( arr, comm::Nothing) = arr
mpi_min!( arr, comm::Nothing) = arr
mpi_max!( arr, comm::Nothing) = arr
mpi_sum!( arr, comm::Nothing) = arr
mpi_mean!(arr, comm::Nothing) = arr

mpi_reduce(arr, op, comm::Nothing) = arr

mpi_bcast!(buf, root::Integer, comm::Nothing) = buf
mpi_bcast( obj, root::Integer, comm::Nothing) = obj
mpi_bcast!(buf, comm::Nothing) = buf
mpi_bcast( obj, comm::Nothing) = obj

# Whether every rank's array has the same size except along the last dimension. Reduced with `&`
# so a mismatch raises on all ranks, not only the one that sees it.
function _check_size(arr::AbstractArray, comm::MPI.Comm)
    size_root = mpi_bcast(size(arr), comm)
    mpi_reduce(size(arr)[1:end-1] == size_root[1:end-1], &, comm)
end

"""
    mpi_gather(arr::AbstractArray, comm::MPI.Comm)
Gathers array along the last dimension to the root rank. The other ranks get an empty array.
"""
function mpi_gather(arr::AbstractArray, comm::MPI.Comm)
    @assert _check_size(arr, comm)
    _sync_device(arr)

    # Size of array in each processors
    counts = MPI.Allgather(Cint(length(arr)), comm)

    # Gather array. Only the root rank owns a receive buffer; `MPI.Gatherv!` takes the root as a
    # keyword defaulting to 0.
    recvbuf = mpi_isroot(comm) ? MPI.VBuffer(similar(arr, sum(counts)), counts) : nothing
    arr_gathered = MPI.Gatherv!(arr, recvbuf, comm)
    if mpi_isroot(comm)
        return reshape(arr_gathered, (size(arr)[1:end-1]..., :))
    else
        return similar(arr, size(arr)[1:end-1]..., 0)
    end
end
mpi_gather(x, comm::Nothing) = x

"""
    mpi_allgather(arr::AbstractArray, comm::MPI.Comm)
Gathers array along the last dimension to all processes
"""
function mpi_allgather(arr::AbstractArray, comm::MPI.Comm)
    @assert _check_size(arr, comm)
    _sync_device(arr)

    # Size of array in each processors
    counts = MPI.Allgather(Cint(length(arr)), comm)

    # Gather array
    arr_gathered = MPI.Allgatherv!(arr, MPI.VBuffer(similar(arr, sum(counts)), counts), comm)
    reshape(arr_gathered, (size(arr)[1:end-1]..., :))
end
mpi_allgather(x, comm::Nothing) = x

"""
Splits an iterator evenly between the processes of `comm` and returns the part handled
by the current process.
"""
function mpi_split_iterator(itr, comm)
    nprocs = mpi_nprocs(comm)
    @assert nprocs <= length(itr)
    split_iterator(itr, nprocs)[1 + MPI.Comm_rank(comm)]  # MPI ranks are 0-based
end
mpi_split_iterator(itr, comm::Nothing) = itr

"""
    mpi_scatter(arr, comm::MPI.Comm)
Scatters array along the last dimension from root to all processes.
"""
function mpi_scatter(arr::Union{AbstractArray,Nothing}, comm::MPI.Comm)
    T = arr !== nothing ? eltype(arr) : nothing
    dims = arr !== nothing ? size(arr) : nothing
    T = mpi_bcast(T, comm)
    dims = mpi_bcast(dims, comm)

    # Size of array in each processors.
    # block_size: first to second last dimensions.
    # tot_count: last dimensions.
    block_size = prod(dims[1:end-1])
    tot_count = dims[end]
    counts = split_count(tot_count, mpi_nprocs(comm)) .* block_size
    counts_cint = Cint.(counts)

    if arr === nothing
        arr = zeros(T, dims[1:end-1]..., 0)
    end

    # Only the root rank owns the send buffer; `MPI.Scatterv!` takes the root as a keyword
    # defaulting to 0.
    _sync_device(arr)
    sendbuf = mpi_isroot(comm) ? MPI.VBuffer(arr, counts_cint) : nothing
    recvbuf = similar(arr, counts_cint[MPI.Comm_rank(comm) + 1])
    arr_scattered = MPI.Scatterv!(sendbuf, recvbuf, comm)
    reshape(arr_scattered, (dims[1:end-1]..., :))
end
mpi_scatter(x, comm::Nothing) = x


# Both hand back the argument, not the reduced inner object: every `mpi_*!` method returns the
# buffer it was given, so a caller cannot see the container change with the communicator.
mpi_sum!(x::Vector{T}, comm::MPI.Comm) where {T <: AbstractArray} = (mpi_sum!.(x, Ref(comm)); x)
mpi_sum!(x::OffsetArray, comm::MPI.Comm) = (mpi_sum!(x.parent, comm); x)
