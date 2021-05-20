
# Adapted from DFTK.jl/src/common/mpi.jl

# Convenience functions for working with MPI
using MPI

export mpi_initialized
export mpi_nprocs
export mpi_isroot
export mpi_myrank

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
mpi_myrank(comm::Nothing) = 0
const MPI_ROOT = 0

mpi_sum( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, +, comm)
mpi_sum!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, +, comm)
mpi_min( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, min, comm)
mpi_min!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, min, comm)
mpi_max( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, max, comm)
mpi_max!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, max, comm)
mpi_mean(arr, comm::MPI.Comm)  = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

mpi_reduce(arr, op, comm::MPI.Comm) = MPI.Allreduce(arr, op, comm)

mpi_bcast!(buf, root::Integer, comm::MPI.Comm) = MPI.Bcast!(buf, root, comm)
mpi_bcast( obj, root::Integer, comm::MPI.Comm) = MPI.bcast( obj, root, comm)
mpi_bcast!(buf, comm::MPI.Comm) = MPI.Bcast!(buf, 0, comm)
mpi_bcast( obj, comm::MPI.Comm) = MPI.bcast( obj, 0, comm)

# Do nothing if comm is nothing
mpi_min( arr, comm::Nothing) = arr
mpi_max( arr, comm::Nothing) = arr
mpi_sum( arr, comm::Nothing) = arr
mpi_min!(arr, comm::Nothing) = nothing
mpi_max!(arr, comm::Nothing) = nothing
mpi_sum!(arr, comm::Nothing) = nothing

function _check_size(arr::AbstractArray, root::Integer, comm::MPI.Comm)
    # Check whether size is equal except for the last dimension
    # size_valid = true if size is okay, false otherwise.
    # Do Allreduce to raise error on all processors
    size_root = mpi_bcast(size(arr), root, comm)
    size_valid = size(arr)[1:end-1] == size_root[1:end-1] ? true : false
    mpi_reduce(size_valid, &, comm)
end

"""
    mpi_gather(arr::AbstractArray, root::Integer, comm::MPI.Comm)
Gathers array along the last dimension
"""
function mpi_gather(arr::AbstractArray, root::Integer, comm::MPI.Comm)
    @assert _check_size(arr, root, comm)

    # Size of array in each processors
    counts = MPI.Allgather([Cint(length(arr))], 1, comm)

    # Gather array
    arr_gathered = MPI.Gatherv(arr, counts, root, comm)
    if mpi_isroot(comm)
        return reshape(arr_gathered, (size(arr)[1:end-1]..., :))
    else
        return typeof(arr)(undef, size(arr)[1:end-1]..., 0)
    end
end
mpi_gather(arr, comm::MPI.Comm) = mpi_gather(arr, MPI_ROOT, comm)
mpi_gather(x, comm::Nothing) = x

"""
    mpi_allgather(arr::AbstractArray, comm::MPI.Comm)
Gathers array along the last dimension to all processes
"""
function mpi_allgather(arr::AbstractArray, comm::MPI.Comm)
    @assert _check_size(arr, MPI_ROOT, comm)

    # Size of array in each processors
    counts = MPI.Allgather([Cint(length(arr))], 1, comm)

    # Gather array
    arr_gathered = MPI.Allgatherv(arr, counts, comm)
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

"""
    mpi_scatter(arr, comm::MPI.Comm)
Scatters array along the last dimension from root to all processes.
"""
function mpi_scatter(arr::AbstractArray, comm::MPI.Comm)
    @assert _check_size(arr, MPI_ROOT, comm)

    dims = mpi_bcast(size(arr), comm)

    # Size of array in each processors.
    # block_size: first to second last dimensions.
    # tot_count: last dimensions.
    block_size = prod(dims[1:end-1])
    tot_count = dims[end]
    counts = split_count(tot_count, mpi_nprocs(comm)) .* block_size
    counts_cint = Cint.(counts)

    arr_scattered = MPI.Scatterv(arr, counts_cint, MPI_ROOT, comm)
    reshape(arr_scattered, (dims[1:end-1]..., :))
end
mpi_scatter(x, comm::Nothing) = x