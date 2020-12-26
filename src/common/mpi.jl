
# Adapted from DFTK.jl/src/common/mpi.jl

# Convenience functions for working with MPI
using MPI

export mpi_root
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
    MPI.Initialized() || MPI.Init_thread(MPI.ThreadLevel(3))
end

"""
Number of processors used in MPI. Can be called without ensuring initialization.
"""
mpi_nprocs(comm=MPI.COMM_WORLD) = (mpi_ensure_initialized(); MPI.Comm_size(comm))
mpi_isroot(comm=MPI.COMM_WORLD) = (mpi_ensure_initialized(); MPI.Comm_rank(comm) == 0)
mpi_myrank(comm=MPI.COMM_WORLD) = (mpi_ensure_initialized(); MPI.Comm_rank(comm))
const mpi_root = 0

mpi_sum( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, +, comm)
mpi_sum!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, +, comm)
mpi_min( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, min, comm)
mpi_min!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, min, comm)
mpi_max( arr, comm::MPI.Comm)  = MPI.Allreduce( arr, max, comm)
mpi_max!(arr, comm::MPI.Comm)  = MPI.Allreduce!(arr, max, comm)
mpi_mean(arr, comm::MPI.Comm)  = mpi_sum(arr, comm) ./ mpi_nprocs(comm)
mpi_mean!(arr, comm::MPI.Comm) = (mpi_sum!(arr, comm); arr ./= mpi_nprocs(comm))

mpi_bcast!(buf, root::Integer, comm::MPI.Comm) = MPI.Bcast!(buf, root, comm)
mpi_bcast( obj, root::Integer, comm::MPI.Comm) = MPI.bcast( obj, root, comm)
mpi_bcast!(buf, comm::MPI.Comm) = MPI.Bcast!(buf, 0, comm)
mpi_bcast( obj, comm::MPI.Comm) = MPI.bcast( obj, 0, comm)

"Gathers array along the last dimension"
function mpi_gather(arr, root::Integer, comm::MPI.Comm)
    # Check whether size is equal except for the last dimension
    # check_dims = 0 if size is okay, 1 otherwise.
    # Do Allreduce to raise error on all processors
    size_root = mpi_bcast(size(arr), root, comm)
    check_dims = size(arr)[1:end-1] == size_root[1:end-1] ? 0 : 1
    check_dims = MPI.Allreduce(check_dims, +, comm)
    @assert check_dims == 0

    # Size of array in each processors
    counts = MPI.Allgather([Cint(length(arr))], 1, comm)

    # Gather array
    arr_gathered = MPI.Gatherv(arr, counts, root, comm)
    if mpi_isroot(comm)
        return reshape(arr_gathered, (size(arr)[1:end-1]..., :))
    else
        return nothing
    end
end

mpi_gather(arr, comm::MPI.Comm) = mpi_gather(arr, 0, comm)

"""
Splits an iterator evenly between the processes of `comm` and returns the part handled
by the current process.
"""
function mpi_split_iterator(itr, comm)
    nprocs = mpi_nprocs(comm)
    @assert nprocs <= length(itr)
    split_evenly(itr, nprocs)[1 + MPI.Comm_rank(comm)]  # MPI ranks are 0-based
end
