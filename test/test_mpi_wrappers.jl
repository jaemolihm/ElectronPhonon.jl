using Test
using ElectronPhonon
using ElectronPhonon: mpi_nprocs, mpi_isroot, mpi_myrank, mpi_sum, mpi_sum!, mpi_min, mpi_min!,
    mpi_max, mpi_max!, mpi_mean, mpi_mean!, mpi_reduce, mpi_bcast, mpi_bcast!, mpi_gather,
    mpi_allgather, mpi_scatter, mpi_split_iterator, mpi_gather_and_scatter,
    Vec3, Kpoints, GridKpoints, BandStates
using OffsetArrays: OffsetArray
import MPI

# The wrappers of src/common/mpi.jl, every contract checked on BOTH communicator types: `nothing`
# (the serial process, where each wrapper is the identity) and `MPI.COMM_SELF` (one rank). A
# caller writes one code path over the two, so what these tests pin is that it gets the same KIND
# of object back either way — the return value, whether it aliases the argument, the element type
# and the shape.
#
# What one rank cannot show: every reduction is the identity there, so a dropped `Allreduce`, or
# one with the wrong operator, passes unnoticed. Cross-rank arithmetic needs a real multi-rank
# launcher; this file is the contract, not the reduction.
@testset "mpi_* wrappers" begin
    MPI.Initialized() || MPI.Init()
    comms = (nothing, MPI.COMM_SELF)

    @testset "rank and size queries" begin
        for comm in comms
            @test mpi_nprocs(comm) == 1
            @test mpi_isroot(comm)
            @test mpi_myrank(comm) == 0
        end
    end

    # The bang forms hand back the buffer they were given, so the reduced value can be used inline
    # (`G = mpi_sum!(Array(M' * M), comm)`); the plain forms must not alias their argument on the
    # `MPI.Comm` side, where they allocate.
    @testset "reductions" begin
        for comm in comms
            for f! in (mpi_sum!, mpi_min!, mpi_max!, mpi_mean!)
                a = [1.0, 2.0, 3.0]
                @test f!(a, comm) === a
                @test a == [1.0, 2.0, 3.0]
            end
            for f in (mpi_sum, mpi_min, mpi_max, mpi_mean)
                a = [1.0, 2.0, 3.0]
                r = f(a, comm)
                @test r == a
                @test comm === nothing ? r === a : r !== a
            end
            # Scalars, not only arrays: the μ-bracket reductions pass one.
            @test mpi_sum(2.5, comm) == 2.5
            @test mpi_min(2.5, comm) == 2.5
            @test mpi_max(2.5, comm) == 2.5

            a = [1.0, 2.0, 3.0]
            r = mpi_reduce(a, +, comm)
            @test r == a
            @test comm === nothing ? r === a : r !== a
        end
    end

    @testset "broadcast" begin
        for comm in comms
            buf = [1.0, 2.0, 3.0]
            @test mpi_bcast!(buf, comm) === buf
            @test mpi_bcast!(buf, 0, comm) === buf
            @test buf == [1.0, 2.0, 3.0]

            obj = (; x = 1, y = "a")
            @test mpi_bcast(obj, comm) == obj
            @test mpi_bcast(obj, 0, comm) == obj
        end
    end

    # Gather and scatter move along the LAST dimension, so a one-rank round trip has to come back
    # in the original shape rather than flattened.
    @testset "gather, allgather, scatter" begin
        for comm in comms, x in ([1.0, 2.0, 3.0], reshape(collect(1.0:6.0), 2, 3))
            for r in (mpi_gather(x, comm), mpi_gather(x, 0, comm),
                      mpi_allgather(x, comm), mpi_scatter(x, comm))
                @test r == x
                @test size(r) == size(x)
                @test eltype(r) == eltype(x)
            end
        end
    end

    @testset "split_iterator" begin
        for comm in comms
            @test collect(mpi_split_iterator(1:7, comm)) == collect(1:7)
        end
    end

    # The two container forwarders reduce something other than the argument itself, and must still
    # return the argument: a caller handed `x.parent` back would see an `OffsetArray`'s index
    # origin change with the communicator.
    @testset "OffsetArray and Vector forwarders" begin
        for comm in comms
            oa = OffsetArray([1.0, 2.0, 3.0], -1:1)
            @test mpi_sum!(oa, comm) === oa
            @test axes(oa, 1) == -1:1
            @test collect(oa) == [1.0, 2.0, 3.0]

            va = [[1.0, 2.0], [3.0, 4.0]]
            inner = va[1]
            @test mpi_sum!(va, comm) === va
            @test va[1] === inner            # the inner arrays are reduced in place, not replaced
            @test va == [[1.0, 2.0], [3.0, 4.0]]
        end
    end

    # The type-specific `mpi_gather`/`mpi_allgather`/`mpi_scatter` methods dispatch on
    # `comm::MPI.Comm`, so `nothing` falls through to the generic wrapper and the object comes back
    # as is. Their `MPI.COMM_SELF` behaviour is checked with the types themselves — `Kpoints` in
    # test_kpoints.jl, `BandStates` in test_band_states.jl.
    @testset "type-specific methods on the serial path" begin
        nk = 4
        kpts = GridKpoints(Kpoints(nk, [Vec3((i - 1) / nk, 0.0, 0.0) for i in 1:nk],
            fill(1 / nk, nk), (nk, 1, 1)))
        states = BandStates(kpts, collect(1:nk), ones(Int, nk), 0.1 .* (1:nk); nw = 1,
            nstates_base = 0.0)
        @test mpi_gather(kpts, nothing) === kpts
        @test mpi_allgather(kpts, nothing) === kpts
        @test mpi_scatter(kpts, nothing) === kpts
        @test mpi_gather_and_scatter(kpts, nothing) === kpts
        @test mpi_allgather(states, nothing) === states
    end
end
