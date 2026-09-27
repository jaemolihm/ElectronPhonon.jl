using Test
using ElectronPhonon
using ElectronPhonon: AbstractBackend, CPUBackend, alloc, alloc_zeros, is_host, to_device,
    to_device_copy, gpu_backend, free_bytes, reclaim_device_memory, backend_from
using SparseArrays: sparse, SparseMatrixCSC, nnz
using LinearAlgebra: mul!

# CUDA is a weak dependency, so load it defensively and run the device arm only when it works.
const BACKEND_ALLOC_GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

# The two allocators that pair with `alloc` and `to_device`. Each exists because its sibling
# cannot be used for a buffer that will be written: `alloc` returns `undef`, and `to_device` is
# the identity on `CPUBackend`. The element type is the CALLER's, not the backend prototype's,
# which `alloc`'s `similar(proto, T, dims...)` is what provides.
@testset "alloc_zeros / to_device_copy" begin
    backends = BACKEND_ALLOC_GPU ? (CPUBackend(), gpu_backend()) : (CPUBackend(),)
    BACKEND_ALLOC_GPU || @info "CUDA not functional — skipping the GPUBackend arm"

    for backend in backends
        @testset "$(nameof(typeof(backend)))" begin
            # `gpu_backend()`'s prototype is a `CuArray{ComplexF64}`, so these element types are
            # all different from it.
            for T in (Float64, ComplexF64, Int)
                Z = alloc_zeros(backend, T, 3, 4, 2)
                @test eltype(Z) === T
                @test size(Z) == (3, 4, 2)
                @test all(iszero, Array(Z))
                @test typeof(Z) === typeof(alloc(backend, T, 3, 4, 2))   # same array type as `alloc`
            end

            A = reshape(collect(1.0:6.0), 2, 3)
            C = to_device_copy(backend, A)
            @test eltype(C) === eltype(A)
            @test size(C) == size(A)
            @test Array(C) == A
            # The point of the routine: distinct from its argument on EVERY backend, where
            # `to_device` is the identity on the host.
            @test C !== A
            @test to_device(CPUBackend(), A) === A          # the sibling it exists to replace
            C .= 0
            @test A == reshape(collect(1.0:6.0), 2, 3)     # writing the copy left `A` alone

            # A vector as well as a matrix, and element types other than Float64: the result's
            # is the ARGUMENT's, exactly, not the backend prototype's and not a promotion.
            v = ComplexF64[1 + 2im, 3 + 4im]
            Cv = to_device_copy(backend, v)
            @test eltype(Cv) === ComplexF64 && size(Cv) == (2,) && Array(Cv) == v
            @test typeof(Cv) === typeof(alloc(backend, ComplexF64, 2))

            Ci = to_device_copy(backend, [1, 2, 3])
            @test eltype(Ci) === Int && Array(Ci) == [1, 2, 3]
            @test eltype(to_device_copy(backend, Complex{Int}[1 + 2im])) === Complex{Int}
        end
    end
end

# `reclaim_device_memory`'s contract, which a residency decision reading `free_bytes` depends on.
#
# The allocation happens in a function, not inline: a `@testset begin … end` body is one top-level
# thunk, and a device array dropped by top-level code stays reachable from that thunk's frame, so
# no GC collects it and the test below would measure nothing.
_alloc_and_drop_device(nbytes) = (fill!(CUDA.CuArray{Float64}(undef, nbytes ÷ sizeof(Float64)),
                                       1.0); nothing)

@testset "reclaim_device_memory" begin
    # The host contract is that this is FREE, not merely harmless: `free_bytes(::CPUBackend)` is
    # `typemax`, so a decision always says resident and there is no pool to trim — a `GC.gc(true)`
    # hoisted into the generic method would be pure cost on every CPU run.
    sweeps = Base.gc_num().full_sweep
    @test reclaim_device_memory(CPUBackend()) === nothing
    @test Base.gc_num().full_sweep == sweeps

    nbytes = 1_000_000_000
    # Establish a clean baseline BEFORE the skip guard and before anything is measured. Without it
    # the guard reads the very pool pollution this test exists to demonstrate and can skip itself,
    # and cached bytes left by the preceding testsets could supply the whole delta below while the
    # 1 GB array is never collected at all.
    BACKEND_ALLOC_GPU && reclaim_device_memory(gpu_backend())
    if !BACKEND_ALLOC_GPU
        @info "CUDA not functional — skipping the device reclaim test"
    elseif free_bytes(gpu_backend()) < 4 * nbytes
        @info "less than $(4 * nbytes) B free on the device — skipping the device reclaim test"
    else
        backend = gpu_backend()
        # A trimmed pool is not an empty one — CUDA.jl keeps a small block and a few kB in use —
        # so every assertion below is a delta against this baseline, not an absolute.
        used0, cached0 = CUDA.used_memory(), CUDA.cached_memory()
        # A dropped (not `unsafe_free!`d) `CuArray` frees through `finalizer(unsafe_free!, obj)`,
        # so its bytes are `used_memory` until a GC runs and pool-held `cached_memory` after. GC is
        # disabled across the allocation so that state is deterministic rather than a race with an
        # incidental collection; `free_bytes` was just checked to have room, so nothing needs to be
        # collected to satisfy it.
        GC.enable(false)
        local used_held, free_held
        try
            _alloc_and_drop_device(nbytes)
            used_held, free_held = CUDA.used_memory(), free_bytes(backend)
        finally
            GC.enable(true)
        end
        @test used_held - used0 >= 9 * nbytes ÷ 10        # held, unfinalized, invisible to a trim
        @test reclaim_device_memory(backend) === nothing
        # Both halves of `RECLAIM_DROP` are pinned: a GC-only hook returns the bytes to the pool and
        # leaves `cached_memory` ~1 GB high, a trim-or-purge-only hook never finalizes the array and
        # leaves `used_memory` ~1 GB high. Both counters are process-local. The `free_bytes` delta
        # is the contract a residency decision actually reads, but it comes from `cuMemGetInfo` and
        # is device-wide, so it is the loosest of the three and not the load-bearing one.
        @test CUDA.used_memory() <= used0
        @test CUDA.cached_memory() <= cached0
        @test free_bytes(backend) - free_held >= 9 * nbytes ÷ 10
    end
end

# A sparse matrix must NOT go through the generic `to_device`, which densifies it. What makes the
# specialization worth having is that `mul!` on the result then runs cuSPARSE SpMM instead of a
# scalar-indexing fallback — including when the dense operand is a `transpose` view, which
# LinearAlgebra unwraps into cuSPARSE's `transb` flag rather than materializing a copy.
@testset "sparse to_device / mul!" begin
    n_f, n_i, r = 4000, 200, 12
    S = sparse(1:n_f, rand(1:n_i, n_f), rand(n_f), n_f, n_i)
    St = SparseMatrixCSC(transpose(S))                 # (n_i × n_f)
    B = rand(n_f, r)
    Bt = permutedims(B)                                # transpose(Bt) == B numerically
    ref = Array(St) * B

    @test to_device(CPUBackend(), St) === St
    Ch = zeros(n_i, r)
    @test mul!(Ch, St, B) ≈ ref
    @test mul!(zeros(n_i, r), St, transpose(Bt)) ≈ ref

    if BACKEND_ALLOC_GPU
        backend = gpu_backend()
        Sd = to_device(backend, St)
        @test !(Sd isa CuArray)                        # densified would be an ordinary CuArray
        @test nnz(Sd) == nnz(St)
        Bd = to_device_copy(backend, B)
        Btd = to_device_copy(backend, Bt)

        # `allowscalar(false)` IS the test: a fallback that iterates the arrays throws under it,
        # and nothing else distinguishes the two — `Base.which` reports LinearAlgebra's entry
        # point (`matmul.jl`) whichever path runs, because cuSPARSE hooks in below it. The setting
        # is process-wide, so restore whatever the caller had.
        probe = CUDA.zeros(Int, 1)
        was_allowed = redirect_stderr(devnull) do
            try (probe[1]; true) catch; false end
        end
        CUDA.allowscalar(false)
        try
            Cd = alloc_zeros(backend, Float64, n_i, r)
            @test Array(mul!(Cd, Sd, Bd)) ≈ ref
            Cdt = alloc_zeros(backend, Float64, n_i, r)
            @test Array(mul!(Cdt, Sd, transpose(Btd))) ≈ ref
        finally
            # restoring the permissive default is itself deprecated and warns; that is noise here
            redirect_stderr(devnull) do
                CUDA.allowscalar(was_allowed)
            end
        end
    end
end

# A backend EP has never seen, standing in for a future one: `is_host` is `true` on a known list
# and `false` by fallback, so such a backend takes the generic (device) route, not the host-only
# one.
struct UnknownBackend <: AbstractBackend end

@testset "is_host" begin
    @test is_host(CPUBackend())
    @test !is_host(UnknownBackend())
    if BACKEND_ALLOC_GPU
        @test !is_host(gpu_backend())
    end
end

# The only `Bool` -> backend mapping in the stack; a driver script's `use_gpu` is consumed here.
@testset "backend_from" begin
    @test backend_from(false) === CPUBackend()
    if BACKEND_ALLOC_GPU
        @test typeof(backend_from(true)) === typeof(gpu_backend())
    elseif !hasmethod(gpu_backend, Tuple{})
        # No extension loaded: the Bool that asks for a GPU must say so, not `MethodError`.
        @test_throws ArgumentError backend_from(true)
    end
end
