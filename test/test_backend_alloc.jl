using Test
using ElectronPhonon
using ElectronPhonon: CPUBackend, alloc, alloc_zeros, to_device, to_device_copy, gpu_backend

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
