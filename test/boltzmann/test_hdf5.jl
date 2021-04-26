using Test
using Random
using StaticArrays
using EPW: _data_julia_to_hdf5, _data_hdf5_to_julia

@testset "BTData hdf5" begin
    Random.seed!(123)
    # Test simple types
    for T in [Int64, Float64, Float32, ComplexF64]
        x = rand(T)
        y = _data_julia_to_hdf5(x)
        z = _data_hdf5_to_julia(y, T)
        @test y ≈ x
        @test y isa T
        @test z ≈ x
        @test z isa T
    end
    # Test Arrays
    for elT in [Int64, Float64, Float32, ComplexF64]
        for dims in [(1,), (2, 1), (1, 2, 3)]
            T = Array{elT, length(dims)}
            x = rand(elT, dims)
            y = _data_julia_to_hdf5(x)
            z = _data_hdf5_to_julia(y, T)
            @test all(y .≈ x)
            @test y isa T
            @test all(z .≈ x)
            @test z isa T
        end
    end
    # Test StaticArray
    for elT in [Int64, Float64, Float32, ComplexF64]
        for T in [SVector{3, elT}, SMatrix{2, 3, elT, 6}]
            x = rand(T)
            y = _data_julia_to_hdf5(x)
            z = _data_hdf5_to_julia(y, T)
            @test all(y .≈ x)
            @test y isa Array{elT, length(size(T))}
            @test all(z .≈ x)
            @test z isa T
        end
    end
    # Test Array of StaticArrays
    for elT in [Int64, Float64]
        for sarrT in [SVector{3, elT}, SMatrix{2, 3, elT, 6}]
            for dims in [(1,), (2, 1), (1, 2, 3)]
                T = Array{sarrT, length(dims)}
                x = reshape([rand(sarrT) for _ in 1:prod(dims)], dims)
                y = _data_julia_to_hdf5(x)
                z = _data_hdf5_to_julia(y, T)
                @test all(vec(y) .≈ vec(reinterpret(elT, x)))
                @test y isa Array{elT, length(size(sarrT)) + length(dims)}
                @test all(reinterpret(elT, z) .≈ reinterpret(elT, x))
                @test z isa T
            end
        end
    end
end

# TODO: Add test of dump_BTData