using Test
using Random
using StaticArrays
using HDF5
using EPW
using EPW: _data_julia_to_hdf5, _data_hdf5_to_julia

function test_hdf_io(data::T) where T
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    h5open(joinpath(tmp_dir, "tmp_data.h5"), "w") do f
        f["data"] = _data_julia_to_hdf5(data)
    end
    data_read = h5open(joinpath(tmp_dir, "tmp_data.h5"), "r") do f
        _data_hdf5_to_julia(read(f, "data"), T)
    end
    @test data_read ≈ data
    @test data_read isa T
    data_read
end

@testset "BTData hdf5" begin
    Random.seed!(123)
    # Test simple types
    for T in [Int64, Float64, Float32, ComplexF64]
        x = rand(T)
        @inferred test_hdf_io(x)
    end
    # Test Arrays
    for elT in [Int64, Float64, Float32, ComplexF64]
        for dims in [(1,), (2, 1), (1, 2, 3)]
            T = Array{elT, length(dims)}
            x = rand(elT, dims)
            @inferred test_hdf_io(x)
        end
    end
    # Test StaticArray
    for elT in [Int64, Float64, Float32, ComplexF64]
        for T in [SVector{3, elT}, SMatrix{2, 3, elT, 6}]
            x = rand(T)
            @inferred test_hdf_io(x)
        end
    end
    # Test Array of StaticArrays
    for elT in [Int64, Float64]
        for sarrT in [SVector{3, elT}, SMatrix{2, 3, elT, 6}]
            for dims in [(1,), (2, 1), (1, 2, 3)]
                T = Array{sarrT, length(dims)}
                x = reshape([rand(sarrT) for _ in 1:prod(dims)], dims)
                @inferred test_hdf_io(x)
            end
        end
    end
    # Test UnitRange
    for x in [4:10]
        @inferred test_hdf_io(x)
    end
end

# Symmetry object
@testset "hdf5: symmetry" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)

    f = h5open(joinpath(tmp_dir, "tmp_symmetry.h5"), "w")
    dump_BTData(f, symmetry)
    close(f)

    f = h5open(joinpath(tmp_dir, "tmp_symmetry.h5"), "r")
    symmetry_read = @inferred load_BTData(f, Symmetry{Float64})
    close(f)

    for name in fieldnames(typeof(symmetry_read))
        @test getfield(symmetry_read, name) ≈ getfield(symmetry, name)
    end
end

# TODO: Add test of dump_BTData