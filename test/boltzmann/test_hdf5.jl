using Test
using Random
using StaticArrays
using OffsetArrays
using HDF5
using ElectronPhonon
using ElectronPhonon: _data_julia_to_hdf5, _data_hdf5_to_julia

function test_hdf_io(data::T) where T
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
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

@testset "hdf5 IO basic" begin
    # Test HDF5 IO of basic types
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
    x = 4:10
    @inferred test_hdf_io(x)
    @inferred test_hdf_io(fill(x, 2, 3))
end

# TODO: Merge test_hdf_io and test_hdf_io_btdata
function test_hdf_io_btdata(data::T) where T
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    h5open(joinpath(tmp_dir, "tmp_data.h5"), "w") do f
        dump_BTData(f, data)
    end
    data_read = h5open(joinpath(tmp_dir, "tmp_data.h5"), "r") do f
        load_BTData(f, T)
    end
    @test data_read isa T
    data_read
end

@testset "hdf5 IO BTData" begin
    # Test HDF5 IO of composite types
    Random.seed!(123)
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    tmp_dir = joinpath(BASE_FOLDER, "test", "tmp")
    mkpath(tmp_dir)

    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]

    # Symmetry
    symmetry = symmetry_operations(lattice, atoms)
    symmetry_read = @inferred test_hdf_io_btdata(symmetry)
    for name in fieldnames(typeof(symmetry_read))
        @test getfield(symmetry_read, name) ≈ getfield(symmetry, name)
    end

    # OffsetArray
    arr = OffsetArray(rand(2, 3, 4), 5:6, -1:1, 1:4)
    arr_read = @inferred test_hdf_io_btdata(arr)
    @test arr ≈ arr_read

    # TODO: Kpoints, GridKpoints, ...
end
