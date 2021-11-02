# Test HDF5 IO
using Test
using EPW
using HDF5

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
    symmetry_read = load_BTData(f, Symmetry{Float64})
    close(f)

    for name in fieldnames(typeof(symmetry_read))
        @test getfield(symmetry_read, name) ≈ getfield(symmetry, name)
    end
end