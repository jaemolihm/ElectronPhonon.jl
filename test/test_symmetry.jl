using Test
using Random
using EPW

@testset "symmetry" begin
    # Silicon structure
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["Si" => [ones(3)/8, -ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)
    @test symmetry.nsym == 48

    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry, disable_time_reversal=true)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((7, 7, 7), symmetry)
    @test kpts.n == 20

    # Cubic Boron Nitride structure
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)
    @test symmetry.nsym == 24

    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry, disable_time_reversal=true)
    @test kpts.n == 22
    kpts = bzmesh_ir_wedge((7, 7, 7), symmetry)
    @test kpts.n == 20

    # Cubic Boron Nitride sturucture imported from file
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, epmat_outer_momentum="ph")
    @test model.symmetry.nsym == 24
    @test model.symmetry.time_reversal == true
    @test model.symmetry.itrevs == [1, -1]
    kpts = bzmesh_ir_wedge((6, 6, 6), model.symmetry)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((7, 7, 7), model.symmetry)
    @test kpts.n == 20
end