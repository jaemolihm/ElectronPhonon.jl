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
    @test symmetry.nsym == 96
    @test symmetry.time_reversal == true
    @test sum(symmetry.is_tr) == 48
    @test sum(symmetry.is_inv) == 48
    @test sum(symmetry.is_inv .& symmetry.is_tr) == 24

    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry, ignore_time_reversal=true)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((7, 7, 7), symmetry)
    @test kpts.n == 20

    # Cubic Boron Nitride structure
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetry = symmetry_operations(lattice, atoms)
    @test symmetry.nsym == 48
    @test symmetry.time_reversal == true
    @test sum(symmetry.is_tr) == 24
    @test sum(symmetry.is_inv) == 24
    @test sum(symmetry.is_inv .& symmetry.is_tr) == 12

    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((6, 6, 6), symmetry, ignore_time_reversal=true)
    @test kpts.n == 22
    kpts = bzmesh_ir_wedge((7, 7, 7), symmetry)
    @test kpts.n == 20

    # Cubic Boron Nitride sturucture imported from file
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, epmat_outer_momentum="ph")
    @test model.symmetry.nsym == 48
    @test model.symmetry.time_reversal == true
    kpts = bzmesh_ir_wedge((6, 6, 6), model.symmetry)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((7, 7, 7), model.symmetry)
    @test kpts.n == 20
end