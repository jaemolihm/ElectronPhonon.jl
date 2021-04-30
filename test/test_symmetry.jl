using Test
using Random
using EPW

@testset "symmetry" begin
    # Silicon structure
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["Si" => [ones(3)/8, -ones(3)/8]]
    symmetries = symmetry_operations(lattice, atoms)
    @test length(symmetries) == 48

    kpts = bzmesh_ir_wedge((6, 6, 6), symmetries, time_reversal=true)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((6, 6, 6), symmetries, time_reversal=false)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((7, 7, 7), symmetries, time_reversal=true)
    @test kpts.n == 20

    # Cubic Boron Nitride structure
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    atoms = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetries = symmetry_operations(lattice, atoms)
    @test length(symmetries) == 24

    kpts = bzmesh_ir_wedge((6, 6, 6), symmetries, time_reversal=true)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((6, 6, 6), symmetries, time_reversal=false)
    @test kpts.n == 22
    kpts = bzmesh_ir_wedge((7, 7, 7), symmetries, time_reversal=true)
    @test kpts.n == 20

    # Cubic Boron Nitride sturucture imported from file
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, epmat_outer_momentum="ph")
    @test length(model.symmetries) == 24
    @test length(model.time_reversal) == true
    kpts = bzmesh_ir_wedge((6, 6, 6), model.symmetries, time_reversal=model.time_reversal)
    @test kpts.n == 16
    kpts = bzmesh_ir_wedge((7, 7, 7), model.symmetries, time_reversal=model.time_reversal)
    @test kpts.n == 20
end