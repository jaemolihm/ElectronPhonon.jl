using StaticArrays
using LinearAlgebra
using ElectronPhonon
using Random
using Test

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

    @test all(inv(symop) * symop ≈ one(symop) for symop in symmetry)
    @test all(symop * inv(symop) ≈ one(symop) for symop in symmetry)
    ElectronPhonon.check_group(symmetry)

    kpts = kpoints_grid((6, 6, 6); symmetry)
    @test kpts.n == 16
    kpts = kpoints_grid((6, 6, 6); symmetry, ignore_time_reversal=true)
    @test kpts.n == 16
    kpts = kpoints_grid((7, 7, 7); symmetry)
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

    @test all(inv(symop) * symop ≈ one(symop) for symop in symmetry)
    @test all(symop * inv(symop) ≈ one(symop) for symop in symmetry)
    ElectronPhonon.check_group(symmetry)

    kpts = kpoints_grid((6, 6, 6); symmetry)
    @test kpts.n == 16
    kpts = kpoints_grid((6, 6, 6); symmetry, ignore_time_reversal=true)
    @test kpts.n == 22
    kpts = kpoints_grid((7, 7, 7); symmetry)
    @test kpts.n == 20

    # Cubic Boron Nitride sturucture imported from file
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model_from_epw_new(folder, "temp", "bn"; load_epmat = false)
    @test model.symmetry.nsym == 48
    @test model.symmetry.time_reversal == true
    @test length(model.symmetry.is_tr) == 48
    @test count(model.symmetry.is_tr) == 24

    @test all(inv(symop) * symop ≈ one(symop) for symop in symmetry)
    @test all(symop * inv(symop) ≈ one(symop) for symop in symmetry)
    ElectronPhonon.check_group(symmetry)

    kpts = kpoints_grid((6, 6, 6); model.symmetry)
    @test kpts.n == 16
    kpts = kpoints_grid((7, 7, 7); model.symmetry)
    @test kpts.n == 20
end

@testset "symmetrize" begin
    Random.seed!(123)

    lattice = 2.0 * [[0 1 1.]; [1 0 1.]; [1 1 0.]];
    atoms = ["A" => [ones(3)/8], "B" => [-ones(3)/9]];
    symmetry = symmetry_operations(lattice, atoms);

    # Scalar
    @test symmetrize(1.0, symmetry) == 1.0
    @test symmetrize(1.0, symmetry, tr_odd=true, axial=false) == 0.0
    @test symmetrize(1.0, symmetry, tr_odd=false, axial=true) == 0.0
    @test symmetrize(1.0, symmetry, tr_odd=true, axial=true) == 0.0

    # Vector
    v = Vec3(1., 2., 3.)
    @test symmetrize(v, symmetry) ≈ Vec3(2., 2., 2.)
    @test symmetrize(v, symmetry, tr_odd=true, axial=false) ≈ zero(v)
    @test symmetrize(v, symmetry, tr_odd=false, axial=true) ≈ zero(v)
    @test symmetrize(v, symmetry, tr_odd=true, axial=true) ≈ zero(v)

    # Matrix
    m = Mat3(rand(3, 3))
    m_sym = symmetrize(m, symmetry)
    for i = 1:3, j = 1:3
        if i == j
            @test m_sym[i, j] ≈ tr(m) / 3
        else
            @test m_sym[i, j] ≈ (sum(m) - tr(m)) / 6
        end
    end
    m_sym = symmetrize(m, symmetry, tr_odd=false, axial=true)
    val_asym = (m[1, 2] + m[2, 3] + m[3, 1] - m[2, 1] - m[3, 2] - m[1, 3]) / 6
    for i = 1:3, j = 1:3
        if i == j
            @test m_sym[i, j] ≈ 0 atol=eps(1.0)
        elseif j == mod1(i + 1, 3)
            @test m_sym[i, j] ≈ val_asym
        else
            @test m_sym[i, j] ≈ -val_asym
        end
    end
    @test symmetrize(m, symmetry, tr_odd=true, axial=false) ≈ zero(m) atol=eps(Float64)
    @test symmetrize(m, symmetry, tr_odd=true, axial=true) ≈ zero(m) atol=eps(Float64)

    # Array
    arr = rand(3, 2)
    @test symmetrize_array(arr, symmetry, order=0) ≈ symmetrize.(arr, Ref(symmetry))
    @test all(symmetrize_array(arr, symmetry, order=1)[:, 1] .≈ sum(arr[:, 1]) / 3)
    @test all(symmetrize_array(arr, symmetry, order=1)[:, 2] .≈ sum(arr[:, 2]) / 3)

    # Axial vector
    Ss = [Mat3{Int}(I(3)), Mat3{Int}(-I(3))]
    symmetry = Symmetry(Ss, zeros(Vec3{Float64}, 2), true, lattice)
    v = Vec3(rand(3))
    @test symmetrize(v, symmetry) ≈ zero(v)
    @test symmetrize(v, symmetry, axial=true) ≈ v
end

@testset "symmetry subset" begin
    lattice = 2.0 * [[0 1 1.];
                     [1 0 1.];
                     [1 1 0.]]
    lattice_low = 2.0 * [[0 1 1.];
                         [1 0 1.];
                         [1 1 0.1]]
    atoms = ["Si" => [ones(3)/8, -ones(3)/8]]
    atoms_low = ["B" => [ones(3)/8], "N" => [-ones(3)/8]]
    symmetry_high = symmetry_operations(lattice, atoms)

    symmetry_low1 = symmetry_operations(lattice, atoms_low)
    symmetry_low2 = symmetry_operations(lattice_low, atoms)

    @test symmetry_is_subset(symmetry_high, symmetry_high) == true
    @test symmetry_is_subset(symmetry_low1, symmetry_high) == true
    @test symmetry_is_subset(symmetry_low2, symmetry_high) == true
    @test symmetry_is_subset(symmetry_high, symmetry_low1) == false
    @test symmetry_is_subset(symmetry_low1, symmetry_low2) == false
    @test symmetry_is_subset(symmetry_low2, symmetry_low1) == false
end
