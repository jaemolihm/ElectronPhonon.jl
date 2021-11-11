using Test
using EPW
using LinearAlgebra
using PyPlot

@testset "plot bandstructure" begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder)

    ref_kcoords = [
        [ 0.000000000000, 0.000000000000, 0.000000000000],
        [ 0.000000000000, 0.055555555555, 0.055555555555],
        [ 0.000000000000, 0.111111111111, 0.111111111111],
        [ 0.000000000000, 0.166666666666, 0.166666666666],
        [ 0.000000000000, 0.222222222222, 0.222222222222],
        [ 0.000000000000, 0.277777777777, 0.277777777777],
        [ 0.000000000000, 0.333333333333, 0.333333333333],
        [ 0.000000000000, 0.388888888888, 0.388888888888],
        [ 0.000000000000, 0.444444444444, 0.444444444444],
        [ 0.000000000000, 0.500000000000, 0.500000000000],
        [ 0.000000000000, 0.541666666666, 0.458333333333],
        [ 0.000000000000, 0.583333333333, 0.416666666666],
        [ 0.000000000000, 0.625000000000, 0.375000000000],
        [-0.375000000000, 0.375000000000, 0.000000000000],
        [-0.333333333333, 0.333333333333, 0.000000000000],
        [-0.291666666666, 0.291666666666, 0.000000000000],
        [-0.250000000000, 0.250000000000, 0.000000000000],
        [-0.208333333333, 0.208333333333, 0.000000000000],
        [-0.166666666666, 0.166666666666, 0.000000000000],
        [-0.125000000000, 0.125000000000, 0.000000000000],
        [-0.083333333333, 0.083333333333, 0.000000000000],
        [-0.041666666666, 0.041666666666, 0.000000000000],
        [ 0.000000000000, 0.000000000000, 0.000000000000],
        [ 0.000000000000, 0.071428571428, 0.000000000000],
        [ 0.000000000000, 0.142857142857, 0.000000000000],
        [ 0.000000000000, 0.214285714285, 0.000000000000],
        [ 0.000000000000, 0.285714285714, 0.000000000000],
        [ 0.000000000000, 0.357142857142, 0.000000000000],
        [ 0.000000000000, 0.428571428571, 0.000000000000],
        [ 0.000000000000, 0.500000000000, 0.000000000000],
        [-0.041666666666, 0.500000000000, 0.041666666666],
        [-0.083333333333, 0.500000000000, 0.083333333333],
        [-0.125000000000, 0.500000000000, 0.125000000000],
        [-0.166666666666, 0.500000000000, 0.166666666666],
        [-0.208333333333, 0.500000000000, 0.208333333333],
        [-0.250000000000, 0.500000000000, 0.250000000000],
        [-0.187500000000, 0.500000000000, 0.312500000000],
        [-0.125000000000, 0.500000000000, 0.375000000000],
        [-0.062500000000, 0.500000000000, 0.437500000000],
        [ 0.000000000000, 0.500000000000, 0.500000000000],
    ]
    ref_x = [0.000000000000, 0.102215475958, 0.204430951917, 0.306646427876, 0.408861903834,
             0.511077379793, 0.613292855752, 0.715508331710, 0.817723807669, 0.919939283628,
             1.028355167916, 1.136771052205, 1.245186936494, 1.245186936494, 1.353602820783,
             1.462018705072, 1.570434589361, 1.678850473650, 1.787266357939, 1.895682242227,
             2.004098126516, 2.112514010805, 2.220929895094, 2.334742865031, 2.448555834969,
             2.562368804906, 2.676181774843, 2.789994744781, 2.903807714718, 3.017620684655,
             3.126036568944, 3.234452453233, 3.342868337522, 3.451284221811, 3.559700106100,
             3.668115990389, 3.783108400842, 3.898100811296, 4.013093221749, 4.128085632203]
    ref_xlabels = ["Γ", "X", "U|K", "Γ", "L", "W", "X"]
    ref_xticks = [0.000000000000, 0.919939283628, 1.245186936494, 2.220929895094,
                  3.017620684655, 3.668115990389, 4.128085632203]

    @test EPW.spglib_spacegroup_number(model) == 216
    @test EPW.get_spglib_lattice(model) ≈ model.alat * I(3)

    kpts, plot_xdata = EPW.high_symmetry_kpath(model, kline_density=10)

    @test kpts.n == length(ref_kcoords)
    for ik in 1:length(ref_kcoords)
        @test kpts.vectors[ik] ≈ ref_kcoords[ik]
    end

    @test plot_xdata.x ≈ ref_x
    @test plot_xdata.xticks ≈ ref_xticks
    @test plot_xdata.xlabels == ref_xlabels

    # Test whether plot_bandstructure runs
    out = plot_bandstructure(model, kline_density=10)
    @test out.fig isa PyPlot.Figure
    @test out.kpts.vectors ≈ kpts.vectors
    @test size(out.e_el) == (model.nw, kpts.n)
    @test size(out.e_ph) == (model.nmodes, kpts.n)
    @test out.plot_xdata.x ≈ plot_xdata.x
    @test out.plot_xdata.xticks ≈ plot_xdata.xticks
    @test out.plot_xdata.xlabels == plot_xdata.xlabels

    # Test whether the k path is the same if different lattice convention is used
    alat = 5.0
    lattice = alat * Mat3([[1 -1/2 0]; [0 sqrt(3)/2 0]; [0 0 sqrt(8/3)]])
    atom_pos_crys = [Vec3(0.0, 0.0, 0.0), Vec3(1/3, 2/3, 1/2)]
    atom_pos = Ref(lattice) .* (atom_pos_crys ./ alat)
    atom_labels = ["A", "B"]
    model = (;alat, lattice, atom_pos, atom_labels)

    @test EPW.atom_pos_crys(model) ≈ atom_pos_crys
    @test EPW.spglib_spacegroup_number(model) == 187
    @test EPW.get_spglib_lattice(model) ≈ lattice

    transf = Mat3([[1 2 3]; [0 1 0]; [0 0 1]]') # transformation matrix to new lattice vector convention
    @assert det(transf) ≈ 1
    lattice_new = model.lattice * transf
    model_new = (;alat, lattice=lattice_new, atom_pos, atom_labels)

    # model_new describes the same physical structure with model, just with a different lattice vector convention.
    @test EPW.spglib_spacegroup_number(model_new) == EPW.spglib_spacegroup_number(model)
    @test EPW.get_spglib_lattice(model_new) ≈ EPW.get_spglib_lattice(model)

    kpts, plot_xdata = EPW.high_symmetry_kpath(model, kline_density=50)
    kpts_new, plot_xdata_new = EPW.high_symmetry_kpath(model_new, kline_density=50)
    @test plot_xdata.x ≈ plot_xdata_new.x
    @test plot_xdata.xticks ≈ plot_xdata_new.xticks
    @test plot_xdata.xlabels == plot_xdata_new.xlabels
    @test kpts.n == kpts_new.n
    # Compare k points in Cartesian coordinates
    @test Ref(inv(lattice)') .* kpts.vectors ≈ Ref(inv(lattice_new)') .* kpts_new.vectors
end

@testset "plot deformation pot." begin
    BASE_FOLDER = dirname(dirname(pathof(EPW)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    model = load_model(folder, epmat_outer_momentum="el")

    # Test that the function runs. No test for the correctness.
    plot_electron_phonon_deformation_potential(model)
    plot_electron_phonon_deformation_potential(model, Vec3(0.0, 0.5, 0.0), band_rng=1:2, kline_density=15, include_polar=false)
end