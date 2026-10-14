using Test
using ElectronPhonon
using LinearAlgebra

# QE pw.x input (namelists + cards) for the tests below.
function _write_qe_input(dir, name, system, cards)
    path = joinpath(dir, name)
    write(path, "&CONTROL\n    calculation = 'scf', prefix = 'x'\n/\n&SYSTEM\n$system\n/\n&ELECTRONS\n/\n$cards")
    path
end

@testset "load_structure_from_qe + write_kpath" begin
    model = _load_model_from_artifacts("cubicBN"; load_epmat = false)
    kpts_ref, xdata_ref = high_symmetry_kpath(model; kline_density = 10)

    dir = mktempdir()
    species = "ATOMIC_SPECIES\nB 10.811 B.upf\nN 14.0067 N.upf\n"

    # fcc from celldm; commas, comments and if_pos columns in the input.
    f_fcc = _write_qe_input(dir, "fcc.in",
        "    ibrav = 2, celldm(1) = 6.83d0  ! alat\n    nat = 2, ntyp = 2\n    ecutwfc = 60",
        species * "ATOMIC_POSITIONS {crystal}\nB 0.125 0.125 0.125 0 0 0\nN -0.125 -0.125 -0.125\n")
    st = load_structure_from_qe(f_fcc)
    @test st.alat ≈ model.alat
    @test st.lattice ≈ model.lattice
    @test st.atom_pos ≈ model.atom_pos
    @test st.atom_labels == ["B", "N"]

    kpts, xdata = high_symmetry_kpath(st; kline_density = 10)
    @test kpts.vectors ≈ kpts_ref.vectors
    @test xdata.x ≈ xdata_ref.x
    @test xdata.xlabels == xdata_ref.xlabels

    # Same structure through ibrav = 0 in angstrom and positions in bohr.
    a = 6.83 / ElectronPhonon.units.Å
    f_ibrav0 = _write_qe_input(dir, "ibrav0.in", "    ibrav = 0\n    nat = 2\n    ntyp = 2",
        species * "CELL_PARAMETERS angstrom\n" *
        join(("$(-a/2) 0.0 $(a/2)", "0.0 $(a/2) $(a/2)", "$(-a/2) $(a/2) 0.0"), "\n") * "\n" *
        "ATOMIC_POSITIONS bohr\nB $(-6.83/8) $(6.83/8) $(6.83/8)\nN $(6.83/8) $(-6.83/8) $(-6.83/8)\n")
    st0 = load_structure_from_qe(f_ibrav0)
    @test st0.lattice ≈ model.lattice
    @test st0.alat ≈ norm(model.lattice[:, 1])  # QE: alat = |a1| for CELL_PARAMETERS bohr/angstrom
    @test st0.atom_pos .* st0.alat ≈ st.atom_pos .* st.alat

    # Hexagonal from A and C (Å), 2D path.
    f_hex = _write_qe_input(dir, "hex.in",
        "    ibrav = 4, A = 3.34, C = 15.0, nat = 3, ntyp = 2",
        "ATOMIC_SPECIES\nTa 180.95 Ta.upf\nS 32.06 S.upf\nATOMIC_POSITIONS crystal\n" *
        "S 0.3333333333 -0.3333333333 0.1041893571\nTa 0.0 0.0 0.0\nS 0.3333333333 -0.3333333333 -0.1041893571\n")
    st_hex = load_structure_from_qe(f_hex)
    A = 3.34 * ElectronPhonon.units.Å
    @test st_hex.lattice ≈ [A -A/2 0; 0 A*sqrt(3)/2 0; 0 0 A*15/3.34]
    _, xdata_hex = high_symmetry_kpath(st_hex; kline_density = 20, is_2d = true)
    @test xdata_hex.xlabels == ["Γ", "M", "K", "Γ"]

    # Output file, for the fcc path with its "U|K" branch break.
    write_kpath("BN", kpts, xdata; dir)
    rows = split.(filter(l -> !startswith(l, "#"), readlines(joinpath(dir, "BN_kpath.dat"))))
    @test length(rows) == kpts.n
    @test [parse(Int, r[1]) for r in rows] == 1:kpts.n
    @test [parse.(Float64, r[2:4]) for r in rows] ≈ kpts.vectors atol = 1e-11
    @test [parse(Float64, r[5]) for r in rows] ≈ xdata.x

    labeled = filter(r -> length(r) == 6, rows)
    @test [r[6] for r in labeled] == ["G", "X", "U", "K", "G", "L", "W", "X"]
    iks = [parse(Int, r[1]) for r in labeled]
    @test iks[1] == 1 && iks[end] == kpts.n
    @test iks[4] == iks[3] + 1 && labeled[3][5] == labeled[4][5]  # U|K: consecutive, same x
    @test all(length(r) == 5 for r in rows if !(r in labeled))
end
