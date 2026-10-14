using PrecompileTools: @setup_workload, @compile_workload

# The EPW loader reads files, so it is precompiled without running it.
@setup_workload begin
    @compile_workload begin
        precompile(Core.kwcall, (@NamedTuple{epmat_outer_momentum::String}, typeof(load_model_from_epw_new), String, String, String))
        precompile(load_model_from_epw_new, (String, String, String))

        # scf.in -> k path files, as run before any DFT calculation.
        dir = mktempdir()
        scf = joinpath(dir, "scf.in")
        write(scf, """
            &SYSTEM
                ibrav = 4, A = 3.34, C = 15.0, nat = 2, ntyp = 2
            /
            ATOMIC_SPECIES
            Ta 180.95 Ta.upf
            S 32.06 S.upf
            ATOMIC_POSITIONS crystal
            Ta 0.0 0.0 0.0
            S 0.3333333333 -0.3333333333 0.1
            """)
        kpts, plot_xdata = high_symmetry_kpath(load_structure_from_qe(scf); kline_density = 10, is_2d = true)
        write_kpath_files("x", kpts, plot_xdata; dir)
        rm(dir; recursive = true)
    end
end
