using PrecompileTools: @setup_workload, @compile_workload

# The EPW loader reads files, so it is precompiled without running it.
@setup_workload begin
    @compile_workload begin
        precompile(Core.kwcall, (@NamedTuple{epmat_outer_momentum::String}, typeof(load_model_from_epw_new), String, String, String))
        precompile(load_model_from_epw_new, (String, String, String))
    end
end
