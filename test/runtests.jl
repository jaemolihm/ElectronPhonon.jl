using TestItemRunner

# `Pkg.test(; test_args = ["test_kpoints.jl", "boltzmann/test_hdf5.jl", ...])` runs only the test
# items in the named files (paths relative to test/). Without arguments it runs every item except
# those tagged `:plotting`: they load PyPlot, and a PyCall finalizer running on a Julia worker
# thread crashes a multithreaded process (#49), so run them alone, single-threaded, with
# `test_args = ["test_plotting.jl"]`. The items tagged `:gpu` run only when CUDA is functional (CI
# drops it from the test target).
for file in ARGS
    isfile(joinpath(@__DIR__, file)) || error("unknown test file $file")
end
const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
HAVE_CUDA || @info "CUDA not functional: skipping the test items tagged :gpu"

@run_package_tests filter = ti ->
    (isempty(ARGS) ? :plotting ∉ ti.tags : relpath(ti.filename, @__DIR__) in ARGS) &&
    (HAVE_CUDA || :gpu ∉ ti.tags)
