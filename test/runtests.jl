using TestItemRunner

# `Pkg.test(; test_args = ["test_kpoints.jl", "boltzmann/test_hdf5.jl", ...])` runs only the test
# items in the named files (paths relative to test/). Without arguments it runs every item except
# those tagged `:plotting`: they load PyPlot, and a PyCall finalizer running on a Julia worker
# thread crashes a multithreaded process (#49), so run them alone, single-threaded, with
# `test_args = ["test_plotting.jl"]`. The items tagged `:gpu` run only when CUDA is functional (CI
# drops it from the test target).
const TEST_FILES = normpath.(ARGS)
for file in TEST_FILES
    isfile(joinpath(@__DIR__, file)) || error("unknown test file $file")
    occursin(r"^@testitem"m, read(joinpath(@__DIR__, file), String)) || error("no @testitem in $file")
end
const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
HAVE_CUDA || @info "CUDA not functional: skipping the test items tagged :gpu"

@run_package_tests filter = ti ->
    (isempty(TEST_FILES) ? :plotting ∉ ti.tags : relpath(ti.filename, @__DIR__) in TEST_FILES) &&
    (HAVE_CUDA || :gpu ∉ ti.tags)
