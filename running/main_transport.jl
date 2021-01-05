using PrettyPrint
using Printf
# using PyPlot
using StaticArrays
using BenchmarkTools
using NPZ
using PyCall
using Profile
using LinearAlgebra
using Base.Threads
using Distributed
using Cthulhu
using Revise
BLAS.set_num_threads(1)

@everywhere begin
    using LinearAlgebra
    using Base.Threads
    using Revise
    push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
    using EPW
    using EPW.WanToBloch
end

world_comm = EPW.mpi_world_comm()

folder = "/home/jmlim/julia_epw/silicon_mobility"
window_max = 7.0 * unit_to_aru(:eV)
window_min = 6.2 * unit_to_aru(:eV)
window = (window_min, window_max)

# model = load_model(folder)
model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")

nklist = (15, 15, 15)
nqlist = (15, 15, 15)

transport_params = TransportParams{Float64}(
    Tlist = [100.0, 200.0, 300.0] .* unit_to_aru(:K),
    n = 1.0e15 * model.volume / unit_to_aru(:cm)^3,
    smearing = 50.0 * unit_to_aru(:meV),
    carrier_type = "e",
    nband_valence = 4,
    spin_degeneracy = 2
)

@time output = EPW.run_eph_outer_loop_q(
    model, nklist, nqlist,
    mpi_comm_q=EPW.mpi_world_comm(),
    fourier_mode="gridopt",
    window=window,
    transport_params=transport_params,
)

EPW.transport_print_mobility(output["transport_σlist"], transport_params, model.volume)

@assert output["transport_σlist"][1, 1, 1] ≈ 0.001409019384286128
@assert output["transport_σlist"][2, 2, 2] ≈ 0.0005270865433373303
@assert output["transport_σlist"][1, 3, 3] ≈ 2.814288672537561e-5

# Profile.clear()
# @profile EPW.run_eph_outer_loop_q(
#     model, nklist, nqlist,
#     mpi_comm_q=EPW.mpi_world_comm(),
#     fourier_mode="gridopt",
#     window=window,
#     transport_params=transport_params,
# )
# Juno.profiler()
