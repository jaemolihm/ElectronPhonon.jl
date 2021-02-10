
# Data and functions for k points
using MPI

export generate_kvec_grid

struct Kpoints{T <: Real}
    n::Int                  # Number of k points
    vectors::Vector{Vec3{T}} # Fractional coordinate of k points
    weights::Vector{T}       # Weight of each k points
end

"""
    generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer)
Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return all k points.
"""
function generate_kvec_grid(nk1, nk2, nk3)
    nk = nk1 * nk2 * nk3
    generate_kvec_grid(nk1, nk2, nk3, 1:nk)
end

"""
    generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer, rng)
Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return k points for global index in the given range.
"""
function generate_kvec_grid(nk1, nk2, nk3, rng::UnitRange{Int})
    # TODO: Type
    nk_grid = nk1 * nk2 * nk3
    @assert rng[1] >= 1
    @assert rng[end] <= nk_grid
    kvecs = Vector{Vec3{Float64}}()
    for ik in rng
        # For (i, j, k), make k the fastest axis
        k = mod(ik-1, nk3)
        j = mod(div(ik-1 - k, nk3), nk2)
        i = mod(div(ik-1 - k - j*nk3, nk2*nk3), nk1)
        push!(kvecs, Vec3{Float64}(i/nk1, j/nk2, k/nk3))
    end
    nk = length(kvecs)
    Kpoints(nk, kvecs, fill(1/nk_grid, (nk,)))
end

"""
    generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer, mpi_comm::MPI.Comm)
Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return all k points. k points are distributed over the MPI communicator mpi_comm.
"""
function generate_kvec_grid(nk1, nk2, nk3, mpi_comm::MPI.Comm)
    nk = nk1 * nk2 * nk3
    range = mpi_split_iterator(1:nk, mpi_comm)
    generate_kvec_grid(nk1, nk2, nk3, range)
end

# "generate grid of k points"
# function generate_kvec_grid_array(nkf)
#     kvecs = zeros(3, nkf[1] * nkf[2] * nkf[3])
#     ind = 0
#     for i in 1:nkf[1], j in 1:nkf[2], k in 1:nkf[3]
#         ind += 1
#         kvecs[:, ind] .= [(i-1) / nkf[1], (j-1) / nkf[2], (k-1) / nkf[3]]
#     end
#     nk = size(kvecs)[2]
#     kvecs, nk
# end

"Filter kpoints using a Boolean vector ik_keep. Retern Kpoints object where
only k points with ik_keep = true are kept."
function get_filtered_kpoints(k, ik_keep)
    @assert length(ik_keep) == k.n
    Kpoints(sum(ik_keep), k.vectors[ik_keep], k.weights[ik_keep])
end

"Collect and uniformly redistribute Kpoints among processers"
function redistribute_kpoints(k::Kpoints, comm::MPI.Comm)
    # TODO: Do this without allgather, by using point-to-point communication.
    # Gather filtered k points
    kvectors = EPW.mpi_allgather(k.vectors, comm)
    weights = EPW.mpi_allgather(k.weights, comm)

    # Redistribute k points
    range = EPW.mpi_split_iterator(1:length(kvectors), comm)
    Kpoints(length(range), kvectors[range], weights[range])
end

redistribute_kpoints(k::Kpoints, comm::Nothing) = k
