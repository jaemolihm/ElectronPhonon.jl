
# Data and functions for k points
using MPI

export Kpoints
export generate_kvec_grid

struct Kpoints{T <: Real}
    n::Int                   # Number of k points
    vectors::Vector{Vec3{T}} # Fractional coordinate of k points
    weights::Vector{T}       # Weight of each k points
    # size of the grid if kpoints is a subset of grid points. (0,0,0) otherwise.
    ngrid::NTuple{3,Int64}
end

Kpoints{T}() where {T} = Kpoints{T}(0, Vector{Vec3{T}}(), Vector{T}(), (0, 0, 0))

function sort!(k::Kpoints)
    inds = sortperm(k.vectors)
    k.vectors .= k.vectors[inds]
    k.weights .= k.weights[inds]
    inds
end

function Kpoints(xks::AbstractArray{T}) where {T <: Real}
    if size(xks, 1) != 3
        throw(ArgumentError("first dimension of xks must be 3"))
    end
    vectors = collect(vec(reinterpret(Vec3{T}, xks)))
    n = length(vectors)
    Kpoints(n, vectors, ones(n) ./ n, (0, 0, 0))
end

"""
    generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer)
Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return all k points.
"""
function generate_kvec_grid(nk1, nk2, nk3; kshift=[0, 0, 0])
    nk = nk1 * nk2 * nk3
    generate_kvec_grid(nk1, nk2, nk3, 1:nk, kshift=kshift)
end

"""
    generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer, rng)
Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return k points for global index in the given range.
-`kshift`: Shift for the grid. Each element can be 0 or 1//2.
"""
function generate_kvec_grid(nk1, nk2, nk3, rng::UnitRange{Int}; kshift=[0, 0, 0])
    # TODO: Type
    nk_grid = nk1 * nk2 * nk3
    @assert rng[1] >= 1
    @assert rng[end] <= nk_grid
    kvecs = Vector{Vec3{Float64}}()

    kshift = Vec3{Rational{Int}}(kshift)
    all(ks in (0, 1//2) for ks in kshift) || error("Only kshifts of 0 or 1//2 implemented.")
    kshift = kshift ./ (nk1, nk2, nk3)

    for ik in rng
        # For (i, j, k), make k the fastest axis
        k = mod(ik-1, nk3)
        j = mod(div(ik-1 - k, nk3), nk2)
        i = mod(div(ik-1 - k - j*nk3, nk2*nk3), nk1)
        push!(kvecs, Vec3{Float64}(i/nk1, j/nk2, k/nk3) + kshift)
    end
    nk = length(kvecs)
    Kpoints(nk, kvecs, fill(1/nk_grid, (nk,)), (nk1, nk2, nk3))
end

"""
    generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer, mpi_comm::MPI.Comm)
Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return all k points. k points are distributed over the MPI communicator mpi_comm.
"""
function generate_kvec_grid(nk1, nk2, nk3, mpi_comm::MPI.Comm; kshift=[0, 0, 0])
    nk = nk1 * nk2 * nk3
    range = mpi_split_iterator(1:nk, mpi_comm)
    generate_kvec_grid(nk1, nk2, nk3, range, kshift=kshift)
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
    Kpoints(sum(ik_keep), k.vectors[ik_keep], k.weights[ik_keep], k.ngrid)
end

function _gather_ngrid(ngrid, comm)
    # If ngrid is not same among processers, set ngrid to (0,0,0).
    ngrid_root = EPW.mpi_bcast(ngrid, comm)
    all_ngrids_same = EPW.mpi_reduce(ngrid == ngrid_root, &, comm)
    new_ngrid = all_ngrids_same ? ngrid_root : (0, 0, 0)
    new_ngrid
end

"mpi_gather(k::Kpoints, comm::MPI.Comm)"
function mpi_gather(k::Kpoints{FT}, comm::MPI.Comm) where {FT}
    kvectors = EPW.mpi_gather(k.vectors, comm)
    weights = EPW.mpi_gather(k.weights, comm)
    new_ngrid = _gather_ngrid(k.ngrid, comm)
    if mpi_isroot(comm)
        Kpoints{FT}(length(kvectors), kvectors, weights, new_ngrid)
    else
        Kpoints{FT}(0, Vector{Vec3{FT}}(), Vector{FT}(), new_ngrid)
    end
end

"""
    mpi_allgather(k::Kpoints, comm::MPI.Comm)
"""
function mpi_allgather(k::Kpoints, comm::MPI.Comm)
    kvectors = EPW.mpi_allgather(k.vectors, comm)
    weights = EPW.mpi_allgather(k.weights, comm)
    new_ngrid = _gather_ngrid(k.ngrid, comm)
    Kpoints(length(kvectors), kvectors, weights, new_ngrid)
end

"""
    mpi_scatter(k::Kpoints{FT}, comm::MPI.Comm) where {FT}
"""
function mpi_scatter(k::Kpoints{FT}, comm::MPI.Comm) where {FT}
    ngrid = mpi_bcast(k.ngrid, comm)
    vectors = mpi_scatter(k.vectors, comm)
    weights = mpi_scatter(k.weights, comm)
    Kpoints{FT}(length(vectors), vectors, weights, ngrid)
end

"Collect and uniformly redistribute Kpoints among processers"
mpi_gather_and_scatter(k::Kpoints, comm::MPI.Comm) = mpi_scatter(mpi_gather(k, comm), comm)
mpi_gather_and_scatter(k::Kpoints, comm::Nothing) = k


"""
    kpoints_create_subgrid(k::EPW.Kpoints, nsubgrid)
For kpoints from a regular grid, divide each k points and generate kpoints from a subgrid.
"""
function kpoints_create_subgrid(k::EPW.Kpoints, nsubgrid)
    # Check arguments
    if any(nsubgrid .< 1)
        throw(ArgumentError("nsubgrid must be positive, not $nsubgrid."))
    end
    if any(k.ngrid .< 1)
        throw(ArgumentError("k must be from a regular grid. k.ngrid = $(k.ngrid)"))
    end
    # If nsubgrid is (1, 1, 1), do nothing.
    if all(nsubgrid .== 1)
        return k
    end

    multiple = prod(nsubgrid)
    new_n = k.n * multiple
    new_ngrid = k.ngrid .* nsubgrid

    new_weights = repeat(k.weights, inner=multiple)
    new_weights ./= multiple

    # Create a list of subsampled k vectors
    new_vectors = zeros(eltype(k.vectors), new_n)
    dk = 1 ./ k.ngrid ./ nsubgrid
    shift = (1 .- nsubgrid) ./ 2 .* dk
    new_ik = 0
    for ik in 1:k.n
        xk = k.vectors[ik]
        for i1 in 0:nsubgrid[1] - 1
            for i2 in 0:nsubgrid[2] - 1
                for i3 in 0:nsubgrid[3] - 1
                    new_ik += 1
                    new_vectors[new_ik] = xk .+ dk .* (i1, i2, i3) .+ shift
                end
            end
        end
    end
    EPW.Kpoints(new_n, new_vectors, new_weights, new_ngrid)
end