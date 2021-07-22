
# Data and functions for k points
using MPI

export Kpoints
export generate_kvec_grid
export GridKpoints
export xk_to_ik

abstract type AbstractKpoints{T <: Real} end

# TODO: Add `shift` field for shifted regular grid.

struct Kpoints{T <: Real}
    n::Int                   # Number of k points
    vectors::Vector{Vec3{T}} # Fractional coordinate of k points
    weights::Vector{T}       # Weight of each k points
    # size of the grid if kpoints is a subset of grid points. (0,0,0) otherwise.
    ngrid::NTuple{3,Int64}
end

Kpoints{T}() where {T} = Kpoints{T}(0, Vector{Vec3{T}}(), Vector{T}(), (0, 0, 0))

function Kpoints(xks::AbstractArray{T}) where {T <: Real}
    if size(xks, 1) != 3
        throw(ArgumentError("first dimension of xks must be 3"))
    end
    vectors = collect(vec(reinterpret(Vec3{T}, xks)))
    n = length(vectors)
    Kpoints(n, vectors, ones(n) ./ n, (0, 0, 0))
end

function Base.sort!(k::Kpoints)
    inds = sortperm(k.vectors)
    k.vectors .= k.vectors[inds]
    k.weights .= k.weights[inds]
    inds
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

"Filter kpoints using a Boolean vector ik_keep. Retern Kpoints object where
only k points with ik_keep = true are kept."
function get_filtered_kpoints(k::Kpoints, ik_keep)
    length(ik_keep) == k.n || error("length of ik_keep must be equal to k.n")
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
    kpoints_create_subgrid(k::Kpoints, nsubgrid)
For kpoints from a regular grid, divide each k points and generate kpoints from a subgrid.
"""
function kpoints_create_subgrid(k::Kpoints, nsubgrid)
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

"""
    add_two_kpoint_grids(kpts, qpts, k_q_to_kq, map_real_to_int, map_int_to_real)
For k and q in kpts and qpts, return Kpoint with `kq = k_q_to_kq(k, q)`.
ngrid_kq: ngrid for kq points
k_q_to_kq: function from (k, q) to kq
map_real_to_int: mapping of kq to kq_int, the integer coordinates on the grid
map_int_to_real: mapping of kq_int to kq. Inverse of map_real_to_int.
TODO: a better name is needed. Not limited to "add"ing.
TODO: Add test.
"""
function add_two_kpoint_grids(kpts, qpts, ngrid_kq, k_q_to_kq, map_real_to_int, map_int_to_real)
    T = eltype(kpts.weights)
    xkqs = Vector{Vec3{T}}()
    map_xkq_int_to_ikq = Dict{NTuple{3, Int}, Int}()
    map_ikq_to_xkq_int = Vector{NTuple{3, Int}}()
    ikq = 0
    for iq in 1:qpts.n
        xq = qpts.vectors[iq]
        for ik in 1:kpts.n
            xk = kpts.vectors[ik]
            xkq = k_q_to_kq(xk, xq)

            # We need to check whether two kq points are same.
            # To do this, we map from the real vector to the integer vector.
            xkq_int = map_real_to_int(xkq)
            if ! isapprox(xkq, map_int_to_real(xkq_int), atol=10*eps(eltype(xkq)))
                @show xkq, map_int_to_real(xkq_int)
                error("xkq not correctly mapped to integer vector")
            end

            # Find new k+q points, append to map_xkq_int_to_ikq and xkqs
            if xkq_int.data ∉ keys(map_xkq_int_to_ikq)
                ikq += 1
                map_xkq_int_to_ikq[xkq_int.data] = ikq
                push!(map_ikq_to_xkq_int, xkq_int.data)
                push!(xkqs, xkq)
            end
        end
    end
    nkq = length(xkqs)
    kqpts = Kpoints{T}(nkq, xkqs, ones(T, nkq) ./ prod(ngrid_kq), ngrid_kq)
    inds = sort!(kqpts)
    for ikq_new = 1:nkq
        key = map_ikq_to_xkq_int[inds[ikq_new]]
        map_xkq_int_to_ikq[key] = ikq_new
    end
    kqpts, map_xkq_int_to_ikq
end


"""k points that form a subset of a regular grid.
All k points should satisfy ``k = (i, j, k) ./ ngrid + shift`` where i, j, k are integers.
It is assumed that k1 and k2 such that mod(k1, 1) == mod(k2, 1) are not present.
(This is needed to use `_hash_xk` which gives a single integer. If not, one should use 3-tuple
of integers, which cost more memory.)
- `n`: number of k points
- `vectors`: fractional coordinates of the k points
- `weights`: weights of the k points for Brillouin zone integration
Additional arguments for `GridKpoints`:
- `ngrid`: size of the grid
- `shift`: shift of the grid from (0, 0, 0)
- `_xk_hash_to_ik`: dictionary for mapping `_hash_xk(kpts.vectors[ik], kpts)` to `ik`
"""
struct GridKpoints{T} <: AbstractKpoints{T}
    n::Int
    vectors::Vector{Vec3{T}}
    weights::Vector{T}
    ngrid::NTuple{3,Int}
    shift::Vec3{T}
    _xk_hash_to_ik::Dict{Int,Int}
end

function GridKpoints(kpts::EPW.Kpoints{T}) where {T}
    all(kpts.ngrid .> 0) || error("kpts must be on a grid to make GridKpoints")
    if kpts.n == 0
        return GridKpoints{T}(0, Vector{Vec3{T}}(), Vector{T}(), kpts.ngrid, zero(Vec3{T}), Dict{Int,Int}())
    end
    shift = mod.(kpts.vectors[1], 1 ./ kpts.ngrid)
    _xk_hash_to_ik = Dict(_hash_xk.(kpts.vectors, Ref(kpts.ngrid), Ref(shift)) .=> 1:kpts.n)
    GridKpoints{T}(kpts.n, kpts.vectors, kpts.weights, kpts.ngrid, shift, _xk_hash_to_ik)
end

function _hash_xk(xk, ngrid, shift)
    xk_int = round.(Int, (xk .- shift) .* ngrid)
    (xk_int[1] * ngrid[2] + xk_int[2]) * ngrid[3] + xk_int[3]
end
_hash_xk(xk, kpts::GridKpoints) = _hash_xk(xk, kpts.ngrid, kpts.shift)

# Retern index of given xk vector
xk_to_ik(xk, kpts) = kpts._xk_hash_to_ik[_hash_xk(xk, kpts)]

function Base.sort!(k::GridKpoints)
    inds = sortperm(k.vectors)
    k.vectors .= k.vectors[inds]
    k.weights .= k.weights[inds]
    for (ik, xk) in enumerate(k.vectors)
        k._xk_hash_to_ik[_hash_xk(xk, k)] = ik
    end
    inds
end