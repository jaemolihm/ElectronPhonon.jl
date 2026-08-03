# Data and functions for k points
using MPI
using Printf
using Base.Threads: @threads

export Kpoints
export kpoints_grid
export GridKpoints
export xk_to_ik
export shift_center!
export split_kpoints

abstract type AbstractKpoints{T <: Real} end

# TODO: Add `shift` field for shifted regular grid.

"""
    Kpoints{T}

Generic type for k points.
Use `kpoints_grid` to generate a regular grid of k points.
"""
struct Kpoints{T} <: AbstractKpoints{T}
    n::Int                   # Number of k points
    vectors::Vector{Vec3{T}} # Fractional coordinate of k points
    weights::Vector{T}       # Weight of each k points
    # size of the grid if kpoints is a subset of grid points. (0,0,0) otherwise.
    ngrid::NTuple{3,Int64}
end

@inline function Base.getproperty(obj::Kpoints, name::Symbol)
    if name === :shift
        zero(eltype(getfield(obj, :vectors)))
    else
        getfield(obj, name)
    end
end


Kpoints{T}() where {T} = Kpoints{T}(0, Vector{Vec3{T}}(), Vector{T}(), (0, 0, 0))

# Initializing Kpoints with a vector of k points
function Kpoints(xks::AbstractVector{Vec3{T}}; ngrid = (0, 0, 0)) where {T <: Real}
    n = length(xks)
    if any(ngrid .> 0)
        all(ngrid .> 0) || error("ngrid must be positive, not $ngrid.")
        for xk in xks
            if !(round.(Int, xk .* ngrid) ≈ xk .* ngrid)
                error("xk must be on the grid")
            end
        end
    end
    Kpoints{T}(n, Vector(xks), ones(n) ./ n, ngrid)
end

# Initializing Kpoints with a k point array
function Kpoints(xks::AbstractArray{T}) where {T <: Real}
    if size(xks, 1) != 3
        throw(ArgumentError("first dimension of xks must be 3"))
    end
    Kpoints(collect(vec(reinterpret(Vec3{T}, xks))))
end

# Initializing Kpoints with a single k point
Kpoints(xk::Vec3{T}) where {T <: Real} = Kpoints{T}(1, [xk], [T(1)], (1, 1, 1))

Base.sortperm(k::AbstractKpoints) = sortperm(k.vectors)

function shift_center!(k::AbstractKpoints, center)
    for ik in 1:k.n
        k.vectors[ik] = mod.(k.vectors[ik] .- center .+ 1//2, 1) .+ center .- 1//2
    end
    k
end

function Base.sort!(k::Kpoints)
    inds = sortperm(k.vectors)
    k.vectors .= k.vectors[inds]
    k.weights .= k.weights[inds]
    k
end

function generate_kvec_grid(nk1, nk2, nk3; shift=(0, 0, 0))
    Base.depwarn("Renamed. Use kpoints_grid instead", :generate_kvec_grid)
    kpoints_grid((nk1, nk2, nk3); shift)
end

function generate_kvec_grid(nk1, nk2, nk3, mpi_comm::MPI.Comm; shift=(0, 0, 0))
    Base.depwarn("Renamed. Use kpoints_grid instead", :generate_kvec_grid)
    kpoints_grid((nk1, nk2, nk3), mpi_comm; shift)
end

"""
    kpoints_grid(ngrid; shift=(0, 0, 0); symmetry=nothing, ignore_time_reversal=false) => Kpoints
Generate regular grid of k points with size ngrid. Shift the grid from the center by `shift`
in crystal coordinates.
If `mpi_comm` is set to a MPI communicator, distribute the k points over it.
# Keyword arguments
- `symmetry`: If given, return the irreducible wedge of a uniform Brillouin zone mesh. The
    `symmetry` is incompatible with `shift`: : mesh always includes the Gamma point.
- `ignore_time_reversal`: If `true` and `symmetry` is given, ignore all symmetries involving
    time reversal.
"""
function kpoints_grid(ngrid, mpi_comm::Union{MPI.Comm, Nothing}=nothing; shift=(0, 0, 0), symmetry=nothing, ignore_time_reversal=false)
    if symmetry isa Symmetry
        if shift != (0, 0, 0)
            error("kpoints_grid with symmetry incompatible with shift")
        end
        if mpi_comm isa MPI.Comm
            # Create the irreducible k points on the root, then scatter them across ranks. The
            # scatter keeps the `GridKpoints` type (each rank's subset is grid-aligned and rebuilds
            # its own xk->ik hash), so both the serial and MPI symmetry branches return `GridKpoints`.
            kpoints = mpi_isroot(mpi_comm) ? kpoints_grid_symmetry(ngrid, symmetry; ignore_time_reversal) : GridKpoints{Float64}()
            return mpi_scatter(kpoints, mpi_comm)
        else
            return kpoints_grid_symmetry(ngrid, symmetry; ignore_time_reversal)
        end
    elseif symmetry === nothing
        nk = prod(ngrid)
        range = mpi_comm isa MPI.Comm ? mpi_split_iterator(1:nk, mpi_comm) : 1:nk
        # TODO: return GridKpoints here too (see kpoints_grid_range) so the non-symmetry branch
        # matches the symmetry branch's GridKpoints return type in both the serial and MPI cases.
        return kpoints_grid_range(ngrid, range; shift)
    else
        error("Wrong input symmetry: must be a Symmetry object or nothing")
    end
end

"""
    kpoints_grid_range(ngrid::NTuple{3, Int}, rng)
Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return k points for global index in the given range.
-`shift`: Shift for the grid in the crystal coordinates.
"""
function kpoints_grid_range(ngrid::NTuple{3, Int}, rng::UnitRange{Int}, ::Type{FT}=Float64; shift=(0, 0, 0)) where FT
    # TODO: Type
    nk1, nk2, nk3 = ngrid
    nk = nk1 * nk2 * nk3
    @assert rng[1] >= 1
    @assert rng[end] <= nk
    kvecs = Vec3{FT}[]

    for ik in rng
        # For (i, j, k), make k the fastest axis
        k = mod(ik-1, nk3)
        j = mod(div(ik-1 - k, nk3), nk2)
        i = mod(div(ik-1 - k - j*nk3, nk2*nk3), nk1)
        push!(kvecs, Vec3{FT}(i/nk1, j/nk2, k/nk3) .+ shift)
    end
    nk = length(kvecs)
    Kpoints(nk, kvecs, fill(one(FT) / (nk1 * nk2 * nk3), nk), (nk1, nk2, nk3))
end

"Filter kpoints using a Boolean vector ik_keep. Retern Kpoints object where
only k points with ik_keep = true are kept."
function get_filtered_kpoints(k::Kpoints, ik_keep)
    length(ik_keep) == k.n || error("length of ik_keep must be equal to k.n")
    Kpoints(sum(ik_keep), k.vectors[ik_keep], k.weights[ik_keep], k.ngrid)
end

function _gather_ngrid(ngrid, comm)
    # If ngrid is not same among processers, set ngrid to (0,0,0).
    ngrid_root = mpi_bcast(ngrid, comm)
    all_ngrids_same = mpi_reduce(ngrid == ngrid_root, &, comm)
    new_ngrid = all_ngrids_same ? ngrid_root : (0, 0, 0)
    new_ngrid
end

"mpi_gather(k::Kpoints, comm::MPI.Comm)"
function mpi_gather(k::Kpoints{FT}, comm::MPI.Comm) where {FT}
    kvectors = mpi_gather(k.vectors, comm)
    weights = mpi_gather(k.weights, comm)
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
    kvectors = mpi_allgather(k.vectors, comm)
    weights = mpi_allgather(k.weights, comm)
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
    Kpoints(new_n, new_vectors, new_weights, new_ngrid)
end

# Two places in this file need an `xk -> ik` map for `xk` on a grid of size `ngrid`:
# `combine_kpoint_grids` below, to de-duplicate k+q, and `GridKpoints` further down, as its inverse
# index. There are two ways to build such a map — a hash `Dict` keyed by the packed grid coordinate,
# or a dense table with one slot per grid node — and they return identical results.
#
# The tradeoff. The `Dict` costs ~34 B per stored point (17 B per slot at a load factor of at most
# 0.75), so its memory scales with the number of points `n`: it is the cheap one on a grid far
# sparser than its point count. The dense table costs a fixed 8 B per grid node whatever `n` is, so
# it wins on a nearly full grid: at `n >= prod(ngrid)/4` it already uses less memory than the Dict,
# which costs ~4x (34/8) the bytes per point that the table costs per node. Dense access is also
# faster — an array load instead of a hash probe.
#
# So each map is built dense while it fits a memory cap and as a `Dict` above it. The caps are per
# map, not shared: the de-duplication table is transient (freed when `combine_kpoint_grids`
# returns), while the index is retained for the lifetime of every `GridKpoints` (and
# `split_kpoints` makes N of them), so the retained one is budgeted much more tightly. A
# `GridKpoints` keeps at most one of the two index maps; `_use_dense_index` is the authority on
# which one it gets.

"""
    _combine_kq_dedup_dense(BkS, BqS, sgn, ngrid_kq, shift_kq)
De-duplicate `k + sgn * q` over all (k, q) pairs with a dense `prod(ngrid_kq)` table indexed by the
integer grid coordinates. Returns the unique k+q points, in grid order.
- `BkS`, `BqS`: integer grid coordinates of the k and of the q points on the common `ngrid_kq`
    grid, i.e. the `b` of `xk = shift + b ./ ngrid_kq`, each already reduced into `0:ng-1` by
    `combine_kpoint_grids` so that `bk + sgn*bq` needs only `_wrap_reduced`.

The table is `(ng3, ng2, ng1)` indexed `[c3+1, c2+1, c1+1]`, making `c3` the unit-stride index so
that a sweep over consecutive `c3` walks the table sequentially. With those dims the linear index
of a slot is exactly `hash + 1` for the `(c1*ng2 + c2)*ng3 + c3` packing, so this table and
`GridKpoints._dense_hash_to_ik` share one layout.

Two passes rather than one. Pass 1 only MARKS which slots occur, in an `Array{Bool}` instead of the
`Array{Int}` of indices the single-pass version needed: the pair loop's access into the table is a
random probe, so at 8 B per node it missed cache on nearly every pair, and `Bool` is 1 B per node,
putting 8× more of the grid in cache (8 MB vs 64 MB at 200³). Marking is also idempotent — every
writer stores the same `true` — which is what lets the pass be threaded with no synchronization.
(A `BitArray` would be smaller still but its writes race at word granularity; `Array{Bool}` stores
one byte per element, so its writes do not.)
Pass 2 sweeps the table in index order and emits the points. That discards first-appearance order,
which is safe because `combine_kpoint_grids` sorts the result on the integer grid coordinates — a
total order, the points being distinct — and `sort!(::GridKpoints)` renumbers the index afterwards.
"""
function _combine_kq_dedup_dense(BkS, BqS, sgn, ngrid_kq, shift_kq::Vec3{T}) where {T}
    ng1, ng2, ng3 = ngrid_kq
    _check_reduced_coords(BkS, BqS, ngrid_kq)
    seen = zeros(Bool, ng3, ng2, ng1)
    _mark_kq_pairs!(seen, BkS, BqS, sgn, ng1, ng2, ng3)

    xkqs = Vector{Vec3{T}}(undef, count(seen))
    ikq = 0
    for c1 in 0:(ng1 - 1), c2 in 0:(ng2 - 1), c3 in 0:(ng3 - 1)
        seen[c3 + 1, c2 + 1, c1 + 1] || continue
        ikq += 1
        xkqs[ikq] = Vec3(c1 / ng1, c2 / ng2, c3 / ng3) + shift_kq
    end
    xkqs
end

# The `0:ng-1` precondition of the two de-duplication routines, enforced once over the two lists
# (O(nk + nkq), microseconds) rather than per pair. Not optional politeness: the pair loops fold
# with `_wrap_reduced` and write the table under `@inbounds`, so an unreduced coordinate is a
# segfault rather than a wrong answer.
function _check_reduced_coords(BkS, BqS, ngrid)
    for (name, B) in (("BkS", BkS), ("BqS", BqS)), b in B
        all(0 .<= b.data .< ngrid) || throw(ArgumentError(
            "$name entry $b is not reduced into 0:ngrid-1 for ngrid = $ngrid; " *
            "call _grid_coords_reduced on the integer grid coordinates first"))
    end
end

# Pass 1 of `_combine_kq_dedup_dense`, as a standalone function so the threaded chunks capture
# typed arguments rather than `Core.Box`. Partitioned over `BqS` (the longer list in the k+q use),
# each chunk sweeping the whole `BkS` — chunks share `seen` but only ever write the same value into
# a slot, so no synchronization is needed and the result does not depend on the partition.
function _mark_kq_pairs!(seen::Array{Bool,3}, BkS::Vector{Vec3{Int}}, BqS::Vector{Vec3{Int}},
                         sgn::Int, ng1::Int, ng2::Int, ng3::Int)
    @threads for iq in eachindex(BqS)
        # `@inbounds` here is load-bearing for performance, not tidiness: this block runs
        # nk*nkq ~ 1e9 times. It is safe because `_check_reduced_coords` has already established
        # `0 <= b < ng` for every entry of both lists, so `_wrap_reduced` returns a value in
        # `0:ng-1` on all three axes and every index below is in range. It also elides
        # `_wrap_reduced`'s own `@boundscheck`, whose precondition that guard is what establishes.
        @inbounds begin
            bq = BqS[iq]
            q1, q2, q3 = sgn * bq[1], sgn * bq[2], sgn * bq[3]
            for bk in BkS
                c1 = _wrap_reduced(bk[1] + q1, ng1)
                c2 = _wrap_reduced(bk[2] + q2, ng2)
                c3 = _wrap_reduced(bk[3] + q3, ng3)
                seen[c3 + 1, c2 + 1, c1 + 1] = true
            end
        end
    end
    seen
end

"""
    _combine_kq_dedup_dict(BkS, BqS, sgn, ngrid_kq, shift_kq)
Same map as `_combine_kq_dedup_dense`, keyed by the packed hash in a `Dict` so the memory is
O(number of unique points) instead of O(prod(ngrid_kq)). Used when the dense table would exceed the
transient memory budget in `combine_kpoint_grids`. Same reduced-coordinate precondition, and it
returns the points in order of FIRST APPEARANCE (a `Dict` has no order to sweep), where the dense
path returns them in grid order — `combine_kpoint_grids` sorts either, so the two agree as sets.
"""
function _combine_kq_dedup_dict(BkS, BqS, sgn, ngrid_kq, shift_kq::Vec3{T}) where {T}
    ng1, ng2, ng3 = ngrid_kq
    _check_reduced_coords(BkS, BqS, ngrid_kq)
    xkqs = Vector{Vec3{T}}()
    cs = Vector{NTuple{3,Int}}()  # integer coords per stored point, for the collision guard
    xkq_hash_to_ikq = Dict{Int, Int}()
    ikq = 0
    for bq in BqS
        q1, q2, q3 = sgn * bq[1], sgn * bq[2], sgn * bq[3]
        for bk in BkS
            c1 = _wrap_reduced(bk[1] + q1, ng1)
            c2 = _wrap_reduced(bk[2] + q2, ng2)
            c3 = _wrap_reduced(bk[3] + q3, ng3)
            xk_hash_value = (c1 * ng2 + c2) * ng3 + c3

            # Find new k+q points, append to xkq_hash_to_ikq and xkqs
            ikq_found = get(xkq_hash_to_ikq, xk_hash_value, 0)
            if ikq_found == 0
                ikq += 1
                xkq_hash_to_ikq[xk_hash_value] = ikq
                push!(cs, (c1, c2, c3))
                push!(xkqs, Vec3(c1 / ng1, c2 / ng2, c3 / ng3) + shift_kq)
            else
                # A hash hit must be the same grid point. A mismatch here means an off-grid input
                # or a hash that overflowed Int (prod(ngrid_kq) too large) — fail loudly.
                @assert cs[ikq_found] == (c1, c2, c3)
            end
        end
    end
    xkqs
end

"""
    combine_kpoint_grids(kpts, qpts, op, ngrid_kq)
For k and q in kpts and qpts, return a `GridKpoints` with `kq = op(k, q)`.
`kpts` and `qpts` must lie on regular grids (they are converted to `GridKpoints`).
The returned points are folded into [-0.5, 0.5)^3 and sorted in grid order.
ngrid_kq: ngrid for kq points
op: function from (k, q) to kq. Only + and -.
"""
function combine_kpoint_grids(kpts, qpts, op, ngrid_kq)
    if Symbol(op) !== :+ && Symbol(op) !== :-
        error("op must be + or -")
    end
    # This function requires the inputs to lie on a regular grid.
    # So convert them to GridKpoints if they are not already.
    kpts = GridKpoints(kpts)
    qpts = GridKpoints(qpts)

    # op(k, q) lies on the ngrid_kq grid only if ngrid_kq is a multiple of both input grids.
    all(mod.(ngrid_kq, kpts.ngrid) .== 0) || throw(ArgumentError(
        "ngrid_kq = $ngrid_kq must be divisible by kpts.ngrid = $(kpts.ngrid)"))
    all(mod.(ngrid_kq, qpts.ngrid) .== 0) || throw(ArgumentError(
        "ngrid_kq = $ngrid_kq must be divisible by qpts.ngrid = $(qpts.ngrid)"))

    T = eltype(kpts.weights)
    shift_k = kpts.vectors[1]
    shift_q = qpts.vectors[1]
    shift_kq = op(shift_k, shift_q)
    sgn = Symbol(op) === :+ ? 1 : -1

    # Integer coordinates on the common grid, in `0:ng-1`. Every input point sits on ngrid_kq exactly
    # (it is a multiple of both input grids, checked above), so `op(k, q)` is integer arithmetic —
    # exact, and the reduced operands let the O(nk·nkq) fold be `_wrap_reduced` rather than `mod`.
    # These coordinates are also the de-duplication key: `_combine_kq_dedup_dict` packs them as
    # `(c1*ng2 + c2)*ng3 + c3` (which equals `_hash_xk(xkq)`), the dense path indexes by them.
    BkS = [_grid_coords_reduced(xk, ngrid_kq, shift_k) for xk in kpts.vectors]
    BqS = [_grid_coords_reduced(xq, ngrid_kq, shift_q) for xq in qpts.vectors]

    # Dense lookup table while it fits in the transient budget, Dict above it. The budget is loose
    # because the table lives only until this function returns. `Int128` because `prod(ngrid_kq)`
    # itself may overflow `Int` here (it is only checked in the `GridKpoints` constructor below,
    # which the Dict path reaches).
    dedup_table_max_bytes = 8 * 1024^3  # 8 GB
    xkqs = if prod(Int128.(ngrid_kq)) * sizeof(Int) <= dedup_table_max_bytes
        _combine_kq_dedup_dense(BkS, BqS, sgn, ngrid_kq, shift_kq)
    else
        _combine_kq_dedup_dict(BkS, BqS, sgn, ngrid_kq, shift_kq)
    end

    nkq = length(xkqs)
    weights = ones(T, nkq) ./ prod(ngrid_kq)
    kqpts = GridKpoints(Kpoints{T}(nkq, xkqs, weights, ngrid_kq))
    shift_center!(kqpts, (0, 0, 0))  # Fold the points into [-0.5, 0.5)^3.
    sort!(kqpts)  # Sort so the points are ordered
    kqpts
end

function add_two_kpoint_grids(kpts, qpts, op, ngrid_kq)
    Base.depwarn("add_two_kpoint_grids is deprecated, use combine_kpoint_grids instead", :add_two_kpoint_grids, force=true)
    combine_kpoint_grids(kpts, qpts, op, ngrid_kq)
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
Both of the following are derived caches of the same `_hash_xk(kpts.vectors[ik], kpts) -> ik` map,
and at most one of them is populated — exactly one when `n > 0`, and `_use_dense_index` picks which;
an empty `GridKpoints` has neither. Read them only through `_ik_from_hash`.
- `_dense_hash_to_ik`: a `(ngrid[3], ngrid[2], ngrid[1])` table of `ik`, where `0` marks a grid node
    that is not a k point. Those dims make the linear index of a node exactly `hash + 1`, which is
    how `_ik_from_hash` reads it. Empty when the grid is too sparse or too large for it.
- `_xk_hash_to_ik`: the fallback for those grids, a `Dict` of `hash => ik`. Empty when the dense
    table was built.
"""
struct GridKpoints{T} <: AbstractKpoints{T}
    n::Int
    vectors::Vector{Vec3{T}}
    weights::Vector{T}
    ngrid::NTuple{3,Int}
    shift::Vec3{T}
    _xk_hash_to_ik::Dict{Int,Int}
    _dense_hash_to_ik::Array{Int,3}
end

# The index-map policy for `GridKpoints`: whether `n` points on a `prod(ngrid)`-node grid get the
# dense inverse index (8 B per grid node) or the `Dict` (~34 B per stored point). See the discussion
# of the tradeoff above `_combine_kq_dedup_dense`.
# `index` overrides the policy: `:auto` applies it, `:dense`/`:dict` force one map regardless of
# size (`:dense` on a huge grid will happily try to allocate it).
function _use_dense_index(n, ngrid; index = :auto)
    index === :dense && return true
    index === :dict && return false
    index === :auto || throw(ArgumentError("index must be :auto, :dense or :dict, got $index"))

    nnode = prod(ngrid)
    # Not for a grid far sparser than its point count: a multigrid/AMR grid can have a huge `ngrid`
    # holding a handful of points, where the Dict is small, L2-resident and already fast.
    dense_enough = nnode <= max(8 * n, 2^20)
    # Not above the size cap either — unless the Dict it would fall back to is the bigger of the
    # two, which is the case below 34/8 = 4.25 nodes per point (rounded down to stay conservative).
    dense_index_max_bytes = 256 * 1024^2  # 256 MB, retained for the lifetime of the GridKpoints
    fits_cap = nnode <= dense_index_max_bytes ÷ sizeof(Int)
    beats_dict = nnode <= 4 * n
    dense_enough && (fits_cap || beats_dict)
end

# Build the dense inverse index. Shaped so that the linear index of `[c3+1, c2+1, c1+1]` is exactly
# `_hash_xk + 1`, which is how `_ik_from_hash` and `sort!` address it.
function _build_dense_index(n, vectors, ngrid, shift)
    table = zeros(Int, ngrid[3], ngrid[2], ngrid[1])
    # Serial and ascending, so a repeated point resolves to the last `ik` — the same tie-break as
    # `Dict(hashes .=> 1:n)`, where later pairs overwrite earlier ones.
    for ik in 1:n
        table[_hash_xk(vectors[ik], ngrid, shift) + 1] = ik
    end
    table
end

# Every construction site that fills all fields goes through this, so the index policy lives in one
# place: exactly one of the two index maps is built, never both. `make_dict` supplies the Dict and
# is called only when the Dict is the index that will be kept, so a caller that would have to build
# one from scratch does not pay for it on a dense grid; callers holding one already can pass
# `() -> dict`. `index` is forwarded to `_use_dense_index`.
function _make_grid_kpoints(n, vectors::Vector{Vec3{T}}, weights, ngrid, shift,
                            make_dict = () -> Dict{Int,Int}(); index = :auto) where {T}
    no_table = zeros(Int, 0, 0, 0)
    if n == 0
        # Nothing to look up, so neither map is built (and `ngrid` may be (0,0,0) here).
        GridKpoints{T}(n, vectors, weights, ngrid, shift, Dict{Int,Int}(), no_table)
    elseif _use_dense_index(n, ngrid; index)
        GridKpoints{T}(n, vectors, weights, ngrid, shift, Dict{Int,Int}(),
                       _build_dense_index(n, vectors, ngrid, shift))
    else
        GridKpoints{T}(n, vectors, weights, ngrid, shift, make_dict(), no_table)
    end
end

"""
    GridKpoints(kpts::Kpoints, ngrid = kpts.ngrid; atol, index = :auto)
`index` selects the xk->ik index map: `:auto` leaves it to `_use_dense_index`, `:dense` and `:dict`
force one of the two.
"""
function GridKpoints(kpts::Kpoints{T}, ngrid = kpts.ngrid; atol = sqrt(eps(T)),
                     index = :auto) where {T}
    all(ngrid .> 0) || throw(ArgumentError("ngrid must be set or provided to make GridKpoints"))
    # `_hash_xk` packs the grid index into a single Int and can reach prod(ngrid) - 1, so
    # prod(ngrid) must fit in Int. Widen the product so prod(ngrid) itself cannot overflow.
    prod(Int128.(ngrid)) < typemax(Int) || throw(ArgumentError(
        "prod(ngrid) = $(prod(Int128.(ngrid))) must be < typemax(Int) to avoid overflow in the k-point hash"))
    if kpts.n == 0
        return _make_grid_kpoints(0, Vector{Vec3{T}}(), Vector{T}(), ngrid, zero(Vec3{T}))
    end

    shift = mod.(first(kpts.vectors) .* ngrid, 1) ./ ngrid

    # Check if all k points are on the shifted grid
    for xk in kpts.vectors
        nxk = (xk - shift) .* ngrid
        if ! isapprox(round.(Int, nxk), nxk; atol)
            throw(ArgumentError("k point $xk is not on the grid of size $ngrid shifted by $shift"))
        end
    end

    _make_grid_kpoints(kpts.n, kpts.vectors, kpts.weights, ngrid, shift,
        () -> Dict(_hash_xk.(kpts.vectors, Ref(ngrid), Ref(shift)) .=> 1:kpts.n); index)
end

GridKpoints(xk::Vec3{T}) where {T <: Real} = GridKpoints(Kpoints(xk))

# Empty GridKpoints (e.g. the receive-side placeholder on non-root ranks before mpi_scatter).
GridKpoints{T}() where {T} = _make_grid_kpoints(0, Vector{Vec3{T}}(), Vector{T}(), (0, 0, 0), zero(Vec3{T}))

# Reduce GridKpoints to Kpoints
Kpoints(k::GridKpoints{T}) where {T} = Kpoints{T}(k.n, k.vectors, k.weights, k.ngrid)
GridKpoints(k::GridKpoints) = k

"""
    mpi_scatter(k::GridKpoints{FT}, comm::MPI.Comm) where {FT}
Scatter the grid-aligned points across `comm`, returning a `GridKpoints` on each rank (the
per-rank xk->ik hash is rebuilt from that rank's subset). Each subset is itself on the grid, so
the result stays a `GridKpoints` — matching the serial symmetry return type.
"""
function mpi_scatter(k::GridKpoints{FT}, comm::MPI.Comm) where {FT}
    ngrid = mpi_bcast(k.ngrid, comm)
    vectors = mpi_scatter(k.vectors, comm)
    weights = mpi_scatter(k.weights, comm)
    GridKpoints(Kpoints{FT}(length(vectors), vectors, weights, ngrid), ngrid)
end

function mpi_gather(k::GridKpoints{FT}, comm::MPI.Comm) where {FT}
    kvectors = mpi_gather(k.vectors, comm)
    weights = mpi_gather(k.weights, comm)
    new_ngrid = _gather_ngrid(k.ngrid, comm)
    if mpi_isroot(comm)
        GridKpoints(Kpoints{FT}(length(kvectors), kvectors, weights, new_ngrid), new_ngrid)
    else
        GridKpoints{FT}()
    end
end

# Gather every rank's k-points to the root and redistribute them evenly. Needed after
# `filter_electron_states`, which drops out-of-window points and so leaves an unbalanced count per
# rank — this restores an even split. Every point is a node of the original regular grid, so any
# split of them is still made of grid nodes ("grid-aligned") and each rank's chunk is again a
# `GridKpoints`. (`filter_electron_states` redistributes the band ranges alongside the k-points with
# the same gather/scatter primitives; this helper covers callers that redistribute a bare grid.)
mpi_gather_and_scatter(k::GridKpoints, comm::MPI.Comm) = mpi_scatter(mpi_gather(k, comm), comm)
mpi_gather_and_scatter(k::GridKpoints, comm::Nothing) = k

# Integer grid coordinates of `xk`, reduced into `0:ngrid[d]-1`: the `c` of `xk ≡ shift + c ./ ngrid`
# (mod 1). These are what `_hash_xk` packs; they are exposed separately so that a loop over pairs of
# grid points can reduce both operands ONCE and then fold their sum or difference with
# `_wrap_reduced`, instead of paying `mod`'s runtime integer division per pair.
@inline _grid_coords_reduced(xk::Vec3, ngrid, shift) = Vec3(
    mod(round(Int, (xk[1] - shift[1]) * ngrid[1]), ngrid[1]),
    mod(round(Int, (xk[2] - shift[2]) * ngrid[2]), ngrid[2]),
    mod(round(Int, (xk[3] - shift[3]) * ngrid[3]), ngrid[3]))

# Fold `d` into `0:ng-1` given `-ng < d < 2ng`, which the sum or difference of two already-reduced
# coordinates satisfies. Two compares instead of a 64-bit division, worth naming because the pair
# loops run it ~1e9 times. Outside that range one fold does not reach `0:ng-1` and the result then
# indexes out of bounds, so it is checked — in a `@boundscheck`, so callers that have established
# the precondition another way (`_check_reduced_coords`) can say `@inbounds` and pay nothing.
@inline function _wrap_reduced(d::Int, ng::Int)
    @boundscheck -ng < d < 2ng || throw(ArgumentError(
        "_wrap_reduced($d, $ng): argument outside (-ng, 2ng), so one fold cannot reduce it into " *
        "0:ng-1; reduce both operands with _grid_coords_reduced before folding"))
    ifelse(d < 0, d + ng, ifelse(d >= ng, d - ng, d))
end

function _hash_xk(xk::Vec3, ngrid, shift)
    c = _grid_coords_reduced(xk, ngrid, shift)
    (c[1] * ngrid[2] + c[2]) * ngrid[3] + c[3]
end
_hash_xk(xk, kpts::GridKpoints) = _hash_xk(xk, kpts.ngrid, kpts.shift)

# Index of the k point with the given hash, 0 if there is none. `ik` is 1-based, so 0 is
# unambiguous even though 0 is itself a legal hash (Γ on an unshifted grid).
# `hash` must be in `0:prod(kpts.ngrid)-1`, which every hash from `_hash_xk` is (it mods each
# coordinate by `ngrid`), as is the equivalent integer-coordinate arithmetic in the GPU outer-k
# loop. The dense index is bounds-checked, so a caller that packs a hash by hand against a different
# `ngrid` gets a `BoundsError` rather than a silently wrong `ik`.
@inline function _ik_from_hash(kpts::GridKpoints, hash::Int)
    dense = kpts._dense_hash_to_ik
    if isempty(dense)
        get(kpts._xk_hash_to_ik, hash, 0)
    else
        dense[hash + 1]
    end
end

# Retern index of given xk vector
function xk_to_ik(xk, kpts)
    ik = _ik_from_hash(kpts, _hash_xk(xk, kpts))
    ik == 0 ? nothing : ik
end

Base.sortperm(k::GridKpoints) = sortperm(map(xk -> round.(Int, (xk - k.shift).data .* k.ngrid), k.vectors))

function Base.sort!(k::GridKpoints)
    inds = sortperm(k)
    k.vectors .= k.vectors[inds]
    k.weights .= k.weights[inds]
    # Renumber whichever index is the populated one, ascending so that a repeated point keeps the
    # same tie-break as construction. A permutation does not change the set of hashes, so no entry
    # goes stale.
    if isempty(k._dense_hash_to_ik)
        for (ik, xk) in enumerate(k.vectors)
            k._xk_hash_to_ik[_hash_xk(xk, k)] = ik
        end
    else
        for (ik, xk) in enumerate(k.vectors)
            k._dense_hash_to_ik[_hash_xk(xk, k) + 1] = ik
        end
    end
    k
end

# The two indices are derived from (vectors, ngrid, shift), so they are not compared.
Base.:(==)(k1::GridKpoints, k2::GridKpoints) = (k1.n ≈ k2.n
    && k1.vectors ≈ k2.vectors
    && k1.weights ≈ k2.weights
    && k1.shift ≈ k2.shift
    && k1.ngrid == k2.ngrid
)

get_filtered_kpoints(k::GridKpoints, ik_keep) = GridKpoints(get_filtered_kpoints(Kpoints(k), ik_keep))
kpoints_create_subgrid(k::GridKpoints, nsubgrid) = GridKpoints(kpoints_create_subgrid(Kpoints(k), nsubgrid))

"""
    unfold_kpoints(kpts::GridKpoints, symmetry; index = :auto) => kpts_unfold, ik_to_ikirr_isym

Unfold k points using symmetry to the full Brillouin zone.
Output `ik_to_ikirr_isym` gives a map ik => (ikirr, isym) such that ``xk[ik] = S[isym](xkirr[ikirr])``.
`index` is the `GridKpoints` index-map override, see `_use_dense_index`.
"""
function unfold_kpoints(kpts::GridKpoints, symmetry; index = :auto)
    # If symmetry is trivial, do nothing and return a copy of input kpts
    if symmetry.nsym == 1
        kpts_unfold = index === :auto ? deepcopy(kpts) : _make_grid_kpoints(
            kpts.n, copy(kpts.vectors), copy(kpts.weights), kpts.ngrid, kpts.shift; index)
        return kpts_unfold, [(ik, 1) for ik in 1:kpts.n]
    end

    ngrid = kpts.ngrid
    shift = kpts.shift

    # For the unfolded kpts to be GridKpoints, all symmetry mapping of kpts.shift must be on the grid.
    for symop in symmetry
        s_shift = symop.is_tr ? -symop.S * shift : symop.S * shift
        dk = s_shift - shift
        if norm(dk - Vec3(round.(Int, dk.data .* ngrid) ./ ngrid)) > 10*eps(eltype(dk))
            error("kpts.shift = $(kpts.shift) does not respect the symmetry $symop. Cannot unfold.")
        end
    end

    # Unfold k points
    sk_hash_dict = Dict{Int, Int}()
    sk_vectors = empty(kpts.vectors)
    sk_weights = empty(kpts.weights)
    ik_to_ikirr_isym = Tuple{Int, Int}[]

    for (isym, symop) in enumerate(symmetry)
        for ik in 1:kpts.n
            xk = kpts.vectors[ik]
            sk = symop.is_tr ? -symop.S * xk : symop.S * xk
            sk = normalize_kpoint_coordinate(sk)
            sk_hash = _hash_xk(sk, ngrid, shift)

            isk = get(sk_hash_dict, sk_hash, nothing)
            if isk === nothing
                # new sk point
                push!(sk_vectors, sk)
                push!(sk_weights, kpts.weights[ik])
                push!(ik_to_ikirr_isym, (ik, isym))
                sk_hash_dict[sk_hash] = length(sk_vectors)
            else
                # sk point already found
                sk_weights[isk] += kpts.weights[ik]
            end
        end
    end
    # Each k point is mapped to length(symmetry) sk points, so divide weights by length(symmetry).
    sk_weights ./= length(symmetry)

    kpts_unfold = _make_grid_kpoints(length(sk_vectors), sk_vectors, sk_weights, ngrid, shift,
                                     () -> sk_hash_dict; index)
    ik_to_ikirr_isym = ik_to_ikirr_isym[sortperm(kpts_unfold)]
    sort!(kpts_unfold)

    return kpts_unfold, ik_to_ikirr_isym
end

"""
    fold_kpoints(kpts, symmetry; index = :auto) => kpts_irr, ik_to_ikirr_isym
Inverse of `unfold_kpoints`. Reduce `kpts` to the irreducible BZ using `symmetry`.
Output ik_to_ikirr_isym gives a map ik => (ikirr, isym) such that ``xk[ik] = S[isym](xkirr[ikirr])``.
`index` is the `GridKpoints` index-map override, see `_use_dense_index`.
"""
function fold_kpoints(kpts::GridKpoints, symmetry; index = :auto)
    if symmetry.nsym == 1
        kpts_irr = index === :auto ? deepcopy(kpts) : _make_grid_kpoints(
            kpts.n, copy(kpts.vectors), copy(kpts.weights), kpts.ngrid, kpts.shift; index)
        return kpts_irr, [(ik, 1) for ik = 1:kpts.n]
    end

    ngrid = kpts.ngrid
    shift = kpts.shift

    hash_dict_irr = Dict{Int, Int}()
    vectors_irr = empty(kpts.vectors)
    weights_irr = empty(kpts.weights)
    ik_to_ikirr_isym = Tuple{Int, Int}[]

    for ik = 1:kpts.n
        xk = kpts.vectors[ik]

        irr_found = false
        for (isym, symop) in enumerate(symmetry)
            # We want the mapping xk = S * xkirr, so we compute sk = inv(S) * xk.
            # FIXME: Optimize by finding inv(S) in symop.
            sk = symop.is_tr ? -inv(symop.S) * xk : inv(symop.S) * xk
            sk = normalize_kpoint_coordinate(sk)
            sk_hash = _hash_xk(sk, ngrid, shift)

            ikirr = get(hash_dict_irr, sk_hash, nothing)
            if ikirr !== nothing
                # xk maps to an existing irreducible k point
                irr_found = true
                weights_irr[ikirr] += kpts.weights[ik]
                push!(ik_to_ikirr_isym, (ikirr, isym))
                break
            end
        end

        # xk is a new irredicuble k point
        if ! irr_found
            push!(vectors_irr, xk)
            push!(weights_irr, kpts.weights[ik])
            push!(ik_to_ikirr_isym, (length(vectors_irr), 1))
            xk_hash = _hash_xk(xk, ngrid, shift)
            hash_dict_irr[xk_hash] = length(vectors_irr)
        end
    end

    kpts_irr = _make_grid_kpoints(length(vectors_irr), vectors_irr, weights_irr, ngrid, shift,
                                  () -> hash_dict_irr; index)

    inds = invperm(sortperm(kpts_irr))
    sort!(kpts_irr)

    ik_to_ikirr_isym = [(inds[ikirr], isym) for (ikirr, isym) in ik_to_ikirr_isym]
    return kpts_irr, ik_to_ikirr_isym
end

fold_kpoints(kpts::GridKpoints, symmetry::Nothing) = kpts, [(ik, 1) for ik = 1:kpts.n]


function split_kpoints(kpts::Kpoints, N)
    rngs = split_iterator(1:kpts.n, N)
    [Kpoints(length(rng), kpts.vectors[rng], kpts.weights[rng], kpts.ngrid) for rng in rngs]
end

function split_kpoints(kpts::GridKpoints, N)
    GridKpoints.(split_kpoints(Kpoints(kpts), N))
end


"""
    print_in_qe_format(kpts::AbstractKpoints, unit = :Crystal; recip_lattice = nothing, alat = nothing)
Print k points in the format of Quantum ESPRESSO input.
- `unit`: Unit of k points. Must be :Crystal, :Cartesian, or :tpiba. Default is :Crystal.
- `recip_lattice`: Reciprocal lattice vectors in Cartesian coordinates. Required if unit is :Cartesian or :tpiba.
- `alat`: Lattice constant. Required if unit is :tpiba.
"""
function print_in_qe_format(kpts :: AbstractKpoints, unit = :Crystal; recip_lattice = nothing, alat = nothing)
    if unit === :Crystal
        println(" $(kpts.n) crystal")
        for xk in kpts.vectors
            @printf "%12.8f %12.8f %12.8f 1.0\n" xk[1] xk[2] xk[3]
        end

    elseif unit === :Cartesian
        recip_lattice === nothing && error("recip_lattice must be given for Cartesian unit")
        println(" $(kpts.n) cartesian")
        for xk in kpts.vectors
            xk = recip_lattice * xk
            @printf "%12.8f %12.8f %12.8f 1.0\n" xk[1] xk[2] xk[3]
        end

    elseif unit === :tpiba
        recip_lattice === nothing && error("recip_lattice must be given for tpiba unit")
        alat === nothing && error("alat must be given for tpiba unit")
        println(" $(kpts.n) tpiba")
        for xk in kpts.vectors
            xk = recip_lattice * xk / (2π / alat)
            @printf "%12.8f %12.8f %12.8f 1.0\n" xk[1] xk[2] xk[3]
        end

    else
        error("Wrong unit $unit. Must :Crystal or :Cartesian or :tpiba")
    end
end
