
# Data and functions for k points
using MPI

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
            # Create the irreducible k points in the root. Then redistribute.
            kpoints = if mpi_isroot(mpi_comm)
                kpoints_grid_symmetry(ngrid, symmetry; ignore_time_reversal)
            else
                Kpoints{Float64}()
            end
            return mpi_scatter(kpoints, mpi_comm)
        else
            return kpoints_grid_symmetry(ngrid, symmetry; ignore_time_reversal)
        end
    elseif symmetry === nothing
        nk = prod(ngrid)
        range = mpi_comm isa MPI.Comm ? mpi_split_iterator(1:nk, mpi_comm) : 1:nk
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
    Kpoints(nk, kvecs, fill(1/nk, (nk,)), (nk1, nk2, nk3))
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

"""
    add_two_kpoint_grids(kpts, qpts, op, ngrid_kq)
For k and q in kpts and qpts, return Kpoint with `kq = op(k, q)`.
ngrid_kq: ngrid for kq points
op: function from (k, q) to kq. Only + and -.
TODO: a better name is needed. Not limited to "add"ing.
TODO: Add test.
"""
function add_two_kpoint_grids(kpts, qpts, op, ngrid_kq)
    if Symbol(op) !== :+ && Symbol(op) !== :-
        error("op must be + or -")
    end
    @assert all(kpts.ngrid .> 0)
    @assert all(qpts.ngrid .> 0)

    T = eltype(kpts.weights)
    xkqs = Vector{Vec3{T}}()
    xkq_hash_to_ikq = Dict{Int, Int}()
    shift_k = kpts.vectors[1]
    shift_q = qpts.vectors[1]
    shift_kq = op(shift_k, shift_q)

    ikq = 0
    for iq in 1:qpts.n
        xq = qpts.vectors[iq]
        xq_rational = round.(Int, (xq - shift_q) .* qpts.ngrid) .// qpts.ngrid
        for ik in 1:kpts.n
            xk = kpts.vectors[ik]
            xk_rational = round.(Int, (xk - shift_k) .* kpts.ngrid) .// kpts.ngrid
            xkq = mod.(op(xk_rational, xq_rational), 1) + shift_kq

            # Find new k+q points, append to xkq_hash_to_ikq and xkqs
            xk_hash_value = _hash_xk(xkq, ngrid_kq, shift_kq)
            if xk_hash_value ∉ keys(xkq_hash_to_ikq)
                ikq += 1
                xkq_hash_to_ikq[xk_hash_value] = ikq
                push!(xkqs, xkq)
            else
                @assert xkq ≈ xkqs[xkq_hash_to_ikq[xk_hash_value]]
            end
        end
    end
    nkq = length(xkqs)
    GridKpoints{T}(nkq, xkqs, ones(T, nkq) ./ prod(ngrid_kq), ngrid_kq, shift_kq, xkq_hash_to_ikq)
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

function GridKpoints(kpts::Kpoints{T}) where {T}
    all(kpts.ngrid .> 0) || error("kpts must be on a grid to make GridKpoints")
    if kpts.n == 0
        return GridKpoints{T}(0, Vector{Vec3{T}}(), Vector{T}(), kpts.ngrid, zero(Vec3{T}), Dict{Int,Int}())
    end
    shift = mod.(kpts.vectors[1], 1 ./ kpts.ngrid)
    _xk_hash_to_ik = Dict(_hash_xk.(kpts.vectors, Ref(kpts.ngrid), Ref(shift)) .=> 1:kpts.n)
    GridKpoints{T}(kpts.n, kpts.vectors, kpts.weights, kpts.ngrid, shift, _xk_hash_to_ik)
end

GridKpoints(xk::Vec3{T}) where {T <: Real} = GridKpoints(Kpoints(xk))

# Reduce GridKpoints to Kpoints
Kpoints(k::GridKpoints{T}) where {T} = Kpoints{T}(k.n, k.vectors, k.weights, k.ngrid)
GridKpoints(k::GridKpoints) = k

function _hash_xk(xk, ngrid, shift)
    # xk_int = mod.(round.(Int, (xk - shift) .* ngrid), ngrid)
    # FIXME: round.(Int, x) allocates, so I use the following temporary fix
    xk_int = Vec3(mod.(round.(Int, (xk - shift).data .* ngrid), ngrid))
    (xk_int[1] * ngrid[2] + xk_int[2]) * ngrid[3] + xk_int[3]
end
_hash_xk(xk, kpts::GridKpoints) = _hash_xk(xk, kpts.ngrid, kpts.shift)

# Retern index of given xk vector
xk_to_ik(xk, kpts) = get(kpts._xk_hash_to_ik, _hash_xk(xk, kpts), nothing)

Base.sortperm(k::GridKpoints) = sortperm(map(xk -> round.(Int, (xk - k.shift).data .* k.ngrid), k.vectors))

function Base.sort!(k::GridKpoints)
    inds = sortperm(k)
    k.vectors .= k.vectors[inds]
    k.weights .= k.weights[inds]
    for (ik, xk) in enumerate(k.vectors)
        k._xk_hash_to_ik[_hash_xk(xk, k)] = ik
    end
    k
end

Base.:(==)(k1::GridKpoints, k2::GridKpoints) = (k1.n ≈ k2.n
    && k1.vectors ≈ k2.vectors
    && k1.weights ≈ k2.weights
    && k1.shift ≈ k2.shift
    && k1.ngrid == k2.ngrid
    && k1._xk_hash_to_ik == k2._xk_hash_to_ik
)

get_filtered_kpoints(k::GridKpoints, ik_keep) = GridKpoints(get_filtered_kpoints(Kpoints(k), ik_keep))
kpoints_create_subgrid(k::GridKpoints, nsubgrid) = GridKpoints(kpoints_create_subgrid(Kpoints(k), nsubgrid))

"""
    unfold_kpoints(kpts::GridKpoints, symmetry) => kpts_unfold, ik_to_ikirr_isym

Unfold k points using symmetry to the full Brillouin zone.
Output `ik_to_ikirr_isym` gives a map ik => (ikirr, isym) such that ``xk[ik] = S[isym](xkirr[ikirr])``.
"""
function unfold_kpoints(kpts::GridKpoints, symmetry)
    # If symmetry is trivial, do nothing and return a copy of input kpts
    if symmetry.nsym == 1
        return deepcopy(kpts)
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

    kpts_unfold = GridKpoints(length(sk_vectors), sk_vectors, sk_weights, ngrid, shift, sk_hash_dict)
    ik_to_ikirr_isym = ik_to_ikirr_isym[sortperm(kpts_unfold)]
    sort!(kpts_unfold)

    return kpts_unfold, ik_to_ikirr_isym
end

"""
    fold_kpoints(kpts, symmetry) => kpts_irr, ik_to_ikirr_isym
Inverse of `unfold_kpoints`. Reduce `kpts` to the irreducible BZ using `symmetry`.
Output ik_to_ikirr_isym gives a map ik => (ikirr, isym) such that ``xk[ik] = S[isym](xkirr[ikirr])``.
"""
function fold_kpoints(kpts::GridKpoints, symmetry)
    if symmetry.nsym == 1
        return deepcopy(kpts), [(ik, 1) for ik = 1:kpts.n]
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

    kpts_irr = GridKpoints(length(vectors_irr), vectors_irr, weights_irr, ngrid, shift, hash_dict_irr)
    sort!(kpts_irr)
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
