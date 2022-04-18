export QMEVector
export data_ik

# TODO: Use AbstractVector? Broadcasting?

"""
`data` represents a block-diagonal sparse matrix `f[(m, n, k)]`, where `m` and `n` are band indices
and `k` is a k-point index. The matrix is diagonal in `k`, so there is only one `k` index.
"""
struct QMEVector{ElType, FT, VT <: AbstractVector{ElType}} <: AbstractVector{ElType}
    state::QMEStates{FT}
    data::VT
    function QMEVector(state::QMEStates{FT}, data::VT) where {FT, ElType, VT <: AbstractVector{ElType}}
        state.n == length(data) || throw(ArgumentError(
            "state.n must be identical to length(data), but got $(state.n) and $(length(data))"))
        axes(data, 1) == 1:state.n || throw(ArgumentError(
            "axes(data, 1) must be identical to 1:state.n, but got $(axes(data, 1))"))
        new{ElType, FT, VT}(state, data)
    end
end
Base.eltype(::Type{QMEVector{ElType, FT, VT}}) where {ElType, FT, VT} = ElType
QMEVector(state, ::Type{ElType}) where ElType = QMEVector(state, zeros(ElType, state.n))

Base.size(A::QMEVector) = (A.state.n,)
Base.getindex(A::QMEVector, i::Int) = A.data[i]
Base.setindex!(A::QMEVector, v, i::Int) = (A.data[i] = v)
Base.IndexStyle(::Type{QMEVector}) = IndexLinear()
Base.similar(A::QMEVector, ::Type{S}) where S = QMEVector(A.state, similar(A.data, S))

Base.copy(x::QMEVector) = QMEVector(x.state, copy(x.data))

Base.:*(A::AbstractMatrix, v::QMEVector) = QMEVector(v.state, A * v.data)
Base.:\(A::AbstractMatrix, v::QMEVector) = QMEVector(v.state, A \ v.data)

# Broadcasting
Base.BroadcastStyle(::Type{<:QMEVector}) = Broadcast.ArrayStyle{QMEVector}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{QMEVector}}, ::Type{ElType}) where ElType
    A = find_entry(bc)
    QMEVector(A.state, similar(Array{ElType}, axes(bc)))
end

"`A = find_entry(As)` returns the first QMEVector among the arguments."
find_entry(x) = x
find_entry(x::QMEVector, rest) = x
find_entry(bc::Base.Broadcast.Broadcasted) = find_entry(bc.args)
find_entry(args::Tuple) = find_entry(find_entry(args[1]), Base.tail(args))
find_entry(::Tuple{}) = nothing
find_entry(::Any, rest) = find_entry(rest)

reinterpret_to_numeric_vector(v::QMEVector{ElType}) where ElType = reinterpret(eltype(ElType), v.data)

function check_state_identity(x::QMEVector, y::QMEVector)
    if x.state !== y.state
        throw(DomainError("Two QMEVector.states are not identical by reference."))
    end
end

function Base.:*(x::QMEVector, y::QMEVector)
    check_state_identity(x, y)
    state = x.state
    z = zeros(typeof(x.data[1] * y.data[1]), length(x.data))
    for ind_z in 1:state.n
        m = state.ib1[ind_z]
        n = state.ib2[ind_z]
        ik = state.ik[ind_z]
        for p in state.ib_rng[ik]
            ind_x = get_1d_index(state, m, p, ik)
            ind_x == 0 && continue
            ind_y = get_1d_index(state, p, n, ik)
            ind_y == 0 && continue
            z[ind_z] += x.data[ind_x] * y.data[ind_y]
        end
    end
    QMEVector(state, z)
end

"""
    data_ik(x::QMEVector{ElType, FT}, ik) where {ElType, FT}
Return the block corresponding to the k point `ik` of the data stored in `x.data` as a matrix.
"""
function data_ik(x::QMEVector{ElType, FT}, ik) where {ElType, FT}
    rng = x.state.ib_rng[ik]
    data_ik = zeros(ElType, rng.stop, rng.stop)
    for i in 1:x.state.n
        if x.state[i].ik == ik
            (; ib1, ib2) = x.state[i]
            data_ik[ib1, ib2] = x.data[i]
        end
    end
    data_ik
end

get_velocity_as_QMEVector(state::QMEStates) = Vec3(QMEVector(state, [v[a] for v in state.v]) for a in 1:3)

"""
    occupation_to_conductivity(δρ::QMEVector, params) => σ
Compute electron conductivity using the density matrix `δρ`. The last axis of `σ` is the
direction of the current, while other axes are determined by the elements of `δρ`.
Currently, works only if `eltype(δρ)` is `Number` or `SVector`.
TODO: Make it work for genearl `SArray`.
"""
function occupation_to_conductivity(δρ::QMEVector, params)
    state = δρ.state
    σ = mapreduce(i -> state.kpts.weights[state.ik[i]] .* δρ.data[i] * state.v[i]', +, 1:state.n)
    return σ * params.spin_degeneracy / params.volume
end



