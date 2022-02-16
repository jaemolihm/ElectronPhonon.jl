export QMEVector
export data_ik

"""
`data` represents a block-diagonal sparse matrix `f[(m, n, k)]`, where `m` and `n` are band indices
and `k` is a k-point index. The matrix is diagonal in `k`, so there is only one `k` index.
"""
struct QMEVector{ElType, FT}
    state::QMEStates{FT}
    data::Vector{ElType}
    function QMEVector(state::QMEStates{FT}, data::AbstractVector{ElType}) where {ElType, FT}
        state.n != length(data) && throw(ArgumentError("state.n must be identical to
            length(data), but got $(state.n) and $(length(data))"))
        new{ElType, FT}(state, data)
    end
end
Base.eltype(::Type{QMEVector{ElType, FT}}) where {ElType, FT} = ElType
QMEVector(state, ::Type{ElType}) where ElType = QMEVector(state, zeros(ElType, state.n))

Base.size(A::QMEVector) = A.state.n
Base.getindex(A::QMEVector, i::Int) = A.data[i]
Base.setindex!(A::QMEVector, v, i::Int) = (A.data[i] = v)
Base.IndexStyle(::Type{QMEVector}) = IndexLinear()
Base.similar(A::QMEVector) = QMEVector(A.state, similar(A.data))
Base.similar(A::QMEVector, ::Type{S}) where S = QMEVector(A.state, similar(A.data, S))

Base.:*(A::AbstractMatrix, v::QMEVector) = QMEVector(v.state, A * v.data)
Base.:\(A::AbstractMatrix, v::QMEVector) = QMEVector(v.state, A \ v.data)

function check_state_identity(x::QMEVector, y::QMEVector)
    if x.state !== y.state
        throw(DomainError("Two QMEVector.states are not identical by reference."))
    end
end

function Base.:*(x::QMEVector, y::QMEVector)
    check_state_identity(x, y)
    state = x.state
    indmap = EPW.states_index_map(state)
    z = zeros(typeof(x.data[1] * y.data[1]), length(x.data))
    for ind_z in 1:state.n
        m = state.ib1[ind_z]
        n = state.ib2[ind_z]
        ik = state.ik[ind_z]
        for p in state.ib_rng[ik]
            ind_x = get(indmap, EPW.CI(m, p, ik), -1)
            ind_x == -1 && continue
            ind_y = get(indmap, EPW.CI(p, n, ik), -1)
            ind_y == -1 && continue
            z[ind_z] += x.data[ind_x] * y.data[ind_y]
        end
    end
    QMEVector(state, z)
end

function Base.:+(x::QMEVector, y::QMEVector)
    check_state_identity(x, y)
    QMEVector(x.state, x.data .+ y.data)
end

function Base.:-(x::QMEVector, y::QMEVector)
    check_state_identity(x, y)
    QMEVector(x.state, x.data .- y.data)
end

Base.:+(a::Number, x::QMEVector) = QMEVector(x.state, x.data .+ a)
Base.:-(a::Number, x::QMEVector) = QMEVector(x.state, x.data .- a)
Base.:*(a::Number, x::QMEVector) = QMEVector(x.state, x.data .* a)
Base.:/(a::Number, x::QMEVector) = QMEVector(x.state, x.data ./ a)
Base.:+(x::QMEVector, a::Number) = QMEVector(x.state, x.data .+ a)
Base.:-(x::QMEVector, a::Number) = QMEVector(x.state, x.data .- a)
Base.:*(x::QMEVector, a::Number) = QMEVector(x.state, x.data .* a)
Base.:/(x::QMEVector, a::Number) = QMEVector(x.state, x.data ./ a)

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
    σ = mapreduce(i -> state.kpts.weights[state.ik[i]] .* real.(δρ.data[i] * state.v[i]'), +, 1:state.n)
    return σ * params.spin_degeneracy / params.volume
end



