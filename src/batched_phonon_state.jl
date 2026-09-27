# Phonon eigenvalues, eigenvectors and derived quantities over a q-point set, as dense stacks

export BatchedPhononState

"""
    BatchedPhononState{T, KT, ET, UT, VT, DT, RT}

The phonons of a q-point set `qpts` as dense stacks, the struct-of-arrays counterpart of a
`Vector{PhononState}`: field `x[..., iq]` is `PhononState.x` at `qpts.vectors[iq]`, which is also
its `xq`. Build one with [`compute_phonon_states_batched`](@ref). The arrays live on the backend
that built them and `qpts` stays on the host, as in [`Eigenpairs`](@ref).

Index legend: `ν` mode, `a` displacement (atom × Cartesian), `d` Cartesian direction, `q` q point.

| field | shape | |
|---|---|---|
| `e` | `(ν, q)` | frequency ω |
| `u` | `(a, ν, q)` | mass-scaled eigenmodes; zero-filled (full size) for `["eigenvalue"]` |
| `vdiag` | `(d, ν, q)` | diagonal velocity; zero-length unless requested, see `has_vdiag` |
| `eph_dipole_coeff` | `(ν, q)` | dipole e-ph coefficients; zero-length unless requested, see `has_dipole` |
| `eph_r_coeff` | `(ν, d, q)` | as `eph_dipole_coeff` |
"""
struct BatchedPhononState{T, KT <: AbstractKpoints{T}, ET <: AbstractMatrix{T},
        UT <: AbstractArray{Complex{T}, 3}, VT <: AbstractArray{T, 3},
        DT <: AbstractMatrix{Complex{T}}, RT <: AbstractArray{Complex{T}, 3}}
    nmodes :: Int
    qpts :: KT               # host
    e :: ET                  # (ν, q)
    u :: UT                  # (a, ν, q)
    vdiag :: VT              # (d, ν, q), or zero-length
    eph_dipole_coeff :: DT   # (ν, q), or zero-length
    eph_r_coeff :: RT        # (ν, d, q), or zero-length

    function BatchedPhononState(nmodes::Int, qpts::KT, e::ET, u::UT, vdiag::VT,
            eph_dipole_coeff::DT, eph_r_coeff::RT) where {T, KT <: AbstractKpoints{T},
            ET <: AbstractMatrix{T}, UT <: AbstractArray{Complex{T}, 3}, VT <: AbstractArray{T, 3},
            DT <: AbstractMatrix{Complex{T}}, RT <: AbstractArray{Complex{T}, 3}}
        nq = qpts.n
        for (name, x, dims, optional) in (("e", e, (nmodes, nq), false),
                ("u", u, (nmodes, nmodes, nq), false), ("vdiag", vdiag, (3, nmodes, nq), true),
                ("eph_dipole_coeff", eph_dipole_coeff, (nmodes, nq), true),
                ("eph_r_coeff", eph_r_coeff, (nmodes, 3, nq), true))
            (size(x) == dims || (optional && isempty(x))) || throw(ArgumentError(
                "$name must be of size $dims$(optional ? " or empty" : ""), got $(size(x))"))
        end
        isempty(eph_dipole_coeff) == isempty(eph_r_coeff) || throw(ArgumentError(
            "eph_dipole_coeff and eph_r_coeff must be both present or both empty"))
        new{T, KT, ET, UT, VT, DT, RT}(nmodes, qpts, e, u, vdiag, eph_dipole_coeff, eph_r_coeff)
    end
end

# A `BatchedPhononState` with every stack in host memory.
const HostBatchedPhononState{T, KT} = BatchedPhononState{T, KT, Matrix{T}, Array{Complex{T}, 3},
    Array{T, 3}, Matrix{Complex{T}}, Array{Complex{T}, 3}}

has_vdiag(b::BatchedPhononState) = !isempty(b.vdiag)
has_dipole(b::BatchedPhononState) = !isempty(b.eph_dipole_coeff)

function Base.show(io::IO, b::BatchedPhononState{T}) where {T}
    print(io, "BatchedPhononState{$T}(nmodes = $(b.nmodes), nq = $(b.qpts.n), " *
              "has_vdiag = $(has_vdiag(b)), has_dipole = $(has_dipole(b)), e::$(typeof(b.e)))")
end
