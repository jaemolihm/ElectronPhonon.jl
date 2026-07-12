
# Functions related to finding Fermi energy

using Roots
using ChunkSplitters
using Base.Threads: @threads

export find_fermi_energy

# Threaded deterministic reduction over the state index used by the ncarrier sums below. The
# chunking is FIXED (size-based, depends only on `n`), each chunk is summed serially, and the
# per-chunk partials are summed serially in chunk order — so the result is run-to-run
# deterministic (independent of thread scheduling and of `nthreads()`). The summation ORDER
# differs from the plain serial loop, so results may differ from it at the floating-point
# rounding level (~1e-15 relative); this shifts the bisected chemical potential by a
# comparable amount. Small problems stay on the serial path (threading overhead dominates).
# `f(i) -> FT` is the per-state summand.
const _NCARRIER_SERIAL_MAX = 1 << 14
const _NCARRIER_CHUNK = 1 << 12

function _ncarrier_sum(f, n::Int, ::Type{FT}) where {FT}
    if n <= _NCARRIER_SERIAL_MAX
        acc = zero(FT)
        for i in 1:n
            acc += f(i)
        end
        return acc
    end
    ck = chunks(1:n; size = _NCARRIER_CHUNK)
    partials = zeros(FT, length(ck))
    @threads for (ic, is) in enumerate(ck)
        acc = zero(FT)
        for i in is
            acc += f(i)
        end
        partials[ic] = acc
    end
    sum(partials)
end

"""
    compute_ncarrier(μ, T, energy, weights; occ_type = :FermiDirac)
Compute number of electrons per unit cell.
- `μ`: Chemical potential.
- `energy`: band energy
- `T`: temperature
- `occ_type`: occupation type, default: `:FermiDirac`. (See [`occ_fermion`](@ref) for all options.)
- `weights`: k-point weights.

Threaded over the states with a deterministic chunked reduction (see `_ncarrier_sum`).
"""
function compute_ncarrier(μ, T, energy::AbstractMatrix, weights; occ_type = :FermiDirac)
    nband, nk = size(energy)
    @assert length(weights) == nk
    _ncarrier_sum(nband * nk, eltype(energy)) do i
        iband = (i - 1) % nband + 1
        ik = (i - 1) ÷ nband + 1
        weights[ik] * occ_fermion(energy[iband, ik] - μ, T; occ_type)
    end
end

function compute_ncarrier(μ, T, energy::Vector, weights; occ_type = :FermiDirac)
    _ncarrier_sum(length(energy), eltype(energy)) do i
        weights[i] * occ_fermion(energy[i] - μ, T; occ_type)
    end
end

"""
    compute_ncarrier(μ, T, energy::AbstractVector, weights; occ_type = :FermiDirac)

Generic (device-capable) method: a plain broadcast + `sum`, so it runs on any array backend
(e.g. `CuArray`) without scalar indexing or a CUDA dependency. `:FermiDirac` only — the other
occupation types need `erf`/`erfc` (SpecialFunctions), which are not device-kernel-safe; callers
keep those on the host. The `Vector` method above (the deterministic chunked-threads host path)
takes precedence for host arrays and is unchanged. The broadcast `sum` reduction order differs
from the host method's, so results agree with it only to the rounding level (~1e-15 relative).
"""
function compute_ncarrier(μ, T, energy::AbstractVector, weights; occ_type = :FermiDirac)
    occ_type === :FermiDirac || throw(ArgumentError(
        "the generic (device) compute_ncarrier supports occ_type = :FermiDirac only (got $occ_type)"))
    if T > sqrt(eps(Float64))
        sum(weights ./ (exp.((energy .- μ) ./ T) .+ 1))
    elseif T >= 0
        sum(weights .* ((1 .- sign.(energy .- μ)) ./ 2))
    else
        throw(ArgumentError("Temperature must be positive"))
    end
end

"""
    compute_ncarrier_hole(μ, T, energy::AbstractVector, weights; occ_type = :FermiDirac)
Compute number of holes per unit cell. See [`compute_ncarrier`](@ref) for arguments.
"""
function compute_ncarrier_hole(μ, T, energy::AbstractVector, weights; occ_type = :FermiDirac)
    _ncarrier_sum(length(energy), eltype(energy)) do i
        weights[i] * (1 - occ_fermion(energy[i] - μ, T; occ_type))
    end
end

"""
    find_chemical_potential(ncarrier, T, energy, weights; occ_type = :FermiDirac)
Find chemical potential for target carrier density using bisection.
- `ncarrier`: target number of carriers (electrons or holes) per unit cell)
- `T`: temperature
- `energy`: band energy
- `weights`: k-point weights.
- `occ_type`: occupation type, default: `:FermiDirac`. (See [`occ_fermion`](@ref) for all options.)
"""
function find_chemical_potential(ncarrier, T, energy, weights; occ_type = :FermiDirac)
    # FIXME: T=0 case
    # TODO: MPI

    # Solve func(μ) = ncarrier(μ) - ncarrier = 0
    func(μ) = compute_ncarrier(μ, T, energy, weights; occ_type) - ncarrier
    Roots.bisection(func, -Inf, Inf)
end

"""
    find_chemical_potential_semiconductor(ncarrier, T, energy_e, energy_h, weights_e, weights_h; occ_type = :FermiDirac)
Find chemical potential for target carrier density using bisection. Minimize floating point
error by computing doped carrier density, not the total carrier density.
- `ncarrier`: target number of electrons per unit cell. `ncarrier > 0` for electron doping, `ncarrier < 0` for hole doping.
See [`compute_ncarrier`](@ref) for other arguments.
"""
function find_chemical_potential_semiconductor(ncarrier, T, energy_e, energy_h, weights_e, weights_h; occ_type = :FermiDirac)
    # FIXME: T=0 case
    # TODO: MPI

    # Solve func(μ) = ncarrier_electron(μ) - ncarrier_hole(μ) - ncarrier = 0
    func(μ) = (  compute_ncarrier(μ, T, energy_e, weights_e; occ_type)
               - compute_ncarrier_hole(μ, T, energy_h, weights_h; occ_type) - ncarrier)
    Roots.bisection(func, -Inf, Inf)
end
