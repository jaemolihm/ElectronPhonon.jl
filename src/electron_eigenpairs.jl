# Full-band electron eigenpairs over a k-point set, computed once and shared by several e-ph runs.

using ChunkSplitters
using Base.Threads: nthreads, @threads

export ElectronEigenpairs
export electron_eigenpairs

"""
    ElectronEigenpairs{T, KT}

Full-band electron eigenvalues `e` (`(nw, kpts.n)`) and eigenvectors `u` (`(nw, nw, kpts.n)`) on
the k-point set `kpts`. Build one with [`electron_eigenpairs`](@ref).

Its purpose is to make two or more runs over *overlapping* k-point sets use the same eigenvector
gauge at every shared k-point. The gauge lives entirely in the full-band `u`, which is independent
of any energy window, so one cache serves runs with different windows and different derived
quantities: a consumer copies `e`/`u` out per k-point and computes its own velocity, position and
occupation from them. That matters wherever a band-resolved quantity is basis-dependent -- inside a
degenerate multiplet `g2 = |g|²` is not invariant per band pair, the per-k CPU eigensolve pins the
multiplet basis with the EPW-mimicking fix of [`solve_eigen_el!`](@ref), and the batched device
eigensolve does not pin it at all.

The value is immutable and read-only; there is no mutating API. Look a k-point up by its crystal
coordinates with `_eigenpair_index`.
"""
struct ElectronEigenpairs{T, KT <: AbstractKpoints{T}}
    nw   :: Int
    kpts :: KT                      # a GridKpoints: gives the O(1) xk -> ik lookup
    e    :: Matrix{T}               # (nw, kpts.n)
    u    :: Array{Complex{T}, 3}    # (nw, nw, kpts.n)

    function ElectronEigenpairs(nw::Int, kpts::KT, e::Matrix{T},
                                u::Array{Complex{T}, 3}) where {T, KT <: AbstractKpoints{T}}
        size(e) == (nw, kpts.n) || throw(ArgumentError(
            "e must be of size ($nw, $(kpts.n)), got $(size(e))"))
        size(u) == (nw, nw, kpts.n) || throw(ArgumentError(
            "u must be of size ($nw, $nw, $(kpts.n)), got $(size(u))"))
        new{T, KT}(nw, kpts, e, u)
    end
end

nk(eig::ElectronEigenpairs) = eig.kpts.n
Base.length(eig::ElectronEigenpairs) = eig.kpts.n

function Base.show(io::IO, eig::ElectronEigenpairs{T}) where {T}
    print(io, "ElectronEigenpairs{$T}(nw = $(eig.nw), nk = $(eig.kpts.n))")
end

"""
    electron_eigenpairs(model, kpts; fourier_mode = "gridopt", backend = CPUBackend())

Compute the full-band electron eigenpairs at every k point of `kpts` and return them as an
[`ElectronEigenpairs`](@ref).

`kpts` is converted to a `GridKpoints`, which both provides the xk -> ik lookup and validates that
every point is a node of the grid. That validation is required, not a convenience: the lookup
rounds `xk` onto the grid, so an off-grid point would alias to the nearest node instead of missing.

On a GPU backend the whole set is solved in one batched eigensolve and the result is brought back
to the host, so the cache is always host-resident; `fourier_mode` is then unused (the batched
interpolator is the only one the device path has), as in `compute_electron_states`. Note also that
the batched eigensolve does not apply the degenerate-multiplet gauge fix of the per-k solve, so a
cache built on the device is self-consistent but not equal to a CPU-built one inside a multiplet.
"""
function electron_eigenpairs(model::Model{FT}, kpts; fourier_mode = "gridopt",
                             backend = CPUBackend()) where {FT}
    (; nw) = model
    gkpts = GridKpoints(kpts)
    e = zeros(FT, nw, gkpts.n)
    u = zeros(Complex{FT}, nw, nw, gkpts.n)
    if backend isa CPUBackend
        @threads for iks in chunks(gkpts.vectors; n=2nthreads())
            # Setup thread-local WannierInterpolator
            ham = get_interpolator(model.el_ham; fourier_mode)
            register_kpoints!(ham, view(gkpts.vectors, iks))
            for ik in iks
                @views get_el_eigen!(e[:, ik], u[:, :, ik], nw, ham, gkpts.vectors[ik])
            end
        end
    else
        itp_elham = get_interpolator(to_device(backend, model.el_ham);
                                     fourier_mode="batched", batch_size=gkpts.n)
        E_dev, U_dev = get_el_eigen_batched(itp_elham, gkpts.vectors)
        copyto!(e, E_dev)
        copyto!(u, U_dev)
    end
    ElectronEigenpairs(nw, gkpts, e, u)
end

# Index of `xk` in the cache, erroring on a miss. `_ik_from_hash` returns 0 for a k point the cache
# does not hold, which a consumer must never silently treat as an index.
function _eigenpair_index(eig::ElectronEigenpairs, xk)
    ik = _ik_from_hash(eig.kpts, _hash_xk(xk, eig.kpts))
    ik == 0 && throw(ArgumentError(
        "k point $xk is not in the ElectronEigenpairs cache (grid $(eig.kpts.ngrid) shifted by " *
        "$(eig.kpts.shift), $(eig.kpts.n) points). The lookup rounds xk onto that grid, so an " *
        "off-grid k point aliases to the nearest node rather than missing -- the cache was built " *
        "with every point validated as a grid node, so this xk is a node the cache does not " *
        "cover."))
    ik
end
