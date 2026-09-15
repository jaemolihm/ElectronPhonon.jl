# Full-band electron eigenpairs over a k-point set, computed once and shared by several e-ph
# runs to ensure eigenvector gauge consistency.

using ChunkSplitters
using Base.Threads: nthreads, @threads

export ElectronEigenpairs
export electron_eigenpairs

"""
    ElectronEigenpairs{T, MT, AT}

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

`e` and `u` are `AbstractArray`s that live wherever the backend that built them put them, as the
`op_r` of a `WannierObject` does: a cache built with `backend = gpu_backend()` stays on the device
and is consumed there without a round trip. The runs that must agree on a gauge are runs on the
*same* backend, so moving the arrays to the host would only mean uploading them again per run;
a consumer on a different backend than the cache is an error, not a silent copy. `kpts`, and the
`xk -> ik` lookup over it, always stay on the host.

The size to keep in mind is `u`: `16 * nw^2 * nk` bytes, resident for the cache's whole lifetime.
That is modest for a windowed selection and large for a dense full-BZ grid.

The value is immutable and read-only; there is no mutating API. Look a k-point up by its crystal
coordinates with `_eigenpair_index`.
"""
struct ElectronEigenpairs{T, MT <: AbstractMatrix{T}, AT <: AbstractArray{Complex{T}, 3}}
    nw   :: Int
    # `GridKpoints` rather than `AbstractKpoints`: the xk -> ik lookup is defined only for a grid
    # (`_hash_xk`), so requiring one here is what makes every cache lookupable.
    kpts :: GridKpoints{T}
    e    :: MT                      # (nw, kpts.n)
    u    :: AT                      # (nw, nw, kpts.n)

    function ElectronEigenpairs(nw::Int, kpts::GridKpoints{T}, e::MT, u::AT) where
            {T, MT <: AbstractMatrix{T}, AT <: AbstractArray{Complex{T}, 3}}
        size(e) == (nw, kpts.n) || throw(ArgumentError(
            "e must be of size ($nw, $(kpts.n)), got $(size(e))"))
        size(u) == (nw, nw, kpts.n) || throw(ArgumentError(
            "u must be of size ($nw, $nw, $(kpts.n)), got $(size(u))"))
        new{T, MT, AT}(nw, kpts, e, u)
    end
end

function Base.show(io::IO, eig::ElectronEigenpairs{T}) where {T}
    print(io, "ElectronEigenpairs{$T}(nw = $(eig.nw), nk = $(eig.kpts.n), e::$(typeof(eig.e)))")
end

"""
    electron_eigenpairs(model, kpts; fourier_mode = "gridopt", backend = CPUBackend())

Compute the full-band electron eigenpairs at every k point of `kpts` and return them as an
[`ElectronEigenpairs`](@ref).

`kpts` is converted to a `GridKpoints`, which provides the xk -> ik lookup. A `Kpoints` argument is
validated against its own `ngrid` on the way in; a `GridKpoints` argument is taken as already being
on the grid it carries. Either way the lookup then rounds a query onto that grid, so
`_eigenpair_index` re-checks every query point (an off-grid one would otherwise alias to the
nearest node instead of missing).

On a GPU backend the whole set is solved in one batched eigensolve and `e`/`u` are left on the
device, so the cache is resident on `backend` and a consumer must run on that same backend.
`fourier_mode` is then unused (the batched interpolator is the only one the device path has), as in
`compute_electron_states`. Two caveats inherited from that path: the batched eigensolve does not
apply the degenerate-multiplet gauge fix of the per-k solve, so a device-built cache is
self-consistent but differs from a CPU-built one inside a multiplet; and the whole set is one batch,
so the device H(k)/U stacks (nw^2 * nk) are unbounded and a very large k-grid can OOM.
"""
function electron_eigenpairs(model::Model{FT}, kpts; fourier_mode = "gridopt",
                             backend = CPUBackend()) where {FT}
    (; nw) = model
    gkpts = GridKpoints(kpts)
    if backend isa CPUBackend
        e = zeros(FT, nw, gkpts.n)
        u = zeros(Complex{FT}, nw, nw, gkpts.n)
        @threads for iks in chunks(gkpts.vectors; n=2nthreads())
            @views begin
                # Setup thread-local WannierInterpolator
                ham = get_interpolator(model.el_ham; fourier_mode)
                register_kpoints!(ham, gkpts.vectors[iks])
                for ik in iks
                    get_el_eigen!(e[:, ik], u[:, :, ik], nw, ham, gkpts.vectors[ik])
                end
            end
        end
        ElectronEigenpairs(nw, gkpts, e, u)
    else
        itp_elham = get_interpolator(to_device(backend, model.el_ham);
                                     fourier_mode="batched", batch_size=gkpts.n)
        E_dev, U_dev = get_el_eigen_batched(itp_elham, gkpts.vectors)
        ElectronEigenpairs(nw, gkpts, E_dev, U_dev)
    end
end

# Index of `xk` in the cache, erroring on a query the cache cannot answer. Both failures are loud:
# `_hash_xk` rounds `xk` onto the cache's grid, so an off-grid query would otherwise alias to the
# nearest cached node, and `_ik_from_hash` returns 0 for a node the cache does not hold -- neither
# may reach a consumer as an index.
function _eigenpair_index(eig::ElectronEigenpairs{T}, xk) where {T}
    (; kpts) = eig
    nxk = (xk - kpts.shift) .* kpts.ngrid
    isapprox(round.(Int, nxk), nxk; atol = sqrt(eps(T))) || throw(ArgumentError(
        "k point $xk is not on the ElectronEigenpairs cache's grid of size $(kpts.ngrid) shifted " *
        "by $(kpts.shift)"))
    ik = _ik_from_hash(kpts, _hash_xk(xk, kpts))
    ik == 0 && throw(ArgumentError(
        "k point $xk is a node of the ElectronEigenpairs cache's grid ($(kpts.ngrid) shifted by " *
        "$(kpts.shift)) but is not one of its $(kpts.n) points"))
    ik
end
