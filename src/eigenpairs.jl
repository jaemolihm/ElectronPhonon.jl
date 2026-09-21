# Full-band eigenpairs over a k-point set, computed once and shared by several e-ph runs to ensure
# eigenvector gauge consistency. The container is species-agnostic (`nbasis` is `nw` for electrons
# and `nmodes` for phonons); the builder below is the electron one, since a phonon cache is
# assembled by its caller from the states of an earlier run rather than solved for.

using ChunkSplitters
using Base.Threads: nthreads, @threads

export Eigenpairs
export electron_eigenpairs

"""
    Eigenpairs{T, MT, AT}

Full-band eigenvalues `e_full` (`(nbasis, kpts.n)`) and eigenvectors `u_full`
(`(nbasis, nbasis, kpts.n)`) on the k-point set `kpts`. Build one with
[`electron_eigenpairs`](@ref).

`nbasis` is the dimension of the eigenproblem, so nothing here is electron-specific: it is `nw`
for electrons and `nmodes` for a phonon cache, whose `(e, u)` per q point have exactly this shape.
Only the builders are per species.

Its purpose is to make two or more runs over *overlapping* k-point sets use the same eigenvector
gauge at every shared k-point. The gauge lives entirely in `u_full`, which is independent of any
energy window, so one cache serves runs with different windows and different derived quantities: a
consumer copies `e_full`/`u_full` out per k-point and computes its own velocity, position and
occupation from them. That matters wherever a band-resolved quantity is basis-dependent -- inside a
degenerate multiplet `g2 = |g|²` is not invariant per band pair, the per-k CPU eigensolve pins the
multiplet basis with the EPW-mimicking fix of [`solve_eigen_el!`](@ref), and the batched device
eigensolve does not pin it at all.

`e_full` and `u_full` are `AbstractArray`s that live wherever the backend that built them put
them, as the `op_r` of a `WannierObject` does: a cache built with `backend = gpu_backend()` stays
on the device and is consumed there without a round trip. The runs that must agree on a gauge are
runs on the *same* backend, so moving the arrays to the host would only mean uploading them again
per run; a consumer on a different backend than the cache is an error, not a silent copy. `kpts`,
and the `xk -> ik` lookup over it, always stay on the host.

The size to keep in mind is `u_full`: `16 * nbasis^2 * nk` bytes, resident for the cache's whole
lifetime. That is modest for a windowed selection and large for a dense full-BZ grid. In the future
the filtering can be combined with the cache, storing only the subset of the eigenvectors that the
runs actually need.

The value is immutable and read-only; there is no mutating API. Look a k-point up by its crystal
coordinates with the checked [`xk_to_ik`](@ref) on `kpts`, which errors rather than aliasing an
off-grid query onto a neighbouring cached node.
"""
struct Eigenpairs{T, MT <: AbstractMatrix{T}, AT <: AbstractArray{Complex{T}, 3}}
    nbasis :: Int                   # nw for electrons, nmodes for phonons
    # `GridKpoints` rather than `AbstractKpoints`: the xk -> ik lookup (`xk_to_ik`) is defined only
    # for a grid, so requiring one here is what makes every cache lookupable.
    kpts   :: GridKpoints{T}
    e_full :: MT                    # (nbasis, kpts.n)
    u_full :: AT                    # (nbasis, nbasis, kpts.n)

    function Eigenpairs(nbasis::Int, kpts::GridKpoints{T}, e_full::MT, u_full::AT) where
            {T, MT <: AbstractMatrix{T}, AT <: AbstractArray{Complex{T}, 3}}
        size(e_full) == (nbasis, kpts.n) || throw(ArgumentError(
            "e_full must be of size ($nbasis, $(kpts.n)), got $(size(e_full))"))
        size(u_full) == (nbasis, nbasis, kpts.n) || throw(ArgumentError(
            "u_full must be of size ($nbasis, $nbasis, $(kpts.n)), got $(size(u_full))"))
        new{T, MT, AT}(nbasis, kpts, e_full, u_full)
    end
end

function Base.show(io::IO, eig::Eigenpairs{T}) where {T}
    print(io, "Eigenpairs{$T}(nbasis = $(eig.nbasis), nk = $(eig.kpts.n), " *
              "e_full::$(typeof(eig.e_full)))")
end

# Why this solves H(k) itself instead of calling `compute_electron_states`, which also solves it
# over a k-point set: that function returns one `ElectronState` per k point, whose `e_full`/`u_full`
# are host-concrete `Vector`/`Matrix`, and its device worker has no device-returning exit
# (`_scatter_electron_states!` takes `Array(E_dev)`/`Array(U_dev)`). Delegating would therefore give
# up the backend residency this type exists to provide -- on a 4-band model at nk = 32768, 2.0x the
# runtime and 10.6x the allocations on the CPU, and a 14.4x host round trip on the device. It also
# interleaves the energy window and the derived quantities into the same thread-local pass, which a
# full-band cache has no use for. `compute_eigenvalues_el` (`compute_eigenvalues.jl`) is the
# precedent for the dense-gather twin, for eigenvalues only.
"""
    electron_eigenpairs(model, kpts; fourier_mode = "gridopt", backend = CPUBackend())

Compute the full-band electron eigenpairs at every k point of `kpts` and return them as an
[`Eigenpairs`](@ref).

`kpts` is converted to a `GridKpoints`, which provides the xk -> ik lookup. A `Kpoints` argument is
validated against its own `ngrid` on the way in; a `GridKpoints` argument is taken as already being
on the grid it carries. Look points up with the checked [`xk_to_ik`](@ref), which validates each
query against that grid; the unchecked `xk_to_ik_unsafe` would alias an off-grid query onto the
nearest cached node instead of missing.

On a GPU backend the whole set is solved in one batched eigensolve and `e_full`/`u_full` are left
on the device, so the cache is resident on `backend` and a consumer must run on that same backend.
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
        e_full = zeros(FT, nw, gkpts.n)
        u_full = zeros(Complex{FT}, nw, nw, gkpts.n)
        @threads for iks in chunks(gkpts.vectors; n=2nthreads())
            @views begin
                # Setup thread-local WannierInterpolator
                ham = get_interpolator(model.el_ham; fourier_mode)
                register_kpoints!(ham, gkpts.vectors[iks])
                for ik in iks
                    get_el_eigen!(e_full[:, ik], u_full[:, :, ik], nw, ham, gkpts.vectors[ik])
                end
            end
        end
        Eigenpairs(nw, gkpts, e_full, u_full)
    else
        itp_elham = get_interpolator(to_device(backend, model.el_ham);
                                     fourier_mode="batched", backend, nk_hint=gkpts.n)
        E_dev, U_dev = get_el_eigen_batched(itp_elham, gkpts.vectors)
        Eigenpairs(nw, gkpts, E_dev, U_dev)
    end
end

# Guards on a caller-supplied cache, checked once per consuming call rather than per k point. Both
# mismatches would otherwise surface badly. A wrong `nbasis` is a `DimensionMismatch` inside the
# per-k copy for every value but one: `nbasis == 1` against a multi-band model *broadcasts* the
# single value into every band and returns wrong numbers silently. A wrong backend is a mixed
# host/device operation -- or, on the eigenvalue-only device path, no error at all.
_check_eigenpairs(::Nothing, nbasis, backend) = nothing

function _check_eigenpairs(eig::Eigenpairs, nbasis, backend)
    eig.nbasis == nbasis || throw(ArgumentError(
        "eigenpairs holds nbasis = $(eig.nbasis), but this run needs nbasis = $nbasis " *
        "(nw for an electron cache, nmodes for a phonon one)"))
    # A cache is resident on the backend that built it, and this run's arrays live on `backend`.
    # Consuming one from the other side would mean moving it per call, or indexing a device array
    # from the host loop, so require a match.
    check_on_backend(backend, eig.e_full, "eigenpairs")
    nothing
end

# Index of `xk` in the cache. Distinct from `xk_to_ik` only in how it phrases a miss: a k point the
# cache does not hold means the cache was built over the wrong k-point set, which is a setup error
# rather than a lookup that legitimately answers `nothing`.
function _eigenpairs_ik(eigenpairs::Eigenpairs, xk)
    ik = xk_to_ik(xk, eigenpairs.kpts)
    ik === nothing && throw(ArgumentError("eigenpairs does not cover k point $xk"))
    ik
end

# The consumers, one method pair per quantity a caller can take from the cache instead of solving
# H(k) itself: `::Nothing` solves (exactly what the call site did before a cache existed) and
# `::Eigenpairs` copies out. The cache therefore arrives as a typed argument, each call site
# specializes on one of the two, and no loop gains a branch. They live here, next to the type,
# rather than beside their callers in compute_states.jl and filter.jl, so that the lookup, the miss
# error and the copy are written once.

# The full eigenpair of one k point, into an `ElectronState`.
_set_eigen_from!(el::ElectronState, ::Nothing, ham, xk) = set_eigen!(el, ham, xk)

function _set_eigen_from!(el::ElectronState, eigenpairs::Eigenpairs, ham, xk)
    ik = _eigenpairs_ik(eigenpairs, xk)
    el.xk = xk
    @views el.e_full .= eigenpairs.e_full[:, ik]
    @views el.u_full .= eigenpairs.u_full[:, :, ik]

    # Reset window to a dummy value
    el.nband = 0
    el.rng = 1:0
end

_set_eigen_valueonly_from!(el::ElectronState, ::Nothing, ham, xk) =
    set_eigen_valueonly!(el, ham, xk)

function _set_eigen_valueonly_from!(el::ElectronState, eigenpairs::Eigenpairs, ham, xk)
    ik = _eigenpairs_ik(eigenpairs, xk)
    el.xk = xk
    @views el.e_full .= eigenpairs.e_full[:, ik]

    # Reset window to a dummy value
    el.nband = 0
    el.rng = 1:0
    el
end

# The full-band eigenvalues of one k point, into `eigenvalues`.
_set_eigenvalues_from!(eigenvalues, nw, ::Nothing, ham, xk) =
    get_el_eigen_valueonly!(eigenvalues, nw, ham, xk)

function _set_eigenvalues_from!(eigenvalues, nw, eigenpairs::Eigenpairs, ham, xk)
    ik = _eigenpairs_ik(eigenpairs, xk)
    @views eigenvalues .= eigenpairs.e_full[:, ik]
end

# The same, batched over a chunk of k points and returned on the host for the window test. The
# device arm gathers off a cache that is resident on that same device.
_eigenvalues_on_host(::Nothing, itp_elham, xks) =
    Array(get_el_eigen_valueonly_batched(itp_elham, xks))

function _eigenvalues_on_host(eigenpairs::Eigenpairs, itp_elham, xks)
    iks = map(xk -> _eigenpairs_ik(eigenpairs, xk), xks)
    Array(eigenpairs.e_full[:, iks])
end

# The full eigenpair of one q point, into a `PhononState`. `e` is the frequency ω and `u` the
# mass-scaled eigenmode, i.e. what `get_ph_eigen!` leaves in a `PhononState` -- a phonon cache is
# built from a previous run's states, so neither the sign(ω²)√|ω²| nor the 1/√mass step is redone
# here. Argument order follows the electron pair above, `(state, cache, what it takes to solve,
# momentum)`; the solver arguments differ because `D(q)` needs the masses and the dipole term
# where `H(k)` needs only its interpolator.
_set_eigen_from!(ph::PhononState, ::Nothing, dyn, mass, polar, xq) =
    set_eigen!(ph, dyn, mass, polar, xq)

function _set_eigen_from!(ph::PhononState, eigenpairs::Eigenpairs, dyn, mass, polar, xq)
    iq = _eigenpairs_ik(eigenpairs, xq)
    ph.xq = xq
    @views ph.e .= eigenpairs.e_full[:, iq]
    @views ph.u .= eigenpairs.u_full[:, :, iq]
    ph
end

# Unlike every other consumer here this one is not inert: `e` comes from the cache's FULL
# eigensolve where the cacheless path runs the value-only driver. The two agree on ω² to
# `eps * ‖D(q)‖`, but ω = sign(ω²)√|ω²| turns that into a `1/(2ω)`-amplified deviation, so a mode
# whose ω² is far below the largest ω² of its own q point -- an acoustic mode at Γ of a cell that
# also carries optical modes -- can differ in the leading digits. Modes away from ω = 0 agree to
# round-off.
_set_eigen_valueonly_from!(ph::PhononState, ::Nothing, dyn, mass, polar, xq) =
    set_eigen_valueonly!(ph, dyn, mass, polar, xq)

function _set_eigen_valueonly_from!(ph::PhononState, eigenpairs::Eigenpairs, dyn, mass, polar, xq)
    iq = _eigenpairs_ik(eigenpairs, xq)
    ph.xq = xq
    @views ph.e .= eigenpairs.e_full[:, iq]
    ph
end
