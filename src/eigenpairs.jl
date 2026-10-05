# Full-band eigenpairs over a k-point set, computed once and shared by several e-ph runs to ensure
# eigenvector gauge consistency. The container is species-agnostic (`nbasis` is `nw` for electrons
# and `nmodes` for phonons), with one builder per species.

using ChunkSplitters
using Base.Threads: nthreads, @threads

export Eigenpairs
export electron_eigenpairs
export phonon_eigenpairs
export set_el_eigen!
export set_ph_eigen!

"""
    Eigenpairs{T, MT, AT}

Full-band eigenvalues `e_full` (`(nbasis, kpts.n)`) and eigenvectors `u_full`
(`(nbasis, nbasis, kpts.n)`) on the k-point set `kpts`. Build one with
[`electron_eigenpairs`](@ref) or [`phonon_eigenpairs`](@ref).

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

# The CPU case is the generic `to_device(::CPUBackend, x) = x`.
"""
    to_device(backend::GPUBackend, eig::Eigenpairs) -> Eigenpairs

`eig` with `e_full`/`u_full` placed on `backend`; `kpts` stays on the host.
"""
to_device(backend::GPUBackend, eig::Eigenpairs) =
    Eigenpairs(eig.nbasis, eig.kpts, to_device(backend, eig.e_full), to_device(backend, eig.u_full))

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
        @threads for iks in index_chunks(gkpts.vectors; n = 2*nthreads())
            @views begin
                # Setup thread-local WannierInterpolator
                ham = get_interpolator(model.el_ham; fourier_mode)
                register_kpoints!(ham, gkpts.vectors[iks])
                for ik in iks
                    compute_el_eigen!(e_full[:, ik], u_full[:, :, ik], nw, ham, gkpts.vectors[ik])
                end
            end
        end
        Eigenpairs(nw, gkpts, e_full, u_full)
    else
        itp_elham = get_interpolator(to_device(backend, model.el_ham);
                                     fourier_mode="batched", backend, nk_hint=gkpts.n)
        E_dev, U_dev = compute_el_eigen_batched(itp_elham, gkpts.vectors)
        Eigenpairs(nw, gkpts, E_dev, U_dev)
    end
end

"""
    phonon_eigenpairs(model, qpts; fourier_mode = "gridopt", backend = CPUBackend())

The phonon twin of [`electron_eigenpairs`](@ref): the frequencies ω (`e_full`) and mass-scaled
eigenmodes (`u_full`) at every q point of `qpts`, `nbasis = nmodes`. They are the `e`/`u` stacks of
[`compute_phonon_states_batched`](@ref) over the same q points, held without a copy, so on the host
they are what [`compute_phonon_states`](@ref) solves for and a run handed this cache as
`ph_eigenpairs` reproduces the run without it.

On a GPU backend the cache stays on the device; `fourier_mode` is then unused and polar phonons are
not supported. The batched eigensolve picks its own basis inside a degenerate mode multiplet, so a
device-built cache differs from a CPU-built one there.
"""
function phonon_eigenpairs(model::Model, qpts; fourier_mode = "gridopt", backend = CPUBackend())
    phs = compute_phonon_states_batched(model, GridKpoints(qpts), [:e, :u]; fourier_mode, backend)
    Eigenpairs(model.nmodes, phs.qpts, phs.e, phs.u)
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

# The consumers. A caller takes the eigenpairs of a k point either by solving H(k) or D(q) itself
# or from the cache, chosen by dispatch on the cache argument: `nothing` solves and an `Eigenpairs`
# copies out. The cache therefore arrives as a typed argument, each call site specializes on one of
# the two, and no loop gains a branch. The choice is made once per species, in the array-level
# `set_el_eigen!` / `set_ph_eigen!` that every host consumer calls: the `set_eigen!` /
# `set_eigen_valueonly!` state setters below, the batched host builders and the window filter. The
# copy does not depend on the species, so the lookup, the miss error and the copy are written once,
# here next to the type.

# Copy the eigenvalues of `xk` into `e` and, unless `u === nothing`, its eigenvectors into `u`.
@views function _copy_eigen_from!(e, u, eigenpairs::Eigenpairs, xk)
    ik = _eigenpairs_ik(eigenpairs, xk)
    e .= eigenpairs.e_full[:, ik]
    u === nothing || (u .= eigenpairs.u_full[:, :, ik])
    nothing
end

"""
    set_el_eigen!(e, u, nw, eigenpairs, ham, xk)
The full-band electron eigenvalues of `xk` into `e` (nw) and, unless `u === nothing`, the
eigenvectors into `u` (nw, nw). `eigenpairs === nothing` diagonalizes H(k) with `ham`
([`compute_el_eigen!`](@ref) or [`compute_el_eigen_valueonly!`](@ref)); an
[`Eigenpairs`](@ref) copies them out of the cache, which must cover `xk`, and `ham` is unused.
"""
set_el_eigen!(e, u, nw, ::Nothing, ham, xk) =
    u === nothing ? compute_el_eigen_valueonly!(e, nw, ham, xk) :
                    compute_el_eigen!(e, u, nw, ham, xk)

set_el_eigen!(e, u, nw, eigenpairs::Eigenpairs, ham, xk) = _copy_eigen_from!(e, u, eigenpairs, xk)

"""
    set_ph_eigen!(e, u, eigenpairs, dyn, mass, polar, xq)
The phonon frequencies ω of `xq` into `e` (nmodes) and, unless `u === nothing`, the mass-scaled
eigenmodes into `u` (nmodes, nmodes), i.e. what [`compute_ph_eigen!`](@ref) returns and what
[`phonon_eigenpairs`](@ref) stores. `eigenpairs === nothing` diagonalizes D(q); an
[`Eigenpairs`](@ref) copies them out of the cache, which must cover `xq`, and the solver arguments
are unused. Argument order follows [`set_el_eigen!`](@ref), `(arrays, cache, what it takes to
solve, momentum)`; `D(q)` needs the masses and the dipole term where `H(k)` needs only `ham`.
"""
set_ph_eigen!(e, u, ::Nothing, dyn, mass, polar, xq) =
    u === nothing ? compute_ph_eigen_valueonly!(e, dyn, mass, polar, xq) :
                    compute_ph_eigen!(e, u, dyn, mass, polar, xq)

# Unlike the other copies this one is not inert for `u === nothing`: `e` comes from the cache's FULL
# eigensolve where the cacheless path runs the value-only driver. The two agree on ω² to
# `eps * ‖D(q)‖`, but ω = sign(ω²)√|ω²| turns that into a `1/(2ω)`-amplified deviation, so a mode
# whose ω² is far below the largest ω² of its own q point -- an acoustic mode at Γ of a cell that
# also carries optical modes -- can differ in the leading digits. Modes away from ω = 0 agree to
# round-off.
set_ph_eigen!(e, u, eigenpairs::Eigenpairs, dyn, mass, polar, xq) =
    _copy_eigen_from!(e, u, eigenpairs, xq)

# The full-band electron eigenvalues of a chunk of k points, batched and returned on the host for
# the window test. The device arm gathers off a cache that is resident on that same device.
_eigenvalues_on_host(::Nothing, itp_elham, xks) =
    Array(compute_el_eigen_valueonly_batched(itp_elham, xks))

function _eigenvalues_on_host(eigenpairs::Eigenpairs, itp_elham, xks)
    iks = map(xk -> _eigenpairs_ik(eigenpairs, xk), xks)
    Array(eigenpairs.e_full[:, iks])
end

# The state setters. They live here rather than in electron_state.jl / phonon_state.jl because the
# cache argument is annotated with `Eigenpairs`, so that a call in the cacheless argument order
# cannot bind to it.

"""
    set_eigen!(el::ElectronState, [eigenpairs,] ham, xk)
Compute electron eigenenergy and eigenvector and save them in `el`, or copy them from
`eigenpairs` (see [`set_el_eigen!`](@ref)). Resets the window to a dummy value.
"""
function set_eigen!(el::ElectronState, eigenpairs::Union{Nothing, Eigenpairs}, ham, xk)
    el.xk = xk
    set_el_eigen!(el.e_full, el.u_full, el.nw, eigenpairs, ham, xk)

    # Reset window to a dummy value
    el.nband = 0
    el.rng = 1:0
    el
end

set_eigen!(el::ElectronState, ham, xk) = set_eigen!(el, nothing, ham, xk)

"""
    set_eigen_valueonly!(el::ElectronState, [eigenpairs,] ham, xk)
Compute electron eigenenergy and save them in `el`, or copy them from `eigenpairs`. Resets the
window to a dummy value.
"""
function set_eigen_valueonly!(el::ElectronState, eigenpairs::Union{Nothing, Eigenpairs}, ham, xk)
    el.xk = xk
    set_el_eigen!(el.e_full, nothing, el.nw, eigenpairs, ham, xk)

    # Reset window to a dummy value
    el.nband = 0
    el.rng = 1:0
    el
end

set_eigen_valueonly!(el::ElectronState, ham, xk) = set_eigen_valueonly!(el, nothing, ham, xk)

# The momentum is annotated so that a call left in an older argument order is a `MethodError`
# rather than a mis-binding.
"""
    set_eigen!(ph::PhononState, [eigenpairs,] dyn, mass, polar, xq)
Compute phonon eigenenergy and eigenvector and save them in `ph`, or copy them from `eigenpairs`
(see [`set_ph_eigen!`](@ref)).
"""
function set_eigen!(ph::PhononState, eigenpairs::Union{Nothing, Eigenpairs}, dyn, mass, polar,
                    xq::Vec3)
    ph.xq = xq
    set_ph_eigen!(ph.e, ph.u, eigenpairs, dyn, mass, polar, xq)
    ph
end

set_eigen!(ph::PhononState, dyn, mass, polar, xq::Vec3) =
    set_eigen!(ph, nothing, dyn, mass, polar, xq)

"""
    set_eigen_valueonly!(ph::PhononState, [eigenpairs,] dyn, mass, polar, xq)
Compute phonon eigenenergy and save them in `ph`, or copy them from `eigenpairs`.
"""
function set_eigen_valueonly!(ph::PhononState, eigenpairs::Union{Nothing, Eigenpairs}, dyn, mass,
                              polar, xq::Vec3)
    ph.xq = xq
    set_ph_eigen!(ph.e, nothing, eigenpairs, dyn, mass, polar, xq)
    ph
end

set_eigen_valueonly!(ph::PhononState, dyn, mass, polar, xq::Vec3) =
    set_eigen_valueonly!(ph, nothing, dyn, mass, polar, xq)
