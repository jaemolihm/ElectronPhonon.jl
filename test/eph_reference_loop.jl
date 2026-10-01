# The test-only reference for the e-ph loops: a plain double loop over (k, k+q) on per-point
# `ElectronState`s and the per-point kernels `get_eph_RR_to_kR!` / `get_eph_kR_to_kq!`, with the
# phonons solved at each q. Its own loop structure, so it is independent of every driver's batching,
# tiling, threading and staging; and recorder calculators that read the same `|g|^2` out of each
# payload of the current drivers. Compared on gauge-invariant quantities: |g|^2 summed over
# degenerate multiplets of the k band, the k+q band and the phonon mode, and the phonon frequencies.

using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPData, EPDataQBatched,
    EPDataKBatched, OuterIteration, OuterIterationBatch, get_eph_RR_to_kR!, get_eph_kR_to_kq!,
    get_next_wannier_object, get_interpolator, Vec3
using OffsetArrays: no_offset_view

# Integer grid coordinates of a k point, folded into 0:n-1: the key of a (k, k+q) pair.
_grid_key(x, ngrid) = Tuple(mod.(round.(Int, x .* ngrid), ngrid))
_pair_key(xk, xkq, ngrid) = (_grid_key(xk, ngrid)..., _grid_key(xkq, ngrid)...)

"""
    eph_reference(model, kpts, kqpts, window_k, window_kq) -> (; g2abs, ωq, el_k, el_kq)

`|ep[m, n, ν]|^2` and `ω[ν]` of every pair `(k, k+q)` of `kpts × kqpts` whose two windows are not
empty, keyed by `_pair_key`, with `m`, `n` physical bands (zero outside the windows). `model` must
have `epmat_outer_momentum = "el"` and no polar terms; the e-ph matrix is the one of the loops
before calculators see it (no dipole term, no 1/2ω).
"""
function eph_reference(model, kpts, kqpts, window_k, window_kq)
    (; nw, nmodes) = model
    (model.polar_phonon.use || model.polar_eph.use) && error("eph_reference has no polar term")
    el_k = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector"], window_k;
                                   fourier_mode = "normal")
    el_kq = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector"], window_kq;
                                    fourier_mode = "normal")
    epmat = get_interpolator(model.epmat; fourier_mode = "normal")
    ep_ekpR_obj = get_next_wannier_object(model.epmat)
    ep_ekpR = get_interpolator(ep_ekpR_obj; fourier_mode = "normal")
    dyn = get_interpolator(model.ph_dyn; fourier_mode = "normal")
    ph = PhononState(nmodes, Float64)
    ngrid = kqpts.ngrid
    g2abs = Dict{NTuple{6, Int}, Array{Float64, 3}}()
    ωq = Dict{NTuple{6, Int}, Vector{Float64}}()
    for (ik, xk) in enumerate(kpts.vectors)
        elk = el_k[ik]
        elk.nband == 0 && continue
        get_eph_RR_to_kR!(ep_ekpR_obj, epmat, xk, no_offset_view(elk.u))
        for (ikq, xkq) in enumerate(kqpts.vectors)
            elkq = el_kq[ikq]
            elkq.nband == 0 && continue
            xq = xkq - xk
            set_eigen!(ph, dyn, model.mass, model.polar_phonon, xq)
            ep = zeros(ComplexF64, elkq.nband, elk.nband, nmodes)
            get_eph_kR_to_kq!(ep, ep_ekpR, xq, ph.u, no_offset_view(elkq.u))
            a = zeros(nw, nw, nmodes)
            a[elkq.rng, elk.rng, :] .= abs2.(ep)
            key = _pair_key(xk, xkq, ngrid)
            g2abs[key] = a
            ωq[key] = copy(ph.e)
        end
    end
    (; g2abs, ωq, el_k = Dict(_grid_key(x, ngrid) => el for (x, el) in zip(kpts.vectors, el_k)),
       el_kq = Dict(_grid_key(x, ngrid) => el for (x, el) in zip(kqpts.vectors, el_kq)))
end

# Indices of `x` grouped into runs of values within `tol` of the previous one (degenerate
# multiplets of sorted energies).
function _multiplets(x, tol)
    groups = Vector{Vector{Int}}()
    for i in eachindex(x)
        if isempty(groups) || abs(x[i] - x[last(groups[end])]) > tol
            push!(groups, [i])
        else
            push!(groups[end], i)
        end
    end
    groups
end

"""
    compare_with_reference(ref, rec; tol_degen = 1e-6) -> (; g2_reldev, ω_dev, npairs, nmissing, nextra)

The largest deviation of the recorded `|g|^2` from the reference over every pair of `ref`, after
summing both over the degenerate multiplets of the in-window k and k+q bands and of the phonon
modes (from the reference's energies and frequencies), relative to the largest reference entry;
and of the frequencies. `nmissing` counts reference pairs absent from `rec`, `nextra` recorded pairs
absent from `ref` with a nonzero coupling.
"""
function compare_with_reference(ref, rec; tol_degen = 1e-6)
    scale = maximum(maximum, values(ref.g2abs))
    g2dev = 0.0; ωdev = 0.0; nmissing = 0
    for (key, a) in ref.g2abs
        haskey(rec.g2abs, key) || (nmissing += 1; continue)
        b = rec.g2abs[key]
        elk = ref.el_k[key[1:3]]; elkq = ref.el_kq[key[4:6]]
        ω = ref.ωq[key]
        # The outer-q batched payload carries no frequencies (recorded as NaN).
        all(isnan, rec.ωq[key]) || (ωdev = max(ωdev, maximum(abs, ω - rec.ωq[key])))
        for gm in _multiplets(collect(elkq.e), tol_degen),
                gn in _multiplets(collect(elk.e), tol_degen), gν in _multiplets(ω, tol_degen)
            mm = elkq.rng[gm]; nn = elk.rng[gn]
            g2dev = max(g2dev, abs(sum(a[mm, nn, gν]) - sum(b[mm, nn, gν])))
        end
    end
    # A loop may hand a calculator pairs with an empty window on one side (the outer-q batched loop
    # masks instead of skipping); they must carry no coupling.
    nextra = count(key -> !haskey(ref.g2abs, key) && !iszero(rec.g2abs[key]), keys(rec.g2abs))
    (; g2_reldev = g2dev / scale, ω_dev = ωdev, npairs = length(ref.g2abs), nmissing, nextra)
end

# ---- recorders: |g|^2 per pair out of each payload --------------------------------------------

# One recorder for every loop and payload. Writes are keyed by the pair, so concurrent chunks
# write different entries; the Dict itself is guarded by a lock.
mutable struct _PairRecorder <: AbstractCalculator
    nw::Int
    nmodes::Int
    ngrid::NTuple{3, Int}
    kpts::Any
    qpts::Any
    kqpts::Any
    g2abs::Dict{NTuple{6, Int}, Array{Float64, 3}}
    ωq::Dict{NTuple{6, Int}, Vector{Float64}}
    lock::ReentrantLock
    _PairRecorder() = new(0, 0, (0, 0, 0), nothing, nothing, nothing, Dict(), Dict(), ReentrantLock())
end
for P in (OuterKLoop, OuterQLoop, EPData, EPDataQBatched, EPDataKBatched)
    @eval ElectronPhonon.supports(::_PairRecorder, ::Type{$P}) = true
end
ElectronPhonon.allowed_eph_phonon_basis(::_PairRecorder) = [:eigenmode]
ElectronPhonon.required_el_k_quantities(::_PairRecorder) = ["eigenvalue", "eigenvector"]
ElectronPhonon.calculator_begin!(::_PairRecorder, ::Any, ctx) = nothing
ElectronPhonon.calculator_end!(::_PairRecorder, ::Any, ctx) = nothing
ElectronPhonon.postprocess_calculator!(c::_PairRecorder; kwargs...) = c
function ElectronPhonon.setup_calculator!(c::_PairRecorder, backend, mode, kpts, qpts, el_states;
        nw, nmodes, kqpts = nothing, kwargs...)
    c.nw, c.nmodes = nw, nmodes
    c.kpts, c.qpts, c.kqpts = kpts, qpts, kqpts
    c.ngrid = kpts.ngrid
    c
end

function _record!(c::_PairRecorder, key, a, ω)
    lock(c.lock) do
        c.g2abs[key] = a
        c.ωq[key] = ω
    end
end

function ElectronPhonon.run_calculator!(c::_PairRecorder, p::EPData, ctx)
    (; epstate, xk, xq) = p
    (; el_k, el_kq, ph) = epstate
    a = zeros(c.nw, c.nw, c.nmodes)
    a[el_kq.rng, el_k.rng, :] .= abs2.(no_offset_view(epstate.ep))
    _record!(c, _pair_key(xk, xk + xq, c.ngrid), a, copy(ph.e))
end

function ElectronPhonon.run_calculator!(c::_PairRecorder, p::EPDataQBatched, ctx)
    eps = Array(p.eps); ωqs = Array(p.ωqs)
    # Box columns past physical band nw are padding.
    nbandkq, nbandk = size(eps, 1), ElectronPhonon.nbandk_physical(p, c.nw)
    xk = c.kpts.vectors[p.ik]
    for (j, ikq) in enumerate(p.ikqs)
        a = zeros(c.nw, c.nw, c.nmodes)
        a[1:nbandkq, p.ibandk_offset .+ (1:nbandk), :] .= abs2.(eps[:, 1:nbandk, :, j])
        _record!(c, _pair_key(xk, c.kqpts.vectors[ikq], c.ngrid), a, ωqs[:, j])
    end
end

function ElectronPhonon.run_calculator!(c::_PairRecorder, p::EPDataKBatched, ctx)
    eps = Array(p.eps)
    xq = c.qpts.vectors[p.iq]
    for (j, xk) in enumerate(p.xks)
        _record!(c, _pair_key(xk, xk + xq, c.ngrid), abs2.(eps[:, :, :, j]), fill(NaN, c.nmodes))
    end
end
