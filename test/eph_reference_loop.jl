# The test-only reference for the e-ph loops: a plain double loop over (k, k+q) on per-point
# `ElectronState`s and the per-point kernels `get_eph_RR_to_kR!` / `get_eph_kR_to_kq!`, with the
# phonons solved at each q. Its own loop structure, so it is independent of every driver's batching,
# tiling, threading and staging; and a recorder calculator that reads the same `|g|^2` out of each
# block of the current drivers. Compared on gauge-invariant quantities: |g|^2 summed over
# degenerate multiplets of the k band, the k+q band and the phonon mode, and the phonon frequencies.

using ElectronPhonon: AbstractCalculator, OuterKLoop, OuterQLoop, EPBlock, get_eph_RR_to_kR!,
    get_eph_kR_to_kq!, get_next_wannier_object, get_interpolator, Vec3
using OffsetArrays: no_offset_view

# Indices of `x` grouped into runs of values within `tol` of the previous one (degenerate
# multiplets of sorted energies).
function contract_multiplets(x, tol)
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

# Integer grid coordinates of a k point, folded into 0:n-1: the key of a (k, k+q) pair.
_grid_key(x, ngrid) = Tuple(mod.(round.(Int, x .* ngrid), ngrid))
_pair_key(xk, xkq, ngrid) = (_grid_key(xk, ngrid)..., _grid_key(xkq, ngrid)...)

"""
    eph_reference(model, kpts, kqpts, window_k, window_kq; ngrid = kqpts.ngrid,
                  kq_indices = ik -> eachindex(kqpts.vectors))
        -> (; ep, g2abs, ωq, el_k, el_kq, wtkq)

`ep[m, n, ν]`, `|ep[m, n, ν]|^2` and `ω[ν]` of every pair `(k, k+q)` of `kpts × kqpts` (for each k
only the k+q points `kq_indices(ik)`) whose two windows are not empty, keyed by `_pair_key` on
`ngrid`, with `m`, `n` physical bands (zero outside the windows); the states and the k+q weights
keyed by `_grid_key`. `model` must have `epmat_outer_momentum = "el"` and no polar terms; the e-ph
matrix is the one of the loops before calculators see it (no dipole term, no 1/2ω).
"""
function eph_reference(model, kpts, kqpts, window_k, window_kq; ngrid = kqpts.ngrid,
                       kq_indices = ik -> eachindex(kqpts.vectors))
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
    ep_pairs = Dict{NTuple{6, Int}, Array{ComplexF64, 3}}()
    g2abs = Dict{NTuple{6, Int}, Array{Float64, 3}}()
    ωq = Dict{NTuple{6, Int}, Vector{Float64}}()
    for (ik, xk) in enumerate(kpts.vectors)
        elk = el_k[ik]
        elk.nband == 0 && continue
        get_eph_RR_to_kR!(ep_ekpR_obj, epmat, xk, no_offset_view(elk.u))
        for ikq in kq_indices(ik)
            xkq = kqpts.vectors[ikq]
            elkq = el_kq[ikq]
            elkq.nband == 0 && continue
            xq = xkq - xk
            set_eigen!(ph, dyn, model.mass, model.polar_phonon, xq)
            ep = zeros(ComplexF64, elkq.nband, elk.nband, nmodes)
            get_eph_kR_to_kq!(ep, ep_ekpR, xq, ph.u, no_offset_view(elkq.u))
            a = zeros(ComplexF64, nw, nw, nmodes)
            a[elkq.rng, elk.rng, :] .= ep
            key = _pair_key(xk, xkq, ngrid)
            ep_pairs[key] = a
            g2abs[key] = abs2.(a)
            ωq[key] = copy(ph.e)
        end
    end
    (; ep = ep_pairs, g2abs, ωq,
       el_k = Dict(_grid_key(x, ngrid) => el for (x, el) in zip(kpts.vectors, el_k)),
       el_kq = Dict(_grid_key(x, ngrid) => el for (x, el) in zip(kqpts.vectors, el_kq)),
       wtkq = Dict(_grid_key(x, ngrid) => w for (x, w) in zip(kqpts.vectors, kqpts.weights)))
end

"""
    eph_reference_k_and_q(model, kpts, qpts, window_k, window_kq, ngrid)

[`eph_reference`](@ref) over the pairs `(k, k + q)` of `kpts × qpts`, the pairs of
`run_eph_over_k_and_q`, keyed on `ngrid`. A fine `ngrid` (`10^6` per axis) keys points on no grid by
their rounded coordinates, which match because the driver forms the same `x_k + x_q`.
"""
function eph_reference_k_and_q(model, kpts, qpts, window_k, window_kq, ngrid)
    xkqs = [xk + xq for xk in kpts.vectors for xq in qpts.vectors]
    kqpts = Kpoints(length(xkqs), xkqs, repeat(qpts.weights, kpts.n), ngrid)
    eph_reference(model, kpts, kqpts, window_k, window_kq; ngrid,
                  kq_indices = ik -> (ik - 1) * qpts.n .+ (1:qpts.n))
end

"""
    eph_reference_dg(model, kpts, kqpts) -> Dict(pair key => (Σ |dg[:, :, :, d]|² for d in 1:3))

The covariant derivative of the e-ph matrix, `dg[m, n, ν, d]`, of every pair `(k, k+q)` of
`kpts × kqpts` on the full band window, in the tight-binding approximation of the per-point outer-k
loop (`covariant_derivative_of_g`): the Fourier transform of `im R_e g(R_e, R_p)` plus
`im (r_j - r_i) g`, rotated to the electron and phonon eigenbases. Summed over the bands and modes,
so the sums do not depend on either basis. `model` must have `epmat_outer_momentum = "el"`.
"""
function eph_reference_dg(model, kpts, kqpts)
    (; nw, nmodes) = model
    epmat_R_obj = ElectronPhonon.wannier_object_multiply_R(model.epmat, model.lattice)
    nrp = length(epmat_R_obj.irvec_next)
    @views for ire in axes(epmat_R_obj.op_r, 2)
        g = Base.ReshapedArray(model.epmat.op_r[:, ire], (nw, nw, nmodes, nrp), ())
        gR = Base.ReshapedArray(epmat_R_obj.op_r[:, ire], (nw, nw, nmodes, nrp, 3), ())
        for idir in 1:3, iw in 1:nw
            ri = model.wann_centers[iw][idir]
            gR[iw, :, :, :, idir] .-= im .* ri .* g[iw, :, :, :]
            gR[:, iw, :, :, idir] .+= im .* ri .* g[:, iw, :, :]
        end
    end
    epmat_R = get_interpolator(epmat_R_obj; fourier_mode = "normal")
    epobj_ekpR_R = get_next_wannier_object(epmat_R_obj)
    ep_ekpR_R = get_interpolator(epobj_ekpR_R; fourier_mode = "normal")
    dyn = get_interpolator(model.ph_dyn; fourier_mode = "normal")
    el_k = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector"]; fourier_mode = "normal")
    el_kq = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector"]; fourier_mode = "normal")
    ph = PhononState(nmodes, Float64)
    ngrid = kqpts.ngrid
    nrp_next = length(epobj_ekpR_R.irvec)
    out = Dict{NTuple{6, Int}, Vector{Float64}}()
    for (ik, xk) in enumerate(kpts.vectors)
        get_fourier!(epmat_R.out, epmat_R, xk)
        tmp = Base.ReshapedArray(epmat_R.out, (nw * nw * nmodes, nrp_next, 3), ())
        epobj_ekpR_R.op_r .= reshape(permutedims(tmp, (1, 3, 2)), (nw * nw * nmodes * 3, nrp_next))
        uk = el_k[ik].u_full
        for (ikq, xkq) in enumerate(kqpts.vectors)
            xq = ElectronPhonon.normalize_kpoint_coordinate(xkq - xk .+ 1/2) .- 1/2
            set_eigen!(ph, dyn, model.mass, model.polar_phonon, xq)
            get_fourier!(ep_ekpR_R.out, ep_ekpR_R, xq)
            dg_wan = reshape(ep_ekpR_R.out, nw, nw, nmodes, 3)
            ukq = el_kq[ikq].u_full
            sums = zeros(3)
            for d in 1:3
                dg_e = stack(ν -> ukq' * dg_wan[:, :, ν, d] * uk, 1:nmodes)       # (m, n, ν') Wannier mode
                dg_ph = reshape(reshape(dg_e, nw * nw, nmodes) * ph.u, nw, nw, nmodes)
                sums[d] = sum(abs2, dg_ph)
            end
            out[_pair_key(xk, xkq, ngrid)] = sums
        end
    end
    out
end

"""
    compare_with_reference(ref, rec; tol_degen = 1e-6)
        -> (; g2_reldev, ω_dev, npairs, nmissing, nextra)

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
        ωdev = max(ωdev, maximum(abs, ω - rec.ωq[key]))
        for gm in contract_multiplets(collect(elkq.e), tol_degen),
                gn in contract_multiplets(collect(elk.e), tol_degen),
                gν in contract_multiplets(ω, tol_degen)
            mm = elkq.rng[gm]; nn = elk.rng[gn]
            g2dev = max(g2dev, abs(sum(a[mm, nn, gν]) - sum(b[mm, nn, gν])))
        end
    end
    # A loop hands a calculator no pair with an empty window on either side.
    nextra = count(key -> !haskey(ref.g2abs, key), keys(rec.g2abs))
    (; g2_reldev = g2dev / scale, ω_dev = ωdev, npairs = length(ref.g2abs), nmissing, nextra)
end

# ---- recorder: |g|^2 per pair out of each block ----------------------------------------------

# One recorder for both loop orders. Writes are keyed by the pair on `ngrid`, the run's k grid unless
# given; the Dict is guarded by a lock.
mutable struct _PairRecorder <: AbstractCalculator
    nw::Int
    nmodes::Int
    ngrid::NTuple{3, Int}
    g2abs::Dict{NTuple{6, Int}, Array{Float64, 3}}
    ωq::Dict{NTuple{6, Int}, Vector{Float64}}
    lock::ReentrantLock
    _PairRecorder(ngrid = (0, 0, 0)) = new(0, 0, ngrid, Dict(), Dict(), ReentrantLock())
end
ElectronPhonon.supports(::_PairRecorder, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_PairRecorder, ::Type{OuterQLoop}) = true
# The loop always provides `e`, `u` and the e-ph matrix elements, which is all this calculator
# reads, so it defines no `required_el_quantities` / `required_ph_quantities`.
ElectronPhonon.calculator_begin!(::_PairRecorder, ctx) = nothing
ElectronPhonon.calculator_end!(::_PairRecorder, ctx) = nothing
ElectronPhonon.postprocess_calculator!(c::_PairRecorder; kwargs...) = c
function ElectronPhonon.setup_calculator!(c::_PairRecorder, backend, els_k, els_kq, phs; kwargs...)
    c.nw, c.nmodes = els_k.nw, phs.nmodes
    all(iszero, c.ngrid) && (c.ngrid = els_k.kpts.ngrid)
    c
end

# Both orders: the side shared by the block has extent 1 along the pair axis.
function ElectronPhonon.run_calculator!(c::_PairRecorder, p::EPBlock, ctx)
    ep = Array(p.ep); ω = Array(p.phs.e)
    offk, nbk = Array(p.els_k.iband_offset), Array(p.els_k.nband)
    offkq, nbkq = Array(p.els_kq.iband_offset), Array(p.els_kq.nband)
    for j in axes(ep, 4)
        jk, jkq = min(j, length(nbk)), min(j, length(nbkq))
        xk = p.xk isa Vec3 ? p.xk : p.xk[j]
        xq = p.xq isa Vec3 ? p.xq : p.xq[j]
        nn, mm = 1:nbk[jk], 1:nbkq[jkq]
        a = zeros(c.nw, c.nw, c.nmodes)
        a[offkq[jkq] .+ mm, offk[jk] .+ nn, :] .= abs2.(ep[mm, nn, :, j])
        lock(c.lock) do
            key = _pair_key(xk, xk + xq, c.ngrid)
            c.g2abs[key] = a
            c.ωq[key] = ω[:, min(j, size(ω, 2))]
        end
    end
end
