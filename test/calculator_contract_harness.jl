# The generic calculator contract harness. Shared by ElectronPhonon.jl's and MigdalEliashberg.jl's
# `test_calculator_contract.jl`, each with its own list of calculator entries.
#
# An entry is a NamedTuple `(; name, make, orders, outputs, reference)`: `make()` returns a fresh
# calculator, `orders` the loop orders it supports (`OuterKLoop`, `OuterQLoop`), and `outputs(c)` a
# `Dict{String, Array}` of what it computed, reduced to gauge-invariant arrays (see
# `contract_multiplet_ids`, `contract_pair_sum`). `reference(c, ref)` rebuilds the fields `outputs`
# reads, on the states of the calculator `c`, from the per-pair data `ref` of the reference double
# loop (`eph_reference`, see `contract_foreach_reference_pair`), so both go through the same
# reduction. For every entry, order and fixture the harness runs every loop setting of
# `contract_settings`, and compares the outputs across the settings and against the reference.

using Test
using ElectronPhonon
using ElectronPhonon: CPUBackend, OuterKLoop, OuterQLoop, unit_to_aru, run_eph_over_k_and_kq,
    run_eph_over_q_and_k, electron_degen_cutoff, omega_acoustic, gpu_backend
isdefined(@__MODULE__, :eph_reference) || include("eph_reference_loop.jl")

# The windowed fixtures of the pb artifact model: nk = 12, E_F +- 0.2 eV (one in-window band per k)
# and nk = 6, E_F - 0.5 / + 3 eV (0 to 3 bands per k, empty windows, windows ending at band nw).
function contract_fixtures()
    eV = unit_to_aru(:eV)
    e_F = 11.68eV
    (; narrow = (; grid = (12, 12, 12), window = (e_F - 0.2eV, e_F + 0.2eV), e_F),
       ragged = (; grid = (6, 6, 6), window = (e_F - 0.5eV, e_F + 3eV), e_F))
end

# The loop settings of each order: the CPU loop with its default widths and several thread chunks,
# at outer batch width 1 and > 1 with many small inner tiles, with the `"normal"` setup
# interpolation, and with `gpu = true` on the GPU.
function contract_settings(::Type{OuterKLoop}; gpu = false)
    s = (default = (; nchunks_threads = 4),
         outer1 = (; backend = CPUBackend(), n_outer_batch = 1, n_inner_tile = 37),
         outer5 = (; backend = CPUBackend(), n_outer_batch = 5, n_inner_tile = 37),
         normal = (; backend = CPUBackend(), fourier_mode = "normal"))
    gpu ? (; s..., gpu = (; backend = gpu_backend(), n_inner_tile = 37)) : s
end
function contract_settings(::Type{OuterQLoop}; gpu = false)
    s = (default = (; nchunks_threads = 4),
         tiles = (; backend = CPUBackend(), n_inner_tile = 37),
         normal = (; backend = CPUBackend(), fourier_mode = "normal"))
    gpu ? (; s..., gpu = (; backend = gpu_backend(), n_inner_tile = 37)) : s
end

function run_contract(entry, order, models, fixture, setting)
    calc = entry.make()
    common = (; calculators = [calc], window_k = fixture.window, window_kq = fixture.window,
              progress_print_step = 10^9, verbosity = 0)
    if order === OuterKLoop
        run_eph_over_k_and_kq(models.el, fixture.grid, fixture.grid; symmetry = nothing, common...,
                              setting...)
    else
        run_eph_over_q_and_k(models.ph, fixture.grid, fixture.grid; symmetry = nothing,
                             common..., setting...)
    end
    calc
end

"""
    contract_multiplet_ids(s; tol = electron_degen_cutoff) -> Vector{Int}

A group id per state of the `BandStates` `s`: states at one k point within `tol` of the next lower
one share an id. A sum over the states of one id (a degenerate multiplet) does not depend on the
eigenvector basis the solver picked inside it, so it can be compared across eigensolvers and
backends.
"""
function contract_multiplet_ids(s; tol = electron_degen_cutoff)
    ids = zeros(Int, s.n)
    perm = sortperm(collect(zip(s.iks, s.es)))
    ngroup = 0
    for (j, i) in enumerate(perm)
        new = j == 1 || s.iks[perm[j-1]] != s.iks[i] || s.es[i] - s.es[perm[j-1]] > tol
        ngroup += new
        ids[i] = ngroup
    end
    ids
end

# `x` summed over the groups `ids` of its axis `d`.
function contract_group_sum(x, d, ids)
    y = zeros(eltype(x), ntuple(k -> k == d ? maximum(ids; init = 0) : size(x, k), ndims(x)))
    for I in CartesianIndices(x)
        y[Base.setindex(Tuple(I), ids[I[d]], d)...] += x[I]
    end
    y
end

"""
    contract_pair_sum(x, ωq, ids_i, ids_f; tol_ω = 1e-6)

A per-mode, per-state-pair quantity `x[ν, i, f]` (`|g|^2`, `|ep|^2`) summed over each degenerate
phonon multiplet of its pair (modes with frequencies `ωq[:, i, f]` within `tol_ω` of the previous
one, as `compare_with_reference` groups them) and over the electron multiplets of both states. The
result `y[ν, gi, gf]` holds a multiplet's sum at its first mode, zero at the others, so it pins
which mode carries the coupling but not the basis inside a multiplet. Modes with
`|ω| < omega_acoustic` are dropped, so the Γ-acoustic `1/(2ω)` entries, which are not physical, do
not set the scale of the comparison.
"""
function contract_pair_sum(x, ωq, ids_i, ids_f; tol_ω = 1e-6)
    y = zero(x)
    for f in axes(x, 3), i in axes(x, 2), g in contract_multiplets(view(ωq, :, i, f), tol_ω)
        abs(ωq[first(g), i, f]) >= omega_acoustic || continue
        y[first(g), i, f] = sum(ν -> x[ν, i, f], g)
    end
    contract_group_sum(contract_group_sum(y, 2, ids_i), 3, ids_f)
end

# Frequencies with `|ω| < omega_acoustic` set to zero: the Γ-acoustic ones are the square root of
# round-off, so they differ between any two eigensolves.
contract_physical_ω(ωq) = ifelse.(abs.(ωq) .>= omega_acoustic, ωq, zero(eltype(ωq)))

"""
    contract_foreach_reference_pair(f, ref, el_i, el_f)

Call `f(i, j, ep, ω, ek, ekq, wtkq)` for every pair of a state `i` of `el_i` and `j` of `el_f`
(`BandStates`), with the reference double loop's data `ref` (`eph_reference`) of that pair:
`ep[ν]` the e-ph element, `ω[ν]` the frequencies, `ek`, `ekq` the two energies and `wtkq` the
weight of the k+q point.
"""
function contract_foreach_reference_pair(f, ref, el_i, el_f)
    for j in 1:el_f.n, i in 1:el_i.n
        key = _pair_key(el_i.kpts.vectors[el_i.iks[i]], el_f.kpts.vectors[el_f.iks[j]],
                        el_i.kpts.ngrid)
        n, m = el_i.ibands[i], el_f.ibands[j]
        f(i, j, view(ref.ep[key], m, n, :), ref.ωq[key], ref.el_k[key[1:3]].e[n],
          ref.el_kq[key[4:6]].e[m], ref.wtkq[key[4:6]])
    end
end

# The reference double loop on a fixture's grid and window.
function contract_reference(model, fixture)
    kpts = GridKpoints(kpoints_grid(fixture.grid))
    eph_reference(model, kpts, kpts, fixture.window, fixture.window)
end

"""
    check_calculator_contract(entries, models; gpu = false, rtol = 1e-10, rtol_gpu = rtol)

Run every entry under every supported order, fixture and setting (`contract_settings(order; gpu)`).
The outputs of every setting must agree with those of the first and with the entry's reference, to
`rtol` (`isapprox`, relative Frobenius norm), and to `rtol_gpu` for the GPU setting. The comparison has teeth only if the largest entry,
perturbed by 1e-7 of itself or swapped with its neighbour, fails it, which is asserted too.
`models` is `(; el, ph)`, the model loaded with `epmat_outer_momentum = "el"` and `"ph"`; the
reference is computed on `models.el`.
"""
function check_calculator_contract(entries, models; gpu = false, rtol = 1e-10, rtol_gpu = rtol)
    for (fname, fixture) in pairs(contract_fixtures())
        ref = contract_reference(models.el, fixture)
        for entry in entries, order in entry.orders
            settings = contract_settings(order; gpu)
            calcs = Dict(name => run_contract(entry, order, models, fixture, setting)
                         for (name, setting) in pairs(settings))
            first_calc = calcs[first(keys(settings))]
            first_out = entry.outputs(first_calc)
            want = entry.outputs(entry.reference(first_calc, ref))
            @testset "$(entry.name), $(nameof(order)), $fname" begin
                for (name, c) in calcs, (key, x) in entry.outputs(c)
                    tol = name === :gpu ? rtol_gpu : rtol
                    @test isapprox(x, first_out[key]; rtol = tol)
                    @test size(x) == size(want[key]) && isapprox(x, want[key]; rtol = tol)
                end
                for (key, x) in first_out
                    j = argmax(abs.(x))
                    y = collect(x); y[j] *= 1 + 1e-7
                    @test !isapprox(y, want[key]; rtol)
                    # Nor does it pass the largest entry trading places with the next one.
                    y = collect(x); j2 = mod1(LinearIndices(y)[j] + 1, length(y))
                    y[j], y[j2] = y[j2], y[j]
                    @test !isapprox(y, want[key]; rtol)
                end
            end
        end
    end
end
