# The generic calculator contract harness. Shared by ElectronPhonon.jl's and MigdalEliashberg.jl's
# `test_calculator_contract.jl`, each with its own list of calculator entries and its own
# golden-value file.
#
# An entry is a NamedTuple `(; name, make, orders, outputs)`: `make()` returns a fresh calculator,
# `orders` the loop orders it supports (`OuterKLoop`, `OuterQLoop`), and `outputs(calc)` a
# `Dict{String, Array}` of what it computed, reduced to gauge-invariant arrays (see
# `contract_multiplet_ids`, `contract_pair_sum`). For every entry, order and fixture the harness
# runs every loop setting of `contract_settings` on the CPU, compares the outputs across the
# settings, and compares them against the golden values recorded from the per-point loops. A test
# file passes `record = (ENV["EP_RECORD_CONTRACT_GOLDEN"] == "1")` so the golden file is regenerated
# by running it with that variable set, which is for a change that is meant to move the numbers.

using Test
using ElectronPhonon
using ElectronPhonon: CPUBackend, OuterKLoop, OuterQLoop, unit_to_aru, run_eph_over_k_and_kq,
    run_eph_over_q_and_k, electron_degen_cutoff, omega_acoustic

# The windowed fixtures of the pb artifact model: nk = 12, E_F +- 0.2 eV (one in-window band per k)
# and nk = 6, E_F - 0.5 / + 3 eV (0 to 3 bands per k, empty windows, windows ending at band nw).
function contract_fixtures()
    eV = unit_to_aru(:eV)
    e_F = 11.68eV
    (; narrow = (; grid = (12, 12, 12), window = (e_F - 0.2eV, e_F + 0.2eV), e_F),
       ragged = (; grid = (6, 6, 6), window = (e_F - 0.5eV, e_F + 3eV), e_F))
end

# The loop settings of each order, all on the CPU: the per-point loop with several thread chunks,
# and the batched loop at outer batch width 1 and > 1 with many small inner tiles. The first is
# the one the golden values are recorded from.
contract_settings(::Type{OuterKLoop}) = (
    per_point = (; nchunks_threads = 4),
    batched_outer1 = (; backend = CPUBackend(), batched = true, nk_outer_batch_max = 1,
                        nq_batch_max = 37),
    batched_outer5 = (; backend = CPUBackend(), batched = true, nk_outer_batch_max = 5,
                        nq_batch_max = 37))
contract_settings(::Type{OuterQLoop}) = (
    per_point = (; nchunks_threads = 4),
    batched = (; backend = CPUBackend(), batched = true, nk_batch_max = 37))

function run_contract(entry, order, models, fixture, setting)
    calc = entry.make()
    common = (; calculators = [calc], window_k = fixture.window, window_kq = fixture.window,
              progress_print_step = 10^9, verbosity = 0)
    if order === OuterKLoop
        run_eph_over_k_and_kq(models.el, fixture.grid, fixture.grid; symmetry = nothing, common...,
                              setting...)
    else
        run_eph_over_q_and_k(models.ph, fixture.grid, fixture.grid; use_symmetry = false,
                             common..., setting...)
    end
    entry.outputs(calc)
end

"""
    contract_multiplet_ids(s; tol = electron_degen_cutoff) -> Vector{Int}

A group id per state of the `BandStates` `s`: states at one k point within `tol` of the next lower
one share an id. A sum over the states of one id (a degenerate multiplet) does not depend on the
eigenvector basis the solver picked inside it, so it is what a golden value can pin across LAPACK
builds and backends.
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

"""
    check_calculator_contract(entries, models; golden_file, record = false, rtol_settings = 1e-10,
                              rtol_golden = 1e-10)

Run every entry under every supported order, fixture and setting. The outputs of all settings must
agree to `rtol_settings` and reproduce the golden arrays in the HDF5 file `golden_file` to
`rtol_golden` (both `isapprox`, relative Frobenius norm); the comparison has teeth only if the
largest entry, perturbed by 1e-7 of itself or swapped with its neighbour, fails it, which is
asserted too. `models` is `(; el, ph)`, the model loaded with `epmat_outer_momentum = "el"` and
`"ph"`. With `record = true` the outputs of the first setting are written to `golden_file` instead
of being compared.
"""
function check_calculator_contract(entries, models; golden_file, record = false,
        rtol_settings = 1e-10, rtol_golden = 1e-10)
    golden = record ? Dict{String, Array}() : read_contract_golden(golden_file)
    for entry in entries, order in entry.orders, (fname, fixture) in pairs(contract_fixtures())
        settings = contract_settings(order)
        results = Dict(name => run_contract(entry, order, models, fixture, setting)
                       for (name, setting) in pairs(settings))
        ref = results[first(keys(settings))]
        @testset "$(entry.name), $(nameof(order)), $fname" begin
            for (name, out) in results, (key, x) in out
                @test isapprox(x, ref[key]; rtol = rtol_settings)
            end
            for (key, x) in ref
                gkey = "$(entry.name)|$(nameof(order))|$fname|$key"
                if record
                    golden[gkey] = collect(x)
                else
                    g = golden[gkey]
                    @test size(x) == size(g) && isapprox(x, g; rtol = rtol_golden)
                    j = argmax(abs.(x))
                    y = collect(x); y[j] *= 1 + 1e-7
                    @test !isapprox(y, g; rtol = rtol_golden)
                    # Nor does it pass the largest entry trading places with the next one.
                    y = collect(x); j2 = mod1(LinearIndices(y)[j] + 1, length(y))
                    y[j], y[j2] = y[j2], y[j]
                    @test !isapprox(y, g; rtol = rtol_golden)
                end
            end
        end
    end
    record && write_contract_golden(golden_file, golden)
    golden
end

# The golden arrays, flat keys `calculator|order|fixture|output`, deflate-compressed.
function write_contract_golden(path, golden)
    ElectronPhonon.HDF5.h5open(path, "w") do f
        for (key, x) in golden
            ElectronPhonon.HDF5.write_dataset(f, key, x; chunk = size(x), deflate = 6)
        end
    end
end

read_contract_golden(path) =
    ElectronPhonon.HDF5.h5open(f -> Dict{String, Array}(k => read(f[k]) for k in keys(f)), path)
