# Builders for the whole-run shared device stacks `ElKDeviceStacks` (defined in AbstractCalculator.jl)
# that the outer-k GPU loop (`run_eph_over_k_and_kq`, `use_gpu`) exposes to batched calculators via
# `LoopContext.el_k_stacks` (see `required_el_k_device_stacks`). Kept next to `ElKDeviceStacks` rather
# than inline in the driver: infrastructure, not loop logic.

# Per-(physical band, k) energy grid `(nw, nk)`, zero outside the window. `el_states[k].e` is an
# OffsetArray over that k's in-window band range; a calculator's flattened per-state view (gathered
# through its `(iband, ik)` map) only ever reads in-window entries, so out-of-window slots stay 0.
function _band_energy_stack(backend, el_states, nw::Integer, ::Type{FT}) where {FT}
    nk = length(el_states)
    host = zeros(FT, nw, nk)
    for (k, el) in enumerate(el_states)
        el.nband == 0 && continue
        @inbounds for ib in el.rng
            host[ib, k] = el.e[ib]
        end
    end
    copyto!(alloc(backend, FT, nw, nk), host)
end

# Build only the requested stacks (union over calculators); undeclared quantities stay `nothing`.
# Returns `nothing` when nothing is requested, so `LoopContext.el_k_stacks` is `nothing` in that
# (common) case.
function _build_el_k_device_stacks(backend, syms, el_k_save, el_kq_save, kpts, kqpts, nw, ::Type{FT}) where {FT}
    isempty(syms) && return nothing
    e_k  = :e_k  in syms ? _band_energy_stack(backend, el_k_save, nw, FT)  : nothing
    e_kq = :e_kq in syms ? _band_energy_stack(backend, el_kq_save, nw, FT) : nothing
    wtk  = :wtk  in syms ? copyto!(alloc(backend, FT, kpts.n),  collect(FT, kpts.weights))  : nothing
    wtq  = :wtq  in syms ? copyto!(alloc(backend, FT, kqpts.n), collect(FT, kqpts.weights)) : nothing
    ElKDeviceStacks(e_k, e_kq, wtk, wtq)
end
