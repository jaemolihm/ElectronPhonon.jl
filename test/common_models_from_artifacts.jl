using Pkg
using OffsetArrays: no_offset_view

function _artifact_folder(prefix)
    toml = Pkg.Artifacts.find_artifacts_toml(@__DIR__)
    Pkg.Artifacts.ensure_artifact_installed(prefix, toml)
end

# Download artifacts that contain large data files if needed, and return the model.
function _load_model_from_artifacts(prefix; kwargs...)
    folder = _artifact_folder(prefix)
    load_model_from_epw_new(folder, "temp", prefix; kwargs...)
end

# Every field `compute_electron_states` fills, compared with `==`. `v`/`rbar`/`occupation` are
# window views, so take them through `no_offset_view` (their axes are covered by `rng`). It lives
# here, not in one test file, because the eigenpair-cache tests assert that a cache leaves a run
# bitwise unchanged: a second copy of this comparator would drift from `ElectronState`'s field list
# and then compare fewer fields while still passing.
function _electron_state_equal(a::ElectronState, b::ElectronState)
    a.xk == b.xk && a.e_full == b.e_full && a.u_full == b.u_full && a.nband == b.nband &&
        a.rng == b.rng && a.vdiag == b.vdiag &&
        no_offset_view(a.v) == no_offset_view(b.v) &&
        no_offset_view(a.rbar) == no_offset_view(b.rbar)
end
