using Pkg

function _artifact_folder(prefix)
    toml = Pkg.Artifacts.find_artifacts_toml(@__DIR__)
    Pkg.Artifacts.ensure_artifact_installed(prefix, toml)
end

# Download artifacts that contain large data files if needed, and return the model.
function _load_model_from_artifacts(prefix; kwargs...)
    folder = _artifact_folder(prefix)
    load_model_from_epw_new(folder, "temp", prefix; kwargs...)
end
