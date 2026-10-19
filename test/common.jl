# Evaluated at the start of every test item with `setup = [Common]`: the artifact-model loader and
# the state comparators shared by several test files.
@testsnippet Common begin
    include("common_models_from_artifacts.jl")
end
