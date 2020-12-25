# __precompile__(true)

module EPW

export Mat3
export Vec3
export Vec2

include("common/types.jl")
include("common/units.jl")
include("common/constants.jl")
include("common/utils.jl")
include("common/split_evenly.jl")

include("gridopt.jl")
include("fourier.jl")
include("fourier_disk.jl")
include("diagonalize.jl")
include("elphdata.jl")
include("electron_selfenergy.jl")

end
