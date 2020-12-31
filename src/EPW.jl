# __precompile__(true)

module EPW

using TimerOutputs

export Mat3
export Vec3
export Vec2

include("common/types.jl")
include("common/units.jl")
include("common/constants.jl")
include("common/timer.jl")
include("common/utils.jl")
include("common/mpi.jl")
include("common/split_evenly.jl")
include("common/kpoints.jl")

include("gridopt.jl")
include("fourier.jl")
include("fourier_disk.jl")
include("diagonalize.jl")
include("filter.jl")
include("elphdata.jl")
include("selfenergy_electron.jl")
include("selfenergy_phonon.jl")
include("model.jl")

end
