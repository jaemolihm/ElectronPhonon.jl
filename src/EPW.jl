# __precompile__(true)

module EPW

using TimerOutputs

export Mat3
export Vec3
export Vec2

include("common/timer.jl")
include("common/types.jl")
include("common/units.jl")
include("common/constants.jl")
include("common/utils.jl")
include("common/mpi.jl")
include("common/split_evenly.jl")
include("common/reshape.jl")
include("common/kpoints.jl")
include("common/symmetry.jl")
include("common/fermi_energy.jl")
include("common/AllocatedLAPACK.jl")

include("external/iterativesolvers.jl")

include("gridopt.jl")
include("fourier.jl")
include("fourier_disk.jl")
include("diagonalize.jl")
include("longrange.jl")
include("model.jl")
include("filter.jl")
include("WanToBloch.jl")
include("electron_state.jl")
include("phonon_state.jl")
include("elphdata.jl")
include("screening_lindhard.jl")
include("selfenergy_electron.jl")
include("selfenergy_phonon.jl")
include("spectral_phonon.jl")
include("transport_electron.jl")
include("compute_states.jl")
include("run_eigenvalues.jl")
include("run_electron_phonon.jl")

include("boltzmann/BTData.jl")
include("boltzmann/BTStates.jl")
include("boltzmann/BTScatterings.jl")

# Quantum master equation transport
include("boltzmann/Vertex.jl")
include("boltzmann/QMEStates.jl")
include("boltzmann/QMEVector.jl")
include("boltzmann/QMEModel.jl")
include("boltzmann/QMEScattering.jl")
include("boltzmann/unfold.jl")
include("boltzmann/covariant_derivative.jl")
include("boltzmann/run_coherence.jl")
include("boltzmann/electron_master_equation.jl")

include("boltzmann/electron_serta.jl")
include("boltzmann/constant_rta.jl")
include("boltzmann/interpolate_energy.jl")
include("boltzmann/doublegrid.jl")
include("boltzmann/gamma_adaptive.jl")
include("boltzmann/electron_lbte.jl")
include("run_transport.jl")
include("run_transport_subgrid_q.jl")
include("boltzmann/electron_transport_linear.jl")
include("boltzmann/electron_transport_hall.jl")

include("postprocess/plot_bandstructure.jl")
include("postprocess/plot_electron_phonon.jl")
include("postprocess/plot_decay.jl")
include("postprocess/check_model_symmetry.jl")

end
