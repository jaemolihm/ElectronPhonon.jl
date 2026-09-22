using ElectronPhonon
using Test

# Test group, selected by `Pkg.test(; test_args=["plotting"])`. "plotting" is the only group that
# loads PyPlot; "core" is everything else and is the default because it is safe to run
# multithreaded. "all" runs both in one process, which is not safe with more than one thread: a
# PyCall finalizer running on a Julia worker thread crashes the process (#49).
const group = isempty(ARGS) ? "core" : only(ARGS)
group in ("all", "core", "plotting") || error("Unknown test group $group; expected one of \"all\", \"core\", \"plotting\"")
@info "Test group: $group"

include("common_models_from_artifacts.jl")

@time @testset "ElectronPhonon.jl" begin
    if group in ("all", "core")
        # Basic tests
        include("test_occupation.jl")
        include("test_smearing.jl")
        include("test_kpoints.jl")
        include("test_band_states.jl")  # (k, band) selection: indexing, filter_states, symmetry stars
        include("test_iq_build.jl")  # GPU outer-k loop's `iq` index build (host arithmetic, CPU-only)
        include("test_symmetry.jl")
        include("test_wannier.jl")
        include("test_holstein_model.jl")  # analytic Holstein Model builder (no artifacts needed)
        include("test_gpu.jl")  # skips gracefully when CUDA is unavailable
        include("test_backend_alloc.jl")  # alloc_zeros / to_device_copy / is_host; GPU arm skips w/o CUDA
        include("test_mpi_wrappers.jl")  # the mpi_* wrappers on the `comm === nothing` serial path
        include("test_diagonalize.jl")
        include("test_iterativesolvers.jl")
        # include("test_symmetry_operator.jl")
        # include("test_unfold.jl")
        include("test_velocity.jl")
        include("test_filter_electron_states.jl")  # unified filter primitive: MPI (COMM_SELF) + shift
        include("test_epmat.jl")
        include("test_ElectronState.jl")
        include("test_eigenpairs.jl")  # shared full-band eigenpair cache; GPU part skips w/o CUDA
        include("test_eph_precomputed_states.jl")  # the eigenpair cache through the e-ph drivers
        include("test_eph_incommensurate_grids.jl")  # run_eph_over_k_and_kq's per-(k,q) phonon solve
        include("test_eph_window_scatter.jl")  # calculator-facing scatter; GPU part skips w/o CUDA
        include("test_high_symmetry_kpath.jl")
        include("test_postprocess_compute.jl")
        # include("test_check_model_symmetry.jl")
        include("test_screening.jl")
        include("test_calculator_guide.jl")  # runs the docs/writing_a_calculator.md example
        include("test_calculator_contract.jl")  # supports trait + driver fail-early checks

        # Integration tests
        include("test_cubicBN_eigenvalues.jl")
        include("test_cubicBN_spectral.jl")
        include("test_cubicBN_selfen.jl")
        # include("test_transport.jl")

        # Boltzmann routines
        include("boltzmann/test_hdf5.jl")
        include("boltzmann/test_BTStates.jl")  # states_index_map, incl. the symmetry-star keys
        include("boltzmann/test_QMEVector.jl")
        include("boltzmann/test_gpu_boltzmann_calculator.jl")  # GPU BTE scatter; GPU part skips w/o CUDA
        include("boltzmann/test_multigrid_weights.jl")  # multigrid k-sampling: pure-BZ quadrature check
        include("boltzmann/test_multigrid_transport.jl")  # multigrid BTE transport; GPU part skips w/o CUDA
        # include("boltzmann/test_QMEModel.jl")
        # include("boltzmann/test_covariant_derivative.jl")
        # include("boltzmann/test_el_constant_rta.jl")
        # include("boltzmann/test_el_transport_semiconductor.jl")
        # include("boltzmann/test_el_transport_metal.jl")
        # include("boltzmann/test_el_master_equation.jl")
        # include("boltzmann/test_el_transport_screening.jl")
        # include("boltzmann/test_el_ac_conductivity.jl")
        # include("boltzmann/test_el_hall_conductivity.jl")
        # include("boltzmann/test_el_transport_finite_efield.jl")
    end
    if group in ("all", "plotting")
        include("test_plotting.jl")
    end
end
