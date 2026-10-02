using Test
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop

# `run_eph_over_k_and_kq` builds its phonons on the q grid the k and k+q grids span, so it needs the
# two grids commensurate, and refuses other grids at entry. No caller in this repo or in
# MigdalEliashberg.jl passes incommensurate grids (the previous release solved the phonons per pair
# there, a branch only this file reached); a q off the k grid is the outer-q loop's case.

isdefined(@__MODULE__, :_load_model_from_artifacts) || include("common_models_from_artifacts.jl")

struct _NeverRunCalc <: AbstractCalculator end
ElectronPhonon.supports(::_NeverRunCalc, ::Type{OuterKLoop}) = true

@testset "run_eph_over_k_and_kq refuses incommensurate grids" begin
    model = _load_model_from_artifacts("pb"; epmat_outer_momentum = "el")
    # (2, 2, 2) and (3, 3, 3): neither divides the other. The refusal comes before any state is
    # built, so the calculator has no methods beyond `supports`.
    for (kgrid, kqgrid) in (((2, 2, 2), (3, 3, 3)), ((3, 3, 3), (2, 2, 2)))
        @test_throws "commensurate k and k+q grids" ElectronPhonon.run_eph_over_k_and_kq(model,
            kgrid, kqgrid; calculators = [_NeverRunCalc()], symmetry = nothing, verbosity = 0)
    end
    # The same grid, one dividing the other, and a prebuilt selection's grid are accepted by the
    # check (it reads the grid of every input kind).
    @test ElectronPhonon._check_run(OuterKLoop(), model, ElectronPhonon.CPUBackend(), [_NeverRunCalc()],
        (2, 2, 2), kpoints_grid((4, 4, 4)), [:u]; energy_conservation = (:None, 0.0),
        covariant_derivative_of_g = false, eph_phonon_basis = :eigenmode, fourier_mode = "gridopt",
        precompute_el_kq = false, screening_params = nothing, mpi_comm_k = nothing,
        el_kq_eigenpairs = nothing) === nothing
end
