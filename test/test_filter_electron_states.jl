using Test
using ElectronPhonon
const EP = ElectronPhonon
using ElectronPhonon: filter_electron_states, band_range, state_xks

# Unit tests for the unified `filter_electron_states` primitive: the MPI construction (redistributing
# the kept k-points and per-k band ranges together so they stay aligned) and the `shift` kwarg on the
# NTuple grid path. The MPI test uses `MPI.COMM_SELF` (single rank), which exercises the
# gather/scatter redistribution path and asserts it reproduces the serial result; a genuine
# multi-rank alignment check needs `mpiexec -n N` (see the repo's test_kpoints TODO).

isdefined(@__MODULE__, :_load_model_from_artifacts) ||
    include(joinpath(@__DIR__, "common_models_from_artifacts.jl"))

@testset "filter_electron_states" begin
    model = _load_model_from_artifacts("pb")
    eV = EP.unit_to_aru(:eV)
    ef = 11.594123 * eV
    window = (ef - 0.3eV, ef + 0.3eV)
    sym = model.symmetry

    # Set of (k-vector, band) states, order-independent (MPI gather/scatter may reorder k).
    stateset(s) = Set((state_xks(s)[i], s.ibands[i]) for i in 1:s.n)

    @testset "serial vs MPI (COMM_SELF) — same state set + nstates_base" begin
        MPI = EP.MPI
        MPI.Initialized() || MPI.Init()
        for symm in (nothing, sym)
            s_ser = filter_electron_states((8, 8, 8), model.nw, model.el_ham, window; symmetry = symm)
            s_mpi = filter_electron_states((8, 8, 8), model.nw, model.el_ham, window;
                symmetry = symm, mpi_comm = MPI.COMM_SELF)
            @test s_mpi.kpts.n == s_ser.kpts.n
            @test s_mpi.n == s_ser.n
            @test stateset(s_mpi) == stateset(s_ser)
            @test s_mpi.nstates_base ≈ s_ser.nstates_base
            @test band_range(s_mpi) == band_range(s_ser)
        end
    end

    @testset "shift round-trip (NTuple grid path)" begin
        shift = (1, 1, 1) ./ 16          # sub-cell shift, as run_transport's k+q path uses
        sel = filter_electron_states((8, 8, 8), model.nw, model.el_ham, (-Inf, Inf); shift)
        ref = EP.GridKpoints(EP.kpoints_grid((8, 8, 8); shift))
        @test sel.kpts.n == ref.n
        @test Set(sel.kpts.vectors) == Set(ref.vectors)
        @test band_range(sel) == 1:model.nw   # trivial window keeps all bands
        # shift is ignored for a prebuilt Kpoints input (only the NTuple path applies it)
        sel2 = filter_electron_states(ref, model.nw, model.el_ham, (-Inf, Inf); shift = (0, 0, 0))
        @test Set(sel2.kpts.vectors) == Set(ref.vectors)
    end

    @testset "shift + symmetry incompatible" begin
        @test_throws ErrorException filter_electron_states((8, 8, 8), model.nw, model.el_ham, window;
            symmetry = sym, shift = (1, 1, 1) ./ 16)
    end
end
