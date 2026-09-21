using Test
using ElectronPhonon
using ElectronPhonon: AbstractCalculator, OuterKLoop, EPData, OuterIteration,
    normalize_kpoint_coordinate

# `run_eph_over_k_and_kq` splits on whether the k and k+q meshes are commensurate. When they are,
# the q-point set is built up front (`combine_kpoint_grids`) and the phonon states with it; when
# they are not, there is no q-point set at all and the loop solves the phonon state of every (k, q)
# pair itself. Every other caller in this repo -- tests and drivers alike -- passes equal or
# commensurate grids, so this file is what covers the second branch.

# Records the phonon state the loop built at each (k, k+q) pair, and the momentum it was built at
# (`set_eigen!` writes `ph.xq`). `iq_is_nothing` is the branch's own signal: without a q-point set
# there is no index into one, so the payload carries `iq === nothing`. It starts `false`, so a pair
# the loop never visited fails the same assertion a wrongly-indexed payload does.
struct _IncommensurateProbe{FT} <: AbstractCalculator
    npayload      :: Threads.Atomic{Int}
    iq_is_nothing :: Matrix{Bool}              # (nk, nkq)
    xq            :: Matrix{Vec3{FT}}          # (nk, nkq)
    e             :: Array{FT, 3}              # (nmodes, nk, nkq)
    u             :: Array{Complex{FT}, 4}     # (nmodes, nmodes, nk, nkq)
    dipole        :: Array{Complex{FT}, 3}     # (nmodes, nk, nkq)
end
_IncommensurateProbe(FT, nmodes, nk, nkq) = _IncommensurateProbe{FT}(
    Threads.Atomic{Int}(0), zeros(Bool, nk, nkq), zeros(Vec3{FT}, nk, nkq),
    zeros(FT, nmodes, nk, nkq), zeros(Complex{FT}, nmodes, nmodes, nk, nkq),
    zeros(Complex{FT}, nmodes, nk, nkq))

ElectronPhonon.supports(::_IncommensurateProbe, ::Type{OuterKLoop}) = true
ElectronPhonon.supports(::_IncommensurateProbe, ::Type{EPData}) = true
ElectronPhonon.calculator_begin!(::_IncommensurateProbe, ::OuterIteration, ctx) = nothing
ElectronPhonon.calculator_end!(::_IncommensurateProbe, ::OuterIteration, ctx) = nothing
ElectronPhonon.setup_calculator!(c::_IncommensurateProbe, backend, mode, kpts, qpts, el_states;
                                 kwargs...) = c
ElectronPhonon.postprocess_calculator!(c::_IncommensurateProbe; kwargs...) = c
function ElectronPhonon.run_calculator!(c::_IncommensurateProbe, d::EPData, ctx)
    ik, ikq = d.ik, d.ikq
    Threads.atomic_add!(c.npayload, 1)
    c.iq_is_nothing[ik, ikq] = d.iq === nothing
    c.xq[ik, ikq] = d.epstate.ph.xq
    c.e[:, ik, ikq] .= d.epstate.ph.e
    c.u[:, :, ik, ikq] .= d.epstate.ph.u
    c.dipole[:, ik, ikq] .= d.epstate.ph.eph_dipole_coeff
    c
end

# `(2, 2, 2)` and `(3, 3, 3)` are the smallest pair of grids neither of which divides the other, so
# this is the cheapest input that reaches the branch: 2^3 * 3^3 = 216 (k, q) pairs.
@testset "run_eph_over_k_and_kq on incommensurate grids" begin
    kgrid, kqgrid = (2, 2, 2), (3, 3, 3)

    # Pb is non-polar and cubicBN is polar. The branch calls `set_eph_dipole_coeff!` under
    # `skip_eph` rather than under `use_polar_dipole`, so both models run it, but only on cubicBN
    # does it produce anything: the Pb arm pins that it leaves the coefficients zero, the cubicBN
    # arm is where the comparison below has content.
    for prefix in ("pb", "cubicBN")
        @testset "$prefix" begin
            model = _load_model_from_artifacts(prefix; epmat_outer_momentum = "el")
            polar = model.use_polar_dipole
            @test polar == (prefix == "cubicBN")

            nk, nkq = prod(kgrid), prod(kqgrid)
            probe = _IncommensurateProbe(Float64, model.nmodes, nk, nkq)
            res = ElectronPhonon.run_eph_over_k_and_kq(model, kgrid, kqgrid;
                calculators = [probe], symmetry = nothing, fourier_mode = "gridopt",
                progress_print_step = 10^9, verbosity = 0)

            # The branch was taken: no q-point set was built, hence no precomputed phonon states,
            # and every payload came without a q index.
            @test res.qpts === nothing
            @test res.ph_save === nothing
            @test all(probe.iq_is_nothing)

            # One payload per (k, k+q) pair, no pair visited twice, over the full grids.
            @test (res.kpts.n, length(res.el_kq_save)) == (nk, nkq)
            @test probe.npayload[] == nk * nkq

            # The momentum each phonon was solved at is the folded k+q - k of its pair.
            xq_expected = [normalize_kpoint_coordinate(
                                res.el_kq_save[ikq].xk - res.kpts.vectors[ik] .+ 1/2) .- 1/2
                           for ik in 1:nk, ikq in 1:nkq]
            @test probe.xq == xq_expected

            # Reference: the phonon states at those same q points from `compute_phonon_states`,
            # which is what the commensurate branch precomputes. It shares `set_eigen!` with the
            # branch under test, so what this pins is the branch's wiring -- that it hands the
            # solver the same model data and momentum the commensurate path does -- not the solver.
            ref = compute_phonon_states(model, Kpoints(vec(probe.xq)),
                ["eigenvalue", "eigenvector", "eph_dipole_coeff"]; fourier_mode = "gridopt")
            @test reshape(probe.e, model.nmodes, :) == stack(ph -> ph.e, ref)
            @test reshape(probe.u, model.nmodes, model.nmodes, :) == stack(ph -> ph.u, ref)
            @test reshape(probe.dipole, model.nmodes, :) == stack(ph -> ph.eph_dipole_coeff, ref)

            # Vacuity guard on the dipole comparison: zero on both sides for a non-polar model.
            @test (maximum(abs, probe.dipole) > 0) == polar
        end
    end
end
