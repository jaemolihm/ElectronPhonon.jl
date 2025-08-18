using ElectronPhonon
using Test
using OffsetArrays: no_offset_view

@testset "velocity" begin
    using ElectronPhonon: get_el_velocity_diag_berry_connection!

    model = _load_model_from_artifacts("cubicBN"; load_epmat = false)
    kpts = kpoints_grid((4, 4, 4)) # cubicBN data is generated using 4*4*4 coarse k grid

    model.el_velocity_mode = :Direct
    els_direct = compute_electron_states(model, kpts, ["velocity"])
    v_direct = reshape(reinterpret(ComplexF64, cat((no_offset_view(el.v) for el in els_direct)..., dims=3)), 3, model.nw, model.nw, kpts.n)

    model.el_velocity_mode = :BerryConnection
    els_berry = compute_electron_states(model, kpts, ["velocity"])
    v_berry = reshape(reinterpret(ComplexF64, cat((no_offset_view(el.v) for el in els_berry)..., dims=3)), 3, model.nw, model.nw, kpts.n)

    # Test whether the full velocity matrix calculated with el_velocity_mode = :Direct and :BerryConnection
    # are similar enough. Use coarse k grid points to reduce interpolation error.
    # Note that the two method does not identical even at the coarse k point because of the error
    # in the finite-difference calculation of A in the BerryConnection method.
    # If the Berry connection contribution is not added, the maximum absolute error is 1.98.
    @test maximum(abs.(v_direct - v_berry)) < 0.4

    # Test diagonal-only calculation for Berry connection method gives the same results with
    # the full velocity matrix calculation.
    # There is no special function for diagonal-only calculation using the direct method, so no test.
    velocity_diag = zeros(3, model.nw)
    ham_R = get_interpolator(model.el_ham_R)
    for ik in 1:kpts.n
        uk = no_offset_view(els_berry[ik].u)
        xk = kpts.vectors[ik]
        get_el_velocity_diag_berry_connection!(velocity_diag, model.nw, ham_R, xk, uk)
        @test velocity_diag ≈ reshape(reinterpret(Float64, els_berry[ik].vdiag), 3, model.nw)
    end
end
