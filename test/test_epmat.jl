using Test
using ElectronPhonon
using OffsetArrays: no_offset_view

@testset "cubicBN epmat" begin
    using ElectronPhonon: WannierObject
    using ElectronPhonon: get_eph_RR_to_Rq!, get_eph_Rq_to_kq!
    using ElectronPhonon: get_eph_RR_to_kR!, get_eph_kR_to_kq!

    # Test fourier transform of electron-phonon matrix elements

    model_ph = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum="ph")
    model_el = _load_model_from_artifacts("cubicBN"; epmat_outer_momentum="el")
    # model_ph_disk = load_model(folder; epmat_on_disk=true, tmpdir=folder, epmat_outer_momentum="ph")
    # model_el_disk = load_model(folder; epmat_on_disk=true, tmpdir=folder_tmp_el, epmat_outer_momentum="el")

    xk = Vec3([0.1, -0.4, 0.7])
    xq = Vec3([0.5, 0.2, -0.5])
    xkq = xk + xq
    window = (5.0, 45.0) .* unit_to_aru(:eV)

    nw = model_ph.nw
    nmodes = model_ph.nmodes
    nband = 7

    # setup electron and phonon states
    ham = get_interpolator(model_ph.el_ham)
    dyn = get_interpolator(model_ph.ph_dyn)
    ph = PhononState(nmodes)
    el_k = ElectronState(nw, nband)
    el_kq = ElectronState(nw, nband)
    set_eigen!(el_k, ham, xk)
    set_window!(el_k, window)
    set_eigen!(el_kq, ham, xkq)
    set_window!(el_kq, window)
    set_eigen!(ph, xq, dyn, model_ph.mass, model_ph.polar_phonon)

    rngk = el_k.rng
    rngkq = el_kq.rng
    @test rngk == 3:8
    @test rngkq == 4:8

    ep_ref = zeros(ComplexF64, (el_kq.nband, el_k.nband, nmodes))

    i = 0
    for fourier_mode in ["normal", "gridopt"]
        for model in [model_ph, model_el]
        # for model in [model_ph, model_ph_disk, model_el, model_el_disk]
            i += 1
            epdata = ElPhData(nw, nmodes, nband)
            epdata.ph = ph
            epdata.el_k = el_k
            epdata.el_kq = el_kq

            @info "$(typeof(model)), epmat_outer_momentum = $(model.epmat_outer_momentum), $fourier_mode"
            epmat = get_interpolator(model.epmat; fourier_mode)
            if model.epmat_outer_momentum == "ph"
                ep_eRpq_obj = WannierObject(model.epmat.irvec_next,
                            zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))
                ep_eRpq = get_interpolator(ep_eRpq_obj; fourier_mode)
                get_eph_RR_to_Rq!(ep_eRpq_obj, epmat, xq, ph.u)
                get_eph_Rq_to_kq!(epdata, ep_eRpq, xk)
            else
                ep_ekpR_obj = WannierObject(model.epmat.irvec_next,
                            zeros(ComplexF64, (nw*epdata.nband_bound*nmodes, length(model.epmat.irvec_next))))
                ep_ekpR = get_interpolator(ep_ekpR_obj; fourier_mode)
                get_eph_RR_to_kR!(ep_ekpR_obj, epmat, xk, no_offset_view(epdata.el_k.u))
                get_eph_kR_to_kq!(epdata, ep_ekpR, xq)
            end

            @test axes(epdata.ep) == (el_kq.rng, el_k.rng, 1:ph.nmodes)
            if i == 1
                ep_ref .= no_offset_view(epdata.ep)
            else
                @test no_offset_view(epdata.ep) ≈ ep_ref
            end
        end
    end
end
