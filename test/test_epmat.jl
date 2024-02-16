using Test
using ElectronPhonon
using OffsetArrays: no_offset_view

@testset "cubicBN epmat" begin
    # Test fourier transform of electron-phonon matrix elements
    BASE_FOLDER = dirname(dirname(pathof(ElectronPhonon)))
    folder = joinpath(BASE_FOLDER, "test", "data_cubicBN")
    folder_tmp_el = joinpath(folder, "tmp_el")
    mkpath(folder_tmp_el)

    model_ph = load_model(folder; epmat_outer_momentum="ph")
    model_el = load_model(folder; epmat_outer_momentum="el")
    model_ph_disk = load_model(folder; epmat_on_disk=true, tmpdir=folder, epmat_outer_momentum="ph")
    model_el_disk = load_model(folder; epmat_on_disk=true, tmpdir=folder_tmp_el, epmat_outer_momentum="el")

    xk = Vec3([0.1, -0.4, 0.7])
    xq = Vec3([0.5, 0.2, -0.5])
    xkq = xk + xq
    window = (5.0, 45.0) .* unit_to_aru(:eV)
    # window = (-Inf, Inf)

    nw = model_ph.nw
    nmodes = model_ph.nmodes
    nband = 7

    # setup electron and phonon states
    ph = PhononState(nmodes)
    el_k = ElectronState(nw, nband)
    el_kq = ElectronState(nw, nband)
    set_eigen!(el_k, model_ph, xk)
    set_window!(el_k, window)  # el_k.nband  = 6
    set_eigen!(el_kq, model_ph, xkq)
    set_window!(el_kq, window) # el_kq.nband = 5
    set_eigen!(ph, model_ph, xq)
    rngk = el_k.rng
    rngkq = el_kq.rng
    ep_ref = zeros(ComplexF64, (el_kq.nband, el_k.nband, nmodes))

    i = 0
    for fourier_mode in ["normal", "gridopt"]
        for model in [model_ph, model_ph_disk, model_el, model_el_disk]
            i += 1
            epdata = ElPhData(nw, nmodes, nband)
            epdata.ph = ph
            epdata.el_k = el_k
            epdata.el_kq = el_kq

            @info "$(typeof(model)), epmat_outer_momentum = $(model.epmat_outer_momentum), $fourier_mode"
            if model.epmat_outer_momentum == "ph"
                epobj_eRpq = WannierObject(model.epmat.irvec_next,
                            zeros(ComplexF64, (nw*nw*nmodes, length(model.epmat.irvec_next))))
                @time ElectronPhonon.get_eph_RR_to_Rq!(epobj_eRpq, model.epmat, xq, ph.u; fourier_mode)
                ElectronPhonon.get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk; fourier_mode)
            else
                epobj_ekpR = WannierObject(model.epmat.irvec_next,
                            zeros(ComplexF64, (nw*epdata.nband_bound*nmodes, length(model.epmat.irvec_next))))
                @time ElectronPhonon.get_eph_RR_to_kR!(epobj_ekpR, model.epmat, xk, no_offset_view(epdata.el_k.u); fourier_mode)
                ElectronPhonon.get_eph_kR_to_kq!(epdata, epobj_ekpR, xq; fourier_mode)
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
