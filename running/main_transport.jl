using PrettyPrint
using Printf
# using PyPlot
using StaticArrays
using BenchmarkTools
using NPZ
using PyCall
using Profile
using LinearAlgebra
using Base.Threads
using Distributed
using Revise
BLAS.set_num_threads(1)

@everywhere begin
    using LinearAlgebra
    using Base.Threads
    using Revise
    push!(LOAD_PATH, "/home/jmlim/julia_epw/EPW.jl")
    using EPW
    using EPW.WanToBloch
end

using MPI
import EPW: mpi_split_iterator, mpi_bcast, mpi_gather, mpi_sum!
world_comm = EPW.mpi_world_comm()

folder = "/home/jmlim/julia_epw/silicon_mobility"
window_max = 7.0 * unit_to_aru(:eV)
window_min = 6.2 * unit_to_aru(:eV)
window = (window_min, window_max)

model = load_model(folder)
# model = load_model(folder, true, "/home/jmlim/julia_epw/tmp")

nkf = [15, 15, 15]
nqf = [15, 15, 15]

# Do not distribute k points
kpoints, ib_min, ib_max = filter_kpoints_grid(nkf..., model.nw, model.el_ham, window)

# Distribute q points
qpoints = generate_kvec_grid(nqf..., world_comm)
qpoints = filter_qpoints(qpoints, kpoints, model.nw, model.el_ham, window)


# Fourier transform electron-phonon matrix from (Re, Rp) -> (Re, q)
function fourier_eph(model::EPW.ModelEPW, kpoints::EPW.Kpoints, qpoints::EPW.Kpoints;
        fourier_mode="normal",
        window=(-Inf,Inf),
        iband_min=1,
        iband_max=model.nw,
        transport_params::TransportParams,
    )

    nw = model.nw
    nmodes = model.nmodes
    nk = kpoints.n
    nq = qpoints.n
    nband = iband_max - iband_min + 1

    epdatas = [ElPhData(Float64, nw, nmodes, nband) for i=1:nthreads()]
    for epdata in epdatas
        epdata.iband_offset = iband_min - 1
    end

    # Compute and save electron matrix elements at k
    ek_full_save = zeros(Float64, nw, nk)
    uk_full_save = Array{ComplexF64,3}(undef, nw, nw, nk)
    vdiagk_save = zeros(Float64, 3, nband, nk)

    Threads.@threads :static for ik in 1:nk
        epdata = epdatas[threadid()]
        xk = kpoints.vectors[ik]

        get_el_eigen!(epdata, "k", model.el_ham, xk, fourier_mode)
        skip_k = epdata_set_window!(epdata, "k", window)
        get_el_velocity_diag!(epdata, "k", model.el_ham_R, xk, fourier_mode)

        # Save matrix elements at k for reusing
        ek_full_save[:, ik] .= epdata.ek_full
        uk_full_save[:, :, ik] .= epdata.uk_full
        vdiagk_save[:, :, ik] .= epdata.vdiagk
    end # ik

    # Compute chemical potential
    μ = transport_get_μ(ek_full_save, kpoints.weights, transport_params)
    @info @sprintf "T = %.1f" transport_params.T / unit_to_aru(:K)
    @info @sprintf "n = %.1e cm^-3" transport_params.n / (model.volume/unit_to_aru(:cm)^3)
    @info @sprintf "μ = %.4f eV" μ / unit_to_aru(:eV)

    omega_save = zeros(nmodes, nq)
    omegas = zeros(nmodes)
    u_ph = zeros(ComplexF64, (nmodes, nmodes))

    # E-ph matrix in electron Wannier, phonon Bloch representation
    epobj_eRpq = WannierObject(model.el_ham.nr, model.el_ham.irvec,
                zeros(ComplexF64, (nw*nw*nmodes, model.el_ham.nr)))

    for iq in 1:nq
        if mod(iq, 100) == 0 && mpi_isroot()
            @info "iq = $iq"
        end
        xq = qpoints.vectors[iq]

        # Phonon eigenvalues
        get_ph_eigen!(omegas, u_ph, model.ph_dyn, model.mass, xq, fourier_mode)
        omega_save[:, iq] .= omegas

        get_eph_RR_to_Rq!(epobj_eRpq, model.epmat, xq, u_ph, fourier_mode)

        Threads.@threads :static for ik in 1:nk
        # for ik in 1:nk
            tid = Threads.threadid()
            epdata = epdatas[tid]
            # phself = phselfs[tid]

            # println("$tid $ik")
            xk = kpoints.vectors[ik]
            xkq = xk + xq

            epdata.wtk = kpoints.weights[ik]
            epdata.wtq = qpoints.weights[iq]
            epdata.omega .= omegas

            # Use saved data for electron eigenstate at k.
            epdata.ek_full .= @view ek_full_save[:, ik]
            epdata.uk_full .= @view uk_full_save[:, :, ik]
            epdata.vdiagk .= @view vdiagk_save[:, :, ik]

            get_el_eigen!(epdata, "k+q", model.el_ham, xkq, fourier_mode)

            # Set energy window, skip if no state is inside the window
            skip_k = epdata_set_window!(epdata, "k", window)
            skip_kq = epdata_set_window!(epdata, "k+q", window)
            if skip_k || skip_kq
                continue
            end

            get_el_velocity_diag!(epdata, "k+q", model.el_ham_R, xkq, fourier_mode)
            get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk, fourier_mode)

            # Now, we are done with matrix elements. All data saved in epdata.

            # Calculate physical quantities.

            # compute_electron_selfen!(elself, epdata, ik;
            #     efermi=efermi, temperature=temperature, smear=degaussw)
            # compute_phonon_selfen!(phself, epdata, iq;
            #     efermi=efermi, temperature=temperature, smear=degaussw)
        end # ik
    end # iq

    # ph_imsigma = sum([phself.imsigma for phself in phselfs])
    #
    # # Average over degenerate states
    # el_imsigma_avg = average_degeneracy(elself.imsigma, ek_full_save[iband_min:iband_max, :])
    # ph_imsigma_avg = average_degeneracy(ph_imsigma, omega_save)
    #
    # (ek=ek_full_save, omega=omega_save,
    # el_imsigma=el_imsigma_avg, ph_imsigma=ph_imsigma_avg,)
end

transport_params = TransportParams(T = 300.0 * unit_to_aru(:K),
                n = 1.0e15 * (model.volume / unit_to_aru(:cm)^3),
                degaussw = 50.0 * unit_to_aru(:meV),
                carrier_type = "e",
                nband_valence = 4,
                spin_degeneracy = 2)

# Electron-phonon coupling
@time output = fourier_eph(model, kpoints, qpoints,
    fourier_mode="gridopt",
    window=window,
    iband_min=ib_min,
    iband_max=ib_max,
    transport_params=transport_params,
)
