# Adaptively sample q points around Gamma.
# Compute matrix elements by using only the long-range part and the matrix element at q=0.
# This approximation reduces the computational cost from O(Nq*Nk) to O(Nq+Nk).

using LinearAlgebra

export run_gamma_adaptive

"""
    run_gamma_adaptive(kpts::GridKpoints{FT}, qpts_original::GridKpoints{FT}, model, params,
    inv_τ_original, window_k, window_kq, nband, nband_ignore, adaptive_subgrid, adaptive_max_iter,
    adaptive_rtol; fourier_mode="gridopt") where {FT}
"""
function run_gamma_adaptive(kpts::GridKpoints{FT}, qpts_original::GridKpoints{FT}, model,
    params, inv_τ_original, window_k, window_kq, nband, nband_ignore, adaptive_subgrid,
    adaptive_max_iter, adaptive_rtol; fourier_mode="gridopt") where {FT}

    model.epmat_outer_momentum != "ph" && error("model.epmat_outer_momentum must be ph")

    nk = kpts.n
    nq = qpts_original.n
    nw = model.nw
    nmodes = model.nmodes

    # 1. Compute electron states and matrix elements at k.
    mpi_isroot() && println("Computing electron states at k")
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_k, nband, nband_ignore; fourier_mode)
    el_i, imap_el_k = electron_states_to_BTStates(el_k_save, kpts)
    g_gamma_save = gamma_adaptive_compute_g_gamma(model, kpts, el_k_save, nband, nband_ignore; fourier_mode)

    # 2. Compute lifetime for original q
    @time inv_τ_subgrid_old = gamma_adaptive_compute_lifetime(kpts, qpts_original, model, el_i, g_gamma_save, params, window_kq, nband, nband_ignore);

    inv_τ_total = copy(inv_τ_original);
    inv_τ_total_save = zeros(size(inv_τ_total)..., adaptive_max_iter+1);
    inv_τ_total_save[:, :, 1] .= inv_τ_total;

    for iter in 1:adaptive_max_iter
        # 3. Find list of q points to make subgrid
        qpts = EPW.kpoints_create_subgrid(qpts_original, adaptive_subgrid)
        indmap = sortperm(qpts)
        sort!(qpts)
        iq_subgrid_to_grid = repeat(1:qpts_original.n, inner=prod(adaptive_subgrid))[indmap]

        # 4. Compute lifetime for subgrid q, map to original q
        inv_τ_subgrid = gamma_adaptive_compute_lifetime(kpts, qpts, model, el_i, g_gamma_save, params, window_kq, nband, nband_ignore);

        inv_τ_subgrid_new = zero(inv_τ_subgrid_old);
        @views for iq in 1:qpts.n
            inv_τ_subgrid_new[:, iq_subgrid_to_grid[iq], :] .+= inv_τ_subgrid[:, iq, :]
        end

        inv_τ_diff = inv_τ_subgrid_new .- inv_τ_subgrid_old

        # 5. Determine which q points to keep in the next iteration by relative error
        @views error_original_grid = zero(inv_τ_diff[:, :, 1])
        @views for iT in 1:size(inv_τ_total, 2)
            error_original_grid .= max.(error_original_grid, abs.(inv_τ_diff[:, :, iT] ./ inv_τ_total[:, iT]))
        end
        iq_original_selected = findall(dropdims(maximum(error_original_grid, dims=1), dims=1) .> adaptive_rtol)

        # 6. Update qpts_original as the subgrid q points that should be further subdivided
        iq_subgrid_keep = map(iq -> iq ∈ iq_original_selected, iq_subgrid_to_grid)
        qpts_original = EPW.get_filtered_kpoints(qpts, iq_subgrid_keep)
        inv_τ_subgrid_old = inv_τ_subgrid[:, iq_subgrid_keep, :]
        inv_τ_total .+= dropdims(sum(inv_τ_diff, dims=2), dims=2)

        inv_τ_total_save[:, :, iter+1] .= inv_τ_total;

        if qpts_original.n == 0
            @info "Converged, iter = $iter"
            break
        else
            @info "Not converged, ngrid = $(qpts.ngrid)"
        end
    end

    # inv_τ_total
    inv_τ_total_save
end

# Internals

function gamma_adaptive_compute_lifetime(kpts, qpts, model, el_i, g_gamma_save, params, window_kq, nband, nband_ignore; do_print=false)
    # Map k and q points to k+q points
    mpi_isroot() && println("Finding the list of k+q points")
    kqpts = add_two_kpoint_grids(kpts, qpts, +, qpts.ngrid)
    sort!(kqpts)

    println("MPI-k rank $(mpi_myrank(nothing)), Number of k   points = $(kpts.n)")
    println("MPI-k rank $(mpi_myrank(nothing)), Number of k+q points = $(kqpts.n)")
    println("MPI-k rank $(mpi_myrank(nothing)), Number of q   points = $(qpts.n)")

    mpi_isroot() && println("Computing phonon states at q")
    ph_save = compute_phonon_states(model, qpts, ["eigenvalue", "eigenvector", "velocity_diagonal", "eph_dipole_coeff"], "gridopt");

    mpi_isroot() && println("Computing electron states at k+q")
    el_kq_save = compute_electron_states(model, kqpts, ["eigenvalue", "eigenvector", "velocity_diagonal"], window_kq, nband, nband_ignore, fourier_mode="gridopt");

    el_f, imap_el_kq = electron_states_to_BTStates(el_kq_save, kqpts)
    ph, imap_ph = phonon_states_to_BTStates(ph_save, qpts)

    # Compute chemical potential
    bte_compute_μ!(params, el_i; do_print)

    inv_τ_q = zeros(Float64, el_i.n, qpts.n, length(params.Tlist));
    inv_τ_single = zeros(Float64, length(params.Tlist));
    nband_kq = el_f.nband

    mpi_isroot() && println("Computing inverse lifetime")
    for ind_el_i in 1:el_i.n
        mpi_isroot() && mod(ind_el_i, 1000) == 0 && println("State $ind_el_i / $(el_i.n)")
        xk = el_i.xks[ind_el_i]
        iband_k = el_i.iband[ind_el_i]
        ik = xk_to_ik(xk, kpts)
        for ind_ph in 1:ph.n
            xq = ph.xks[ind_ph]
            imode = ph.iband[ind_ph]
            iq = xk_to_ik(xq, qpts)
            xkq = xk + xq
            ikq = xk_to_ik(xkq, kqpts)

            for iband_kq in 1:nband_kq
                ind_el_f = imap_el_kq[iband_kq, ikq]
                ind_el_f == 0 && continue

                g = g_gamma_save[iband_kq, iband_k, imode, ik]
                if iband_k == iband_kq
                    g += ph_save[iq].eph_dipole_coeff[imode]
                end
                mel = abs2(g) / 2 / ph.e[ind_ph]

                @views for sign_ph in (-1, 1)
                    s = (; ind_el_i, ind_el_f, ind_ph, sign_ph, mel)
                    EPW._compute_lifetime_serta_single_scattering!(inv_τ_single, el_i, el_f, ph, params, s, model.recip_lattice)
                    inv_τ_q[ind_el_i, iq, :] .+= inv_τ_single
                end
            end
        end
    end
    inv_τ_q
end

function gamma_adaptive_compute_g_gamma(model, kpts, el_k_save, nband, nband_ignore; fourier_mode="gridopt")
    model.epmat_outer_momentum != "ph" && error("model.epmat_outer_momentum must be ph")
    FT = Float64 # FIXME

    nw = model.nw
    nmodes = model.nmodes
    nk = kpts.n
    xq_gamma = Vec3{FT}(0, 0, 0)

    ph_gamma = compute_phonon_states(model, Kpoints(xq_gamma), ["eigenvalue", "eigenvector"], "gridopt")[1]

    epobj_eRpq = WannierObject(model.epmat.irvec_next, zeros(Complex{FT}, (nw*nw*nmodes, length(model.epmat.irvec_next))))
    get_eph_RR_to_Rq!(epobj_eRpq, model.epmat, xq_gamma, ph_gamma.u, fourier_mode)

    epdata = ElPhData{Float64}(nw, nmodes, nband)
    epdata.ph = ph_gamma

    g_gamma_save = zeros(Complex{FT}, nband, nband, nmodes, nk)

    for ik in 1:nk
        el_k = el_k_save[ik]
        epdata.el_k = el_k
        epdata.el_kq = el_k
        xk = kpts.vectors[ik]
        get_eph_Rq_to_kq!(epdata, epobj_eRpq, xk, fourier_mode)
        g_gamma_save[:, :, :, ik] .= epdata.ep
    end
    g_gamma_save
end
