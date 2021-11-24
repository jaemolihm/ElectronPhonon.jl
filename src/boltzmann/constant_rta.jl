export run_transport_constant_relaxation_time

"""
Calculate electron conductivity in the constant relaxation time approximation.
Output
- `σ_vdiag`: intra-band term (consider only the ``m=n`` case in `σ_intra_degen`)
- `σ_intra_degen`: intra-degenerate-group term
- `σ_full`: intra- and inter-degenerate-group term

Intra-degenerate-group term:
``σ_intra_degen[a,b] = ∑_{k,m,n; ε_mk = ε_nk} τ * (-df/dε)_mk * v^a_mn * v^b_nm``

Inter-degenerate-group term:
``σ_full[a,b] = σ_intra_degen[a,b]
              + ∑_{k,m,n; ε_mk /= ε_nk} -(f_mk - f_nk) / (ε_mk - ε_nk) * v^a_mn * v^b_nm
                                        *  Γ / ((ε_mk - ε_nk)^2 + Γ^2),``
where Γ = 1 / τ.
"""
function run_transport_constant_relaxation_time(model, k_input, transport_params;
        inv_τ_constant = 8.0 * unit_to_aru(:meV),
        fourier_mode = "gridopt",
        window = (-Inf, Inf),
        use_irr_k = true,
        mpi_comm_k = nothing,
        do_print = true
    )

    mpi_comm_k !== nothing && error("mpi_comm_k not implemented")

    nw = model.nw
    τ = 1 / inv_τ_constant

    # Filter k points
    symmetry = use_irr_k ? model.symmetry : nothing
    kpts, iband_min, iband_max, nstates_base = filter_kpoints(k_input, nw, model.el_ham, window, mpi_comm_k; symmetry)

    nband = iband_max - iband_min + 1
    nband_ignore = iband_min - 1

    # Calculate electron states
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"], window, nband, nband_ignore; fourier_mode);

    # Calculate chemical potential
    energies = vcat([el.e[el.rng] for el in el_k_save]...)
    weights = vcat([fill(kpts.weights[ik], el.nband) for (ik, el) in enumerate(el_k_save)]...)
    transport_set_μ!(transport_params, energies, weights, nstates_base; do_print)

    # Calculate conductivity
    σ_vdiag = zeros(3, 3, length(transport_params.Tlist))
    σ_intra_degen = zeros(3, 3, length(transport_params.Tlist))
    σ_full  = zeros(3, 3, length(transport_params.Tlist))
    for iT in 1:length(transport_params.Tlist)
        T = transport_params.Tlist[iT]
        μ = transport_params.μlist[iT]
        for (ik, el_k) in enumerate(el_k_save)
            @views for m in el_k.rng
                emk = el_k.e[m]
                dfocc = -occ_fermion_derivative(emk - μ, T)

                # Use only diagonal velocity
                vmk = el_k.vdiag[m]
                σ_vdiag[:, :, iT] .+= (vmk * transpose(vmk)) .* (kpts.weights[ik] * dfocc * τ)

                # Use velocity matrix inside degenerate subspace
                for n in el_k.rng
                    if abs(el_k.e[n] - emk) .< EPW.electron_degen_cutoff
                        vv = real(el_k.v[m, n] * transpose(el_k.v[n, m]))
                        σ_intra_degen[:, :, iT] .+= vv .* (kpts.weights[ik] * dfocc * τ)
                        σ_full[:, :, iT] .+= vv .* (kpts.weights[ik] * dfocc * τ)
                    end
                end

                # Use full velocity matrix, including nondegenerate band pairs
                for n in el_k.rng
                    enk = el_k.e[n]
                    fmk = occ_fermion(emk - μ, T)
                    fnk = occ_fermion(enk - μ, T)
                    if abs(emk - enk) >= EPW.electron_degen_cutoff
                        vv = real(el_k.v[m, n] * transpose(el_k.v[n, m]))
                        coeff = -(fmk - fnk) / (emk - enk) * inv_τ_constant / ((emk - enk)^2 + inv_τ_constant^2)
                        σ_full[:, :, iT] .+= vv .* (coeff * kpts.weights[ik])
                    end
                end
            end
        end
    end
    σ_vdiag .*= transport_params.spin_degeneracy / transport_params.volume
    σ_intra_degen .*= transport_params.spin_degeneracy / transport_params.volume
    σ_full .*= transport_params.spin_degeneracy / transport_params.volume

    # Symmetrize conductivity (if using irreducible k points)
    σ_vdiag = symmetrize_array(σ_vdiag, symmetry, order=2)
    σ_intra_degen = symmetrize_array(σ_intra_degen, symmetry, order=2)
    σ_full = symmetrize_array(σ_full, symmetry, order=2)

    # Calculate and print conductivity and mobility in SI units
    if do_print
        println("# Using only diagonal velocity")
        transport_print_mobility(σ_vdiag, transport_params)
        println("# Using intra-degenerate-bands velocity matrix elements")
        transport_print_mobility(σ_intra_degen, transport_params)
        println("# Using full velocity matrix for all bands in window")
        transport_print_mobility(σ_full, transport_params)
    end

    (; σ_vdiag, σ_intra_degen, σ_full)
end
