export run_transport_constant_relaxation_time

# Calculate electron conductivity in the constant relaxation time approximation.
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
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"], window, nband, nband_ignore, fourier_mode);

    # Calculate chemical potential
    energies = vcat([el.e[el.rng] for el in el_k_save]...)
    weights = vcat([fill(kpts.weights[ik], el.nband) for (ik, el) in enumerate(el_k_save)]...)
    transport_set_μ!(transport_params, energies, weights, nstates_base; do_print)

    # Calculate conductivity
    σlist_vdiag = zeros(3, 3, length(transport_params.Tlist))
    σlist_full_velocity = zeros(3, 3, length(transport_params.Tlist))
    for iT in 1:length(transport_params.Tlist)
        T = transport_params.Tlist[iT]
        μ = transport_params.μlist[iT]
        for (ik, el_k) in enumerate(el_k_save)
            for ib in el_k.rng
                enk = el_k.e[ib]
                dfocc = -occ_fermion_derivative(enk - μ, T)

                # Use only diagonal velocity
                vnk = el_k.vdiag[ib]
                σlist_vdiag[:, :, iT] .+= (vnk * transpose(vnk)) .* (kpts.weights[ik] * dfocc * τ)

                # Use full velocity matrix for degenerate bands
                ib_degen = el_k.rng[findall(abs.(el_k.e[el_k.rng] .- enk) .< electron_degen_cutoff)]
                for ib2 in ib_degen, ib1 in ib_degen
                    vv = real(el_k.v[ib1, ib2] * transpose(el_k.v[ib2, ib1]))
                    σlist_full_velocity[:, :, iT] .+= vv .* (kpts.weights[ik] * dfocc * τ / length(ib_degen))
                end
            end
        end
    end
    σlist_vdiag .*= transport_params.spin_degeneracy / transport_params.volume
    σlist_full_velocity .*= transport_params.spin_degeneracy / transport_params.volume

    # Symmetrize conductivity (if using irreducible k points)
    σlist_vdiag = symmetrize_array(σlist_vdiag, symmetry, order=2)
    σlist_full_velocity = symmetrize_array(σlist_full_velocity, symmetry, order=2)

    # Calculate and print conductivity and mobility in SI units
    do_print && println("# Using only diagonal velocity")
    σ_vdiag_SI, mobility_vdiag_SI = transport_print_mobility(σlist_vdiag, transport_params; do_print)
    do_print && println("# Using full velocity matrix for degenerate bands")
    σ_full_velocity_SI, mobility_full_velocity_SI = transport_print_mobility(σlist_full_velocity, transport_params; do_print)

    (; σlist_vdiag, σ_vdiag_SI, mobility_vdiag_SI, σlist_full_velocity, σ_full_velocity_SI, mobility_full_velocity_SI)
end