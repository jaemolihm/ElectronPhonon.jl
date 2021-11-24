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
    el_k_save = compute_electron_states(model, kpts, ["eigenvalue", "eigenvector", "velocity"], window, nband, nband_ignore; fourier_mode);

    # Calculate chemical potential
    energies = vcat([el.e[el.rng] for el in el_k_save]...)
    weights = vcat([fill(kpts.weights[ik], el.nband) for (ik, el) in enumerate(el_k_save)]...)
    transport_set_μ!(transport_params, energies, weights, nstates_base; do_print)

    # Calculate conductivity
    σ_vdiag = zeros(3, 3, length(transport_params.Tlist))
    σ_full_velocity = zeros(3, 3, length(transport_params.Tlist))
    for iT in 1:length(transport_params.Tlist)
        T = transport_params.Tlist[iT]
        μ = transport_params.μlist[iT]
        for (ik, el_k) in enumerate(el_k_save)
            for m in el_k.rng
                emk = el_k.e[m]
                dfocc = -occ_fermion_derivative(emk - μ, T)

                # Use only diagonal velocity
                vmk = el_k.vdiag[m]
                σ_vdiag[:, :, iT] .+= (vmk * transpose(vmk)) .* (kpts.weights[ik] * dfocc * τ)

                # Use full velocity matrix for degenerate bands
                for n in el_k.rng
                    if abs(el_k.e[n] - emk) .< EPW.electron_degen_cutoff
                        vv = real(el_k.v[m, n] * transpose(el_k.v[n, m]))
                        σ_full_velocity[:, :, iT] .+= vv .* (kpts.weights[ik] * dfocc * τ)
                    end
                end
            end
        end
    end
    σ_vdiag .*= transport_params.spin_degeneracy / transport_params.volume
    σ_full_velocity .*= transport_params.spin_degeneracy / transport_params.volume

    # Symmetrize conductivity (if using irreducible k points)
    σ_vdiag = symmetrize_array(σ_vdiag, symmetry, order=2)
    σ_full_velocity = symmetrize_array(σ_full_velocity, symmetry, order=2)

    # Calculate and print conductivity and mobility in SI units
    if do_print
        println("# Using only diagonal velocity")
        transport_print_mobility(σ_vdiag, transport_params)
        println("# Using full velocity matrix for degenerate bands")
        transport_print_mobility(σ_full_velocity, transport_params)
    end

    (; σ_vdiag, σ_full_velocity)
end
