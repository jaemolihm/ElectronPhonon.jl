using Printf
using EPW.WanToBloch: get_symmetry_representation_wannier!

export check_electron_symmetry_of_model

"""
    check_electron_symmetry_of_model(model::EPW.ModelEPW{FT}, ngrid; fourier_mode="gridopt")
Check whether the electron operators in Wannier function basis follow the symmetry of the system.
Symmetry in `model.el_sym.symmetry`, not `model.symmetry` are tested because one has symmetry
operators in the Wannier basis only for the former.
# Quantities that are calculated:
- Energy eigenvalues at symmetry-equivalent k points
- Unitarity of symmetry operators in Wannier basis
- Symmetry of Hamiltonian in Wannier basis
- Symmetry of velocity matrix in Wannier basis (direct interpolation)
- Symmetry of velocity matrix computed using Berry curvature and converted back to Wannier basis

Note: the symmetry of position or H * R matrices cannot be directly tested because
symmetry operation involves translation and shift of Wannier functions. So we use the velocity
matrix computed using the Berry curvature method as an indirect check of position matrix.
"""
function check_electron_symmetry_of_model(model::EPW.ModelEPW{FT}, ngrid; fourier_mode="gridopt") where FT
    model.el_sym === nothing && error("model.el_sym must be set. Pass load_symmetry_operators=true in load_model.")

    error_keys = [:Energy, :SymMatrixUnitarity, :Hamiltonian, :Velocity_Direct, :Velocity_BerryConnection]
    max_errors = Dict(key => 0.0 for key in error_keys)
    rms_errors = Dict(key => 0.0 for key in error_keys)

    # Reasonable bounds for each error item.
    # The error for Velocity_BerryConnection is larger than Velocity_Direct because of
    # finite-difference error in the position matrix elements.
    # TODO: Check whether this bound is reasonable.
    rms_error_bounds = Dict(
        :Energy => EPW.electron_degen_cutoff,
        :SymMatrixUnitarity => 1e-3,
        :Hamiltonian => 1.0 * unit_to_aru(:meV),
        :Velocity_Direct => 1.0 * unit_to_aru(:meV) * unit_to_aru(:Å),
        :Velocity_BerryConnection => 1e3 * unit_to_aru(:meV) * unit_to_aru(:Å),
    )

    kpts = GridKpoints(kpoints_grid(ngrid))
    nw = model.nw
    symmetry = model.el_sym.symmetry

    # Calculate velocity using the BerryConnection method
    el_velocity_mode_save = model.el_velocity_mode
    model.el_velocity_mode = :BerryConnection
    el_states = compute_electron_states(model, kpts, ["eigenvalue", "velocity"])
    model.el_velocity_mode = el_velocity_mode_save

    # Compute operators in Wannier basis
    op_Hk = [zeros(Complex{FT}, nw, nw) for _ in 1:kpts.n]
    op_vk_direct = [zeros(Complex{FT}, nw, nw, 3) for _ in 1:kpts.n]
    op_vk_berry = [zeros(Complex{FT}, nw, nw, 3) for _ in 1:kpts.n]
    for ik in 1:kpts.n
        xk = kpts.vectors[ik]
        get_fourier!(op_Hk[ik], model.el_ham, xk; fourier_mode)
        get_fourier!(op_vk_direct[ik], model.el_vel, xk; fourier_mode)
        v_bc = el_states[ik].u_full * el_states[ik].v * el_states[ik].u_full'
        for i in 1:3, jw in 1:nw, iw in 1:nw
            op_vk_berry[ik][iw, jw, i] = v_bc[iw, jw][i]
        end
    end

    sym_W = zeros(Complex{FT}, nw, nw)
    S_vk = [zeros(Complex{FT}, nw, nw) for _ in 1:3]

    for (isym, symop) in enumerate(symmetry)
        for ik in 1:kpts.n
            xk = kpts.vectors[ik]
            sk = symop.is_tr ? -symop.S * xk : symop.S * xk
            isk = xk_to_ik(sk, kpts)
            if ! (kpts.vectors[isk] ≈ EPW.normalize_kpoint_coordinate(sk))
                error("k point grid is not symmetric: sk=$sk (xk=$xk, isym=$isym) not in the grid.")
            end

            # SymMatrixUnitarity: Check unitarity of symmetry operators in Wannier basis
            get_symmetry_representation_wannier!(sym_W, model.el_sym.operators[isym], xk, symop.is_tr; fourier_mode)
            sym_W_error = sqrt(norm(sym_W' * sym_W - I(nw)))
            max_errors[:SymMatrixUnitarity] = max(max_errors[:SymMatrixUnitarity], sym_W_error)
            rms_errors[:SymMatrixUnitarity] += sum(sym_W_error^2)

            # Energy: check e_{m,Sk} = e_{m,k}
            max_errors[:Energy] = max(max_errors[:Energy], maximum(abs.(el_states[ik].e .- el_states[isk].e)))
            rms_errors[:Energy] += sum((el_states[ik].e .- el_states[isk].e).^2) / nw

            # Hamiltonian: check H_W(Sk) = S_W(k) * H_W(k) * S_W(k)'
            if symop.is_tr
                H_error = norm(sym_W * conj.(op_Hk[ik]) * sym_W' - op_Hk[isk])
            else
                H_error = norm(sym_W * op_Hk[ik] * sym_W' - op_Hk[isk])
            end
            max_errors[:Hamiltonian] = max(max_errors[:Hamiltonian], H_error)
            rms_errors[:Hamiltonian] += sum(H_error^2)

            # Velocity: check V_W(Sk) = S( S_W(k) * V_W(k) * S_W(k)' )
            # Outermost S indicates a rotation and time reversal operation.
            @views for (op_vk, key) in [(op_vk_direct, :Velocity_Direct), (op_vk_berry, :Velocity_BerryConnection)]
                for i in 1:3
                    if symop.is_tr
                        S_vk[i] .= sym_W * conj.(op_vk[ik][:, :, i]) * sym_W'
                    else
                        S_vk[i] .= sym_W * op_vk[ik][:, :, i] * sym_W'
                    end
                end
                # Rotate S_vk along the last dimension
                S_vk .= symop.Scart * S_vk
                # Velocity is time reveral odd
                if symop.is_tr
                    S_vk .*= -1
                end

                # FIXME: Type instability
                v_error = sum(norm(S_vk[i] - op_vk[isk][:, :, i]) for i in 1:3)
                max_errors[key] = max(max_errors[key], v_error)
                rms_errors[key] += sum(v_error^2)
            end
        end
    end

    # Compute RMS values and print.
    for key in keys(rms_errors)
        rms_errors[key] = sqrt(rms_errors[key] / kpts.n / symmetry.nsym)
    end
    @printf "%25s %12s %12s\n" "Item" "RMS error" "MAX error"
    for key in error_keys
        @printf "%25s %12.3e %12.3e\n" key rms_errors[key] max_errors[key]
    end

    # Warn if RMS error is larger than the bounds
    for key in keys(rms_error_bounds)
        if rms_errors[key] > rms_error_bounds[key]
            @warn ("Symmetry warning for $key: RMS error $(rms_errors[key]) is larger than " *
                   "the bound $(rms_error_bounds[key])")
        end
    end

    (; rms_errors, max_errors)
end
