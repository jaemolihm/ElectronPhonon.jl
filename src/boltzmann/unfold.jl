export unfold_QMEStates

"""
Unfold `ElectronState` definde on the irreducible BZ `kpts_irr` to the full BZ `kpts`.
By default, only unfold the energy eigenvalues and eigenvectors. Then, unfold quantities
listed in `quantities`.
- `quantities`: Quantities to unfold. Can contain "velocity_diagonal", "velocity", "position".
"""
function unfold_ElectronStates(model, states_irr::AbstractVector{ElectronState{FT}}, kpts_irr, kpts, ik_to_ikirr_isym, symmetry; quantities=[], fourier_mode="gridopt") where FT
    states = ElectronState{FT}[]
    sym_k = zeros(Complex{FT}, nw, nw)
    for ik = 1:kpts.n
        xk = kpts.vectors[ik]
        ikirr, isym = ik_to_ikirr_isym[ik]
        xkirr = kpts_irr.vectors[ikirr]
        symop = symmetry[isym]

        # Energy eigenvalues and ranges are automatically unfolded by copying.
        push!(states, deepcopy(states_irr[ikirr]))

        # Skip if symmetry is trivial.
        # FIXME: Change to isone(symop)
        isym === 1 && continue

        # Unfold eigenvectors.
        # Find symop in model.el_sym.
        isym_op = findfirst(s -> s ≈ symop, model.el_sym.symmetry)
        isym_op === nothing && error("Symmetry $isym not found in model.el_sym.symmetry.")
        # Compute symmetry operator in Wannier k basis and multiply it to u_full.
        get_fourier!(sym_k, model.el_sym.operators[isym_op], xkirr; mode=fourier_mode)
        # Apply SVD to make sym_k unitary.
        u, s, v = svd(sym_k)
        mul!(sym_k, u, v')
        mul!(states[ik].u_full, sym_k, states_irr[ikirr].u_full)

        if "velocity_diagonal" ∈ quantities
            if symop.is_tr
                states[ik].vdiag .= .-Ref(symop.Scart) .* states_irr[ikirr].vdiag
            else
                states[ik].vdiag .= Ref(symop.Scart) .* states_irr[ikirr].vdiag
            end
        end

        if "velocity" ∈ quantities
            if symop.is_tr
                states[ik].v .= .-Ref(symop.Scart) .* states_irr[ikirr].v
            else
                states[ik].v .= Ref(symop.Scart) .* states_irr[ikirr].v
            end
        end

        if "position" ∈ quantities
            # Symmetry operation of position matrix element involves derivative of the symmetry
            # matrix. So, we just recalculate it using the new eigenvector.
            set_position!(states[ik], model, xk, fourier_mode)
        end
    end
    states
end


"""
    unfold_QMEStates(el::QMEStates, symmetry) => el_unfold, isk_to_ik_isym
Unfold QMEStates in a symmetry-reduced k grid to the full grid using `symmetry`.
To ensure gauge consistency, all states in a given unfolded k point are unfolded from the
same folded k point and symmetry operation.

Output:
- `el_unfold`: Unfolded QMEStates object.
- `isk_to_ik_isym`: Vector containing (ik, isym) that unfolds to k point isk.
"""
function unfold_QMEStates(el::QMEStates, symmetry)
    kpts_unfold = unfold_kpoints(el.kpts, symmetry)
    isk_to_ik_isym = fill((-1, -1), kpts_unfold.n)
    ik_unfold = empty(el.ik)
    ib1_unfold = empty(el.ib1)
    ib2_unfold = empty(el.ib2)
    e1_unfold = empty(el.e1)
    e2_unfold = empty(el.e2)
    v_unfold = empty(el.v)
    for i in 1:el.n
        ik = el.ik[i]
        xk = el.kpts.vectors[ik]
        for (isym, symop) in enumerate(symmetry)
            sk = symop.is_tr ? -symop.S * xk : symop.S * xk
            isk = xk_to_ik(sk, kpts_unfold)

            # To ensure gauge consistency, each isk point must be mapped from a single
            # (ik, isym) pair. If isk_to_ik_isym is not set or set to (ik, isym), use this
            # (ik, isym). If not, skip because other (ik, isym) will be used.
            if isk_to_ik_isym[isk] == (-1, -1) || isk_to_ik_isym[isk] == (ik, isym)
                isk_to_ik_isym[isk] = (ik, isym)
                push!(ik_unfold, isk)
                push!(ib1_unfold, el.ib1[i])
                push!(ib2_unfold, el.ib2[i])
                push!(e1_unfold, el.e1[i])
                push!(e2_unfold, el.e2[i])
                push!(v_unfold, symop.is_tr ? -symop.Scart * conj(el.v[i]) : symop.Scart * el.v[i])
            end
        end
    end

    if any(isk_to_ik_isym .== Ref((-1, -1)))
        error("Some isk point is not found by unfolding.")
    end

    ib_rng_unfold = [el.ib_rng[isk_to_ik_isym[isk][1]] for isk in 1:kpts_unfold.n]
    el_unfold = QMEStates(n=length(ik_unfold), nband=el.nband, e1=e1_unfold, e2=e2_unfold,
        ib1=ib1_unfold, ib2=ib2_unfold, ik=ik_unfold, v=v_unfold, ib_rng=ib_rng_unfold,
        nstates_base=el.nstates_base, kpts=kpts_unfold)
    el_unfold, isk_to_ik_isym
end

"""
    unfold_scattering_out_matrix!(qme_model::QMEIrreducibleKModel)
Unfold the scattering-out matrix on the irreducible k grid to the one defined on the full
k grid. The input is `qme_model.S_out_irr`, which must be set before calling this function.
The output is stored in `qme_model.S_out`.
"""
function unfold_scattering_out_matrix!(qme_model::QMEIrreducibleKModel)
    (; S_out_irr, el_irr, el, ik_to_ikirr_isym) = qme_model
    qme_model.S_out = [unfold_scattering_out_matrix(first(S_out_irr), el_irr, el, ik_to_ikirr_isym)]
    for iT in eachindex(S_out_irr)[2:end]
        push!(qme_model.S_out, unfold_scattering_out_matrix(S_out_irr[iT], el_irr, el, ik_to_ikirr_isym))
    end
    qme_model.S_out
end

# Do nothing for a `QMEModel`.
unfold_scattering_out_matrix!(qme_model::QMEModel) = qme_model.S_out

function unfold_scattering_out_matrix(S_out_irr, el_irr, el, ik_to_ikirr_isym)
    # Assume that S_out_irr is diagonal in k.
    sp_i = Int[]
    sp_j = Int[]
    sp_val = eltype(S_out_irr)[]
    indmap = states_index_map(el)
    indmap_irr = states_index_map(el_irr)
    for i in 1:el.n
        (; ik, ib1, ib2) = el[i]
        ikirr, _ = ik_to_ikirr_isym[ik]
        i_irr = get(indmap_irr, CI(ib1, ib2, ikirr), -1)
        i_irr == -1 && continue
        for ib3 in el.ib_rng[ik], ib4 in el.ib_rng[ik]
            j = get(indmap, CI(ib3, ib4, ik), -1)
            j == -1 && continue
            j_irr = get(indmap_irr, CI(ib3, ib4, ikirr), -1)
            j_irr == -1 && continue
            val = S_out_irr[i_irr, j_irr]
            if abs(val) > 0
                push!(sp_i, i)
                push!(sp_j, j)
                push!(sp_val, val)
            end
        end
    end
    dropzeros!(sparse(sp_i, sp_j, sp_val, el.n, el.n))
end


"""
    unfold_QMEVector(f_irr::QMEVector, model::QMEIrreducibleKModel, trodd, invodd)
Unfold QMEVector defined on `model.el_irr` to `model.el`` using `model.symmetry`.
TODO: Generalize ``symop.Scart * x[i]`` to work with any datatype (scalar, vector, tensor).
"""
function unfold_QMEVector(f_irr::QMEVector{ElType, FT}, model::QMEIrreducibleKModel, trodd, invodd) where {ElType, FT}
    @assert f_irr.state === model.el_irr
    indmap_irr = states_index_map(model.el_irr)
    f = QMEVector(model.el, ElType)
    for i in 1:model.el.n
        (; ik, ib1, ib2) = model.el[i]

        ik_irr, isym = model.ik_to_ikirr_isym[ik]
        symop = model.symmetry[isym]
        i_irr = indmap_irr[CI(ib1, ib2, ik_irr)]

        f[i] = symop.Scart * f_irr[i_irr]
        if trodd && symop.is_tr
            f[i] *= -1
        end
        if invodd && symop.is_inv
            f[i] *= -1
        end
    end
    f
end

# Since QMEModel does not use symmetry, unfolding is a do-nothing operation.
function unfold_QMEVector(f_irr::QMEVector, model::QMEModel, trodd, invodd)
    QMEVector(f_irr.state, copy(f_irr.data))
end
