export unfold_QMEStates

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
    kpts_unfold = EPW.unfold_kpoints(el.kpts, symmetry)
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