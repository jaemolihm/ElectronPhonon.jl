export unfold_QMEStates

"""
Unfold `ElectronState` definde on the irreducible BZ `kpts_irr` to the full BZ `kpts`.
By default, only unfold the energy eigenvalues and eigenvectors. Then, unfold quantities
listed in `quantities`.
- `quantities`: Quantities to unfold. Can contain "velocity_diagonal", "velocity", "position".
"""
function unfold_ElectronStates(model, states_irr::AbstractVector{ElectronState{FT}}, kpts_irr, kpts, ik_to_ikirr_isym, symmetry; quantities=[], fourier_mode="gridopt") where FT
    # FIXME: Add test
    # If symmetry is trivial, do nothing and return the original states.
    if symmetry === nothing || symmetry.nsym == 1
        return states_irr
    end

    states = ElectronState{FT}[]

    sym_k = zeros(Complex{FT}, model.nw, model.nw)

    for ik = 1:kpts.n
        xk = kpts.vectors[ik]
        ikirr, isym = ik_to_ikirr_isym[ik]
        xkirr = kpts_irr.vectors[ikirr]
        symop = symmetry[isym]

        # Energy eigenvalues and ranges are automatically unfolded by copying.
        push!(states, deepcopy(states_irr[ikirr]))
        el = states[ik]

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
        mul!(el.u_full, sym_k, states_irr[ikirr].u_full)

        if "position" ∈ quantities
            # Symmetry operation of position matrix element involves derivative of the symmetry
            # matrix. So, we just recalculate it using the new eigenvector.
            set_position!(el, model, xk, fourier_mode)
        end

        if "velocity_diagonal" ∈ quantities
            if symop.is_tr
                el.vdiag .= .-Ref(symop.Scart) .* states_irr[ikirr].vdiag
            else
                el.vdiag .= Ref(symop.Scart) .* states_irr[ikirr].vdiag
            end
        end

        if "velocity" ∈ quantities
            if symop.is_tr
                el.v .= .-Ref(symop.Scart) .* states_irr[ikirr].v
            else
                el.v .= Ref(symop.Scart) .* states_irr[ikirr].v
            end
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
function unfold_QMEVector(f_irr::QMEVector{ElType, FT}, model::QMEIrreducibleKModel, trodd, invodd) where {ElType <: Vec3, FT}
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

function unfold_QMEVector(f_irr::QMEVector{ElType, FT}, model::QMEIrreducibleKModel, trodd, invodd) where {ElType <: Number, FT}
    @assert f_irr.state === model.el_irr
    indmap_irr = states_index_map(model.el_irr)
    f = QMEVector(model.el, ElType)
    for i in 1:model.el.n
        (; ik, ib1, ib2) = model.el[i]

        ik_irr, isym = model.ik_to_ikirr_isym[ik]
        symop = model.symmetry[isym]
        i_irr = indmap_irr[CI(ib1, ib2, ik_irr)]

        f[i] = f_irr[i_irr]
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

"""
    symmetrize_QMEVector(x::QMEVector{ElType, FT}, qme_model::QMEIrreducibleKModel,
    trodd, invodd) where {ElType <: Vec3, FT}
Symmetrize a QMEVector defined on the irreducible BZ. Symmetrization acts only to the
high-symmetry k points which have nontrivial symmetry operation that maps k to itself.
FIXME: Make this work for general data (not only Vec3).
"""
function symmetrize_QMEVector(x::QMEVector{ElType, FT}, qme_model::QMEIrreducibleKModel,
        trodd, invodd) where {ElType <: Vec3, FT}
    @assert x.state === qme_model.el_irr
    indmap = states_index_map(x.state)
    f = h5open(qme_model.filename, "r")
    g = open_group(f, "gauge_self")
    ik_list = read(g, "ik_list")::Vector{Int}

    x_symmetrized = copy(x)

    # Count number of symmetry operations that map k to itself. 1 by default because of identity.
    cnt_symm = fill(1, x.state.n)

    for ik in ik_list
        g_ik = open_group(g, "ik$ik")
        isym_list = read(g_ik, "isym")::Vector{Int}
        sym_gauge = load_BTData(open_group(g_ik, "gauge_matrix"), OffsetArray{Complex{FT}, 3, Array{Complex{FT}, 3}})
        is_degenerate = load_BTData(open_group(g_ik, "is_degenerate"), OffsetArray{Bool, 2, Array{Bool, 2}})
        for ind = 1:x.state.n
            x.state.ik[ind] == ik || continue
            (; ib1, ib2) = x.state[ind]

            # <u_{jb1,k}|S|u_{ib1,k}> * x_{ib1, ib2, k} * <u_{ib2, k}|S|u_{jb2, k}>
            for jb2 in x.state.ib_rng[ik]
                is_degenerate[jb2, ib2] || continue
                for jb1 in x.state.ib_rng[ik]
                    is_degenerate[jb1, ib1] || continue
                    ind_symm = get(indmap, CI(jb1, jb2, ik), nothing)
                    ind_symm === nothing && continue

                    for (ind_isym, isym) in enumerate(isym_list)
                        symop = qme_model.symmetry[isym]
                        gauge_coeff = sym_gauge[jb1, ib1, ind_isym] * sym_gauge[jb2, ib2, ind_isym]'
                        data_new = (symop.Scart * x.data[ind]) * gauge_coeff
                        if symop.is_tr && trodd
                            data_new *= -1
                        end
                        if symop.is_inv && invodd
                            data_new *= -1
                        end
                        x_symmetrized.data[ind_symm] += data_new
                    end
                end
            end
            cnt_symm[ind] += length(isym_list)
        end
    end
    close(f)
    x_symmetrized.data ./= cnt_symm
    x_symmetrized
end

symmetrize_QMEVector(x::QMEVector, qme_model::QMEModel, trodd, invodd) = copy(x)

"""
Compute a map that maps a `QMEVector` data defined on `el` by S to `el_f` using symmetry
`S = qme_model.symmetry[isym]`.
``y_{m',n',k'} = ∑_{m, n} <u^(f)_{m'k'}|S|u^(i)_{mk}> x_{m,n,k} <u^(i)_{nk}|S^{-1}|u^(f)_{n'k'}>``
where ``k' = S * k``.
"""
function _el_to_el_f_symmetry_maps(qme_model::QMEIrreducibleKModel{FT}) where FT
    (; el, el_f, symmetry) = qme_model

    # Read gauge information from file
    # TODO: Reading gauge information is the bottleneck.
    sym_gauge_list = OffsetArray{Complex{FT}, 3, Array{Complex{FT}, 3}}[]
    is_degenerate_list = OffsetArray{Bool, 3, Array{Bool, 3}}[]
    fid = h5open(qme_model.filename, "r")
    for isym = 1:symmetry.nsym
        group_sym = open_group(fid, "gauge/isym$isym")
        push!(sym_gauge_list, load_BTData(open_group(group_sym, "gauge_matrix"),
                                          OffsetArray{Complex{FT}, 3, Array{Complex{FT}, 3}}))
        push!(is_degenerate_list, load_BTData(open_group(group_sym, "is_degenerate"),
                                              OffsetArray{Bool, 3, Array{Bool, 3}}))
    end
    close(fid)

    indmap_f = states_index_map(el_f)

    i_to_f_maps = SparseMatrixCSC{Complex{FT}, Int}[]

    for symop in symmetry
        sp_ind_f = Int[]
        sp_ind_i = Int[]
        sp_val = Complex{FT}[]
        for ind_i = 1:el.n
            (; ib1, ib2, ik) = el[ind_i]
            xk = el.kpts.vectors[ik]
            sk = symop.is_tr ? -symop.S * xk : symop.S * xk
            isk = xk_to_ik(sk, el_f.kpts)
            isk === nothing && continue

            # We know <u^(f)_Sk|S|u^(i)_k> only for irreducible k points. To compute the gauge
            # for general k points, we use k = S_irr * k_irr and
            # <u^(f)_{m'k'}|S|u^(i)_{mk}> = <u^(f)_{m'k'}|S * S_irr|u^(i)_{m,k_irr}>.
            ik_irr, isym_irr = qme_model.ik_to_ikirr_isym[ik]
            symop_prod = symop * symmetry[isym_irr]
            isym_prod = findfirst(s -> s ≈ symop_prod, symmetry)

            is_degenerate = is_degenerate_list[isym_prod]
            sym_gauge = sym_gauge_list[isym_prod]

            for jb2 in el_f.ib_rng[isk]
                is_degenerate[jb2, ib2, ik_irr] || continue
                for jb1 in el_f.ib_rng[isk]
                    is_degenerate[jb1, ib1, ik_irr] || continue
                    ind_f = get(indmap_f, CI(jb1, jb2, isk), -1)
                    ind_f == -1 && continue
                    gauge_coeff = sym_gauge[jb1, ib1, ik_irr] * sym_gauge[jb2, ib2, ik_irr]'
                    push!(sp_ind_f, ind_f)
                    push!(sp_ind_i, ind_i)
                    push!(sp_val, gauge_coeff)
                end
            end
        end
        push!(i_to_f_maps, dropzeros!(sparse(sp_ind_f, sp_ind_i, sp_val, el_f.n, el.n)))
    end
    i_to_f_maps
end



# function symmetrize_scattering_out_matrix(S_out_irr, qme_model::EPW.QMEIrreducibleKModel{FT}) where FT
#     (; el_irr) = qme_model
#     @assert size(S_out_irr) == (el_irr.n, el_irr.n)
#     indmap = EPW.states_index_map(el_irr)
#     f = h5open(qme_model.filename, "r")
#     g = open_group(f, "gauge_self")
#     ik_list = read(g, "ik_list")::Vector{Int}

#     is = Int[]
#     js = Int[]
#     vals = eltype(S_out_irr)[]

#     # Count number of symmetry operations that map k to itself. 1 by default because of identity.
#     cnt_symm = fill(1, el_irr.n)

#     for ik in ik_list
#         g_ik = open_group(g, "ik$ik")
#         isym_list = read(g_ik, "isym")::Vector{Int}
#         sym_gauge = load_BTData(open_group(g_ik, "gauge_matrix"), OffsetArray{Complex{FT}, 3, Array{Complex{FT}, 3}})
#         is_degenerate = load_BTData(open_group(g_ik, "is_degenerate"), OffsetArray{Bool, 2, Array{Bool, 2}})
#         for j = 1:el_irr.n
#             el_irr.ik[j] == ik || continue
#             cnt_symm[j] += length(isym_list)
#             jb1, jb2 = el_irr.ib1[j], el_irr.ib2[j]
#             for i = 1:el_irr.n
#                 el_irr.ik[i] == ik || continue
#                 ib1, ib2 = el_irr.ib1[i], el_irr.ib2[i]

#                 abs(S_out_irr[i, j]) > 0 || continue

#                 # S_{pb1, pb2, k <- qb1, qb2, k} <-- (symmetry) -- S_{ib1, ib2, k <- jb1, jb2, k}
#                 # gauge_coeff = S[pb1, ib1] * S[pb2, ib2]' * S[qb1, jb1]' * S[qb2, jb2]
#                 # where S[i, j] = <u_{pb1,k}|S|u_{ib1,k}>.
#                 for qb2 in el_irr.ib_rng[ik]
#                     is_degenerate[qb2, jb2] || continue
#                     for qb1 in el_irr.ib_rng[ik]
#                         is_degenerate[qb1, jb1] || continue

#                         q = get(indmap, CI(qb1, qb2, ik), nothing)
#                         q === nothing && continue

#                         for pb2 in el_irr.ib_rng[ik]
#                             is_degenerate[pb2, ib2] || continue
#                             for pb1 in el_irr.ib_rng[ik]
#                                 is_degenerate[pb1, ib1] || continue

#                                 p = get(indmap, CI(pb1, pb2, ik), nothing)
#                                 p === nothing && continue

#                                 gauge_coeff = 0.0im
#                                 for indsym = 1:size(sym_gauge, 3)
#                                     gauge_coeff += (  sym_gauge[pb1, ib1, indsym]  * sym_gauge[pb2, ib2, indsym]'
#                                                     * sym_gauge[qb1, jb1, indsym]' * sym_gauge[qb2, jb2, indsym])
#                                     # FIXME: Time reversal, Inversion (now assumed to be all even)
#                                 end
#                                 push!(is, p)
#                                 push!(js, q)
#                                 push!(vals, S_out_irr[i, j] * gauge_coeff)
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
#     close(f)
#     S_out_irr_symmetrized = sparse(is, js, vals, size(S_out_irr)...) + S_out_irr
#     S_out_irr_symmetrized ./= cnt_symm
#     dropzeros!(S_out_irr_symmetrized)
#     S_out_irr_symmetrized
# end