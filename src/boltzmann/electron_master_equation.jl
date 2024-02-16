using SparseArrays
using OffsetArrays

# Constructing and solving quantum master equation for electrons

export compute_qme_scattering_matrix

"""
    compute_qme_scattering_matrix(filename, params, el_i, el_f, ph)
Compute the scattering matrix element for quantum master equation of electrons.

# Input
- `use_mrta`: If true, compute Sₒ using MRTA (momentum relaxation time approximation). See
    Eq. (49) of Ponce et al, Rep. Prog. Phys. (2020). Since the MRTA is an approximate way to
    describe the scattering-in term, one must set `compute_Sᵢ = false` when using MRTA.
- `compute_Sᵢ`: If true, compute the scattering-in term.
- `use_eph_dipole=true`: If false, do not include dipole e-ph term even if present.
"""
function compute_qme_scattering_matrix(filename, params, el_i::QMEStates{FT}, el_f::QMEStates{FT}, ph;
        compute_Sᵢ=true, use_mrta=false, use_eph_dipole=true) where {FT}
    if compute_Sᵢ && use_mrta
        throw(ArgumentError("compute_Sᵢ = true and use_mrta = true is not compatible."))
    end
    nT = length(params.Tlist)

    ind_ph_map = states_index_map(ph)

    fid = h5open(filename, "r")
    group_scattering = open_group(fid, "scattering")
    mpi_isroot() && println("Original grid: Total $(length(group_scattering)) groups of scattering")

    # Disable eph dipole if eph_dipole data is not present.
    use_eph_dipole = use_eph_dipole && "eph_dipole" ∈ keys(open_group(fid, "phonon"))

    @assert params.smearing[1] == :Gaussian
    η = params.smearing[2]
    inv_η = 1 / η

    # Preallocate buffer arrays
    p_mel = zeros(Complex{FT}, nT)
    p_mel_ikq = zeros(Complex{FT}, nT)
    s_mel_ikq = zeros(Complex{FT}, nT)
    eph_mel_1 = zeros(Complex{FT}, nT)
    eph_mel_2 = zeros(Complex{FT}, nT)
    gg = zeros(Complex{FT}, nT)

    # Scattering-out and scattering-in matrices
    # Sₒ is sparse (block diagonal) while Sᵢ is not sparse in general.
    Sₒ = [spzeros(Complex{FT}, el_i.n, el_i.n) for _ in 1:nT]
    if compute_Sᵢ
        Sᵢ = [zeros(Complex{FT}, el_i.n, el_f.n) for _ in 1:nT]
    else
        Sᵢ = nothing
    end

    # Compute occupation factors
    focc_el_i_all = compute_occupations_electron(el_i, params.Tlist, params.μlist)
    focc_el_f_all = compute_occupations_electron(el_f, params.Tlist, params.μlist)
    nocc_ph_all = zeros(nT, ph.n)
    for i in 1:ph.n
        @. nocc_ph_all[:, i] = occ_boson(ph.e[i], params.Tlist)
    end

    if use_eph_dipole
        phonon_eph_dipole = read(fid, "phonon/eph_dipole")::Vector{Complex{FT}}
        if "ϵ_screen" in keys(fid)
            ϵ_screen = read(fid, "ϵ_screen")::Matrix{Complex{FT}}
        else
            ϵ_screen = ones(Complex{FT}, nT, ph.n)
        end
    end

    for ik in 1:el_i.kpts.n
        mpi_isroot() && mod(ik, 50) == 0 && println("Calculating scattering for group $ik")

        @timing "read scat" scat = load_BTData(open_group(group_scattering, "ik$ik"), ElPhVertexDataset{FT})

        # Read mmat = <u(k+q)|u(k)> at ik
        if use_eph_dipole
            mmat = load_BTData(open_group(fid, "mmat/ik$ik"), MatrixElementDataset{Complex{FT}})
        end

        # 1. Scattering-out term
        # P_{ib1, ib2} = sum_{ikq, imode, jb, ±} g*_{jb, ib1} * g_{jb, ib2}
        #              * δ^{1/2}(e_ib1 - e_jb ± ω_imode) * δ^{1/2}(e_ib2 - e_jb ± ω_imode)
        #              * (n_imode + f_jb)       (+)
        #                (n_imode + 1 - f_jb)   (-)
        # dδρ_{ib1, ib2}/dt = - [P, δρ]_{ib1, ib2} / 2

        @timing "scat out" for ib2 in el_i.ib_rng[ik], ib1 in el_i.ib_rng[ik]
            # Calculate P_{ib1, ib2} only if ∃ ib3 such that both (ib1, ib3) and (ib2, ib3)
            # or both (ib3, ib1) and (ib3, ib2) are in el_i.
            found = false
            local ind_el_i1, ind_el_i2
            for ib3 in el_i.ib_rng[ik]
                if hasstate(el_i, ib1, ib3, ik) && hasstate(el_i, ib2, ib3, ik)
                    found = true
                    ind_el_i1 = get_1d_index(el_i, ib1, ib3, ik)
                    ind_el_i2 = get_1d_index(el_i, ib2, ib3, ik)
                    break
                end
                if hasstate(el_i, ib3, ib1, ik) && hasstate(el_i, ib3, ib2, ik)
                    found = true
                    ind_el_i1 = get_1d_index(el_i, ib1, ib3, ik)
                    ind_el_i2 = get_1d_index(el_i, ib2, ib3, ik)
                    break
                end
            end
            found || continue
            e_i1 = el_i.e1[ind_el_i1]
            e_i2 = el_i.e1[ind_el_i2]
            v_i1 = el_i.v[ind_el_i1]
            v_i2 = el_i.v[ind_el_i2]
            p_mel .= 0

            for ind_el_f in 1:el_f.n
                jb1 = el_f.ib1[ind_el_f]
                jb2 = el_f.ib2[ind_el_f]
                jb1 != jb2 && continue # Only diagonal part contributes to the scattering-out term
                jb = jb1
                e_f = el_f.e1[ind_el_f]
                ikq = el_f.ik[ind_el_f]
                v_f = el_f.v[ind_el_f]
                focc_f = @view focc_el_f_all[:, jb1, ikq]

                xq = el_f.kpts.vectors[ikq] - el_i.kpts.vectors[ik]
                xq_int = mod.(round.(Int, xq.data .* ph.ngrid), ph.ngrid)
                ind_ph_list = get(ind_ph_map, CI(xq_int...), nothing)
                ind_ph_list === nothing && continue # skip if this xq is not in ph

                p_mel_ikq .= 0

                for imode in 1:ph.nband
                    ind_ph = ind_ph_list[imode]
                    ind_ph == 0 && continue # skip if this imode is not in ph
                    ω_ph = ph.e[ind_ph]
                    # Skip if phonon frequency is too close to 0 (acoustic phonon at q=0)
                    ω_ph < omega_acoustic && continue
                    nocc_ph = @view nocc_ph_all[:, ind_ph]

                    s1 = scat[ikq, ib1, jb, imode]
                    s1 === nothing && continue
                    s2 = scat[ikq, ib2, jb, imode]
                    s2 === nothing && continue

                    # Matrix element factor
                    eph_mel_1 .= s1.mel
                    eph_mel_2 .= s2.mel
                    @views if use_eph_dipole
                        # Add long-range term
                        @. eph_mel_1 += phonon_eph_dipole[ind_ph] * mmat[ikq, ib1, jb] / sqrt(2ω_ph) / ϵ_screen[:, ind_ph]
                        @. eph_mel_2 += phonon_eph_dipole[ind_ph] * mmat[ikq, ib2, jb] / sqrt(2ω_ph) / ϵ_screen[:, ind_ph]
                    end
                    @. gg = conj(eph_mel_1) * eph_mel_2

                    if s1.econv_p && s2.econv_p
                        if use_mrta == true
                            _compute_p_matrix_element_mrta!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, +1, inv_η, focc_f, nocc_ph, v_i1, v_f)
                        else
                            _compute_p_matrix_element!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, +1, inv_η, focc_f, nocc_ph)
                        end
                    end
                    if s1.econv_m && s1.econv_m
                        if use_mrta == true
                            _compute_p_matrix_element_mrta!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, -1, inv_η, focc_f, nocc_ph, v_i1, v_f)
                        else
                            _compute_p_matrix_element!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, -1, inv_η, focc_f, nocc_ph)
                        end
                    end
                end
                p_mel .+= p_mel_ikq .* 2FT(π) .* el_f.kpts.weights[ikq]
            end # ikq

            # Calculate the scattering matrix. Contribution of P_{ib1, ib2} are:
            # Sₒ[(ib1,ib3), (ib2,ib3)] += - P_{ib1, ib2} / 2
            # Sₒ[(ib3,ib2), (ib3,ib1)] += - P_{ib1, ib2} / 2
            for ib3 in el_i.ib_rng[ik]
                if hasstate(el_i, ib1, ib3, ik) && hasstate(el_i, ib2, ib3, ik)
                    i1 = get_1d_index(el_i, ib1, ib3, ik)
                    i2 = get_1d_index(el_i, ib2, ib3, ik)
                    for iT in 1:nT
                        Sₒ[iT][i1, i2] += -p_mel[iT] / 2
                    end
                end
                if hasstate(el_i, ib3, ib1, ik) && hasstate(el_i, ib3, ib2, ik)
                    i1 = get_1d_index(el_i, ib3, ib2, ik)
                    i2 = get_1d_index(el_i, ib3, ib1, ik)
                    for iT in 1:nT
                        Sₒ[iT][i1, i2] += -p_mel[iT] / 2
                    end
                end
            end
        end # ib1, ib2

        if ! compute_Sᵢ
            continue
        end

        # 2. Scattering-in term
        # dδρ_{ib1,ib2,k}/dt = sum_{ikq, imode, jb1, jb2, ±} g*_{jb1, ib1} * g_{jb2, ib2}
        #                    * δ^{1/2}(e_ib1 - e_jb1 ± ω_imode) * δ^{1/2}(e_ib2 - e_jb2 ± ω_imode)
        #                      (n_imode + 1 - (f_ib1 + f_ib2) / 2 )       (+)
        #                    * (n_imode +     (f_ib1 + f_ib2) / 2 )       (-)

        @timing "scat in" for ib2 in el_i.ib_rng[ik], ib1 in el_i.ib_rng[ik]
            # Calculate only if (ib1, ib2, ik) ∈ el_i
            ind_el_i = get_1d_index(el_i, ib1, ib2, ik)
            ind_el_i == 0 && continue
            e_i1 = el_i.e1[ind_el_i]
            e_i2 = el_i.e2[ind_el_i]
            focc_i1 = @view focc_el_i_all[:, ib1, ik]
            focc_i2 = @view focc_el_i_all[:, ib2, ik]

            for ind_el_f in 1:el_f.n
                jb1 = el_f.ib1[ind_el_f]
                jb2 = el_f.ib2[ind_el_f]
                e_f1 = el_f.e1[ind_el_f]
                e_f2 = el_f.e2[ind_el_f]

                # Find q point
                ikq = el_f.ik[ind_el_f]
                # ikq > length(scat) && continue # skip if this ikq is not in scat

                # DEBUG: 0.2 sec
                xq = el_f.kpts.vectors[ikq] - el_i.kpts.vectors[ik]
                xq_int = mod.(round.(Int, xq.data .* ph.ngrid), ph.ngrid)
                ind_ph_list = get(ind_ph_map, CI(xq_int...), nothing)
                ind_ph_list === nothing && continue # skip if this xq is not in ph

                s_mel_ikq .= 0

                for imode in 1:ph.nband
                    # DEBUG: 0.2 sec
                    ind_ph = ind_ph_list[imode]
                    ind_ph == 0 && continue # skip if this imode is not in ph
                    ω_ph = ph.e[ind_ph]
                    # Skip if phonon frequency is too close to 0 (acoustic phonon at q=0)
                    ω_ph < omega_acoustic && continue
                    nocc_ph = @view nocc_ph_all[:, ind_ph]

                    # DEBUG: 0.3 sec
                    s1 = scat[ikq, ib1, jb1, imode]
                    s1 === nothing && continue
                    s2 = scat[ikq, ib2, jb2, imode]
                    s2 === nothing && continue

                    # Matrix element factor
                    eph_mel_1 .= s1.mel
                    eph_mel_2 .= s2.mel
                    @views if use_eph_dipole
                        # Add long-range term
                        @. eph_mel_1 += phonon_eph_dipole[ind_ph] * mmat[ikq, ib1, jb1] / sqrt(2ω_ph) / ϵ_screen[:, ind_ph]
                        @. eph_mel_2 += phonon_eph_dipole[ind_ph] * mmat[ikq, ib2, jb2] / sqrt(2ω_ph) / ϵ_screen[:, ind_ph]
                    end
                    @. gg = conj(eph_mel_1) * eph_mel_2

                    # DEBUG: 1.2 sec -> 0.9 sec
                    if s1.econv_p && s2.econv_p
                        _compute_s_in_matrix_element!(s_mel_ikq, gg, e_i1, e_i2, e_f1, e_f2, ω_ph, +1, inv_η, focc_i1, focc_i2, nocc_ph)
                    end
                    if s1.econv_m && s1.econv_m
                        _compute_s_in_matrix_element!(s_mel_ikq, gg, e_i1, e_i2, e_f1, e_f2, ω_ph, -1, inv_η, focc_i1, focc_i2, nocc_ph)
                    end
                end
                s_mel_ikq .*= 2FT(π) * el_f.kpts.weights[ikq]

                for iT in 1:nT
                    Sᵢ[iT][ind_el_i, ind_el_f] += s_mel_ikq[iT]
                end
            end
        end

    end # ik
    close(fid)
    Sₒ, Sᵢ
end

@inline function _compute_p_matrix_element!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, sign_ph, inv_η, f_kq, n_ph)
    # energy conservation factor
    delta1 = gaussian((e_i1 - e_f - sign_ph * ω_ph) * inv_η) * inv_η
    delta2 = gaussian((e_i2 - e_f - sign_ph * ω_ph) * inv_η) * inv_η
    delta = sqrt(delta1 * delta2)
    for iT in 1:length(p_mel_ikq)
        # occupation factor
        n = sign_ph == 1 ? n_ph[iT] + 1 - f_kq[iT] : n_ph[iT] + f_kq[iT]
        # P matrix element
        p_mel_ikq[iT] += gg[iT] * delta * n
    end
end

@inline function _compute_p_matrix_element_mrta!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, sign_ph, inv_η, f_kq, n_ph, v_i1, v_f)
    # Momentum relaxation time approximation: See Eq. (49) of Ponce et al, Rep. Prog. Phys. (2020).
    # energy conservation factor
    delta1 = gaussian((e_i1 - e_f - sign_ph * ω_ph) * inv_η) * inv_η
    delta2 = gaussian((e_i2 - e_f - sign_ph * ω_ph) * inv_η) * inv_η
    delta = sqrt(delta1 * delta2)
    vfac = 1 - dot(v_i1, v_f) / dot(v_i1, v_i1) # Original Eq. (49)
    # vfac = 1 - dot(v_i1, v_f) / norm(v_i1) / norm(v_f) # Modified form to make 0 <= vfac <= 1
    for iT in 1:length(p_mel_ikq)
        # occupation factor
        n = sign_ph == 1 ? n_ph[iT] + 1 - f_kq[iT] : n_ph[iT] + f_kq[iT]
        # P matrix element
        p_mel_ikq[iT] += gg[iT] * delta * n * vfac
    end
end

@inline function _compute_s_in_matrix_element!(s_mel_ikq, gg, e_i1, e_i2, e_f1, e_f2, ω_ph, sign_ph, inv_η, f_kq1, f_kq2, n_ph)
    # energy conservation factor
    delta1 = gaussian((e_i1 - e_f1 - sign_ph * ω_ph) * inv_η) * inv_η
    delta2 = gaussian((e_i2 - e_f2 - sign_ph * ω_ph) * inv_η) * inv_η
    delta = sqrt(delta1 * delta2)
    for iT in 1:length(s_mel_ikq)
        # occupation factor
        favg = (f_kq1[iT] + f_kq2[iT]) / 2
        n = sign_ph == 1 ? n_ph[iT] + favg : n_ph[iT] + 1 - favg
        # scattering matrix element
        s_mel_ikq[iT] += gg[iT] * delta * n
    end
end

"""
Invert Sₒ, which is a matrix that is block-diagonal in k. The output is returned as a
sparse matrix, keeping the block diagonality.
"""
function invert_scattering_out_matrix(Sₒ, el)
    @assert size(Sₒ) == (el.n, el.n)
    ElType = eltype(Sₒ)
    Sₒ⁻¹ = SparseMatrixCSC{ElType, Int}[]

    ind_block_to_el = zeros(Int, el.nband^2)
    Sₒ_block = zeros(ElType, el.nband^2, el.nband^2)

    # Take the ik block of Sₒ and invert it.
    Is = Int[]
    Js = Int[]
    Values = ElType[]
    for ik in 1:el.kpts.n
        nstates = 0
        ind_block_to_el .= 0
        Sₒ_block .= 0

        # Find the indices for ik.
        for ib2 in el.ib_rng[ik], ib1 in el.ib_rng[ik]
            i = get_1d_index(el, ib1, ib2, ik)
            if i != 0
                nstates += 1
                ind_block_to_el[nstates] = i
            end
        end

        # Extract the ik-th block from Sₒ
        for block_j = 1:nstates, block_i = 1:nstates
            i, j = ind_block_to_el[block_i], ind_block_to_el[block_j]
            Sₒ_block[block_i, block_j] = Sₒ[i, j]
        end
        inv_Sₒ_block = inv(Sₒ_block[1:nstates, 1:nstates])

        # Append nonzero elements of inv_Sₒ_block to the list
        for block_j = 1:nstates, block_i = 1:nstates
            if abs(inv_Sₒ_block[block_i, block_j]) > 0
                push!(Is, ind_block_to_el[block_i])
                push!(Js, ind_block_to_el[block_j])
                push!(Values, inv_Sₒ_block[block_i, block_j])
            end
        end
    end
    Sₒ⁻¹ = dropzeros!(sparse(Is, Js, Values, el.n, el.n))
    Sₒ⁻¹
end

"""
If symmetry involves time reversal, complex conjuate must be taken before applying the unfold
map. So, we return `unfold_map` and `unfold_map_tr`. For the latter, one must apply complex
conjugate and then multiply the map.
"""
function _qme_linear_response_unfold_map(el_i::QMEStates{FT}, el_f::QMEStates{FT}, filename) where FT
    # FIXME: Do not write symmetry twice. Use qme_model.
    fid = h5open(filename, "r")
    symmetry = load_BTData(open_group(fid, "gauge/symmetry"), Symmetry{FT})

    cnt_inds_f = zeros(Int, el_f.n);
    sp_inds_f = Int[]
    sp_inds_i = Int[]
    sp_vals = Mat3{Complex{FT}}[]

    sp_inds_f_tr = Int[]
    sp_inds_i_tr = Int[]
    sp_vals_tr = Mat3{Complex{FT}}[]

    for (isym, symop) in enumerate(symmetry)
        # Read symmetry gauge matrix elements
        group_sym = open_group(fid, "gauge/isym$isym")
        sym_gauge = load_BTData(open_group(group_sym, "gauge_matrix"), OffsetArray{Complex{FT}, 3, Array{Complex{FT}, 3}})
        is_degenerate = load_BTData(open_group(group_sym, "is_degenerate"), OffsetArray{Bool, 3, Array{Bool, 3}})

        for ik in 1:el_i.kpts.n
            xk = el_i.kpts.vectors[ik]
            sxk = symop.is_tr ? -symop.S * xk : symop.S * xk
            isk = xk_to_ik(sxk, el_f.kpts)
            isk === nothing && continue

            # Set unfolding matrix
            for ib2 in el_i.ib_rng[ik], ib1 in el_i.ib_rng[ik]
                ind_el_i = get_1d_index(el_i, ib1, ib2, ik)
                ind_el_i == 0 && continue
                # continue only if ib1 and jb1 are degenerate, and ib2 and jb2 are degenerate.
                for jb2 in el_f.ib_rng[isk]
                    is_degenerate[jb2, ib2, ik] || continue
                    for jb1 in el_f.ib_rng[isk]
                        is_degenerate[jb1, ib1, ik] || continue
                        ind_el_f = get_1d_index(el_f, jb1, jb2, isk)
                        ind_el_f == 0 && continue
                        gauge_coeff = sym_gauge[jb1, ib1, ik] * sym_gauge[jb2, ib2, ik]'
                        if symop.is_tr
                            push!(sp_inds_f_tr, ind_el_f)
                            push!(sp_inds_i_tr, ind_el_i)
                            push!(sp_vals_tr, -symop.Scart * gauge_coeff)
                        else
                            push!(sp_inds_f, ind_el_f)
                            push!(sp_inds_i, ind_el_i)
                            push!(sp_vals, symop.Scart * gauge_coeff)
                        end
                        # Count number of k points that are mapped to this Sk point
                        if (ib1 == jb1) && (ib2 == jb2)
                            cnt_inds_f[ind_el_f] += 1
                        end
                    end
                end
            end
        end
    end
    close(fid)

    unfold_map = sparse(sp_inds_f, sp_inds_i, sp_vals, el_f.n, el_i.n)
    unfold_map_tr = sparse(sp_inds_f_tr, sp_inds_i_tr, sp_vals_tr, el_f.n, el_i.n)

    inv_cnt_inds_f = 1 ./ cnt_inds_f
    inv_cnt_inds_f[cnt_inds_f .== 0] .= 0
    unfold_map .*= inv_cnt_inds_f
    unfold_map_tr .*= inv_cnt_inds_f

    unfold_map, unfold_map_tr
end


function _qme_linear_response_unfold_map_nosym(el_i::QMEStates{FT}, el_f::QMEStates{FT}, filename) where FT
    fid = h5open(filename, "r")
    gauge = load_BTData(open_group(fid, "gauge/gauge_matrix"), OffsetArray{Complex{FT}, 3, Array{Complex{FT}, 3}})
    is_degenerate = load_BTData(open_group(fid, "gauge/is_degenerate"), OffsetArray{Bool, 3, Array{Bool, 3}})

    # We assume that all el_i and el_f use the same grid and same shift.
    δk = el_i.kpts.shift ≈ el_f.kpts.shift
    @assert all(δk - round.(δk) .≈ 0)
    @assert el_i.kpts.ngrid == el_f.kpts.ngrid

    sp_inds_f = Int[]
    sp_inds_i = Int[]
    sp_vals = Complex{FT}[]

    for ind_el_i in 1:el_i.n
        ik = el_i.ik[ind_el_i]
        xk = el_i.kpts.vectors[ik]
        ik_f = xk_to_ik(xk, el_f.kpts)
        ik_f === nothing && continue

        ib1 = el_i.ib1[ind_el_i]
        ib2 = el_i.ib2[ind_el_i]

        # continue only if ib1 and jb1 are degenerate and ib2 and jb2 are degenerate.
        for jb2 in el_f.ib_rng[ik_f]
            is_degenerate[jb2, ib2, ik] || continue
            for jb1 in el_f.ib_rng[ik_f]
                is_degenerate[jb1, ib1, ik] || continue

                ind_el_f = get_1d_index(el_f, jb1, jb2, ik_f)
                ind_el_f == 0 && continue

                gauge_coeff = gauge[jb1, ib1, ik] * conj(gauge[jb2, ib2, ik])
                push!(sp_inds_f, ind_el_f)
                push!(sp_inds_i, ind_el_i)
                push!(sp_vals, gauge_coeff)
            end
        end
    end
    close(fid)
    unfold_map = sparse(sp_inds_f, sp_inds_i, sp_vals, el_f.n, el_i.n)
    unfold_map
end

# TODO: Cleanup. Give SERTA and exact output at the same time.
function compute_transport_distribution_function(out_qme, δρ=out_qme.δρ, el::QMEStates=out_qme.el; elist, smearing, symmetry=nothing)
    Σ_tdf = zeros(length(elist), 3, 3, length(out_qme.params.Tlist))
    e_gaussian = zero(elist)
    for iT in 1:length(out_qme.params.Tlist)
        @views for i in 1:el.n
            @. e_gaussian = gaussian((elist - el.e1[i]) / smearing) / smearing
            σ_i = (el.kpts.weights[el.ik[i]] * real.(δρ[iT][i] * el.v[i]'))
            for b in 1:3, a in 1:3
                Σ_tdf[:, a, b, iT] .+= e_gaussian .* σ_i[a, b]
            end
        end
    end
    Σ_tdf .*= out_qme.params.spin_degeneracy / out_qme.params.volume
    for arr in eachslice(Σ_tdf, dims=1)
        arr .= symmetrize_array(arr, symmetry; order=2)
    end
    Σ_tdf
end
