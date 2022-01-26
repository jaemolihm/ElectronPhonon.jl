using SparseArrays

# Constructing and solving quantum master equation for electrons

# TODO: Modularize. Make QMEModel type.

# TODO: Accessing scat[ikq][CI(ib, jb, imode)] is the bottleneck (takes half of the time in
#       scat_in). Need to optimize.

export compute_qme_scattering_matrix, solve_electron_qme

"""
    compute_qme_scattering_matrix(filename, params, el_i, el_f, ph)
Compute the scattering matrix element for quantum master equation of electrons.
"""
function compute_qme_scattering_matrix(filename, params, el_i::QMEStates{FT}, el_f::QMEStates{FT}, ph; compute_S_in=true) where {FT}
    nT = length(params.Tlist)

    indmap_el_i = states_index_map(el_i)
    ind_ph_map = states_index_map(ph)

    fid = h5open(filename, "r")
    group_scattering = open_group(fid, "scattering")
    mpi_isroot() && println("Original grid: Total $(length(group_scattering)) groups of scattering")

    @assert params.smearing[1] == :Gaussian
    η = params.smearing[2]
    inv_η = 1 / η

    p_mel = zeros(Complex{FT}, nT)
    p_mel_ikq = zeros(Complex{FT}, nT)
    s_mel_ikq = zeros(Complex{FT}, nT)

    # Scattering-out and scattering-in matrices
    # S_out is sparse (block diagonal) while S_in is not sparse in general.
    S_out = [spzeros(Complex{FT}, el_i.n, el_i.n) for _ in 1:nT]
    S_in = [zeros(Complex{FT}, el_i.n, el_f.n) for _ in 1:nT]

    for ik in 1:el_i.kpts.n
        mpi_isroot() && mod(ik, 10) == 0 && println("Calculating scattering for group $ik")

        @timing "read scat" scat = load_BTData(open_group(group_scattering, "ik$ik"), QMEElPhScatteringData{FT})

        # 1. Scattering-out term
        # P_{ib1, ib2} = sum_{ikq, imode, jb, ±} g*_{jb, ib1} * g_{jb, ib2}
        #              * δ^{1/2}(e_ib1 - e_jb ± ω_imode) * δ^{1/2}(e_ib2 - e_jb ± ω_imode)
        #              * (n_imode + f_jb)       (+)
        #                (n_imode + 1 - f_jb)   (-)
        # dδρ_{ib1, ib2}/dt = - [P, δρ]_{ib1, ib2} / 2

        @timing "scat out" for ib2 in 1:el_i.nband, ib1 in 1:el_i.nband
            # Calculate P_{ib1, ib2} only if ∃ ib3 such that both (ib1, ib3) and (ib2, ib3)
            # or both (ib3, ib1) and (ib3, ib2) are in el_i.
            found = false
            local ib3, e_i1, e_i2
            for outer ib3 in 1:el_i.nband
                if haskey(indmap_el_i, CI(ib1, ib3, ik)) && haskey(indmap_el_i, CI(ib2, ib3, ik))
                    found = true
                    e_i1 = el_i.e1[indmap_el_i[CI(ib1, ib3, ik)]]
                    e_i2 = el_i.e1[indmap_el_i[CI(ib2, ib3, ik)]]
                    break
                end
                if haskey(indmap_el_i, CI(ib1, ib3, ik)) && haskey(indmap_el_i, CI(ib2, ib3, ik))
                    found = true
                    e_i1 = el_i.e2[indmap_el_i[CI(ib3, ib1, ik)]]
                    e_i2 = el_i.e2[indmap_el_i[CI(ib3, ib2, ik)]]
                    break
                end
            end
            found || continue
            p_mel .= 0

            for ind_el_f in 1:el_f.n
                jb1 = el_f.ib1[ind_el_f]
                jb2 = el_f.ib2[ind_el_f]
                jb1 != jb2 && continue # Only diagonal part contributes to the scattering-out term
                jb = jb1
                e_f = el_f.e1[ind_el_f]

                ikq = el_f.ik[ind_el_f]
                ikq > length(scat) && continue # skip if this ikq is not in scat

                xq = el_f.kpts.vectors[ikq] - el_i.kpts.vectors[ik]
                xq_int = mod.(round.(Int, xq .* ph.ngrid), ph.ngrid)
                ind_ph_list = get(ind_ph_map, CI(xq_int...), nothing)
                ind_ph_list === nothing && continue # skip if this xq is not in ph

                p_mel_ikq .= 0

                for imode in 1:ph.nband
                    ind_ph = ind_ph_list[imode]
                    ind_ph == 0 && continue # skip if this imode is not in ph
                    ω_ph = ph.e[ind_ph]
                    # Skip if phonon frequency is too close to 0 (acoustic phonon at q=0)
                    ω_ph < EPW.omega_acoustic && continue

                    # Matrix element factor
                    s1 = get(scat[ikq], CI(ib1, jb, imode), nothing)
                    s1 === nothing && continue
                    s2 = get(scat[ikq], CI(ib2, jb, imode), nothing)
                    s2 === nothing && continue
                    gg = conj(s1.mel) * s2.mel

                    if s1.econv_p && s2.econv_p
                        _compute_p_matrix_element!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, +1, inv_η, params.Tlist, params.μlist)
                    end
                    if s1.econv_m && s1.econv_m
                        _compute_p_matrix_element!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, -1, inv_η, params.Tlist, params.μlist)
                    end
                end
                p_mel .+= p_mel_ikq .* 2FT(π) .* el_f.kpts.weights[ikq]
            end # ikq

            # Calculate the scattering matrix. Contribution of P_{ib1, ib2} are:
            # S_out[(ib1,ib3), (ib2,ib3)] += - P_{ib1, ib2} / 2
            # S_out[(ib3,ib2), (ib3,ib1)] += - P_{ib1, ib2} / 2
            for ib3 in 1:el_i.nband
                if haskey(indmap_el_i, CI(ib1, ib3, ik)) && haskey(indmap_el_i, CI(ib2, ib3, ik))
                    i1 = indmap_el_i[CI(ib1, ib3, ik)]
                    i2 = indmap_el_i[CI(ib2, ib3, ik)]
                    for iT in 1:nT
                        S_out[iT][i1, i2] += -p_mel[iT] / 2
                    end
                end
                if haskey(indmap_el_i, CI(ib3, ib1, ik)) && haskey(indmap_el_i, CI(ib3, ib2, ik))
                    i1 = indmap_el_i[CI(ib3, ib2, ik)]
                    i2 = indmap_el_i[CI(ib3, ib1, ik)]
                    for iT in 1:nT
                        S_out[iT][i1, i2] += -p_mel[iT] / 2
                    end
                end
            end
        end # ib1, ib2

        if ! compute_S_in
            continue
        end

        # 2. Scattering-in term
        # dδρ_{ib1,ib2,k}/dt = sum_{ikq, imode, jb1, jb2, ±} g*_{jb1, ib1} * g_{jb2, ib2}
        #                    * δ^{1/2}(e_ib1 - e_jb1 ± ω_imode) * δ^{1/2}(e_ib2 - e_jb2 ± ω_imode)
        #                    * (n_imode +     (f_jb1 + f_jb2) / 2 )       (+)
        #                      (n_imode + 1 - (f_jb1 + f_jb2) / 2 )       (-)

        @timing "scat in" for ib2 in 1:el_i.nband, ib1 in 1:el_i.nband
            # Calculate only if (ib1, ib2, ik) ∈ el_i
            ind_el_i = get(indmap_el_i, CI(ib1, ib2, ik), nothing)
            ind_el_i === nothing && continue
            e_i1 = el_i.e1[ind_el_i]
            e_i2 = el_i.e2[ind_el_i]

            for ind_el_f in 1:el_f.n
                jb1 = el_f.ib1[ind_el_f]
                jb2 = el_f.ib2[ind_el_f]
                e_f1 = el_f.e1[ind_el_f]
                e_f2 = el_f.e2[ind_el_f]

                # Find q point
                ikq = el_f.ik[ind_el_f]
                ikq > length(scat) && continue # skip if this ikq is not in scat

                xq = el_f.kpts.vectors[ikq] - el_i.kpts.vectors[ik]
                xq_int = mod.(round.(Int, xq .* ph.ngrid), ph.ngrid)
                ind_ph_list = get(ind_ph_map, CI(xq_int...), nothing)
                ind_ph_list === nothing && continue # skip if this xq is not in ph

                s_mel_ikq .= 0

                for imode in 1:ph.nband
                    ind_ph = ind_ph_list[imode]
                    ind_ph == 0 && continue # skip if this imode is not in ph
                    ω_ph = ph.e[ind_ph]
                    # Skip if phonon frequency is too close to 0 (acoustic phonon at q=0)
                    ω_ph < EPW.omega_acoustic && continue

                    # Matrix element factor
                    s1 = get(scat[ikq], CI(ib1, jb1, imode), nothing)
                    s1 === nothing && continue
                    s2 = get(scat[ikq], CI(ib2, jb2, imode), nothing)
                    s2 === nothing && continue
                    gg = conj(s1.mel) * s2.mel

                    if s1.econv_p && s2.econv_p
                        _compute_s_in_matrix_element!(s_mel_ikq, gg, e_i1, e_i2, e_f1, e_f2, ω_ph, +1, inv_η, params.Tlist, params.μlist)
                    end
                    if s1.econv_m && s1.econv_m
                        _compute_s_in_matrix_element!(s_mel_ikq, gg, e_i1, e_i2, e_f1, e_f2, ω_ph, -1, inv_η, params.Tlist, params.μlist)
                    end
                end
                s_mel_ikq .*= 2FT(π) * el_f.kpts.weights[ikq]

                for iT in 1:nT
                    S_in[iT][ind_el_i, ind_el_f] += s_mel_ikq[iT]
                end
            end
        end

    end # ik
    close(fid)
    S_out, S_in
end

function _compute_p_matrix_element!(p_mel_ikq, gg, e_i1, e_i2, e_f, ω_ph, sign_ph, inv_η, Tlist, μlist)
    # energy conservation factor
    delta1 = gaussian((e_i1 - e_f - sign_ph * ω_ph) * inv_η) * inv_η
    delta2 = gaussian((e_i2 - e_f - sign_ph * ω_ph) * inv_η) * inv_η
    delta = sqrt(delta1 * delta2)
    for (iT, (T, μ)) in enumerate(zip(Tlist, μlist))
        # occupation factor
        n_ph = occ_boson(ω_ph, T)
        f_kq = occ_fermion(e_f - μ, T)
        n = sign_ph == 1 ? n_ph + 1 - f_kq : n_ph + f_kq
        # P matrix element
        p_mel_ikq[iT] += gg * delta * n
    end
end

function _compute_s_in_matrix_element!(s_mel_ikq, gg, e_i1, e_i2, e_f1, e_f2, ω_ph, sign_ph, inv_η, Tlist, μlist)
    # energy conservation factor
    delta1 = gaussian((e_i1 - e_f1 - sign_ph * ω_ph) * inv_η) * inv_η
    delta2 = gaussian((e_i2 - e_f2 - sign_ph * ω_ph) * inv_η) * inv_η
    delta = sqrt(delta1 * delta2)
    for (iT, (T, μ)) in enumerate(zip(Tlist, μlist))
        # occupation factor
        n_ph = occ_boson(ω_ph, T)
        f_kq1 = occ_fermion(e_i1 - μ, T)
        f_kq2 = occ_fermion(e_i2 - μ, T)
        favg = (f_kq1 + f_kq2) / 2
        n = sign_ph == 1 ? n_ph + favg : n_ph + 1 - favg
        # scattering matrix element
        s_mel_ikq[iT] += gg * delta * n
    end
end

"""
    occupation_to_conductivity(δρ, el::QMEStates, params)
Compute electron conductivity using the density matrix `δρ`.
"""
function occupation_to_conductivity(δρ, el::QMEStates, params)
    @assert length(δρ) == el.n
    σ = mapreduce(i -> el.kpts.weights[el.ik[i]] .* real.(δρ[i] * el.v[i]'), +, 1:el.n)
    return σ * params.spin_degeneracy / params.volume
end

"""
    function solve_electron_qme(params, el_i::QMEStates{FT}, el_f::QMEStates{FT}, scat_mat_out,
        scat_mat_in=nothing; symmetry=nothing, max_iter=100, rtol=1e-10, qme_offdiag_cutoff=Inf) where {FT}
Solve quantum master equation for electrons.
Linearized quantum master equation (stationary state case):
```math
0 = ∂ δρ / ∂t
  = -i(e1[i] - e2[i]) * δρ[i] + drive_efield[i] + ∑_j (S_out + S_in * map_i_to_f)[i, j] * δρ[j],
```
where
```math
drive_efield[i] = - v[i] * (df/dε)_{ε=e1[i]}              : if e_mk  = e_nk
                = - v[i] * (f_mk - f_nk) / (e_mk - e_nk)  : if e_mk /= e_nk
```
and ``i = (m, n, k)``, ``δρ_i = δρ_{mn;k}``, ``e1[i], e2[i] = e_mk, e_nk``, and ``v[i] = v_{mn;k}``.

We need `map_i_to_f` because `S_in` maps states in `el_f` to `el_i` (i.e. has size `(el_i.n, el_f.n)`),
while `δρ` is for states in `el_i`. `el_i` and `el_f` can differ due to use of irreducible grids,
different windows, different grids, etc. So, we need to first map `δρ` to states `el_f` using `map_i_to_f`.
"""
function solve_electron_qme(params, el_i::QMEStates{FT}, el_f::Union{QMEStates{FT},Nothing}, scat_mat_out,
        scat_mat_in=nothing; filename="", symmetry=nothing, max_iter=100, rtol=1e-10, qme_offdiag_cutoff=Inf) where {FT}
    if scat_mat_in !== nothing
        ! isfile(filename) && error("filename = $filename is not a valid file.")
        el_f === nothing && throw(ArgumentError("If scat_mat_in is used (exact LBTE), el_f must be provided."))
    end

    σ_serta = zeros(FT, 3, 3, length(params.Tlist))
    σ = zeros(FT, 3, 3, length(params.Tlist))
    δρ_serta = zeros(Vec3{Complex{FT}}, el_i.n, length(params.Tlist))
    δρ = zeros(Vec3{Complex{FT}}, el_i.n, length(params.Tlist))
    σ_iter = fill(FT(NaN), (max_iter+1, 3, 3, length(params.Tlist)))

    drive_efield = zeros(Vec3{Complex{FT}}, el_i.n)
    δρ_iter = zeros(Vec3{Complex{FT}}, el_i.n)
    δρ_iter_tmp = zeros(Vec3{Complex{FT}}, el_i.n)

    inds_exclude = (el_i.ib1 .!= el_i.ib2) .&& (abs.(el_i.e1 .- el_i.e2) .> qme_offdiag_cutoff)

    # setup map_i_to_f. This is needed only when solving the linear equation iteratively.
    @timing "unfold map" if scat_mat_in !== nothing
        if symmetry === nothing
            map_i_to_f = _qme_linear_response_unfold_map_nosym(el_i, el_f, filename)
        else
            map_i_to_f = _qme_linear_response_unfold_map(el_i, el_f, filename)
        end
    end

    for (iT, (T, μ)) in enumerate(zip(params.Tlist, params.μlist))
        # Compute the E-field drive term
        @timing "drive_efield" for i in 1:el_i.n
            e1, e2 = el_i.e1[i], el_i.e2[i]
            if abs(e1 - e2) < EPW.electron_degen_cutoff
                drive_efield[i] = - el_i.v[i] * occ_fermion_derivative(e1 - μ, T)
            else
                drive_efield[i] = - el_i.v[i] * (occ_fermion(e1 - μ, T) - occ_fermion(e2 - μ, T)) / (e1 - e2)
            end
        end
        drive_efield[inds_exclude] .= Ref(zero(Vec3{Complex{FT}}))

        # Add the scattering-out term and the bare Hamiltonian term into S_serta
        S_serta = copy(scat_mat_out[iT])
        for i in 1:el_i.n
            e1, e2 = el_i.e1[i], el_i.e2[i]
            if abs(e1 - e2) >= EPW.electron_degen_cutoff
                S_serta[i, i] += -im * (e1 - e2)
            end
        end
        S_serta_factorize = factorize(S_serta)

        # QME-SERTA: Solve S_out * δρ + drive_efield = 0
        @views _solve_qme_direct!(δρ_serta[:, iT], S_serta_factorize, .-drive_efield)
        δρ_serta[inds_exclude, iT] .= Ref(zero(Vec3{Complex{FT}}))
        σ_serta[:, :, iT] .= symmetrize(occupation_to_conductivity(δρ_serta[:, iT], el_i, params), symmetry)

        # QME-exact: Solve (S_out + S_in) * δρ + drive_efield = 0
        # Solve iteratively the fixed point equation δρ = S_out^{-1} * (-S_in * δρ - drive_efield)
        if scat_mat_in !== nothing
            # Scattering matrix: first unfold to el_f and then apply scat_mat_in.
            S_in = scat_mat_in[iT] * map_i_to_f

            # Initial guess: SERTA density matrix
            @views δρ_iter .= δρ_serta[:, iT]
            σ_new = symmetrize(occupation_to_conductivity(δρ_iter, el_i, params), symmetry)
            σ_iter[1, :, :, iT] .= σ_new

            # Fixed-point iteration
            for iter in 1:max_iter
                σ_old = σ_new

                # Compute δρ_iter_next = S_out^{-1} * (-S_in * δρ_iter_prev - drive_efield)
                #                      = - S_out^{-1} * S_in * δρ_iter_prev + δρ_serta[:, iT]
                @timing "S_in" mul!(δρ_iter_tmp, S_in, δρ_iter, -1, 0)
                δρ_iter_tmp[inds_exclude] .= Ref(zero(Vec3{Complex{FT}}))
                _solve_qme_direct!(δρ_iter, S_serta_factorize, δρ_iter_tmp)
                δρ_iter[inds_exclude] .= Ref(zero(Vec3{Complex{FT}}))
                @views δρ_iter .+= δρ_serta[:, iT]
                σ_new = symmetrize(occupation_to_conductivity(δρ_iter, el_i, params), symmetry)
                σ_iter[iter+1, :, :, iT] .= σ_new

                # Check convergence
                if norm(σ_new - σ_old) / norm(σ_new) < rtol
                    @info "iT=$iT, converged at iteration $iter"
                    break
                elseif iter == max_iter
                    @info "iT=$iT, convergence not reached at maximum iteration $max_iter"
                end
            end

            σ[:, :, iT] .= σ_new
            δρ[:, iT] .= δρ_iter
        else
            σ[:, :, iT] .= NaN
            # δρ[:, iT] .= NaN # FIXME
        end
    end
    (;σ, σ_serta, δρ_serta, δρ, σ_iter, el=el_i, params)
end

# Solve S * δρ = δρ0 using left division.
@timing "_solve_qme_direct!" function _solve_qme_direct!(δρ::AbstractVector{Vec3{Complex{FT}}}, S, δρ0::AbstractVector{Vec3{Complex{FT}}}) where FT
    # Here, we use \ although it allocates because non-allocating ldiv! not supported for
    # sparse matrix: https://github.com/JuliaLang/SuiteSparse.jl/issues/19
    @views for a in 1:3
        b = reshape(reinterpret(Complex{FT}, δρ0), 3, :)[a, :]
        reshape(reinterpret(Complex{FT}, δρ), 3, :)[a, :] .= S \ b;
    end
end

# TODO: check performance of unfold_map (adding to sparse matrix)
function _qme_linear_response_unfold_map(el_i::QMEStates{FT}, el_f::QMEStates{FT}, filename) where FT
    indmap_el_i = EPW.states_index_map(el_i);
    indmap_el_f = EPW.states_index_map(el_f);

    fid = h5open(filename, "r")
    symmetry = load_BTData(open_group(fid, "gauge/symmetry"), Symmetry{FT})

    cnt_inds_f = zeros(Int, el_f.n);
    sp_inds_f = Int[]
    sp_inds_i = Int[]
    sp_vals = Mat3{Complex{FT}}[]
    for isym in 1:symmetry.nsym
        # Read symmetry gauge matrix elements
        Scart = symmetry[isym].Scart
        group_sym = open_group(fid, "gauge/isym$isym")
        sym_gauge = read(group_sym, "gauge_matrix")::Array{Complex{FT}, 3}
        is_degenerate = read(group_sym, "is_degenerate")::Array{Bool, 3}

        for ik in 1:el_i.kpts.n
            xk = el_i.kpts.vectors[ik]
            sxk = symmetry[isym].S * xk
            isk = xk_to_ik(sxk, el_f.kpts)

            # Set unfolding matrix
            for ib2 in 1:el_i.nband, ib1 in 1:el_i.nband
                ind_el_i = get(indmap_el_i, EPW.CI(ib1, ib2, ik), -1)
                ind_el_i == -1 && continue
                # continue only if ib1 and jb1 are degenerate, and ib2 and jb2 are degenerate.
                for jb2 in 1:el_f.nband
                    is_degenerate[jb2, ib2, ik] || continue
                    for jb1 in 1:el_f.nband
                        is_degenerate[jb1, ib1, ik] || continue
                        ind_el_f = get(indmap_el_f, EPW.CI(jb1, jb2, isk), -1)
                        ind_el_f == -1 && continue
                        gauge_coeff = sym_gauge[jb1, ib1, ik] * conj(sym_gauge[jb2, ib2, ik])
                        push!(sp_inds_f, ind_el_f)
                        push!(sp_inds_i, ind_el_i)
                        push!(sp_vals, Scart * gauge_coeff)
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

    inv_cnt_inds_f = 1 ./ cnt_inds_f
    inv_cnt_inds_f[cnt_inds_f .== 0] .= 0
    unfold_map .*= inv_cnt_inds_f
    unfold_map
end


function _qme_linear_response_unfold_map_nosym(el_i::QMEStates{FT}, el_f::QMEStates{FT}, filename) where FT
    fid = h5open(filename, "r")
    gauge = read(open_group(fid, "gauge"), "gauge_matrix")::Array{Complex{FT}, 3}
    is_degenerate = read(open_group(fid, "gauge"), "is_degenerate")::Array{Bool, 3}

    # We assume that all el_i and el_f use the same grid and same shift.
    δk = el_i.kpts.shift ≈ el_f.kpts.shift
    @assert all(δk - round.(δk) .≈ 0)
    @assert el_i.kpts.ngrid == el_f.kpts.ngrid

    indmap_el_f = EPW.states_index_map(el_f);
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

        # continue only if ib1 and jb1 are degenerate, and ib2 and jb2 are degenerate.
        # FIXME: 1:el_f.nband is not safe. One should use iband_min:iband_max.
        for jb2 in 1:el_f.nband
            is_degenerate[jb2, ib2, ik] || continue
            for jb1 in 1:el_f.nband
                is_degenerate[jb1, ib1, ik] || continue

                ind_el_f = get(indmap_el_f, EPW.CI(jb1, jb2, ik_f), -1)
                ind_el_f == -1 && continue

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
function compute_transport_distribution_function(out_qme, δρ=out_qme.δρ, el::EPW.QMEStates=out_qme.el; elist, smearing, symmetry=nothing)
    Σ_tdf = zeros(length(elist), 3, 3, length(out_qme.params.Tlist))
    e_gaussian = zero(elist)
    for iT in 1:length(out_qme.params.Tlist)
        @views for i in 1:el.n
            @. e_gaussian = gaussian((elist - el.e1[i]) / smearing) / smearing
            σ_i = (el.kpts.weights[el.ik[i]] * real.(δρ[i, iT] * el.v[i]'))
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
