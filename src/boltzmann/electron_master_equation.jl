using SparseArrays

# Constructing and solving quantum master equation for electrons

# TODO: Scattering in term

export compute_qme_scattering_matrix, solve_electron_qme

"""
    compute_qme_scattering_matrix(filename, params, el_i, el_f, ph)
Compute the scattering matrix element for quantum master equation of electrons.
"""
function compute_qme_scattering_matrix(filename, params, el_i::QMEStates{FT}, el_f, ph) where {FT}
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
    S_in = [zeros(Complex{FT}, el_i.n, el_i.n) for _ in 1:nT]

    for ik in 1:el_i.kpts.n
        mpi_isroot() && mod(ik, 100) == 0 && println("Calculating scattering for group $ik")

        scat = load_BTData(open_group(group_scattering, "ik$ik"), QMEElPhScatteringData{FT})

        # 1. Scattering-out term
        # P_{ib1, ib2} = sum_{ikq, imode, jb, ±} g*_{jb, ib1} * g_{jb, ib2}
        #              * δ^{1/2}(e_ib1 - e_jb ± ω_imode) * δ^{1/2}(e_ib2 - e_jb ± ω_imode)
        #              * (n_imode + f_jb)       (+)
        #                (n_imode + 1 - f_jb)   (-)
        # dδρ_{ib1, ib2}/dt = - [P, δρ]_{ib1, ib2} / 2

        for ib2 in 1:el_i.nband, ib1 in 1:el_i.nband
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
                    s1 = get(scat, CI(ik, ib1, ikq, jb, imode), nothing)
                    s1 === nothing && continue
                    s2 = get(scat, CI(ik, ib2, ikq, jb, imode), nothing)
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

        # 2. Scattering-in term
        # dδρ_{ib1,ib2,k}/dt = sum_{ikq, imode, jb1, jb2, ±} g*_{jb1, ib1} * g_{jb2, ib2}
        #                    * δ^{1/2}(e_ib1 - e_jb1 ± ω_imode) * δ^{1/2}(e_ib2 - e_jb2 ± ω_imode)
        #                    * (n_imode +     (f_jb1 + f_jb2) / 2 )       (+)
        #                      (n_imode + 1 - (f_jb1 + f_jb2) / 2 )       (-)

        for ib2 in 1:el_i.nband, ib1 in 1:el_i.nband
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
                    s1 = get(scat, CI(ik, ib1, ikq, jb1, imode), nothing)
                    s1 === nothing && continue
                    s2 = get(scat, CI(ik, ib2, ikq, jb2, imode), nothing)
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
    occupation_to_conductivity(δρ::Vector{Vec3{FT}}, el::QMEStates{FT}, params) where {FT}
Compute electron conductivity using the density matrix `δρ`.
"""
function occupation_to_conductivity(δρ::Vector{Vec3{Complex{FT}}}, el::QMEStates{FT}, params) where {FT}
    @assert length(δρ) == el.n
    σ = zero(Mat3{FT})
    @views for i in 1:el.n
        σ += el.kpts.weights[el.ik[i]] * real.(δρ[i] * el.v[i]')
    end
    σ * params.spin_degeneracy / params.volume
end

"""
    function solve_electron_qme(params, el_i::QMEStates{FT}, el_f::QMEStates{FT}, scat_mat_out,
        scat_mat_in=nothing; symmetry=nothing, max_iter=100, rtol=1e-10) where {FT}
Solve quantum master equation for electrons.
TODO: symmetry
TODO: scattering out term (i.e. beyond serta)
"""
function solve_electron_qme(params, el_i::QMEStates{FT}, el_f::QMEStates{FT}, scat_mat_out,
        scat_mat_in=nothing; symmetry=nothing, max_iter=100, rtol=1e-10) where {FT}
    σ_serta = zeros(FT, 3, 3, length(params.Tlist))
    σ = zeros(FT, 3, 3, length(params.Tlist))
    δρ_serta = zeros(Vec3{Complex{FT}}, el_i.n, length(params.Tlist))
    δρ = zeros(Vec3{Complex{FT}}, el_i.n, length(params.Tlist))

    δρ_external = zeros(Vec3{Complex{FT}}, el_i.n)
    δρ_iter_old = zeros(Vec3{Complex{FT}}, el_i.n)
    δρ_iter_new = zeros(Vec3{Complex{FT}}, el_i.n)
    δρ_iter_tmp = zeros(Vec3{Complex{FT}}, el_i.n)

    if symmetry === nothing
        # FIXME: no symmetry but different grid
        @assert el_i.n == el_f.n
        map_i_to_f = I(el_i.n)
    else
        error("symmetry not implemented")
        # map_i_to_f = vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry)
    end

    for (iT, (T, μ)) in enumerate(zip(params.Tlist, params.μlist))
        @. δρ_external = el_i.v * -EPW.occ_fermion_derivative(el_i.e1 - μ, T);
        S_out = scat_mat_out[iT]
        S_out_factorize = factorize(S_out)

        # QME-SERTA: Solve S_out * δρ + δρ_external = 0
        @views _solve_qme_direct!(δρ_serta[:, iT], S_out_factorize, .-δρ_external)
        σ_serta[:, :, iT] .= symmetrize(occupation_to_conductivity(δρ_serta[:, iT], el_i, params), symmetry)

        # QME-exact: Solve (S_out + S_in) * δρ + δρ_external = 0
        # Solve iteratively the fixed point equation δρ = S_out^{-1} * (-S_in * δρ - δρ_external)
        if scat_mat_in !== nothing
            S_in = scat_mat_in[iT]

            # Initial guess: SERTA density matrix
            @views δρ_iter_new .= δρ_serta[:, iT]
            σ_new = symmetrize(occupation_to_conductivity(δρ_iter_new, el_i, params), symmetry)

            # Fixed-point iteration
            for iter in 1:max_iter
                σ_old = σ_new
                δρ_iter_old .= δρ_iter_new

                # TODO: Symmetry
                # Unfold from δf_i to δf_f
                # δf_f = map_i_to_f * δf_i

                # Compute δρ_iter_new = S_out^{-1} * (-S_in * δρ_iter_old - δρ_external)
                #                     = - S_out^{-1} * S_in * δρ_iter_old + δρ_serta[:, iT]
                mul!(δρ_iter_tmp, S_in, δρ_iter_old, -1, 0)
                _solve_qme_direct!(δρ_iter_new, S_out_factorize, δρ_iter_tmp)
                @views δρ_iter_new .+= δρ_serta[:, iT]
                σ_new = symmetrize(occupation_to_conductivity(δρ_iter_new, el_i, params), symmetry)

                # Check convergence
                if norm(σ_new - σ_old) / norm(σ_new) < rtol
                    @info "iT=$iT, converged at iteration $iter"
                    break
                elseif iter == max_iter
                    @info "iT=$iT, convergence not reached at maximum iteration $max_iter"
                end
            end

            σ[:, :, iT] .= σ_new
            δρ[:, iT] .= δρ_iter_new
        else
            σ[:, :, iT] .= NaN
            # δρ[:, iT] .= NaN # FIXME
        end
    end
    (;σ, σ_serta, δρ_serta, δρ)
end

# Solve S * δρ = δρ0 using left division.
function _solve_qme_direct!(δρ::AbstractVector{Vec3{Complex{FT}}}, S, δρ0::AbstractVector{Vec3{Complex{FT}}}) where FT
    # Here, we use \ although it allocates because non-allocating ldiv! not supported for
    # sparse matrix: https://github.com/JuliaLang/SuiteSparse.jl/issues/19
    @views for a in 1:3
        b = reshape(reinterpret(Complex{FT}, δρ0), 3, :)[a, :]
        reshape(reinterpret(Complex{FT}, δρ), 3, :)[a, :] .= S \ b;
    end
end
