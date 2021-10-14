using SparseArrays

# Constructing and solving quantum master equation for electrons

# TODO: Scattering in term

export compute_qme_scattering_matrix, solve_electron_qme

"""
    compute_qme_scattering_matrix(filename, params, el_i, el_f, ph)
Compute the scattering matrix element for quantum master equation of electrons.
"""
function compute_qme_scattering_matrix(filename, params, el_i, el_f, ph)
    FT = Float64
    nT = length(params.Tlist)

    indmap_el_i = states_index_map(el_i)
    ind_ph_map = EPW.states_index_map(ph)

    # FIXME: JLD2 group is not iterable...
    fid = jldopen(filename, "r")
    mpi_isroot() && println("Original grid: Total $(el_i.kpts.n) groups of scattering")

    @assert params.smearing[1] == :Gaussian
    η = params.smearing[2]
    inv_η = 1 / η

    p_mel = zeros(Complex{FT}, nT)
    p_mel_ikq = zeros(Complex{FT}, nT)

    # Scattering-out matrix
    S_out = [spzeros(Complex{FT}, el_i.n, el_i.n) for iT in 1:nT]

    @time for ik in 1:el_i.kpts.n
        scat = load_scatteringdata(fid["scattering/ik$ik"])

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

                p_mel_ikq .= 0

                for imode in 1:ph.nband
                    ind_ph = ind_ph_map[(xq_int..., imode)]
                    ω_ph = ph.e[ind_ph]
                    # Skip if phonon frequency is too close to 0 (acoustic phonon at q=0)
                    ω_ph < EPW.omega_acoustic && continue

                    # Matrix element factor
                    haskey(scat, CI(ik, ib1, ikq, jb, imode)) || continue
                    haskey(scat, CI(ik, ib2, ikq, jb, imode)) || continue
                    s1 = scat[CI(ik, ib1, ikq, jb, imode)]
                    s2 = scat[CI(ik, ib2, ikq, jb, imode)]
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
        # TODO
    end # ik
    close(fid)
    S_out
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
    function solve_electron_qme(el_i::QMEStates{FT}, el_f::QMEStates{FT}, scat_mat_out, params,
        symmetry=nothing; max_iter=100, rtol=1e-10) where {FT}
Solve quantum master equation for electrons.
TODO: symmetry
TODO: scattering out term (i.e. beyond serta)
"""
function solve_electron_qme(el_i::QMEStates{FT}, el_f::QMEStates{FT}, scat_mat_out, params,
        symmetry=nothing; max_iter=100, rtol=1e-10) where {FT}
    σ_serta = zeros(FT, 3, 3, length(params.Tlist))
    σ = zeros(FT, 3, 3, length(params.Tlist))
    δρ_serta = zeros(Vec3{Complex{FT}}, el_i.n, length(params.Tlist))
    δρ = zeros(Vec3{Complex{FT}}, el_i.n, length(params.Tlist))

    δρ_external = zeros(Vec3{Complex{FT}}, el_i.n)
    δρ_iter_old = zeros(Vec3{Complex{FT}}, el_i.n)
    δρ_iter_new = zeros(Vec3{Complex{FT}}, el_i.n)

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
        Sout = scat_mat_out[iT]

        # SERTA: Solve S_out * δρ + δρ_external = 0
        @views for a in 1:3
            x = reshape(reinterpret(Complex{FT}, δρ_external), 3, :)[a, :]
            y = Sout \ x;
            reshape(reinterpret(Complex{FT}, δρ_serta[:, iT]), 3, :)[a, :] .= .-y
        end
        σ_serta[:, :, iT] .= symmetrize(occupation_to_conductivity(δρ_serta[:, iT], el_i, params), symmetry)

        # QME: Solve (S_out + S_in) * δρ + δρ_external = 0
        # Initial guess: SERTA density matrix
        δρ_iter_new .= δρ_serta[:, iT]
        σ_new = symmetrize(occupation_to_conductivity(δρ_iter_new, el_i, params), symmetry)

        # TODO
        # for iter in 1:max_iter
        #     σ_old = σ_new
        #     δρ_iter_old .= δρ_iter_new

        #     # TODO: Symmetry
        #     # Unfold from δf_i to δf_f
        #     # δf_f = map_i_to_f * δf_i

        #     # Multiply scattering matrix, and
        #     δρ_iter_new .= δρ_iter_old .* 0.999 + (scat_mat[iT] * δρ_iter_old) .* 0.001
        #     σ_new = symmetrize(occupation_to_conductivity(δρ_iter_new, el_i, params), symmetry)
        #     @show σ_new[1, 1]
        #     if norm(σ_new - σ_old) / norm(σ_new) < rtol
        #         @info "iT=$iT, converged at iteration $iter"
        #         break
        #     elseif iter == max_iter
        #         @info "iT=$iT, convergence not reached at maximum iteration $max_iter"
        #     end
        # end

        # σ_list[:, :, iT] .= σ
        # δf_i_list[:, iT] .= δf_i
    end
    (;σ, σ_serta, δρ_serta, δρ)
end

