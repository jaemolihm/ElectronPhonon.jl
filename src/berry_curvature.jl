
"""
    compute_berry_curvature(el_states, kpts, model; fourier_mode="gridopt")
Compute the Berry curvature using Eq. (A1) of [1].
Reference: [1] Wang et al, PRB 74, 195118 (2006).
"""
function compute_berry_curvature(el_states, kpts, model; fourier_mode="gridopt", γ = 10.0 * unit_to_aru(:meV))
    el_pos_R = get_interpolator(wannier_object_multiply_R(model.el_pos, model.lattice); fourier_mode)
    Ωbar_W = zeros(ComplexF64, model.nw, model.nw, 3, 3)
    Ωbar_H = zeros(ComplexF64, model.nw, model.nw, 3)
    tmp = zeros(ComplexF64, model.nw, model.nw)

    Ω = zeros(3, model.nw, kpts.n)
    for ik in 1:kpts.n
        el = el_states[ik]
        xk = kpts.vectors[ik]

        # Compute Ωbar_H: see Eq. (40) and Eq. (21) of [1].
        # Note that Ωbar_W[m, n, a, b] = ∑_R im * R[b] * <m|r[a]|n>.
        get_fourier!(Ωbar_W, el_pos_R, xk)
        @views @inbounds for c = 1:3
            a, b = mod1(c + 1, 3), mod1(c + 2, 3)
            mul!(tmp, Ωbar_W[:, :, b, a], el.u)
            mul!(Ωbar_H[:, :, c], el.u', tmp)
            mul!(tmp, Ωbar_W[:, :, a, b], el.u)
            mul!(Ωbar_H[:, :, c], el.u', tmp, -1, 1)
        end

        # Second and third terms of Eq. (A1)
        for ib1 in el.rng, ib2 in el.rng
            e1, e2 = el.e[ib1], el.e[ib2]
            v12 = el.v[ib1, ib2]
            rbar = el.rbar[ib1, ib2]
            abs(e1 - e2) < electron_degen_cutoff && continue
            Ω[:, ib1, ik] += -imag(cross(v12, conj(v12)) / ((e1 - e2)^2 + γ^2))
            Ω[:, ib1, ik] +=  imag(cross(rbar, conj(rbar)))
        end

        # First term of Eq. (A1)
        for ib1 in el.rng
            Ω[:, ib1, ik] .+= real.(Ωbar_H[ib1, ib1, :])
        end
    end
    return Ω
end

"""
    compute_berry_curvature_AC(el_states, kpts, model; fourier_mode="gridopt")
Compute the Berry curvature using Eq. (A1) of [1].
Include frequency factor using Eq. (C20) of [2].
Frequency factor is include only to the second and third terms of Eq. (A1) of [1].
Reference: [1] Wang et al, PRB 74, 195118 (2006), [2] Tsirkin et al, PRB 97, 035158 (2018).
"""
function compute_berry_curvature_AC(el_states, kpts, model, ω; fourier_mode="gridopt")
    el_pos_R = get_interpolator(wannier_object_multiply_R(model.el_pos, model.lattice); fourier_mode)
    Ωbar_W = zeros(ComplexF64, model.nw, model.nw, 3, 3)
    Ωbar_H = zeros(ComplexF64, model.nw, model.nw, 3)
    tmp = zeros(ComplexF64, model.nw, model.nw)

    γ = 10.0 * unit_to_aru(:meV)

    Ω = zeros(3, model.nw, kpts.n)
    for ik in 1:kpts.n
        el = el_states[ik]
        xk = kpts.vectors[ik]

        # Compute Ωbar_H: see Eq. (40) and Eq. (21) of [1].
        # Note that Ωbar_W[m, n, a, b] = ∑_R im * R[b] * <m|r[a]|n>.
        get_fourier!(Ωbar_W, el_pos_R, xk)
        @views @inbounds for c = 1:3
            a, b = mod1(c + 1, 3), mod1(c + 2, 3)
            mul!(tmp, Ωbar_W[:, :, b, a], el.u)
            mul!(Ωbar_H[:, :, c], el.u', tmp)
            mul!(tmp, Ωbar_W[:, :, a, b], el.u)
            mul!(Ωbar_H[:, :, c], el.u', tmp, -1, 1)
        end

        # Second and third terms of Eq. (A1)
        for ib1 in el.rng, ib2 in el.rng
            e1, e2 = el.e[ib1], el.e[ib2]
            v12 = el.v[ib1, ib2]
            rbar = el.rbar[ib1, ib2]
            abs(e1 - e2) < electron_degen_cutoff && continue
            ac_factor = real(((e1 - e2)^2 + γ^2) / ((e1 - e2)^2 - (ω + im * γ)^2))
            Ω[:, ib1, ik] += -imag(cross(v12, conj(v12)) / ((e1 - e2)^2 + γ^2)) * ac_factor
            Ω[:, ib1, ik] +=  imag(cross(rbar, conj(rbar))) * ac_factor
        end

        # First term of Eq. (A1)
        for ib1 in el.rng
            Ω[:, ib1, ik] .+= real.(Ωbar_H[ib1, ib1, :])
        end
    end
    return Ω
end

function compute_berry_curvature_el(model, el, ω=nothing; fourier_mode="gridopt", γ=10*unit_to_aru(:meV))
    el_states = compute_electron_states(model, el.kpts, ["eigenvalue", "eigenvector", "position", "velocity"]; fourier_mode)
    if ω === nothing
        Ω = compute_berry_curvature(el_states, el.kpts, model; fourier_mode, γ)
    else
        Ω = compute_berry_curvature_AC(el_states, el.kpts, model, ω; fourier_mode)
    end

    Ω_el = QMEVector(el, Vec3{Float64});
    for i in 1:el.n
        (; ib1, ib2, ik) = el[i]
        @views if ib1 == ib2
            Ω_el[i] = Vec3(Ω[:, ib1, ik])
        else
            continue
        end
    end
    Ω_el
end
