"""
Screening in the Random phase approximation.
"""

# TODO: Use n instead of μ

export RPAScreeningParams
export RPAScreening

using StaticArrays
using LinearAlgebra

Base.@kwdef struct RPAScreeningParams{T<:Real}
    degeneracy::Int64 # degeneracy of bands. spin and/or valley degeneracy.
    temperature::T # Temperature
    μ::T # Chemical potential
    ϵM::T # Macroscopic dielectric constant. Unitless.
    smearing::T # Smearing of frequency in Rydberg.
end

function compute_χ0(el_k_save, el_kq_save, ph_save, kpts, kqpts, qpts, symmetry, params::RPAScreeningParams)
    μ = params.μ
    T = params.temperature
    η = params.smearing

    for el in el_k_save
        set_occupation!(el, μ, T)
    end
    for el in el_kq_save
        set_occupation!(el, μ, T)
    end

    iband_min = minimum(el.rng.start for el in el_k_save if el.nband > 0)
    iband_max = maximum(el.rng.stop  for el in el_k_save if el.nband > 0)
    rng_max = iband_min:iband_max
    nband_max = length(rng_max)
    mmat = OffsetArray(zeros(eltype(first(el_k_save).u), nband_max, nband_max), rng_max, rng_max)
    χ0 = zeros(ComplexF64, first(ph_save).nmodes, qpts.n)

    for iq = 1:qpts.n
        xq = qpts.vectors[iq]
        ph = ph_save[iq]
        xq == Vec3(0, 0, 0) && continue # skip q = 0

        for ik = 1:kpts.n
            xk = kpts.vectors[ik]
            ikq = xk_to_ik(xk + xq, kqpts)
            ikq === nothing && continue

            el_k = el_k_save[ik]
            el_kq = el_kq_save[ikq]

            ek = el_k.e
            ekq = el_kq.e
            f_k = el_k.occupation
            f_kq = el_kq.occupation

            # Compute
            mmat_in_rng = view(mmat, el_kq.rng, el_k.rng)
            mul!(no_offset_view(mmat_in_rng), no_offset_view(el_kq.u)', no_offset_view(el_k.u))

            @views for ib = el_k.rng, jb = el_kq.rng
                abs(f_k[ib] - f_kq[jb]) < 1E-10 && continue
                numerator = (f_k[ib] - f_kq[jb]) * abs(mmat[jb, ib])^2 * kpts.weights[ik]
                @. χ0[:, iq] += numerator / (ph.e + ek[ib] - ekq[jb] + im * η)
            end
        end
    end

    # Symmetrize χ0_avg.
    # We summed only the k points in the irreducible BZ. So, we need to sum χ0_avg
    # over symmetry-equivalent q points.
    χ0_symmetrized = zero(χ0)
    for (iq, xq) in enumerate(qpts.vectors)
        for symop in symmetry
            sxq = symop.is_tr ? -symop.S * xq : symop.S * xq
            isq = xk_to_ik(sxq, qpts)
            isq === nothing && continue
            @views χ0_symmetrized[:, iq] .+= χ0[:, isq]
        end
    end
    χ0_symmetrized ./= length(symmetry)

    χ0_symmetrized
end