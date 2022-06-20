"""
Screening in the Random phase approximation.
"""

# TODO: Use n instead of μ

export RPAScreeningParams
export RPAScreening

using StaticArrays
using LinearAlgebra

# TODO: Anisotropic ϵM

"""
    RPAScreeningParams{FT<:Real}
Parameters for free carrier screening based on RPA. Only a single n and T is allowed.
Internally, T, n, and μ are stored as a size-1 vector to use bte_compute_μ! (which requires
vector access).

# Arguments:
- `ϵM::Mat3{FT}`: Macroscopic dielectric constant.
- `T::FT`: Temperature.
- `n::FT`: Number of electrons per unit cell, relative to the reference configuration where
    `spin_degeneracy * nband_valence` bands are filled. Includes the spin degeneracy.
- `nband_valence::Int`: Number of valence bands, excluding spin degeneracy.
- `volume::FT`: Volume of the unit cell.
- `smearing::FT`: Smearing of frequency for RPA calculation.
- `spin_degeneracy::Int`: Spin degeneracy of bands.
- `type::Symbol`: Type of the carrier. `:Metal` or `:Semiconductor`. Defaults to `:Metal` if
    abs(n) >= 1 and to `:Semiconductor` otherwise.
"""
struct RPAScreeningParams{FT<:Real}
    ϵM::Mat3{FT}
    Tlist::Vector{FT}
    nlist::Vector{FT}
    nband_valence::Int
    volume::FT
    smearing::FT
    spin_degeneracy::Int
    μlist::Vector{FT}
    type::Symbol
    function RPAScreeningParams(; ϵM, T, n, nband_valence, volume, smearing, spin_degeneracy)
        FT = typeof(T)
        Tlist = [T]
        nlist = [n]
        μlist = fill(convert(eltype(Tlist), NaN), length(Tlist))
        type = maximum(abs.(nlist)) >= 1 ? :Metal : :Semiconductor
        new{FT}(ϵM, Tlist, nlist, nband_valence, volume, smearing, spin_degeneracy, μlist, type)
    end
end

function Base.getproperty(obj::RPAScreeningParams, name::Symbol)
    if name === :n
        first(getfield(obj, :nlist))
    elseif name === :T
        first(getfield(obj, :Tlist))
    elseif name === :μ
        first(getfield(obj, :μlist))
    else
        getfield(obj, name)
    end
end

function compute_χ0(el_k_save, el_kq_save, ph_save, kpts, kqpts, qpts, symmetry, params::RPAScreeningParams)
    (; μ, T) = params
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

function compute_epsilon_rpa(el_k_save, el_kq_save, ph_save, kpts, kqpts, qpts, symmetry,
        recip_lattice, params::RPAScreeningParams)
    χ0 = compute_χ0(el_k_save, el_kq_save, ph_save, kpts, kqpts, qpts, symmetry, params)
    ϵ = zero(χ0)
    for (iq, xq) in enumerate(qpts.vectors)
        if all(abs.(xq) .< 1e-8) # skip q = 0
            ϵ[:, iq] .= 1
        else
            xq_cart = recip_lattice * xq
            ϵM = xq_cart' * params.ϵM * xq_cart / norm(xq_cart)^2
            coeff = EPW.e2 * 4π / norm(xq_cart)^2 / params.volume * params.spin_degeneracy / ϵM
            @. ϵ[:, iq] = 1 - χ0[:, iq] * coeff
        end
    end
    ϵ
end