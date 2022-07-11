"""
Screening in the Random phase approximation.
"""

export RPAScreeningParams
export RPAScreening

using StaticArrays
using LinearAlgebra

# TODO: Cleanup indmap_ph

"""
    RPAScreeningParams{FT<:Real}
Parameters for free carrier screening based on RPA.

# Arguments:
- `ϵM::Mat3{FT}`: Macroscopic dielectric constant.
- `Tlist::Vector{FT}`: Temperature.
- `nlist::Vector{FT}`: Number of electrons per unit cell, relative to the reference configuration where
    `spin_degeneracy * nband_valence` bands are filled. Includes the spin degeneracy.
- `nband_valence::Int`: Number of valence bands, excluding spin degeneracy.
- `volume::FT`: Volume of the unit cell.
- `smearing::FT`: Smearing of frequency for RPA calculation.
- `spin_degeneracy::Int`: Spin degeneracy of bands.
- `type::Symbol`: Type of the carrier. `:Metal` or `:Semiconductor`. Defaults to `:Metal` if
    `maximum(abs(nlist)) >= 1` and to `:Semiconductor` otherwise.
"""
Base.@kwdef struct RPAScreeningParams{FT<:Real}
    ϵM::Mat3{FT}
    Tlist::Vector{FT}
    nlist::Vector{FT}
    nband_valence::Int
    volume::FT
    smearing::FT
    spin_degeneracy::Int
    μlist::Vector{FT} = fill(convert(eltype(Tlist), NaN), length(Tlist))
    type::Symbol = maximum(abs.(nlist)) >= 1 ? :Metal : :Semiconductor
end

function compute_χ0(ph, indmap_ph, el_k_save, el_kq_save, kpts, kqpts, symmetry, params::RPAScreeningParams)
    η = params.smearing

    iband_min = minimum(el.rng.start for el in el_k_save if el.nband > 0)
    iband_max = maximum(el.rng.stop  for el in el_k_save if el.nband > 0)
    rng_max = iband_min:iband_max
    nband_max = length(rng_max)
    mmat = OffsetArray(zeros(eltype(first(el_k_save).u), nband_max, nband_max), rng_max, rng_max)
    χ0 = zeros(ComplexF64, length(params.nlist), ph.n)

    for (iT, (μ, T)) in enumerate(zip(params.μlist, params.Tlist))
        for el in el_k_save
            set_occupation!(el, μ, T)
        end
        for el in el_kq_save
            set_occupation!(el, μ, T)
        end

        for ik in 1:kpts.n
            for iq in 1:ph.nk
                i = indmap_ph[1, iq]
                xq = ph[i].xks
                xq == Vec3(0, 0, 0) && continue # skip q = 0

                xk = kpts.vectors[ik]
                ikq = xk_to_ik(xk + xq, kqpts)
                ikq === nothing && continue

                el_k = el_k_save[ik]
                el_kq = el_kq_save[ikq]

                ek = el_k.e
                ekq = el_kq.e
                f_k = el_k.occupation
                f_kq = el_kq.occupation

                # Compute overlap matrix
                mmat_in_rng = view(mmat, el_kq.rng, el_k.rng)
                mul!(no_offset_view(mmat_in_rng), no_offset_view(el_kq.u)', no_offset_view(el_k.u))

                @views for ib = el_k.rng, jb = el_kq.rng
                    abs(f_k[ib] - f_kq[jb]) < 1E-10 && continue
                    numerator = (f_k[ib] - f_kq[jb]) * abs(mmat[jb, ib])^2 * kpts.weights[ik]
                    for imode in 1:ph.nband
                        i = indmap_ph[imode, iq]
                        χ0[iT, i] += numerator / (ph[i].e + ek[ib] - ekq[jb] + im * η)
                    end
                end
            end
        end
    end

    # Symmetrize χ0_avg.
    # We summed only the k points in the irreducible BZ. So, we need to sum χ0_avg
    # over symmetry-equivalent q points.
    ind_ph_map = states_index_map(ph)
    χ0_symmetrized = zero(χ0)
    for i in 1:ph.n
        xq = ph[i].xks
        imode = ph[i].iband
        for symop in symmetry
            # Find index of (imode, S(xq))
            sxq = symop.is_tr ? -symop.S * xq : symop.S * xq
            sxq_int = mod.(round.(Int, sxq.data .* ph.ngrid), ph.ngrid)
            ind_ph_list = get(ind_ph_map, CI(sxq_int...), nothing)
            ind_ph_list === nothing && continue # skip if this xq is not in ph
            j = ind_ph_list[imode]
            j == 0 && continue # skip if this imode is not in ph
            @views χ0_symmetrized[:, j] .+= χ0[:, i]
        end
    end
    χ0_symmetrized ./= length(symmetry)

    χ0_symmetrized
end

function compute_epsilon_rpa(ph, indmap_ph, el_k_save, el_kq_save, kpts, kqpts, symmetry,
        recip_lattice, params::RPAScreeningParams)
    χ0 = compute_χ0(ph, indmap_ph, el_k_save, el_kq_save, kpts, kqpts, symmetry, params)
    ϵ = zero(χ0)
    @views for i in 1:ph.n
        xq = ph[i].xks
        if all(abs.(xq) .< 1e-8) # skip q = 0
            ϵ[:, i] .= 1
        else
            xq_cart = recip_lattice * xq
            ϵM = xq_cart' * params.ϵM * xq_cart / norm(xq_cart)^2
            coeff = EPW.e2 * 4π / norm(xq_cart)^2 / params.volume * params.spin_degeneracy / ϵM
            @. ϵ[:, i] = 1 - χ0[:, i] * coeff
        end
    end
    ϵ
end