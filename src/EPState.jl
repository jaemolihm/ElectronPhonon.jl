
# For computing electron-phonon coupling at fine a k and q point

using Base: @kwdef
using OffsetArrays
using OffsetArrays: no_offset_view

export EPState
export epstate_set_g2!
export epstate_set_mmat!
export epstate_compute_eph_dipole!

# Energy and matrix elements at a single k and q point
@kwdef mutable struct EPState{T <: Real}
    nw::Int # Number of Wannier functions
    nmodes::Int # Number of modes
    nband_bound::Int # Maximum allowed number of bands inside the window
    wtk::T # Weight of the k point
    wtq::T # Weight of the q point

    # Electron states
    el_k::ElectronState{T} # electron state at k
    el_kq::ElectronState{T} # electron state at k+q

    # Phonon state
    ph::PhononState{T} # phonon state at q

    # U(k+q)' * U(k)
    mmat::Matrix{Complex{T}} = zeros(Complex{T}, nband_bound, nband_bound)

    # Electron-phonon coupling
    ep::Array{Complex{T}, 3} = zeros(Complex{T}, nband_bound, nband_bound, nmodes)
    g2::Array{T, 3} = zeros(T, nband_bound, nband_bound, nmodes)

    # Preallocated buffers
    buffer::Matrix{Complex{T}} = zeros(Complex{T}, nw, nw)
    buffer2::Array{T, 3} = zeros(T, nband_bound, nband_bound, nmodes)
end

function EPState{T}(nw, nmodes, nband_bound=nw) where {T}
    @assert nw > 0
    @assert nmodes > 0
    @assert nband_bound > 0

    EPState{T}(nw=nw, nmodes=nmodes, nband_bound=nband_bound, wtk=T(0), wtq=T(0),
        el_k = ElectronState{T}(nw),
        el_kq = ElectronState{T}(nw),
        ph = PhononState(nmodes, T),
        ep = zeros(Complex{T}, nband_bound * nband_bound * nmodes, 1, 1),
        g2 = zeros(Complex{T}, nband_bound * nband_bound * nmodes, 1, 1),
        buffer2 = zeros(Complex{T}, nband_bound * nband_bound * nmodes, 1, 1),
    )
end

EPState(nw, nmodes, nband_bound=nw) = EPState{Float64}(nw, nmodes, nband_bound)

@inline function Base.getproperty(epstate::EPState, name::Symbol)
    if name === :mmat
        OffsetArray(view(getfield(epstate, name), 1:getfield(epstate, :el_kq).nband, 1:getfield(epstate, :el_k).nband),
            getfield(epstate, :el_kq).rng, getfield(epstate, :el_k).rng)
    elseif name === :ep || name === :g2 || name === :buffer2
        n1 = getfield(epstate, :el_kq).nband
        n2 = getfield(epstate, :el_k).nband
        n3 = getfield(epstate, :nmodes)
        OffsetArray(reshape(view(getfield(epstate, name), 1:n1*n2*n3, 1, 1), n1, n2, n3),
            getfield(epstate, :el_kq).rng, getfield(epstate, :el_k).rng, :)
    else
        getfield(epstate, name)
    end
end

" Set epstate.g2[:, :, imode] = |epstate.ep[:, :, imode]|^2 / (2 omega)"
function epstate_set_g2!(epstate)
    for imode in 1:epstate.nmodes
        # The lower bound for phonon frequency is not set here. If ω is close to 0, g2 may
        # be very large. This should be handled when calculating physical quantities.
        ω = epstate.ph.e[imode]
        inv_2ω = 1 / (2 * ω)
        @views epstate.g2[:, :, imode] .= (abs2.(epstate.ep[:, :, imode]) .* inv_2ω)
    end
end

"Set mmat = ukq' * uk"
@timing "setmmat" function epstate_set_mmat!(epstate)
    @views mul!(no_offset_view(epstate.mmat), no_offset_view(epstate.el_kq.u)', no_offset_view(epstate.el_k.u))
end

# Define wrappers of wannier_to_bloch functions

"""
    get_eph_Rq_to_kq!(epstate::EPState, epobj_eRpq, xk)
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
"""
function get_eph_Rq_to_kq!(epstate::EPState, epobj_eRpq, xk)
    ep_kq = no_offset_view(epstate.ep)
    get_eph_Rq_to_kq!(ep_kq, epobj_eRpq, xk, no_offset_view(epstate.el_k.u), no_offset_view(epstate.el_kq.u))
end

"""
    get_eph_kR_to_kq!(epstate::EPState, epobj_ekpR, xq)
Compute electron-phonon coupling matrix in electron and phonon Bloch basis.
"""
function get_eph_kR_to_kq!(epstate::EPState, epobj_ekpR, xq)
    ep_kq = no_offset_view(epstate.ep)
    get_eph_kR_to_kq!(ep_kq, epobj_ekpR, xq, epstate.ph.u, no_offset_view(epstate.el_kq.u))
end

"""
    epstate_compute_eph_dipole!(epstate::EPState, polar::Polar, factor=1)
Compute electron-phonon coupling matrix elements using pre-computed `ph.eph_dipole_coeff` and `mmat`.
Divide by `factor` if given. Can be used to screen the dipole term (`factor = ϵ`)
or to subtracting the dipole term from `epstate.ep` (`factor = -1`).
"""
function epstate_compute_eph_dipole!(epstate::EPState, factor=nothing; model = nothing)
    # epstate.ep .= 0
    # return
    coeff = epstate.ph.eph_dipole_coeff
    coeff_r = epstate.ph.eph_r_coeff
    if factor === nothing
        @views for imode in 1:epstate.nmodes
            @. epstate.ep[:, :, imode] += coeff[imode] * epstate.mmat
        end
    else
        # THIS
        # r = epstate.el_kq.u' * epstate.el_k.u * epstate.el_k.rbar

        # r = epstate.el_kq.rbar * epstate.el_kq.u' * epstate.el_k.u

        # epstate.ep .= 0

        @views for imode in 1:epstate.nmodes
            # @. epstate.ep[:, :, imode] += (coeff[imode] / factor[imode]) * epstate.mmat
            for m in epstate.el_kq.rng, n in epstate.el_k.rng
                for iw in 1:epstate.nw
                    epstate.ep[m, n, imode] += (coeff[imode] / factor[imode]) * epstate.el_kq.u[iw, m]' * epstate.el_k.u[iw, n]
                end
            end
        end

        @views for imode in 1:epstate.nmodes
            for m in epstate.el_kq.rng, n in epstate.el_k.rng
                for i in 1:3
                    # epstate.ep[m, n, imode] += (coeff_r[imode, i] / factor[imode]) * r[m, n][i]
                end
            end
        end
    end

    return


    polar = model.polar_eph
    (; alat, recip_lattice, volume, atom_pos) = polar.cell
    (; η, cutoff) = polar.method
    natom = length(atom_pos)
    metric = (2π / alat)^2  # Conversion factor for G^2, unit bohr⁻²
    xq = epstate.ph.xq
    xq = normalize_kpoint_coordinate(xq .+ 0.5) .- 0.5

    fac = 1im * ElectronPhonon.e2 * 4π / volume

    # temporary vectors of size (nmodes,)
    tmp = zeros(Vec3{ComplexF64}, polar.nmodes)

    for G_crystal in polar.Glist
        qG = recip_lattice * (xq .+ G_crystal)  # In bohr⁻¹
        GϵG = qG' * polar.ϵ * qG  # In bohr⁻²

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * metric * η) >= cutoff)
            continue
        end

        # After EPW v5.7: sqrt(metric) factor is removed
        fac2 = fac * exp(-GϵG / (4 * metric * η)) / GϵG

        # Until EPW v5.6: sqrt(metric) is included to keep compatibility
        # fac2 = fac * exp(-GϵG * sqrt(metric) / (4 * metric * polar.η)) / GϵG  # The exponent is unitless

        for iatom in 1:natom
            phasefac = cis(-alat * dot(qG, atom_pos[iatom]))
            GZi = qG' * polar.Z[iatom]
            for ipol in 1:3
                tmp[3*(iatom-1)+ipol] += im * qG * (fac2 * phasefac * GZi[ipol])
            end
        end
    end

    coeff_r = Transpose(epstate.ph.u) * tmp

    r = epstate.el_kq.u' * epstate.el_k.u * epstate.el_k.rbar

    @views for imode in 1:epstate.nmodes
        for m in epstate.el_kq.rng, n in epstate.el_k.rng
            epstate.ep[m, n, imode] += conj.(coeff_r[imode])' * r[m, n]
        end
    end
end


"""
    epstate_g2_degenerate_average!(epstate::EPState)
Avearage g2 over degenerate bands of el_k and el_kq
"""
function epstate_g2_degenerate_average!(epstate::EPState{FT}) where {FT}
    el_k = epstate.el_k
    el_kq = epstate.el_kq
    g2_avg = epstate.buffer2

    # average over bands at k
    g2_avg .= 0
    @views for ib in el_k.rng
        ndegen = 0
        for jb in el_k.rng
            if abs(el_k.e[ib] - el_k.e[jb]) <= electron_degen_cutoff
                g2_avg[el_kq.rng, ib, :] .+= epstate.g2[el_kq.rng, jb, :]
                ndegen += 1
            end
        end
        g2_avg[:, ib, :] ./= ndegen
    end
    epstate.g2 .= g2_avg

    # average over bands at k+q
    g2_avg .= 0
    @views for ib in el_kq.rng
        ndegen = 0
        for jb in el_kq.rng
            if abs(el_kq.e[ib] - el_kq.e[jb]) <= electron_degen_cutoff
                g2_avg[ib, el_k.rng, :] .+= epstate.g2[jb, el_k.rng, :]
                ndegen += 1
            end
        end
        g2_avg[ib, :, :] ./= ndegen
    end
    epstate.g2 .= g2_avg
    nothing
end


# Per-thread EPState channel used by the CPU inner loops of the outer-k and over-k-and-kq drivers.
function get_epstates_channel(::Type{FT}, nw, nmodes, nband_max) where {FT}
    ch = Channel{EPState{FT}}(Base.Threads.nthreads())
    foreach(1:Base.Threads.nthreads()) do _
        put!(ch, EPState{FT}(nw, nmodes, nband_max))
    end
    ch
end
