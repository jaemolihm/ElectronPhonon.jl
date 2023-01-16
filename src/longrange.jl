
# Long-range parts of phonon dynamical matrix and electron-phonon coupling

# TODO: If xq = 0, return.

using Optim

export Polar
export dynmat_dipole!
export eph_dipole!

Base.@kwdef struct Polar{T <: Real}
    use::Bool # If true, use polar correction
    # Structure information
    alat::T
    volume::T
    nmodes::Int
    recip_lattice::Mat3{T}
    atom_pos::Vector{Vec3{T}} # Atom position in alat units
    # Dipole term information
    ϵ::Mat3{T} # Dielectric tensor
    Z::Vector{Mat3{T}} = zeros(Mat3{T}, length(atom_pos))  # Born effective charge tensor
    Q::Vector{Vec3{Mat3{T}}} = zeros(Vec3{Mat3{T}}, length(atom_pos))  # Quadrupole tensor
    nxs::NTuple{3, Int} # Bounds of reciprocal space grid points to do Ewald sum
    cutoff::T # Maximum exponent for the reciprocal space Ewald sum
    η::T # Ewald parameter
    # G vectors to use in the Ewald sum
    Glist::Vector{Vec3{Int}} = get_Glist(nxs, η, recip_lattice, alat, ϵ, cutoff)
    # preallocaetd buffer
    tmp::Vector{Vector{Complex{T}}} = [zeros(Complex{T}, nmodes) for _ in 1:nthreads()]
end

function Base.show(io::IO, obj::Polar)
    print(io, typeof(obj), "(use=$(obj.use), nmodes=$(obj.nmodes), nxs=$(obj.nxs), η=$(obj.η))")
end

"""
Compute list of G vectors such that (q+G)ϵ(q+G) / (2π / alat)^2 / (4 * η) < cutoff for some q in [-0.5, 0.5]^3
Do this by computing minval = min_{q ∈ [-0.5, 0.5]^3} (q+G) * ϵ * (q+G)
Select the G vector if minval / (2π / alat)^2 / (4 * η) < cutoff.
"""
function get_Glist(nxs, η, recip_lattice, alat, ϵ, cutoff)
    FT = typeof(η)
    ϵ_crystal = recip_lattice' * ϵ * recip_lattice

    f(qG) = qG' * ϵ_crystal * qG
    function g!(G, qG)
        G .= 2 .* (ϵ_crystal * qG)
    end

    xq_upper = Vec3{FT}(1//2, 1//2, 1//2)
    xq_lower = -xq_upper
    xq_initial = Vec3{FT}(0, 0, 0)

    Glist = Vector{Vec3{Int}}()

    for ci in CartesianIndices((-nxs[1]:nxs[1], -nxs[2]:nxs[2], -nxs[3]:nxs[3]))
        G_crystal = Vec3{Int}(ci.I)

        lower = Vector(xq_lower + G_crystal)
        upper = Vector(xq_upper + G_crystal)
        initial_x = Vector(xq_initial + G_crystal)
        inner_optimizer = Optim.LBFGS()
        results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))

        minval = Optim.minimum(results)
        GϵG = minval / (2π / alat)^2 / (4 * η)

        if GϵG < cutoff
            push!(Glist, G_crystal)
        end
    end
    Glist
end

# Null initialization for non-polar case
# FIXME: defining Polar{T}() does not work. (maybe overriden by @kwdef.)
Polar{T}(::Nothing) where {T} = Polar{T}(use=false, alat=0, volume=0, nmodes=0, recip_lattice=zeros(Mat3{T}),
    atom_pos=[], ϵ=zeros(Mat3{T}), nxs=(0,0,0), cutoff=0, η=0, Glist=[])

"""
    parse_epw_quadrupole_fmt(filename)
parse quadrupole.fmt file in the EPW format (as of EPW v5.5)
"""
function parse_epw_quadrupole_fmt(filename)
    Q = Vec3{Mat3{Float64}}[]
    Q_iatm = zeros(Float64, 3, 3, 3)
    open(filename, "r") do f
        readline(f)  # dummy header
        for line in readlines(f)
            iatm, idir = parse.(Int, split(line)[1:2])
            Qxx, Qyy, Qzz, Qyz, Qxz, Qxy = parse.(Float64, split(line)[3:end])
            Q_iatm[idir, :, :] .= [Qxx Qxy Qxz; Qxy Qyy Qyz; Qxz Qyz Qzz]
            idir == 3 && push!(Q, Vec3{Mat3{Float64}}(eachslice(Q_iatm, dims=1)))
        end
    end
    Q
end

# Compute dynmat += sign * (dynmat from dipole-dipole interaction)
# TODO: Multiply phase factor only once
@timing "lr_dyn_dip" function dynmat_dipole!(dynmat, xq, polar::Polar{T}, sign=1) where {T}
    polar.use || return dynmat

    metric = (2T(π) / polar.alat)^2  # Conversion factor for G^2, unit bohr⁻²

    # Map xq to inside [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate(xq .+ T(1/2)) .- T(1/2)

    @assert eltype(dynmat) == Complex{T}

    atom_pos = polar.atom_pos
    natom = length(atom_pos)
    fac = sign * EPW.e2 * 4T(π) / polar.volume

    # First term: q-independent part.
    for G_crystal in polar.Glist
        G = polar.recip_lattice * G_crystal  # In bohr⁻¹
        GϵG = G' * polar.ϵ * G  # In bohr⁻²

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * metric * polar.η) >= polar.cutoff)
            continue
        end

        fac2 = fac * exp(-GϵG / (4 * metric * polar.η)) / GϵG  # The exponent is unitless
        for iatom = 1:natom
            GZi = G' * polar.Z[iatom]
            GQi = (Ref(G') .* polar.Q[iatom] .* Ref(G))'

            GZ_phase = zero(Vec3{complex(T)})'
            GQ_phase = zero(Vec3{complex(T)})'
            for jatom = 1:natom
                GZj = G' * polar.Z[jatom]
                GQj = (Ref(G') .* polar.Q[jatom] .* Ref(G))'
                phasefac = cis(polar.alat * G' * (atom_pos[iatom] - atom_pos[jatom]))
                GZ_phase += GZj * phasefac
                GQ_phase += GQj * phasefac
            end

            dyn_tmp = fac2 * (GZi' * GZ_phase)  # dipole-dipole
            dyn_tmp += fac2 * (GQi' * GZ_phase - GZi' * GQ_phase) / 2 * im  # dipole-quadrupole
            dyn_tmp += fac2 * (GQi' * GQ_phase) / 4  # quadrupole-quadrupole
            for j in 1:3
                for i in 1:3
                    # Minus sign because we are subtracting the q=0 contribution as the
                    # correction to satisfy the acoustic sum rule
                    # Ref: Gonze and Lee (1997), see also Lin, Ponce, Marzari (2022)
                    dynmat[3*(iatom-1)+i, 3*(iatom-1)+j] -= dyn_tmp[i, j]
                end
            end
        end
    end

    # Second term: q-dependent part.
    # Note that the definition of G is different: xq is added.
    for G_crystal in polar.Glist
        G = polar.recip_lattice * (xq .+ G_crystal)  # In bohr⁻¹
        GϵG = G' * polar.ϵ * G  # In bohr⁻²

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * metric * polar.η) >= polar.cutoff)
            continue
        end

        fac2 = fac * exp(-GϵG / (4 * metric * polar.η)) / GϵG  # The exponent is unitless
        for jatom = 1:natom
            GZj = G' * polar.Z[jatom]
            GQj = (Ref(G') .* polar.Q[jatom] .* Ref(G))'
            for iatom = 1:natom
                GZi = G' * polar.Z[iatom]
                GQi = (Ref(G') .* polar.Q[iatom] .* Ref(G))'
                phasefac = cis(polar.alat * G' * (atom_pos[iatom] - atom_pos[jatom]))

                dyn_tmp = (fac2 * phasefac) * (GZi' * GZj)  # dipole-dipole
                dyn_tmp += (fac2 * phasefac) * (GQi' * GZj - GZi' * GQj) / 2 * im  # dipole-quadrupole
                dyn_tmp += (fac2 * phasefac) * (GQi' * GQj) / 4  # quadrupole-quadrupole
                for j in 1:3
                    for i in 1:3
                        dynmat[3*(iatom-1)+i, 3*(jatom-1)+j] += dyn_tmp[i, j]
                    end
                end
            end
        end
    end
    dynmat
end

# Compute coefficients for dipole e-ph coupling. The coefficients depend only on the phonon properties.
function get_eph_dipole_coeffs!(coeff, xq, polar::Polar{T}, u_ph) where {T}
    if ! polar.use || all(abs.(xq) .<= 1.0e-8)
        coeff .= 0
        return coeff
    end
    metric = (2T(π) / polar.alat)^2  # Conversion factor for G^2, unit bohr⁻²

    # Map xq to inside [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate(xq .+ 0.5) .- 0.5

    atom_pos = polar.atom_pos
    natom = length(atom_pos)
    fac = 1im * EPW.e2 * 4T(π) / polar.volume

    # temporary vectors of size (nmodes,)
    tmp = polar.tmp[threadid()]
    tmp .= 0

    for G_crystal in polar.Glist
        G = polar.recip_lattice * (xq .+ G_crystal)  # In bohr⁻¹
        GϵG = G' * polar.ϵ * G  # In bohr⁻²

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * metric * polar.η) >= polar.cutoff)
            continue
        end

        # sqrt(metric) is included to keep compatibility with EPW v5.6
        fac2 = fac * exp(-GϵG * sqrt(metric) / (4 * metric * polar.η)) / GϵG  # The exponent is unitless
        for iatom in 1:natom
            phasefac = cis(-polar.alat * dot(G, atom_pos[iatom]))
            GZi = G' * polar.Z[iatom]
            GQi = (Ref(G') .* polar.Q[iatom] .* Ref(G))' / 2
            for ipol in 1:3
                tmp[3*(iatom-1)+ipol] += fac2 * phasefac * (GZi[ipol] - im * GQi[ipol])
            end
        end
    end

    mul!(coeff, Transpose(u_ph), tmp)
    coeff
end

# Compute eph_kq += sign * (eph_kq from dipole potential)
@timing "lr_eph_dip" function eph_dipole!(eph, xq, polar::Polar{T}, u_ph, mmat, sign=1) where {T}
    polar.use || return eph

    @assert eltype(eph) == Complex{T}

    nmodes = polar.nmodes
    coeff = polar.tmp[threadid()]
    get_eph_dipole_coeffs!(coeff, xq, polar, u_ph)

    @views @inbounds for imode = 1:nmodes
        eph[:, :, imode] .+= (sign * coeff[imode]) .* mmat
    end
    eph
end
