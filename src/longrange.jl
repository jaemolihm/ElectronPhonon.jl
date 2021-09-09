
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
    atom_pos::Vector{Vec3{T}}
    # Dipole term information
    ϵ::Mat3{T} # Dielectric tensor
    Z::Vector{Mat3{T}} # Born effective charge tensor
    nxs::NTuple{3, Int} # Bounds of reciprocal space grid points to do Ewald sum
    cutoff::T # Maximum exponent for the reciprocal space Ewald sum
    η::T # Ewald parameter
    # G vectors to use in the Ewald sum
    Glist::Vector{Vec3{Int}} = get_Glist(nxs, η, recip_lattice, alat, ϵ, cutoff)
    # preallocaetd buffer
    tmp1::Vector{Complex{T}} = zeros(Complex{T}, nmodes)
    tmp2::Vector{Complex{T}} = zeros(Complex{T}, nmodes)
end

function Base.show(io::IO, obj::Polar)
    print(io, typeof(obj), "(use=$(obj.use), nmodes=$(obj.nmodes), nxs=$(obj.nxs), η=$(obj.η))")
end

"""
Compute list of G vectors such that (q+G)/ϵ(q+G) / (2π / alat)^2 / (4 * η) < cutoff for some q in [-0.5, 0.5]^3
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
    atom_pos=[], ϵ=zeros(Mat3{T}), Z=[], nxs=(0,0,0), cutoff=0, η=0)

# Compute dynmat += sign * (dynmat from dipole-dipole interaction)
@timing "lr_dyn_dip" function dynmat_dipole!(dynmat, xq, polar::Polar{T}, sign=1) where {T}
    if ! polar.use
        return
    end

    # Map xq to inside [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate(xq .+ 0.5) .- 0.5

    @assert eltype(dynmat) == Complex{T}

    atom_pos = polar.atom_pos
    natom = length(atom_pos)
    fac = sign * EPW.e2 * 4T(π) / polar.volume

    # First term: q-independent part.
    for G_crystal in polar.Glist
        G = polar.recip_lattice * G_crystal / (2π / polar.alat)
        GϵG = G' * polar.ϵ * G

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * polar.η) >= polar.cutoff)
            continue
        end

        fac2 = fac * exp(-GϵG / (4 * polar.η)) / GϵG
        for iatom = 1:natom
            GZi = G' * polar.Z[iatom]
            f = zero(G')
            for jatom = 1:natom
                GZj = G' * polar.Z[jatom]
                phasefac = cospi(2 * G' * (atom_pos[iatom] - atom_pos[jatom]))
                f .+= GZj * phasefac
            end

            dyn_tmp = -fac2 .* (GZi' * f)
            for j in 1:3
                for i in 1:3
                    dynmat[3*(iatom-1)+i, 3*(iatom-1)+j] += dyn_tmp[i, j]
                end
            end
        end
    end

    # Second term: q-dependent part.
    # Note that the definition of G is different: xq is added.
    for G_crystal in polar.Glist
        G = polar.recip_lattice * (xq .+ G_crystal) / (2π / polar.alat)
        GϵG = G' * polar.ϵ * G

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * polar.η) >= polar.cutoff)
            continue
        end

        fac2 = fac * exp(-GϵG / (4 * polar.η)) / GϵG
        for jatom = 1:natom
            GZj = G' * polar.Z[jatom]
            for iatom = 1:natom
                GZi = G' * polar.Z[iatom]
                phasefac = cis(2T(π) * G' * (atom_pos[iatom] - atom_pos[jatom]))

                dyn_tmp = (fac2 * phasefac) .* (GZi' * GZj)
                for j in 1:3
                    for i in 1:3
                        dynmat[3*(iatom-1)+i, 3*(jatom-1)+j] += dyn_tmp[i, j]
                    end
                end
            end
        end
    end
end

# Compute coefficients for dipole e-ph coupling. The coefficients depend only on the phonon properties.
function get_eph_dipole_coeffs!(coeff, xq, polar::Polar{T}, u_ph) where {T}
    if ! polar.use || all(abs.(xq) .<= 1.0e-8)
        coeff .= 0
        return coeff
    end

    # Map xq to inside [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate(xq .+ 0.5) .- 0.5

    atom_pos = polar.atom_pos
    natom = length(atom_pos)
    fac = 1im * EPW.e2 * 4T(π) / polar.volume

    # temporary vectors of size (nmodes,)
    tmp = polar.tmp1
    tmp .= 0

    for G_crystal in polar.Glist
        G = polar.recip_lattice * (xq .+ G_crystal) / (2T(π) / polar.alat)
        GϵG = G' * polar.ϵ * G

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * polar.η) >= polar.cutoff)
            continue
        end

        GϵG *= 2T(π) / polar.alat
        fac2 = fac * exp(-GϵG / (4 * polar.η)) / GϵG
        for iatom in 1:natom
            phasefac = cispi(-2 * dot(G, atom_pos[iatom]))
            @views for ipol in 1:3
                GZ = dot(G, polar.Z[iatom][:, ipol])
                tmp[3*(iatom-1)+ipol] += fac2 * phasefac * GZ
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
    coeff = polar.tmp2
    get_eph_dipole_coeffs!(coeff, xq, polar, u_ph)

    @views @inbounds for imode = 1:nmodes
        eph[:, :, imode] .+= (sign * coeff[imode]) .* mmat
    end
    eph
end
