
# Long-range parts of phonon dynamical matrix and electron-phonon coupling

export Polar
export dynmat_dipole!

Base.@kwdef struct Polar{T<:Real}
    ϵ::Mat3{T} # Dielectric tensor
    Z::Vector{Mat3{T}} # Born effective charge tensor
    nxs::NTuple{3, Int} # Bounds of reciprocal space grid points to do Ewald sum
    cutoff::T # Maximum exponent for the reciprocal space Ewald sum
    η::T # Ewald parameter
end

# Null initialization for non-polar case
Polar(T) = Polar{T}(ϵ=zeros(Mat3{T}), Z=[], nxs=(0,0,0), cutoff=0, η=0)

# Compute dynmat += sign * (dynmat from dipole-dipole interaction)
function dynmat_dipole!(dynmat, xq, recip_lattice, volume, atom_pos, alat, polar::Polar, sign=1)
    T = eltype(recip_lattice)
    @assert typeof(volume) == T
    @assert eltype(atom_pos[1]) == T
    @assert typeof(alat) == T
    @assert eltype(dynmat) == Complex{T}

    natom = length(atom_pos)
    nxs = polar.nxs
    fac = sign * EPW.e2 * 4T(π) / volume

    # First term: q-independent part.
    for n1 in -nxs[1]:nxs[1], n2 in -nxs[2]:nxs[2], n3 in -nxs[3]:nxs[3]
        G = recip_lattice * Vec3{Int}(n1, n2, n3) ./ (2π / alat)
        GϵG = G' * polar.ϵ * G

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * polar.η) >= polar.cutoff)
            continue
        end

        coeff = fac * exp(-GϵG / (4 * polar.η)) / GϵG
        for iatom = 1:natom
            GZi = G' * polar.Z[iatom]
            f = zero(G')
            for jatom = 1:natom
                GZj = G' * polar.Z[jatom]
                phasefac = cospi(2 * G' * (atom_pos[iatom] - atom_pos[jatom]))
                f .+= GZj * phasefac
            end

            dyn_tmp = -coeff .* (GZi' * f)
            for j in 1:3
                for i in 1:3
                    dynmat[3*(iatom-1)+i, 3*(iatom-1)+j] += dyn_tmp[i, j]
                end
            end
        end
    end

    # Second term: q-dependent part.
    # Note that the definition of G is different: xq is added.
    for n1 in -nxs[1]:nxs[1], n2 in -nxs[2]:nxs[2], n3 in -nxs[3]:nxs[3]
        G = recip_lattice * (xq .+ Vec3{Int}(n1, n2, n3)) ./ (2π / alat)
        GϵG = G' * polar.ϵ * G

        # Skip if G=0 or if exponenent GϵG is large
        if (GϵG <= 0) || (GϵG / (4 * polar.η) >= polar.cutoff)
            continue
        end

        coeff = fac * exp(-GϵG / (4 * polar.η)) / GϵG
        for jatom = 1:natom
            GZj = G' * polar.Z[jatom]
            for iatom = 1:natom
                GZi = G' * polar.Z[iatom]
                phasefac = cis(2T(π) * G' * (atom_pos[iatom] - atom_pos[jatom]))

                dyn_tmp = (coeff * phasefac) .* (GZi' * GZj)
                for j in 1:3
                    for i in 1:3
                        dynmat[3*(iatom-1)+i, 3*(jatom-1)+j] += dyn_tmp[i, j]
                    end
                end
            end
        end
    end
end
