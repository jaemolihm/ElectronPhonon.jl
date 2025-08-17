
# Long-range parts of phonon dynamical matrix and electron-phonon coupling

# TODO: If xq = 0, return.

using Optim

export Polar
export dynmat_dipole!

abstract type AbstractPolarMethods; end

struct Polar3D <: AbstractPolarMethods
    # Maximum exponent for the reciprocal space Ewald sum
    cutoff :: Float64
    # Ewald parameter
    η :: Float64
end

struct Polar2D <: AbstractPolarMethods
    # 2D polar correction. (system_2d == 'dipole_sp' and 'quadrupole' in EPW.)
    # M. Royo and M. Stengel, Phys. Rev. X 11, 041027 (2021).
    # S. Ponce et al., Phys. Rev. B 107, 155424 (2023).

    # Maximum argument of f(x) = 1 - tanh(x) to include.
    cutoff :: Float64

    # Vertical distance parameter
    L :: Float64
end

Base.@kwdef struct Polar{PM <: AbstractPolarMethods}
    use::Bool # If true, use polar correction

    # Method to use for polar correction
    method :: PM

    # --- Structure information ---
    # Cell and atomic structure
    cell :: Structure

    # Number of phonon modes
    nmodes :: Int

    # --- Long-range electrostatics information ---
    # Dielectric tensor
    ϵ :: Mat3{Float64}
    # Born effective charge tensor
    Z :: Vector{Mat3{Float64}}
    # Quadrupole tensor (zero if not available)
    Q :: Vector{Vec3{Mat3{Float64}}}

    # Bounds of reciprocal space grid points to do Ewald sum
    nxs :: NTuple{3, Int}

    # G vectors to use in the Ewald sum
    Glist :: Vector{Vec3{Int}}

    # q-independent part of the dynamical matrix to be added
    dynmat_asr :: Matrix{ComplexF64}
end

function Polar(cell :: Structure; use, ϵ, Z, Q, L = nothing, mode = :Polar3D)

    if mode === :Polar3D
        # EPW hard-coded parameters (see EPW/src/rigid_f90)
        cutoff = 14.0  # gmax
        η = 1.0  # alph

        method = Polar3D(cutoff, η)

    elseif mode === :Polar2D
        @assert L !== nothing "Polar2D requires L"
        @assert cell.lattice[1, 3] == 0 "Polar2D requires 2d system with vacuum along z"
        @assert cell.lattice[2, 3] == 0 "Polar2D requires 2d system with vacuum along z"
        @assert cell.lattice[3, 1] == 0 "Polar2D requires 2d system with vacuum along z"
        @assert cell.lattice[3, 2] == 0 "Polar2D requires 2d system with vacuum along z"

        cutoff = 15.0  # 1 - tanh(15.0) = 2e-13
        method = Polar2D(cutoff, L)

    else
        error("Unknown mode: $mode")
    end

    # Compute nxs, the box that includes all vectors up to |G| < 4*η*cutoff.
    # See SUBROUTINE rgd_blk of EPW
    nx1 = floor(Int, sqrt(4 * 1.0 * 14.0) / norm(cell.recip_lattice[:, 1])) + 1
    nx2 = floor(Int, sqrt(4 * 1.0 * 14.0) / norm(cell.recip_lattice[:, 2])) + 1
    nx3 = floor(Int, sqrt(4 * 1.0 * 14.0) / norm(cell.recip_lattice[:, 3])) + 1
    if mode === :Polar3D
        nxs = (nx1, nx2, nx3)
    elseif mode === :Polar2D
        nxs = (nx1, nx2, 0)
    end

    nmodes = 3 * length(cell.atom_pos)
    Glist = get_Glist(method, cell, nxs, ϵ)

    dynmat_asr = zeros(ComplexF64, nmodes, nmodes)
    polar = Polar(; use, method, cell, nmodes, ϵ, Z, Q, nxs, Glist, dynmat_asr)

    # In the long-range dynamical matrix, we have to subtract the q=0 contribution as the
    # correction to satisfy the acoustic sum rule. We subtract only on the diagonal part:
    # dynmat(q)[iatm, iatm] -= ∑_jatm dynmat(q=0)[iatm, jatm]
    # See Gonze and Lee (1997). See also Lin, Ponce, Marzari (2022)

    # To simplify the implementation, we only implement the q-dependent part in
    # the function dynmat_dipole!. We then call it to compute the correction term
    # and store it in dynmat_asr.

    dynmat_q0 = zeros(ComplexF64, nmodes, nmodes)
    dynmat_dipole!(dynmat_q0, zero(Vec3{Float64}), polar)

    natom = length(cell.atom_pos)
    for jatom in 1:natom, iatom in 1:natom
        for j in 1:3, i in 1:3
            polar.dynmat_asr[3*(iatom-1)+i, 3*(iatom-1)+j] -= dynmat_q0[3*(iatom-1)+i, 3*(jatom-1)+j]
        end
    end

    polar
end

function Polar(::Nothing)
    # Null initialization for non-polar case
    cell = Structure(nothing)
    method = Polar3D(0., 0.)
    nmodes = 0
    ϵ = zeros(Mat3{Float64})
    Z = Mat3{Float64}[]
    Q = Vec3{Mat3{Float64}}[]
    nxs = (0, 0, 0)
    Glist = Vec3{Int}[]
    dynmat_asr = zeros(ComplexF64, 0, 0)
    Polar(; use = false, method, cell, nmodes, ϵ, Z, Q, nxs, Glist, dynmat_asr)
end


function Base.show(io::IO, obj::Polar)
    print(io, typeof(obj), "(use=$(obj.use), method=$(obj.method), nxs=$(obj.nxs))")
end


"""
Compute list of G vectors such that (q+G)ϵ(q+G) / (2π / alat)^2 / (4 * η) < cutoff for some q in [-0.5, 0.5]^3
Do this by computing minval = min_{q ∈ [-0.5, 0.5]^3} (q+G) * ϵ * (q+G)
Select the G vector if minval / (2π / alat)^2 / (4 * η) < cutoff.
"""
function get_Glist(method :: Polar3D, cell :: Structure, nxs, ϵ)
    (; alat, recip_lattice) = cell
    (; cutoff, η) = method
    ϵ_crystal = recip_lattice' * ϵ * recip_lattice

    metric = (2π / alat)^2  # Conversion factor for G^2, unit bohr⁻²
    # metric = 6.0796001256436796 * (2π / alat)^2

    f(qG) = qG' * ϵ_crystal * qG
    function g!(G, qG)
        G .= 2 .* (ϵ_crystal * qG)
    end

    xq_upper = Vec3(1/2, 1/2, 1/2)
    xq_lower = -xq_upper
    xq_initial = Vec3(0., 0, 0)

    Glist = Vector{Vec3{Int}}()

    for ci in CartesianIndices((-nxs[1]:nxs[1], -nxs[2]:nxs[2], -nxs[3]:nxs[3]))
        G_crystal = Vec3{Int}(ci.I)

        lower = Vector(xq_lower + G_crystal)
        upper = Vector(xq_upper + G_crystal)
        initial_x = Vector(xq_initial + G_crystal)
        inner_optimizer = Optim.LBFGS()
        results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))

        minval = Optim.minimum(results)
        GϵG = minval / (4 * metric * η)

        if GϵG < cutoff
            push!(Glist, G_crystal)
        end
    end
    Glist
end

function get_Glist(method :: Polar2D, cell :: Structure, nxs, ϵ)
    # Truncate by |q+G| * L / 2 < cutoff => |q+G|^2 < (2 * cutoff / L)²
    if nxs[3] != 0
        error("Polar2D is only implemented for 2D systems. nxs[3] must be 1.")
    end
    (; recip_lattice) = cell

    f(qG) = norm(recip_lattice[1:2, 1:2] * qG)^2
    function g!(G, qG)
        G .= 2 .* recip_lattice[1:2, 1:2]' * (recip_lattice[1:2, 1:2] * qG)
    end

    xq_upper = Vec3(1/2, 1/2, 0)
    xq_lower = -xq_upper
    xq_initial = Vec3(0., 0, 0)

    Glist = Vector{Vec3{Int}}()

    for ci in CartesianIndices((-nxs[1]:nxs[1], -nxs[2]:nxs[2], -nxs[3]:nxs[3]))
        G_crystal = Vec3{Int}(ci.I)

        lower = Vector((xq_lower + G_crystal)[1:2])
        upper = Vector((xq_upper + G_crystal)[1:2])
        initial_x = Vector((xq_initial + G_crystal)[1:2])
        inner_optimizer = Optim.LBFGS()
        results = optimize(f, g!, lower, upper, initial_x, Fminbox(inner_optimizer))

        minval = Optim.minimum(results)

        if minval < (2 * method.cutoff / method.L)^2
            push!(Glist, G_crystal)
        end
    end
    Glist
end


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

@timing "lr_dyn_dip" function dynmat_dipole!(dynmat, xq, polar::Polar{Polar3D}, sign=1)
    # Compute dynmat += sign * (dynmat from dipole-dipole interaction)
    polar.use || return dynmat

    (; recip_lattice) = polar.cell
    natom = length(polar.cell.atom_pos)

    # Map xq to [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate_centered(xq)

    Zq = zeros(Vec3{ComplexF64}, natom)

    # Compute only the q-dependent part.
    # The q-independent part is precomputed and stored in polar.dynmat_asr.
    for G_crystal in polar.Glist
        qG = recip_lattice * (xq + G_crystal)  # In bohr⁻¹

        # Skip if G=0 or if exponenent GϵG is large
        _polar_check_skip(polar, qG) && continue

        fac, qϵq = _polar_compute_ϵ(polar, qG)

        for iatom in 1:natom
            Zq[iatom] = _polar_compute_Z(polar, qG, iatom)
        end

        for jatom in 1:natom, iatom in 1:natom
            dyn_tmp = sign * fac * conj.(Zq[iatom] * Zq[jatom]') / qϵq

            @inbounds for j in 1:3, i in 1:3
                dynmat[3*(iatom-1)+i, 3*(jatom-1)+j] += dyn_tmp[i, j]
            end
        end
    end

    dynmat .+= polar.dynmat_asr .* sign

    dynmat
end

"""
    get_eph_dipole_coeffs!(coeff, xq, polar::Polar, u_ph)
Compute coefficients for dipole e-ph coupling. The coefficients depend only on the phonon properties.
- `xq` : q point in crystal coordinates
"""
function get_eph_dipole_coeffs!(coeff_δ, coeff_r, xq, polar::Polar{Polar3D}, u_ph)
    if ! polar.use
        coeff_δ .= 0
        coeff_r .= 0
        return coeff_δ, coeff_r
    end

    (; recip_lattice) = polar.cell
    natom = length(polar.cell.atom_pos)

    # Map xq to [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate_centered(xq)

    # temporary vectors of size (nmodes,)
    tmp = zeros(ComplexF64, polar.nmodes)

    for G_crystal in polar.Glist
        qG = recip_lattice * (xq + G_crystal)  # In bohr⁻¹

        # Skip if G=0 or if exponenent GϵG is large
        _polar_check_skip(polar, qG) && continue

        fac, qϵq = _polar_compute_ϵ(polar, qG)

        for iatom in 1:natom
            Zq = _polar_compute_Z(polar, qG, iatom)
            for ipol in 1:3
                tmp[3*(iatom-1)+ipol] += im * fac * Zq[ipol] / qϵq
            end
        end
    end

    mul!(coeff_δ, Transpose(u_ph), tmp)

    # TODO: coeff_r term

    coeff_δ, coeff_r
end


@views function _polar_check_skip(polar :: Polar{Polar3D}, q :: Vec3)
    (; η, cutoff) = polar.method
    metric = (2π / polar.cell.alat)^2  # Conversion factor for G^2, unit bohr⁻²
    # metric = 6.0796001256436796 * (2π / polar.cell.alat)^2

    # Skip if G=0 or if exponenent GϵG is large
    qϵq = q' * polar.ϵ * q
    skip = (qϵq <= 0) || (qϵq / (4 * metric * η) >= cutoff)
    return skip
end

@views function _polar_compute_ϵ(polar :: Polar{Polar3D}, q :: Vec3)
    # NOTE: Until EPW v5.6, there was sqrt(metric) term in the factor. This was removed in EPW v5.7.
    #       Here, we use the EPW v5.7 formula.
    # fac2 = fac * exp(-GϵG * sqrt(metric) / (4 * metric * polar.η)) / GϵG

    (; η) = polar.method
    fac = ElectronPhonon.e2 * 4π / polar.cell.volume
    metric = (2π / polar.cell.alat)^2
    # metric = 6.0796001256436796 * (2π / polar.cell.alat)^2

    qϵq = q' * polar.ϵ * q  # In bohr⁻²
    fac2 = fac * exp(-qϵq / (4 * metric * η))  # The exponent is unitless

    (; fac2, qϵq)
end

@views function _polar_compute_Z(polar :: Polar{Polar3D}, q :: Vec3, iatom :: Integer)
    phasefac = cis(-polar.cell.alat * q' * polar.cell.atom_pos[iatom])
    Zq_dip  = polar.Z[iatom] * q
    Zq_quad = -im / 2 * Vec3(q' * polar.Q[iatom][i] * q for i in 1:3)
    (Zq_dip + Zq_quad) * phasefac
end




# 2D long-range electrostatics

@timing "lr_dyn_dip" function dynmat_dipole!(dynmat, xq, polar::Polar{Polar2D}, sign = 1)
    # Compute dynmat += sign * (dynmat from dipole-dipole interaction)
    polar.use || return dynmat

    (; recip_lattice) = polar.cell
    natom = length(polar.cell.atom_pos)

    # Map xq to [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate_centered(xq)

    # Compute only the q-dependent part.
    # The q-independent part is precomputed and stored in polar.dynmat_asr.

    GZ_para = zeros(Vec3{ComplexF64}, natom)
    GZ_perp = zeros(Vec3{ComplexF64}, natom)

    @views for G_crystal in polar.Glist
        qG = recip_lattice * (xq + G_crystal)  # In bohr⁻¹

        # Skip if qG is too large
        _polar_check_skip(polar, qG) && continue

        fac, ϵ_para, ϵ_perp = _polar_compute_ϵ(polar, qG)

        for iatom in 1:natom
            GZ_para_iatom, GZ_perp_iatom = _polar_compute_Z(polar, qG, iatom)
            GZ_para[iatom] = GZ_para_iatom
            GZ_perp[iatom] = GZ_perp_iatom
        end

        for jatom in 1:natom, iatom in 1:natom
            dyn_tmp = sign * fac * (  conj.(GZ_para[iatom] * GZ_para[jatom]') / ϵ_para
                                    - conj.(GZ_perp[iatom] * GZ_perp[jatom]') / ϵ_perp )
            @inbounds for j in 1:3, i in 1:3
                dynmat[3*(iatom-1)+i, 3*(jatom-1)+j] += dyn_tmp[i, j]
            end
        end
    end

    dynmat .+= polar.dynmat_asr .* sign

    dynmat
end


function get_eph_dipole_coeffs!(coeff_δ, coeff_r, xq, polar::Polar{Polar2D}, u_ph)
    if ! polar.use
        coeff_δ .= 0
        coeff_r .= 0
        return coeff_δ, coeff_r
    end

    (; alat, recip_lattice, atom_pos) = polar.cell
    natom = length(atom_pos)

    # Map xq to [-0.5, 0.5]^3
    xq = normalize_kpoint_coordinate_centered(xq)

    # temporary vectors of size (nmodes,)
    tmp_δ = zeros(ComplexF64, polar.nmodes)
    tmp_r = zeros(ComplexF64, polar.nmodes, 3)

    @views for G_crystal in polar.Glist
        qG = recip_lattice * (xq + G_crystal)  # In bohr⁻¹
        qG_2d = Vec2(qG[1], qG[2])
        qnorm = norm(qG_2d)

        # Skip if qG is too large
        _polar_check_skip(polar, qG) && continue

        fac, ϵ_para, ϵ_perp = _polar_compute_ϵ(polar, qG)

        for iatom in 1:natom
            phasefac = cis(-alat * dot(qG, atom_pos[iatom]))
            GZi_para = (qG_2d' * SMatrix{2, 3}(polar.Z[iatom][1:2, :]))'
            GZi_perp = qnorm * polar.Z[iatom][3, :]

            GQi_para = (
                Vec3(qG_2d' * SMatrix{2, 2}(polar.Q[iatom][i][1:2, 1:2]) * qG_2d / 2 for i in 1:3)
                - qnorm^2 * Vec3(polar.Q[iatom][i][3, 3] / 2 for i in 1:3)
            )

            for ipol in 1:3
                tmp_δ[3*(iatom-1)+ipol] += fac * phasefac * (im * GZi_para[ipol] + GQi_para[ipol]) / ϵ_para
            end

            for ipol in 1:3
                tmp_r[3*(iatom-1)+ipol, 1] -= fac * phasefac * qG[1] * GZi_para[ipol] / ϵ_para
                tmp_r[3*(iatom-1)+ipol, 2] -= fac * phasefac * qG[2] * GZi_para[ipol] / ϵ_para
                tmp_r[3*(iatom-1)+ipol, 3] += fac * phasefac * qnorm * GZi_perp[ipol] / ϵ_perp
            end
        end
    end

    mul!(coeff_δ, Transpose(u_ph), tmp_δ)
    mul!(coeff_r, Transpose(u_ph), tmp_r)

    coeff_δ, coeff_r
end

@views function _polar_check_skip(polar :: Polar{Polar2D}, q :: Vec3)
    (; L, cutoff) = polar.method
    q_2d = Vec2(q[1], q[2])
    qnorm = norm(q_2d)

    # Skip if G=0 or if argument of tanh is large
    skip = (qnorm <= 0) || (qnorm * L / 2 > cutoff)
    return skip
end

@views function _polar_compute_ϵ(polar :: Polar{Polar2D}, q :: Vec3)
    L = polar.method.L
    c = polar.cell.lattice[3, 3]
    fac = ElectronPhonon.e2 * 2π / polar.cell.volume * c

    q_2d = Vec2(q[1], q[2])
    qnorm = norm(q_2d)
    f = 1 - tanh(qnorm * L / 2)
    fac2 = fac * f / qnorm

    α_para = c / 4π * q_2d' * (SMatrix{2, 2}(polar.ϵ[1:2, 1:2]) - I) * q_2d
    α_perp = c / 4π * (polar.ϵ[3, 3] - 1)

    ϵ_para = 1 + 2π * f * α_para / qnorm
    ϵ_perp = 1 - 2π * f * α_perp * qnorm

    (; fac2, ϵ_para, ϵ_perp)
end

@views function _polar_compute_Z(polar :: Polar{Polar2D}, q :: Vec3, iatom :: Integer)
    q_2d = Vec2(q[1], q[2])
    phasefac = cis(-polar.cell.alat * q' * polar.cell.atom_pos[iatom])

    qnorm = norm(q_2d)
    Z_para = (
        (q_2d' * SMatrix{2, 3}(polar.Z[iatom][1:2, :]))'
        - 0.5im * Vec3(q_2d' * SMatrix{2, 2}(polar.Q[iatom][i][1:2, 1:2]) * q_2d for i in 1:3)
        + 0.5im * qnorm^2 * Vec3(polar.Q[iatom][i][3, 3] for i in 1:3)
    ) * phasefac

    Z_perp = (
        qnorm * polar.Z[iatom][3, :]
        - im * qnorm * Vec3(Vec2(polar.Q[iatom][i][3, 1:2])' * q_2d for i in 1:3)
    ) * phasefac

    Z_para, Z_perp
end
