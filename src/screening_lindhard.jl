"""
Lindhard screening.
Reference: L. Hedin, Phys. Rev. 139, A796 (1965)
"""

export LindhardParams

using Parameters
using StaticArrays
using LinearAlgebra

@with_kw struct LindhardParams{T<:Real}
    degeneracy::Int64 # degeneracy of bands. spin and/or valley degeneracy.
    m_eff::T # Ratio of effective mass and electron mass. Unitless.
    n::T # Absolute carrier density per unit cell in Bohr^-3.
    ϵM::T # Macroscopic dielectric constant. Unitless.
    smearing::T # Smearing of frequency in Rydberg.
end


# See Eq. (56) of Hedin (1965)
H_lindhard(z) = 2*z + (1-z^2) * log((z+1)/(z-1))


"""
    epsilon_lindhard(xq, ω, params::LindhardParams; verbose=false)
Compute dielectric function using Lindhard theory. Use Eq. (56) of Hedin (1965).
Assume an isotropic, 3d parabolic band with effective mass m_eff, assume zero temperature.
- xq: the crystal momentum in Cartesian basis, 1/Bohr.
- ω: frequency in Ry.
- verbose: if true, print Lindhard screening parameters.
"""
function epsilon_lindhard(xq, ω, params::LindhardParams; verbose=false)
    degeneracy = params.degeneracy
    m_eff = params.m_eff
    n = params.n
    ϵM = params.ϵM
    smearing = params.smearing

    if norm(xq) < 1E-10
        return Complex{eltype(xq)}(1)
    end

    # Compute parameters
    kFermi = (6 * π^2 * n / degeneracy)^(1/3)
    EFermi = kFermi^2 / (2 * m_eff) # in Hartree
    rs = m_eff / ϵM * (3/(4π*n))^(1/3)
    coeff = (4/9π)^(1/3) * rs / 8π
    if verbose
        # TODO: Here, degeneracy==2 is assumed.
        println("Lindhard screening parameters")
        @show rs
        @show kFermi
        println("EFermi = ", EFermi * unit_to_aru(:Ha))
        println("k_TF = ", sqrt((16/3π^2)^(2/3)*rs*kFermi^2))
    end

    # Unitless variables
    q = norm(xq) / (2 * kFermi)
    u = (ω + im * smearing) / unit_to_aru(:Ha) / (4 * EFermi)

    # Dielectric function
    ϵ = 1 + coeff * (H_lindhard(q + u/q) + H_lindhard(q - u/q)) / q^3
    ϵ
end
