"""
Lindhard screening.
Reference: L. Hedin, Phys. Rev. 139, A796 (1965)
"""

# TODO: Test plasma frequency

export LindhardScreeningParams

using Parameters
using StaticArrays
using LinearAlgebra

@with_kw struct LindhardScreeningParams{T<:Real}
    degeneracy::Int64 # degeneracy of bands. spin and/or valley degeneracy.
    m_eff::T # Ratio of effective mass and electron mass. Unitless.
    n::T # Absolute carrier density in Bohr^-3.
    ϵM::T # Macroscopic dielectric constant. Unitless.
    smearing::T # Smearing of frequency in Rydberg.
end


# See Eq. (56) of Hedin (1965)
H_lindhard(z) = 2*z + (1-z^2) * log((z+1)/(z-1))


"""
    epsilon_lindhard(xq, ω, params::LindhardScreeningParams; verbose=false)
Compute dielectric function using Lindhard theory. Use Eq. (56) of Hedin (1965).
Assume an isotropic 3d parabolic band with effective mass m_eff, and assume zero temperature.
Generalized to work with arbitrary number of degeneracy (spin and/or valley).

# Inputs
- `xq`: the crystal momentum in Cartesian basis, 1/Bohr.
- `ω`: frequency in Ry.
- `params::LindhardScreeningParams`: parameters.
- `verbose`: if true, print Lindhard screening parameters.
"""
function epsilon_lindhard(xq, ω, params::LindhardScreeningParams; verbose=false)
    (; degeneracy, m_eff, n, ϵM, smearing) = params

    if norm(xq) < 1E-10
        return Complex{eltype(xq)}(1)
    end

    # Compute parameters
    k_fermi = (6 * π^2 * abs(n) / degeneracy)^(1/3) # k0 in Hedin (1965)
    e_fermi = k_fermi^2 / abs(m_eff) # in Rydberg units
    rs = abs(m_eff) / ϵM * (3/(4π*abs(n)))^(1/3)
    coeff = (4/9π)^(1/3) * rs / 8π * (degeneracy / 2)^(4/3)
    if verbose
        println("Lindhard screening parameters")
        println("rs = $rs (bohr)")
        println("k_fermi = $k_fermi (bohr⁻¹)")
        println("e_fermi = $(e_fermi / unit_to_aru(:eV)) (eV)")
        # println("k_TF = ", sqrt((16/3π^2)^(2/3)*rs*k_fermi^2)) # FIXME: This equation assumes degeneracy == 2
    end

    # Unitless variables
    q = norm(xq) / (2 * k_fermi)
    u = (ω + im * smearing) / (4 * e_fermi)

    # Dielectric function
    ϵ = 1 + coeff * (H_lindhard(q + u/q) + H_lindhard(q - u/q)) / q^3
    ϵ
end
