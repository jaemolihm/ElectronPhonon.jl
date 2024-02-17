"""
Lindhard screening.
Reference: L. Hedin, Phys. Rev. 139, A796 (1965)
"""

# TODO: Test plasma frequency

export LindhardScreeningParams

using Parameters
using StaticArrays
using LinearAlgebra
using QuadGK

@with_kw struct LindhardScreeningParams{T<:Real}
    degeneracy::Int64 # degeneracy of bands. spin and/or valley degeneracy.
    m_eff::T # Ratio of effective mass and electron mass. Unitless.
    nlist::Vector{T} # Absolute carrier density in Bohr^-3.
    ϵM::Mat3{T} # Macroscopic dielectric constant. Unitless.
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
    (; degeneracy, m_eff, nlist, ϵM, smearing) = params
    _epsilon_lindhard.(Ref(xq), ω, nlist, m_eff, Ref(ϵM), degeneracy, smearing; verbose)
end

function _epsilon_lindhard(xq, ω, n, m_eff, ϵM, degeneracy, smearing; verbose=false)
    if norm(xq) < 1E-10
        return Complex{eltype(n)}(1)
    end

    ϵM_q = xq' * ϵM * xq / norm(xq)^2

    # Compute parameters
    k_fermi = (6 * π^2 * abs(n) / degeneracy)^(1/3) # k0 in Hedin (1965)
    e_fermi = k_fermi^2 / abs(m_eff) # in Rydberg units
    rs = abs(m_eff) / ϵM_q * (3/(4π*abs(n)))^(1/3)
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


function epsilon_lindhard_finite_temperature(xq, ω, m, T, μ, ϵM, degeneracy, smearing)
    q = norm(xq)
    ϵM_q = xq' * ϵM * xq
    χ0 = _epsilonlindhard_finite_temperature_χ0(q, ω + im * smearing, m, T, μ)
    ϵ = 1 - degeneracy * χ0 * 4π * ElectronPhonon.e2 / ϵM_q
    ϵ
end

function _epsilonlindhard_finite_temperature_χ0(q, ω, m, T, μ)
    εq = q^2 / m
    function f(k)
        occ_fermion(k^2/m - μ, T) * k * (  log(( ω - εq - 2k*q/m) / ( ω - εq + 2k*q/m))
                                         + log((-ω - εq - 2k*q/m) / (-ω - εq + 2k*q/m)))
    end
    quadgk(f, 0, Inf)[1] * -m / 2 / 4π^2 / q
end

function epsilon_lindhard_multivalley(xq, ω, m_list, T, μ, ϵM, degeneracy, smearing)
    χ0 = mapreduce(+, m_list) do m
        m_avg = det(m)^(1/3)
        q_avg = sqrt(m_avg * dot(xq, m \ xq))
        _epsilonlindhard_finite_temperature_χ0(q_avg, ω + im * smearing, m_avg, T, μ)
    end
    ϵM_q = xq' * ϵM * xq
    ϵ = 1 - degeneracy * χ0 * 4π * ElectronPhonon.e2 / ϵM_q
    ϵ
end
