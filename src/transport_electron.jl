
# Functions related to calculation of electron conductivity

export transport_get_μ
export TransportParams

Base.@kwdef struct TransportParams{Type <: Real}
    T::Type # Temperature
    n::Type # Carrier density
    carrier_type::String # e (electron) or h (hole)
    nband_valence::Int # Number of valence bands (used only for semiconductors)
    degaussw::Type # Smearing parameter for delta function
    spin_degeneracy::Int # Spin degeneracy
end

function transport_get_μ(energy, weights, parameters::TransportParams)
    if parameters.carrier_type == "e"
        @views e_carrier = energy[parameters.nband_valence+1:end, :]
    elseif parameters.carrier_type == "h"
        @views e_carrier = energy[1:parameters.nband_valence, :]
    else
        error("carrier_type must be e or h, not $carrier_type")
    end

    ncarrier_target = parameters.n / parameters.spin_degeneracy
    find_fermi_energy(ncarrier_target, parameters.T, e_carrier, weights)
end
