
"Scatterings in Boltzmann transport calculation"

struct ElPhScatteringData{T <: Real} <: AbstractBTData{T}
    # Each index (i = 1, ..., n) represents a single scattering process.
    # State indices refers to a state defined by a BTStates object.
    n::Int # Number of scattering processes
    ind_el_i::Vector{Int} # Index of electron initial states
    ind_el_f::Vector{Int} # Index of electron final states
    ind_ph::Vector{Int} # Index of phonon states
    sign_ph::Vector{Int} # Sign of phonon energy. +1 for emission, -1 for absorption.
    mel::Vector{T} # Scattering matrix elements. (Squared e-ph matrix elements.)
end

function ElPhScatteringData{T}() where {T}
    ElPhScatteringData{T}(0, Vector{Int}(), Vector{Int}(), Vector{Int}(), Vector{Int}(),
        Vector{T}())
end

function concatenate_scattering(scatterings::ElPhScatteringData{T}...) where {T}
    n = sum([scat.n for scat in scatterings])
    ind_el_i = vcat([scat.ind_el_i for scat in scatterings]...)
    ind_el_f = vcat([scat.ind_el_f for scat in scatterings]...)
    ind_ph = vcat([scat.ind_ph for scat in scatterings]...)
    sign_ph = vcat([scat.sign_ph for scat in scatterings]...)
    mel = vcat([scat.mel for scat in scatterings]...)
    ElPhScatteringData{T}(n, ind_el_i, ind_el_f, ind_ph, sign_ph, mel)
end
