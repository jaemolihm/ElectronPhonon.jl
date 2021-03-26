# __precompile__(true)

using Statistics

# TODO: separate to smearing.jl and kpoints.jl

# module Utils
export occ_fermion
export occ_boson
export gaussian
export generate_kvec_grid
export average_degeneracy
export bisect

"""
Occupation of a fermion at energy `e` and temperature `T`.
"""
function occ_fermion(e, T; occ_type="fd")
    if occ_type == "fd"
        if T > 1.0E-8
            return 1 / (exp(e/T) + 1)
        else
            return (sign(e) + 1) / 2
        end
    else
        throw(ArgumentError("unknown occ_type $occ_type"))
    end
end

"""
Derivative of the fermion occupation function with respect to `e`. Approximation to minus
the delta function divided by temperature.
"""
function occ_fermion_derivative(e, T; occ_type="fd")
    if occ_type == "fd"
        if T > 1.0E-8
            # occ = occ_fermion(e, T, occ_type="fd")
            # return occ * (1 - occ) / T
            x = e / T
            return -1 / (2 + exp(x) + exp(-x)) / T
        else
            return zero(e)
        end
    else
        throw(ArgumentError("unknown occ_type $occ_type"))
    end
end

"""
Occupation of a boson at energy `e` and temperature `T`.
"""
function occ_boson(e, T, occ_type="be")
    if occ_type == "be"
        if T > 1.0E-8
            return 1 / (exp(e/T) - 1)
        else
            throw(ArgumentError("Temperature for occ_boson cannot be zero or negative"))
        end
    else
        throw(ArgumentError("unknown occ_type $occ_type"))
    end
end

const inv_sqrtpi = 1 / sqrt(pi)
function gaussian(value)
    # One without the conditional is slightly (~2.5 %) faster.
    exp(-value^2) * inv_sqrtpi
    # abs(value) < 15.0 ? exp(-value^2) * inv_sqrtpi : 0.0
end

"Average data for degenerate states using energy. Non allocating version."
function average_degeneracy!(data_averaged, data, energy, degeneracy_cutoff=1.e-6)
    @assert size(data) == size(energy)
    @assert size(data) == size(data_averaged)
    nbnd, nk = size(data)

    iblist_degen = zeros(Bool, nbnd)

    # This is fast enough, so no need for threading.
    # Threads.@threads :static for ik in 1:nk
    for ik in 1:nk
        for ib in 1:nbnd
            iblist_degen .= abs.(energy[:, ik] .- energy[ib, ik]) .< degeneracy_cutoff
            data_averaged[ib, ik] = mean(data[iblist_degen, ik])
        end # ib
    end # ik
    nothing
end

"Average data for degenerate states using energy"
function average_degeneracy(data, energy, degeneracy_cutoff=1.e-6)
    data_averaged = similar(data)
    average_degeneracy!(data_averaged, data, energy, degeneracy_cutoff)
    data_averaged
end
# end
