# TODO: separate to smearing.jl and kpoints.jl
using SpecialFunctions: erf

# module Utils
export occ_fermion
export occ_fermion_derivative
export occ_boson
export gaussian
export delta_smeared
export average_degeneracy
export bisect

"""
    occ_fermion(e, T; occ_type=:FermiDirac)
Occupation of a fermion at energy `e` and temperature `T`.
"""
@inline function occ_fermion(e, T; occ_type=:FermiDirac)
    if T > sqrt(eps(eltype(T)))
        x = e / T

        if occ_type == :FermiDirac
            return 1 / (exp(x) + 1)

        elseif occ_type == :ColdSmearing || occ_type == :MarzariVanderbilt || occ_type == :MV
            # Cold smearing (Marzari, Vanderbilt, DeVita, Payne, PRL 82, 3296 (1999))
            xp = x + 1 / sqrt(2)
            return (sqrt(2/π) * exp(-xp^2) + 1 - erf(xp)) / 2

            # TODO: Implement Methfessel-Paxton

        else
            throw(ArgumentError("unknown occ_type $occ_type"))
        end

    elseif T >= 0
        return (1 - sign(e)) / 2

    else
        throw(ArgumentError("Temperature must be positive"))
    end
end

"""
    occ_fermion_derivative(e, T; occ_type=:FermiDirac)
Derivative of the fermion occupation function with respect to `e` at temperature `T`.
Approximation to minus the delta function divided by temperature.
"""
@inline function occ_fermion_derivative(e, T; occ_type=:FermiDirac)
    if occ_type == :FermiDirac
        x = e / T
        return -1 / (2 + exp(x) + exp(-x)) / T
        # The following is mathematically equivalent, but is numerically unstable if e << -T
        # occ = occ_fermion(e, T, occ_type=:FermiDirac)
        # return -occ * (1 - occ) / T
    else
        throw(ArgumentError("unknown occ_type $occ_type"))
    end
end

"""
    occ_boson(e, T, occ_type=:BoseEinstein)
Occupation of a boson at energy `e` and temperature `T`.
"""
@inline function occ_boson(e, T, occ_type=:BoseEinstein)
    if occ_type == :BoseEinstein
        if T > sqrt(eps(eltype(T)))
            return e == 0 ? zero(e) : 1 / expm1(e / T)
        elseif T >= 0
            return zero(e)
        else
            throw(ArgumentError("Temperature must be positive"))
        end
    else
        throw(ArgumentError("unknown occ_type $occ_type"))
    end
end

const inv_sqrtpi = 1 / sqrt(pi)
@inline function gaussian(value)
    # TODO: type stability
    # One without the conditional is slightly (~2.5 %) faster.
    exp(-value^2) * inv_sqrtpi
    # abs(value) < 15.0 ? exp(-value^2) * inv_sqrtpi : 0.0
end


"""
    delta_smeared(Δe, smearing :: Tuple{Symbol, Float64})

Smeared delta function.

- `smearing == (:Gaussian, η)`: Gaussian smearing.
- `smearing == (:Lorentzian, η)`: Lorentzian smearing.
- `smearing == (:Tetrahedron, η)`: Tetrahedron smearing.
- `smearing == (:GaussianTetrahedron, η)`: Gaussian Tetrahedron smearing.
"""
function delta_smeared(Δe, smearing :: Tuple{Symbol, Float64})
    type, η = smearing

    if type == :Gaussian
        delta = gaussian(Δe / η) / η

    elseif type == :Lorentzian
        delta = η / (Δe^2 + η^2) / π

    elseif type == :Tetrahedron
        # v_cart = - el_f.vdiag[ind_el_f] - sign_ph * ph.vdiag[ind_ph]
        # v_delta_e = recip_lattice' * v_cart
        # delta = delta_parallelepiped(zero(FT), delta_e, v_delta_e, 1 ./ ngrid)
        throw(ArgumentError("Smearing Tetrahedron not implemented yet"))

    elseif type == :GaussianTetrahedron
        # v_cart = - el_f.vdiag[ind_el_f] - sign_ph * ph.vdiag[ind_ph]
        # v_delta_e = recip_lattice' * v_cart
        # delta = gaussian_parallelepiped(η, delta_e, v_delta_e, 1 ./ ngrid) / η
        throw(ArgumentError("Smearing GaussianTetrahedron not implemented yet"))

    else
        throw(ArgumentError("Unknown smearing type: $type"))
    end
end


"Average data for degenerate states using energy. Non allocating version."
function average_degeneracy!(data_averaged, data, energy, degeneracy_cutoff=1.e-6)
    # TODO: Use electron_degen_cutoff
    @assert size(data) == size(data_averaged)
    @assert axes(data) == axes(data_averaged)

    _mean(x) = sum(x) / length(x)

    iblist_degen = zeros(Bool, axes(data, 1))

    # This is fast enough, so no need for threading.
    @views for ik in axes(data, 2)
        for ib in axes(data, 1)
            for jb in axes(data, 1)
                iblist_degen[jb] = abs(energy[jb, ik] - energy[ib, ik]) < degeneracy_cutoff
            end
            data_averaged[ib, ik] = _mean(data[iblist_degen, ik])
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
