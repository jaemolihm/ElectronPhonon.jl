"""
    ElectronOccupationParams

Parameters for defining electron occupation.

`nlist` and `nelec` includes the spin degeneracy (e.g. set `nelec == 8` for bulk silicon).

**NOTE**: If `type == :Semiconductor`, it is assumed that nelec is an integer and that
the maximum of the `nelec`-th band is below the minimum of the `nelec+1`-th band.
If this condition does not hold, manually set `type = :Metal`.
"""
struct ElectronOccupationParams
    # List of temperatures
    Tlist :: Vector{Float64}

    # List of the number of doped electrons per unit cell.
    # The total number of electrons is `nlist + nelec`.
    nlist :: Vector{Float64}

    # List of chemical potential
    μlist :: Vector{Float64}

    # Number of electrons per unit cell in the undoped configuration.
    nelec :: Float64

    # Volume of the unit cell
    volume :: Float64

    # Spin degeneracy
    spin_degeneracy :: Int

    # Type of the carrier. `:Metal` or `:Semiconductor`.
    type :: Symbol

    function ElectronOccupationParams(; Tlist, nlist, μlist = nothing, volume, nelec, spin_degeneracy, type = nothing)
        # If either T or n is a scalar, convert it to a vector.
        if Tlist isa Number && nlist isa Number
            Tlist = [Tlist]
            nlist = [nlist]

        elseif Tlist isa Number && nlist isa Vector
            Tlist = fill(Tlist, length(nlist))

        elseif Tlist isa Vector && nlist isa Number
            nlist = fill(nlist, length(Tlist))

        elseif Tlist isa Vector && nlist isa Vector
            nothing

        else
            throw(ArgumentError("Tlist and nlist must be either a scalar or a vector."))
        end

        if μlist === nothing
            # If μlist is not set, fill it with NaN
            μlist = fill(NaN, length(nlist))
        elseif μlist isa Number
            μlist = fill(μlist, length(nlist))
        end

        if length(Tlist) !== length(nlist)
            throw(ArgumentError("Length of Tlist and nlist must be the same."))
        end
        if length(Tlist) !== length(μlist)
            throw(ArgumentError("Length of Tlist and μlist must be the same."))
        end

        if type === nothing
            # If doping (`nlist`) is small and nelec is an integer, assume the system
            # is a semiconductor. Otherwise, assume it is a metal.
            if maximum(abs.(nlist)) < 1 && round(Int, nelec) ≈ nelec
                type = :Semiconductor
            else
                type = :Metal
            end
        end

        new(Tlist, nlist, μlist, nelec, volume, spin_degeneracy, type)
    end
end

function Base.getproperty(occ::ElectronOccupationParams, name::Symbol)
    # TODO: Rename all "nband_valence" to "nelec"
    if name === :nband_valence
        return getfield(occ, :nelec) / getfield(occ, :spin_degeneracy)
    else
        return getfield(occ, name)
    end
end

"""
    chemical_potential_is_computed(occ :: ElectronOccupationParams)
If `μlist` is set to a finite value, return true.
Otherwise, `μlist` has to be computed, so return false.
"""
function chemical_potential_is_computed(occ :: ElectronOccupationParams)
    all(.!isnan.(occ.μlist))
end


# Iterator interface
Base.length(occ::ElectronOccupationParams) = length(occ.μlist)
Base.firstindex(occ) = 1
Base.lastindex(occ) = length(occ)

function Base.getindex(occ::ElectronOccupationParams, i)
    1 <= i <= length(occ) || throw(BoundsError(occ, i))
    return (; T=occ.Tlist[i], μ=occ.μlist[i], n=occ.nlist[i])
end

function Base.iterate(occ::ElectronOccupationParams, i=1)
    i > length(occ) ? nothing : (occ[i], i+1)
end
