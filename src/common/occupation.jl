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

    # Type of the occupation function
    occ_type :: Symbol

    function ElectronOccupationParams(; Tlist, nlist, μlist = nothing, volume, nelec, spin_degeneracy, type = nothing, occ_type = :FermiDirac)
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

        if occ_type ∉ (:FermiDirac, :MV)
            throw(ArgumentError("Invalid occupation type: $occ_type"))
        end

        new(Tlist, nlist, μlist, nelec, volume, spin_degeneracy, type, occ_type)
    end
end


# Iterator interface
Base.length(occ::ElectronOccupationParams) = length(occ.μlist)
Base.firstindex(occ) = 1
Base.lastindex(occ) = length(occ)
Base.keys(occ::ElectronOccupationParams) = 1:length(occ)
function Base.getindex(occ::ElectronOccupationParams, i)
    1 <= i <= length(occ) || throw(BoundsError(occ, i))
    return (; T=occ.Tlist[i], μ=occ.μlist[i], n=occ.nlist[i])
end
function Base.iterate(occ::ElectronOccupationParams, i=1)
    i > length(occ) ? nothing : (occ[i], i+1)
end

function occ_fermion(e, occ :: ElectronOccupationParams, i :: Integer)
    (; μ, T) = occ[i]
    occ_fermion(e - μ, T; occ.occ_type)
end
function occ_fermion_derivative(e, occ :: ElectronOccupationParams, i :: Integer)
    (; μ, T) = occ[i]
    occ_fermion_derivative(e - μ, T; occ.occ_type)
end


"""
    chemical_potential_is_computed(occ :: ElectronOccupationParams)
If `μlist` is set to a finite value, return true.
Otherwise, `μlist` has to be computed, so return false.
"""
function chemical_potential_is_computed(occ :: ElectronOccupationParams)
    all(.!isnan.(occ.μlist))
end

"""
    set_chemical_potential!(occ, el_states, kpts, nelec_below_window; do_print = true)
If `occ.μlist` is not set, compute the chemical potential and set it.
If `occ.μlist` is already set to some value that is not NaN, do nothing.
"""
function set_chemical_potential!(occ, el_states, kpts, nelec_below_window; do_print = true)
    if !chemical_potential_is_computed(occ)
        el, _ = electron_states_to_BTStates(el_states, kpts, nelec_below_window)
        bte_compute_μ!(occ, el; do_print)
    end
    occ
end


function bte_compute_μ!(occ :: ElectronOccupationParams, el; do_print=true)
    if occ.type == :Metal
        # For metals, compute the total electron density.
        # Since occ.n is the difference of number of electrons per cell from nband_valence,
        # nband_valence should be added for the real target ncarrier.
        # Also, el.nstates_base is the contribution to the ncarrier from occupied states
        # outside the window (i.e. not included in `energy`). So it is subtracted.
        ncarrier_target = @. (occ.nlist + occ.nelec) / occ.spin_degeneracy - el.nstates_base
    elseif occ.type == :Semiconductor
        # For semiconductors, count the doped carriers to minimize floating point error.
        # FIXME: nband_valence needs to be a field
        nband_valence = round(Int, occ.nelec / occ.spin_degeneracy)
        ncarrier_target = @. occ.nlist / occ.spin_degeneracy
        e_e = el.e[el.iband .>  nband_valence]
        e_h = el.e[el.iband .<= nband_valence]
        w_e = el.k_weight[el.iband .>  nband_valence]
        w_h = el.k_weight[el.iband .<= nband_valence]
    else
        throw(DomainError("Invalid occ.type $(occ.type)"))
    end

    for i in axes(occ.Tlist, 1)
        T = occ.Tlist[i]
        if occ.type == :Metal
            μ = find_chemical_potential(ncarrier_target[i], T, el.e, el.k_weight; occ.occ_type)
        elseif occ.type == :Semiconductor
            μ = find_chemical_potential_semiconductor(ncarrier_target[i], T, e_e, e_h, w_e, w_h; occ.occ_type)
        end
        occ.μlist[i] = μ
        if do_print && mpi_isroot()
            @info @sprintf "n = %.1e cm^-3 , T = %.1f K (%s) , μ = %.4f eV" occ.nlist[i] / (occ.volume/unit_to_aru(:cm)^3) T/unit_to_aru(:K) occ.occ_type μ/unit_to_aru(:eV)
        end
    end
    occ.μlist
end
