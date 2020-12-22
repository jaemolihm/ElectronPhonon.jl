
# Adapted from units.jl of the DFTK.jl package
# https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/common/units.jl
# There, Hartree atomic units is used. Here, we use Rydberg atomic units.

export unit_to_aru

# Commonly used constants. The factors convert from the respective unit
# to atomic units
module units
    const Ha = 2                   # Hartree -> Rydberg
    const Ry = 1                   # Rydberg -> Rydberg
    const eV = 0.03674932248 * 2   # electron volt -> Rydberg
    const meV = 0.03674932248 * 2 * 1E-3  # mili electron volt -> Rydberg
    const Å  = 1.8897261254578281  # Ångström -> Bohr
    const K  = 3.166810508e-6 * 2  # Kelvin -> Rydberg
end

"""
    unit_to_aru(symbol)
Get the factor converting from the unit `symbol` to atomic Rydberg units.
E.g. `unit_to_aru(:eV)` returns the conversion factor from electron volts to Rydberg.
"""
unit_to_aru(symbol::Symbol) = getfield(units, symbol)
