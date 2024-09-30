
# Adapted from units.jl of the DFTK.jl package
# https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/common/units.jl
# There, Hartree atomic units is used. Here, we use Rydberg atomic units.

export unit_to_aru

const e2 = 2 # e^2 in Rydberg atomic units
const e_SI = 1.602176634e-19 # e in SI units

# Commonly used constants. The factors convert from the respective unit
# to atomic units
module units
    using ElectronPhonon: e2, e_SI
    const Ha = 2                   # Hartree -> Rydberg
    const Ry = 1                   # Rydberg -> Rydberg
    const eV = 0.03674932248 * 2   # electron volt -> Rydberg
    const meV = 0.03674932248 * 2 * 1E-3  # mili electron volt -> Rydberg
    const Å  = 1.8897261254578281  # Ångström -> Bohr
    const cm = 1.8897261254578281 * 1E8  # cm -> Bohr
    const m  = 1.8897261254578281 * 1E10  # m -> Bohr
    const K  = 3.166810508e-6 * 2  # Kelvin -> Rydberg
    const ħ  = 1 / 1.054571800E-34 # J*s -> Rydberg units
    const s = 1 / (4.8377687 * 1e-17) # second -> Rydberg units
    const C = sqrt(e2) / e_SI # Coulomb
    const J = eV / e_SI # 1 Joule = 1 eV / e
    const V = J / C # 1 Volt = 1 J / C
    const A = C / s # 1 Ampere = 1 C / s
    const THz = 2π * 1e12 / s  # THz -> Rydberg
end

"""
    unit_to_aru(symbol)
Get the factor converting from the unit `symbol` to atomic Rydberg units.
E.g. `unit_to_aru(:eV)` returns the conversion factor from electron volts to Rydberg.
"""
unit_to_aru(symbol::Symbol) = getfield(units, symbol)
