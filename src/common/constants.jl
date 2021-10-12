
# Threshold for the zero-frequency acoustic mode.
# Phonon modes with energy below omega_acoustic are ignored.
# const omega_acoustic = 6.1992E-04 * unit_to_aru(:eV)
const omega_acoustic = 0.01 * unit_to_aru(:meV)
const electron_degen_cutoff = 0.01 * unit_to_aru(:meV)
