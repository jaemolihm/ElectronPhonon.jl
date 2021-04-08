## Note on comparing results with EPW
There are several differences that affects the results of EPW.jl and EPW, even if same
parameters are used.

1. In EPW.jl, the window is applied to each state and filters out states outside the window.
In EPW, the window filters out k points that have no states inside the window. But if a
k point has at least one state inside the window, then all states of that k point are
included, whether or not they are inside the window.
This difference affects following quantities.
    * quantities involving sum over off-shell states (e.g. the real part of self-energy)
    * imaginary self-energy of electron states whose energy difference with the window
    boundary is comparable to smearing (because in EPW, states right outside the window
    may be included, while in EPW.jl, they are not included.)

2. The phonon spectral function implementation of EPW is not consistent with the equation
in the literature, because it takes the absolute value of the imaginary part of the
self-energy.

3. Degeneracy may lead to differences in the matrix elements. Electron and phonon
self-energy is aveaged over degenerate states, so they are not affected by degeneracy.
But, the self-energy for phonon spectral function is not averaged over degenerate states.
So, one should compare only the q points without any degeneracy between phonon modes.