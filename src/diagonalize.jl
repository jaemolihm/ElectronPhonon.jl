# __precompile__(true)

using LinearAlgebra
using EPW.AllocatedLAPACK: epw_syev!

export solve_eigen_el!
export solve_eigen_el_valueonly!
export solve_eigen_ph!
export solve_eigen_ph_valueonly!

function degeneracy_perturbation_matrix!(P, n)
    # See subroutine hamwan2bloch of EPW
    P .= 0
    @inbounds for i in 1:n, j in 1:n
        P[j, i] = (0.25644832 + 0.005 * (i + j)) + im * (0.0125 * (j  - i))
    end
    P
end

@inline function is_degenerate(e, cutoff)
    @inbounds for i in 1:length(e)-1
        if abs(e[i+1] - e[i]) < cutoff
            return true
        end
    end
    return false
end

"Get eigenenergy and eigenvector of electrons at a single k point.
Input hk is not destroyed at output."
@timing "eig_el" function solve_eigen_el!(eigvalues, eigvectors, hk)
    @assert size(eigvectors) == size(hk)
    # Directly calling LAPACK.syev! is more efficient than
    # eigen(Hermitian(hk)) for small matrices, because eigen uses
    # SYEVR which is slower than SYEV for small matrices
    # eigvalues, eigvectors[:, :] = eigen(Hermitian(hk))
    # Use AllocatedLAPACK module for preallocating and reusing workspaces
    eigvectors .= hk
    epw_syev!('V', 'U', eigvectors, eigvalues)

    # If there are no degenerate bands, return
    is_degenerate(eigvalues, electron_degen_cutoff) || return eigvalues, eigvectors

    # FIXME: This is a fix to mimic EPW. Later, a correct gauge-invariant formula should be implemented.
    # Fix eigenvector gauge for degenerate bands. Follow what is done in EPW.
    # Generate a pertubation matrix of size (nw x nw) made of hard-coded number.
    # For each degenerate group, diagonalize the corresponding block of the perturbation
    # matrix and use it as the eigenvector
    n = size(hk, 1)
    degeneracy_perturbation_matrix!(hk, n)

    degen_e = eigvalues[1]
    degen_from = 1
    degen_to = -1
    for i in 1:n-1
        # Check if the degenerate group is terminated
        if abs(degen_e - eigvalues[i+1]) > electron_degen_cutoff
            degen_to = i
        elseif i == n-1
            degen_to = i + 1
        else
            continue # degenerate group is not terminated, continue to next band.
        end

        # fix gauge
        ndegen = degen_to - degen_from + 1
        @views if ndegen > 1
            rng = degen_from:degen_to
            perturbation_matrix = Adjoint(eigvectors[:, rng]) * hk * eigvectors[:, rng]
            epw_syev!('V', 'U', perturbation_matrix)
            eigvectors[:, rng] .= eigvectors[:, rng] * perturbation_matrix
        end

        # setup for next degenerate group
        degen_e = eigvalues[i+1]
        degen_from = i + 1
    end
    eigvalues, eigvectors
end

"Get eigenenergy of electrons at a single k point.
Input hk is destroyed at output."
@timing "eig_el_val" function solve_eigen_el_valueonly!(eigvalues, hk)
    # eigvalues = eigvals(Hermitian(hk))
    # Use AllocatedLAPACK module for preallocating and reusing workspaces
    epw_syev!('N', 'U', hk, eigvalues)[1]
end

"Get frequency and eigenmode of phonons at a single q point.
Input dynq is destroyed at output."
@timing "eig_ph" function solve_eigen_ph!(eigvalues, eigvectors, dynq, mass)
    @assert size(dynq)[1] == length(mass)
    @assert size(eigvectors) == size(dynq)
    # F = eigen!(Hermitian(dynq))
    # eigvalues .= F.values
    # eigvectors .= F.vectors
    # Use AllocatedLAPACK module for preallocating and reusing workspaces
    eigvectors .= dynq
    epw_syev!('V', 'U', eigvectors, eigvalues)

    # Computed eigenvalues are omega^2. Return sign(omega^2) * omega.
    @inbounds for i in eachindex(eigvalues)
        eigvalues[i] = eigvalues[i] >= 0 ? sqrt(eigvalues[i]) : -sqrt(-eigvalues[i])
    end

    # Multiply mass factor to the eigenvectors
    for row in eachcol(eigvectors)
        row ./= sqrt.(mass)
    end
    eigvalues
end

"Get frequency of phonons at a single q point.
Input dynq is destroyed at output."
@timing "eig_ph_val" function solve_eigen_ph_valueonly!(eigvalues, dynq)
    # Directly calling LAPACK.syev! is more efficient than eigen(Hermitian(hk))
    # But, for the sake of type stability, we use eigen and eigvals.
    # eigvalues = LAPACK.syev!('N', 'U', dynq)
    # eigvalues = eigvals!(Hermitian(dynq))

    # Use AllocatedLAPACK module for preallocating and reusing workspaces
    epw_syev!('N', 'U', dynq, eigvalues)[1]

    # Computed eigenvalues are omega^2. Return sign(omega^2) * omega.
    @. eigvalues = sqrt(abs(eigvalues)) * sign(eigvalues)
    return eigvalues
end
