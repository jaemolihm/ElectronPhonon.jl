# __precompile__(true)

module Diagonalize
    using LinearAlgebra

    using EPW: @timing

    export solve_eigen_el!
    export solve_eigen_el_valueonly!
    export solve_eigen_ph!
    export solve_eigen_ph_valueonly!

    "Get eigenenergy and eigenvector of electrons at a single k point.
    Input hk is not destroyed at output."
    @timing "eig_el" function solve_eigen_el!(eigvectors, hk)
        @assert size(eigvectors) == size(hk)
        # Directly calling LAPACK.syev! is more efficient than
        # eigen(Hermitian(hk)) for small matrices, because eigen uses
        # SYEVR which is slower than SYEV for small matrices
        # eigvalues, eigvectors[:, :] = eigen(Hermitian(hk))
        eigvectors .= hk
        eigvalues, = LAPACK.syev!('V', 'U', eigvectors)
        return eigvalues::Vector{eltype(hk).parameters[1]}
    end

    "Get eigenenergy of electrons at a single k point.
    Input hk is destroyed at output."
    @timing "eig_el_val" function solve_eigen_el_valueonly!(hk)
        # eigvalues = eigvals(Hermitian(hk))
        eigvalues = LAPACK.syev!('N', 'U', hk)
        return eigvalues::Vector{eltype(hk).parameters[1]}
    end

    "Get frequency and eigenmode of phonons at a single q point.
    Input dynq is destroyed at output."
    @timing "eig_ph" function solve_eigen_ph!(eigvectors, dynq, mass)
        @assert size(dynq)[1] == length(mass)
        @assert size(eigvectors) == size(dynq)
        # Directly calling LAPACK.syev! is more efficient than eigen(Hermitian(hk))
        # But, for the sake of type stability, we use eigen and eigvals.
        # eigvalues, eigvectors = LAPACK.syev!('V', 'U', dynq)
        F = eigen!(Hermitian(dynq))
        eigvalues = F.values
        eigvectors .= F.vectors

        # Computed eigenvalues are omega^2. Return sign(omega^2) * omega.
        eigvalues_sqrt = sqrt.(abs.(eigvalues))
        eigvalues_sqrt[eigvalues .< 0.0] .*= -1

        # Multiply mass factor to the eigenvectors
        for row in eachcol(eigvectors)
            row ./= sqrt.(mass)
        end
        return eigvalues_sqrt
    end

    "Get frequency of phonons at a single q point.
    Input dynq is destroyed at output."
    @timing "eig_ph_val" function solve_eigen_ph_valueonly!(dynq)
        # Directly calling LAPACK.syev! is more efficient than eigen(Hermitian(hk))
        # But, for the sake of type stability, we use eigen and eigvals.
        # eigvalues = LAPACK.syev!('N', 'U', dynq)
        eigvalues = eigvals!(Hermitian(dynq))

        # Computed eigenvalues are omega^2. Return sign(omega^2) * omega.
        eigvalues_sqrt = sqrt.(abs.(eigvalues))
        eigvalues_sqrt[eigvalues .< 0.0] .*= -1
        return eigvalues_sqrt
    end
end
