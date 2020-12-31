
"Functions for filtering k points and bands outside a certain energy window"

# TODO: Threads

using MPI
using EPW: Kpoints
using EPW.Diagonalize

export filter_kpoints
export filter_kpoints_grid

"Test whether e is inside the window."
function inside_window(e, window_min, window_max)
    findall((e .< window_max) .& (e .> window_min))
end

"Filter Kpoints object"
function filter_kpoints(kpoints::Kpoints, nw, el_ham, window)
    hk = zeros(ComplexF64, nw, nw)
    ik_keep = zeros(Bool, kpoints.n)
    for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        get_fourier!(hk, el_ham, xk, mode="normal")
        eigenvalues = solve_eigen_el_valueonly!(hk)
        ik_keep[ik] = !isempty(inside_window(eigenvalues, window...))
    end
    EPW.get_filtered_kpoints(kpoints, ik_keep)
end

"""
    filter_kpoints_grid(nk1, nk2, nk3, nw, window, el_ham) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points
"""
function filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window)
    kpoints = generate_kvec_grid(nk1, nk2, nk3)

    # If the window is trivial, return the whole grid
    if window === (-Inf, Inf)
        return kpoints
    end

    hk = zeros(ComplexF64, nw, nw)
    ik_keep = zeros(Bool, kpoints.n)

    for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        get_fourier!(hk, el_ham, xk, mode="gridopt")
        eigenvalues = solve_eigen_el_valueonly!(hk)
        ik_keep[ik] = !isempty(inside_window(eigenvalues, window...))
    end
    EPW.get_filtered_kpoints(kpoints, ik_keep)
end

"""
    filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window, mpi_comm::MPI.Comm) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points.
Parallelized over the MPI communicator mpi_comm
"""
function filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window, mpi_comm::MPI.Comm)
    # Distribute k points
    range = EPW.mpi_split_iterator(1:nk1*nk2*nk3, mpi_comm)
    kpoints = generate_kvec_grid(nk1, nk2, nk3, range)

    # If the window is trivial, return the whole grid
    if window === (-Inf, Inf)
        return kpoints
    end

    hk = zeros(ComplexF64, nw, nw)
    ik_keep = zeros(Bool, kpoints.n)

    for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        get_fourier!(hk, el_ham, xk, mode="gridopt")
        eigenvalues = solve_eigen_el_valueonly!(hk)
        ik_keep[ik] = !isempty(inside_window(eigenvalues, window...))
    end
    k_filtered = EPW.get_filtered_kpoints(kpoints, ik_keep)

    # TODO: Make function kpoints_gather
    # TODO: Make gather + redistribute as a single function for Kpoints

    # Gather filtered k points
    kvectors = EPW.mpi_allgather(k_filtered.vectors, mpi_comm)
    weights = EPW.mpi_allgather(k_filtered.weights, mpi_comm)

    # Redistribute k points
    range = EPW.mpi_split_iterator(1:length(kvectors), mpi_comm)
    Kpoints(length(range), kvectors[range], weights[range])
end
