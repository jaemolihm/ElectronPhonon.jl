
"Functions for filtering k points and bands outside a certain energy window"

# TODO: Threads

using Base.Threads
using MPI
using EPW: Kpoints

# export inside_window
export filter_kpoints
export filter_kpoints_grid
export filter_qpoints

"Test whether e is inside the window. Assume e is sorted"
function inside_window(e, window_min, window_max)
    searchsortedfirst(e, window_min):searchsortedlast(e, window_max)
end

"Filter Kpoints object"
function filter_kpoints(kpoints::Kpoints, nw, el_ham, window)
    # If the window is trivial, return the original kpoints
    if window === (-Inf, Inf)
        return kpoints, 1, nw
    end

    hk = zeros(ComplexF64, nw, nw)
    ik_keep = zeros(Bool, kpoints.n)
    band_min = nw
    band_max = 1
    for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        get_fourier!(hk, el_ham, xk, mode="normal")
        eigenvalues = solve_eigen_el_valueonly!(hk)
        bands_in_window = inside_window(eigenvalues, window...)
        if ! isempty(bands_in_window)
            ik_keep[ik] = true
            band_min = min(bands_in_window[1], band_min)
            band_max = max(bands_in_window[end], band_max)
        end
    end
    EPW.get_filtered_kpoints(kpoints, ik_keep), band_min, band_max
end

"""
    filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points
"""
function filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window)
    kpoints = generate_kvec_grid(nk1, nk2, nk3)

    # If the window is trivial, return the whole grid
    if window === (-Inf, Inf)
        return kpoints, 1, nw
    end

    hk = zeros(ComplexF64, nw, nw)
    ik_keep = zeros(Bool, kpoints.n)
    band_min = nw
    band_max = 1

    for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        get_fourier!(hk, el_ham, xk, mode="gridopt")
        eigenvalues = solve_eigen_el_valueonly!(hk)
        bands_in_window = inside_window(eigenvalues, window...)
        if ! isempty(bands_in_window)
            ik_keep[ik] = true
            band_min = min(bands_in_window[1], band_min)
            band_max = max(bands_in_window[end], band_max)
        end
    end
    EPW.get_filtered_kpoints(kpoints, ik_keep), band_min, band_max
end

"""
    filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window, mpi_comm::MPI.Comm) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points.
Obtained k points are distributed over the MPI communicator mpi_comm.
"""
function filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window, mpi_comm::MPI.Comm)
    # Distribute k points
    kpoints = generate_kvec_grid(nk1, nk2, nk3, mpi_comm)

    # If the window is trivial, return the whole grid
    if window === (-Inf, Inf)
        return kpoints, 1, nw
    end

    hk = zeros(ComplexF64, nw, nw)
    ik_keep = zeros(Bool, kpoints.n)
    band_min = nw
    band_max = 1

    for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        get_fourier!(hk, el_ham, xk, mode="gridopt")
        eigenvalues = solve_eigen_el_valueonly!(hk)
        bands_in_window = inside_window(eigenvalues, window...)
        if ! isempty(bands_in_window)
            ik_keep[ik] = true
            band_min = min(bands_in_window[1], band_min)
            band_max = max(bands_in_window[end], band_max)
        end
    end
    k_filtered = EPW.get_filtered_kpoints(kpoints, ik_keep)

    band_min = mpi_min(band_min, mpi_comm)
    band_max = mpi_max(band_max, mpi_comm)

    # TODO: Make function kpoints_gather
    # TODO: Make gather + redistribute as a single function for Kpoints

    # Gather filtered k points
    kvectors = EPW.mpi_allgather(k_filtered.vectors, mpi_comm)
    weights = EPW.mpi_allgather(k_filtered.weights, mpi_comm)

    # Redistribute k points
    range = EPW.mpi_split_iterator(1:length(kvectors), mpi_comm)
    new_kpoints = Kpoints(length(range), kvectors[range], weights[range])
    new_kpoints, band_min, band_max
end

"""
    filter_qpoints(qpoints, kpoints, nw, el_ham)

Filter only q points which have k point such that the energy at k+q is inside
the window.
"""
function filter_qpoints(qpoints, kpoints, nw, el_ham, window)
    iq_keep = zeros(Bool, qpoints.n)
    hks = [zeros(ComplexF64, nw, nw) for i=1:nthreads()]
    @threads for iq in 1:qpoints.n
        hk = hks[threadid()]
        xq = qpoints.vectors[iq]
        for xk in kpoints.vectors
            xkq = xq + xk
            get_fourier!(hk, el_ham, xkq, mode="gridopt")
            eigenvalues = solve_eigen_el_valueonly!(hk)

            # If k+q is inside window, use this q point
            if ! isempty(inside_window(eigenvalues, window...))
                iq_keep[iq] = true
                break
            end
        end
    end
    EPW.get_filtered_kpoints(qpoints, iq_keep)
end
