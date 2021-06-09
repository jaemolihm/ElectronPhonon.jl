
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

"""Filter Kpoints object
# Output
- `kpts`: Filtered kpoints.
- `band_min`, `band_max`: Minimum and maximum index of bands inside the window.
- `nelec_below_window`: Number of bands below the window, weighted by the k-point weights.
"""
function filter_kpoints(kpoints::Kpoints, nw, el_ham, window, fourier_mode="normal")
    # If the window is trivial, return the original kpoints
    if window == (-Inf, Inf)
        return kpoints, 1, nw
    end
    nelec_below_window = zero(eltype(window))

    hk = zeros(ComplexF64, nw, nw)
    ik_keep = zeros(Bool, kpoints.n)
    band_min = nw
    band_max = 1
    eigenvalues = zeros(Float64, nw)
    for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        get_fourier!(hk, el_ham, xk, mode=fourier_mode)
        solve_eigen_el_valueonly!(eigenvalues, hk)
        bands_in_window = inside_window(eigenvalues, window...)
        nelec_below_window += (bands_in_window.start - 1) * kpoints.weights[ik]
        if ! isempty(bands_in_window)
            ik_keep[ik] = true
            band_min = min(bands_in_window[1], band_min)
            band_max = max(bands_in_window[end], band_max)
        end
    end
    EPW.get_filtered_kpoints(kpoints, ik_keep), band_min, band_max, nelec_below_window
end

"""
    filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points
# Output
- `kpts`: Filtered kpoints.
- `band_min`, `band_max`: Minimum and maximum index of bands inside the window.
- `nelec_below_window`: Number of bands below the window, weighted by the k-point weights.
"""
function filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window; symmetry=nothing, kshift=[0, 0, 0])
    if symmetry !== nothing && (! all(kshift .== 0))
        error("nonzero kshift and symmetry incompatible (not implemented)")
    end

    if symmetry === nothing
        kpoints = generate_kvec_grid(nk1, nk2, nk3, kshift=kshift)
    else
        kpoints = bzmesh_ir_wedge((nk1, nk2, nk3), symmetry)
    end

    # If the window is trivial, return the whole grid
    if window == (-Inf, Inf)
        return kpoints, 1, nw, zero(eltype(window))
    end

    ik_keep, band_min, band_max, nelec_below_window = _filter_kpoints(nw, kpoints, el_ham, window)
    nelec_below_window /= nk1 * nk2 * nk3
    EPW.get_filtered_kpoints(kpoints, ik_keep), band_min, band_max, nelec_below_window
end

function _filter_kpoints(nw, kpoints, el_ham, window)
    eigenvalues_ = [zeros(Float64, nw) for _ in 1:nthreads()]
    ik_keep_ = [zeros(Bool, kpoints.n) for _ in 1:nthreads()]
    nelec_below_window_ = [zero(eltype(window)) for _ in 1:nthreads()]
    band_min_ = [nw for _ in 1:nthreads()]
    band_max_ = [1 for _ in 1:nthreads()]

    @threads :static for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        eigenvalues = eigenvalues_[threadid()]
        get_el_eigen_valueonly!(eigenvalues, nw, el_ham, xk, "gridopt")
        bands_in_window = inside_window(eigenvalues, window...)
        nelec_below_window_[threadid()] += bands_in_window.start - 1
        if ! isempty(bands_in_window)
            ik_keep_[threadid()][ik] = true
            band_min_[threadid()] = min(bands_in_window[1], band_min_[threadid()])
            band_max_[threadid()] = max(bands_in_window[end], band_max_[threadid()])
        end
    end

    ik_keep = reduce(.|, ik_keep_)
    nelec_below_window = sum(nelec_below_window_)
    band_min = minimum(band_min_)
    band_max = maximum(band_max_)
    ik_keep, band_min, band_max, nelec_below_window
end

"""
    filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window, mpi_comm::MPI.Comm) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points.
Obtained k points are distributed over the MPI communicator mpi_comm.
# Output
- `new_kpoints`: Filtered kpoints, equally redistributed among `mpi_comm`.
- `band_min`, `band_max`: Minimum and maximum index of bands inside the window.
- `nelec_below_window`: Number of bands below the window, weighted by the k-point weights.
"""
function filter_kpoints_grid(nk1, nk2, nk3, nw, el_ham, window, mpi_comm::MPI.Comm; symmetry=nothing, kshift=[0, 0, 0])
    if symmetry !== nothing && (! all(kshift .== 0))
        error("nonzero kshift and symmetry incompatible (not implemented)")
    end
    # Distribute k points
    if symmetry === nothing
        kpoints = generate_kvec_grid(nk1, nk2, nk3, mpi_comm, kshift=kshift)
    else
        # Create the irreducible k points in the root. Then redistribute.
        if mpi_isroot(mpi_comm)
            kpoints = bzmesh_ir_wedge((nk1, nk2, nk3), symmetry)
        else
            kpoints = Kpoints{Float64}()
        end
        kpoints = mpi_scatter(kpoints, mpi_comm)
    end

    # If the window is trivial, return the whole grid
    if window == (-Inf, Inf)
        return kpoints, 1, nw, zero(eltype(window))
    end

    ik_keep, band_min, band_max, nelec_below_window = _filter_kpoints(nw, kpoints, el_ham, window)

    nelec_below_window /= nk1 * nk2 * nk3

    k_filtered = EPW.get_filtered_kpoints(kpoints, ik_keep)

    band_min = mpi_min(band_min, mpi_comm)
    band_max = mpi_max(band_max, mpi_comm)
    nelec_below_window = mpi_sum(nelec_below_window, mpi_comm)

    new_kpoints = mpi_gather_and_scatter(k_filtered, mpi_comm)
    new_kpoints, band_min, band_max, nelec_below_window
end

"""
    filter_qpoints(qpoints, kpoints, nw, el_ham)

Filter only q points which have k point such that the energy at k+q is inside
the window.
"""
function filter_qpoints(qpoints, kpoints, nw, el_ham, window)
    iq_keep = zeros(Bool, qpoints.n)
    eigenvalues_threads = [zeros(Float64, nw) for i=1:nthreads()]
    @threads for iq in 1:qpoints.n
        eigenvalues = eigenvalues_threads[threadid()]
        xq = qpoints.vectors[iq]
        for xk in kpoints.vectors
            xkq = xq + xk
            get_el_eigen_valueonly!(eigenvalues, nw, el_ham, xkq, "gridopt")

            # If k+q is inside window, use this q point
            if ! isempty(inside_window(eigenvalues, window...))
                iq_keep[iq] = true
                break
            end
        end
    end
    EPW.get_filtered_kpoints(qpoints, iq_keep)
end
