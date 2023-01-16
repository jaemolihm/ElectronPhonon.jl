
"Functions for filtering k points and bands outside a certain energy window"

# TODO: Threads
# TODO: Rename filter_kpoints -> filter_kpoints_by_energy / initialize_and_filter_kpoints

using Base.Threads
using MPI
using EPW: Kpoints

export filter_kpoints
export filter_qpoints

# range of bands inside the window. Assume e is sorted in ascending order.
inside_window(e, window_min, window_max) = searchsortedfirst(e, window_min):searchsortedlast(e, window_max)

"""Filter Kpoints object
# Output
- `kpts`: Filtered kpoints.
- `band_min`, `band_max`: Minimum and maximum index of bands inside the window.
- `nelec_below_window`: Number of bands below the window, weighted by the k-point weights.
"""
function filter_kpoints(kpoints::AbstractKpoints, nw, el_ham, window; fourier_mode="normal", symmetry=nothing, shift=nothing)
    # If the window is trivial, return the original kpoints
    window == (-Inf, Inf) && return kpoints, 1, nw, zero(eltype(window))
    ik_keep, band_min, band_max, nelec_below_window = _filter_kpoints(nw, kpoints, el_ham, window; fourier_mode)
    get_filtered_kpoints(kpoints, ik_keep), band_min, band_max, nelec_below_window
end

"""
    filter_kpoints(nks::NTuple{3,Integer}, nw, el_ham, window) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points, where nks = (nk1, nk2, nk3).
# Output
- `kpts`: Filtered kpoints.
- `band_min`, `band_max`: Minimum and maximum index of bands inside the window.
- `nelec_below_window`: Number of bands below the window, weighted by the k-point weights.
"""
function filter_kpoints(nks::NTuple{3,Integer}, nw, el_ham, window; fourier_mode="gridopt", symmetry=nothing, shift=(0, 0, 0))
    if symmetry !== nothing && (! all(shift .== 0))
        error("nonzero shift and symmetry incompatible (not implemented)")
    end
    kpoints = kpoints_grid(nks; symmetry, shift)

    # If the window is trivial, return the whole grid
    window == (-Inf, Inf) && return kpoints, 1, nw, zero(eltype(window))

    ik_keep, band_min, band_max, nelec_below_window = _filter_kpoints(nw, kpoints, el_ham, window; fourier_mode)
    EPW.get_filtered_kpoints(kpoints, ik_keep), band_min, band_max, nelec_below_window
end

filter_kpoints(k_input, nw, el_ham, window, mpi_comm::Nothing; kwargs...) = filter_kpoints(k_input, nw, el_ham, window; kwargs...)

"""
    filter_kpoints(nks::NTuple{3,Integer}, nw, el_ham, window, mpi_comm::MPI.Comm) -> Kpoints

Generate k grid of size nk1 * nk2 * nk3 and filter the k points, where nks = (nk1, nk2, nk3).
Obtained k points are distributed over the MPI communicator mpi_comm.
# Output
- `new_kpoints`: Filtered kpoints, equally redistributed among `mpi_comm`.
- `band_min`, `band_max`: Minimum and maximum index of bands inside the window.
- `nelec_below_window`: Number of bands below the window, weighted by the k-point weights.
"""
function filter_kpoints(nks::NTuple{3,Integer}, nw, el_ham, window, mpi_comm::MPI.Comm; fourier_mode="gridopt", symmetry=nothing, shift=[0, 0, 0])
    if symmetry !== nothing && (! all(shift .== 0))
        error("nonzero shift and symmetry incompatible (not implemented)")
    end
    # Distribute k points
    kpoints = kpoints_grid(nks, mpi_comm; shift, symmetry)

    # If the window is trivial, return the whole grid
    window == (-Inf, Inf) && return kpoints, 1, nw, zero(eltype(window))

    ik_keep, band_min, band_max, nelec_below_window = _filter_kpoints(nw, kpoints, el_ham, window; fourier_mode)

    k_filtered = EPW.get_filtered_kpoints(kpoints, ik_keep)

    band_min = mpi_min(band_min, mpi_comm)
    band_max = mpi_max(band_max, mpi_comm)
    nelec_below_window = mpi_sum(nelec_below_window, mpi_comm)

    new_kpoints = mpi_gather_and_scatter(k_filtered, mpi_comm)
    new_kpoints, band_min, band_max, nelec_below_window
end

function _filter_kpoints(nw, kpoints, el_ham, window; fourier_mode="normal")
    eigenvalues_ = [zeros(Float64, nw) for _ in 1:nthreads()]
    ik_keep_ = [zeros(Bool, kpoints.n) for _ in 1:nthreads()]
    nelec_below_window_ = zeros(eltype(window), kpoints.n)
    band_min_ = [nw for _ in 1:nthreads()]
    band_max_ = [1 for _ in 1:nthreads()]

    @threads :static for ik in 1:kpoints.n
        xk = kpoints.vectors[ik]
        eigenvalues = eigenvalues_[threadid()]
        get_el_eigen_valueonly!(eigenvalues, nw, el_ham, xk; fourier_mode)
        bands_in_window = inside_window(eigenvalues, window...)
        nelec_below_window_[ik] = (bands_in_window.start - 1) * kpoints.weights[ik]
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
    filter_qpoints(qpoints, kpoints, nw, el_ham, window; fourier_mode="gridopt")

Filter only q points which have k point such that the energy at k+q is inside
the window.
"""
function filter_qpoints(qpoints, kpoints, nw, el_ham, window; fourier_mode="gridopt")
    iq_keep = zeros(Bool, qpoints.n)
    eigenvalues_threads = [zeros(Float64, nw) for i=1:nthreads()]
    @threads for iq in 1:qpoints.n
        eigenvalues = eigenvalues_threads[threadid()]
        xq = qpoints.vectors[iq]
        for xk in kpoints.vectors
            xkq = xq + xk
            get_el_eigen_valueonly!(eigenvalues, nw, el_ham, xkq; fourier_mode)

            # If k+q is inside window, use this q point
            if ! isempty(inside_window(eigenvalues, window...))
                iq_keep[iq] = true
                break
            end
        end
    end
    EPW.get_filtered_kpoints(qpoints, iq_keep)
end
