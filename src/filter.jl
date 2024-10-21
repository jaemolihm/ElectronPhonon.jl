
"Functions for filtering k points and bands outside a certain energy window"

# TODO: Threads
# TODO: Rename filter_kpoints -> filter_kpoints_by_energy / initialize_and_filter_kpoints

using Base.Threads
using ChunkSplitters
using Folds
using MPI
using ElectronPhonon: Kpoints
using Interpolations: AbstractInterpolation

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
    get_filtered_kpoints(kpoints, ik_keep), band_min, band_max, nelec_below_window
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

    k_filtered = get_filtered_kpoints(kpoints, ik_keep)

    band_min = mpi_min(band_min, mpi_comm)
    band_max = mpi_max(band_max, mpi_comm)
    nelec_below_window = mpi_sum(nelec_below_window, mpi_comm)

    new_kpoints = mpi_gather_and_scatter(k_filtered, mpi_comm)
    new_kpoints, band_min, band_max, nelec_below_window
end

function _filter_kpoints(nw, kpoints, el_ham, window; fourier_mode="normal")
    ik_keep = zeros(Bool, kpoints.n)
    nelec_below_window_ = zeros(eltype(window), kpoints.n)
    band_min_ = zeros(Int, kpoints.n)
    band_max_ = zeros(Int, kpoints.n)

    @threads for iks in chunks(kpoints.vectors; n=2*nthreads())
        ham = get_interpolator(el_ham; fourier_mode)
        eigenvalues = zeros(real(eltype(el_ham)), nw)

        for ik in iks
            xk = kpoints.vectors[ik]
            get_el_eigen_valueonly!(eigenvalues, nw, ham, xk)
            bands_in_window = inside_window(eigenvalues, window...)

            nelec_below_window_[ik] = (bands_in_window.start - 1) * kpoints.weights[ik]
            if ! isempty(bands_in_window)
                ik_keep[ik] = true
                band_min_[ik] = bands_in_window[1]
                band_max_[ik] = bands_in_window[end]
            end
        end
    end

    nelec_below_window = sum(nelec_below_window_)
    band_min = minimum(band_min_)
    band_max = maximum(band_max_)
    (; ik_keep, band_min, band_max, nelec_below_window)
end


"""
    filter_qpoints(qpoints, kpoints, nw, el_ham, window; fourier_mode="gridopt")

Filter only q points which have k point such that the energy at k+q is inside
the window.
"""
function filter_qpoints(qpoints, kpoints, nw, el_ham, window; fourier_mode="gridopt")
    iq_keep = zeros(Bool, qpoints.n)

    @threads for iqs in chunks(qpoints.vectors; n=2*nthreads())
        ham = get_interpolator(el_ham; fourier_mode)
        eigenvalues = zeros(real(eltype(el_ham)), nw)

        for iq in iqs
            xq = qpoints.vectors[iq]
            for xk in kpoints.vectors
                xkq = xq + xk
                get_el_eigen_valueonly!(eigenvalues, nw, ham, xkq)

                # If k+q is inside window, use this q point
                if ! isempty(inside_window(eigenvalues, window...))
                    iq_keep[iq] = true
                    break
                end
            end
        end
    end
    get_filtered_kpoints(qpoints, iq_keep)
end

function filter_qpoints(qpoints, kpoints, itp_el::Dict{Int, <: AbstractInterpolation}, window)
    iq_keep = Folds.map(1:qpoints.n) do iq
        xq = qpoints.vectors[iq]
        for xk in kpoints.vectors, iw in keys(itp_el)
            xkq = xq + xk
            ekq = itp_el[iw](xkq...)
            if window[1] <= ekq <= window[2]
                # If k+q is inside window, use this q point
                return true
            end
        end
        return false
    end
    get_filtered_kpoints(qpoints, iq_keep)
end



"""
    filter_kpoints_multigrid(nks1, nks2, window1, window2, nw, el_ham;
                             fourier_mode="gridopt", symmetry=nothing)
Generate k point grid using the double grid method
1) Mesh with size nks1 inside window1 (denser mesh, narrower window)
2) Mesh with size nks2 inside window2 (coarser mesh, wider window)
"""
function filter_kpoints_multigrid(nks1, nks2, window1, window2, nw, el_ham;
                                  fourier_mode="gridopt", symmetry=nothing)

    all(mod.(nks1, nks2) .== 0) || throw(ArgumentError("nks1 must be a multiple of nks2"))

    kpts1, = filter_kpoints(nks1, nw, el_ham, window1; symmetry, fourier_mode)
    kpts1 = GridKpoints(kpts1)
    kpts2, = filter_kpoints(nks2, nw, el_ham, window2; symmetry, fourier_mode)
    kpts2 = GridKpoints(kpts2)

    # Merge kpts1 and kpts2, remove duplicates

    (; vectors, weights, ngrid) = kpts1
    for i in 1:kpts2.n
        xk = kpts2.vectors[i]
        if xk_to_ik(xk, kpts1) === nothing
            push!(vectors, xk)
            push!(weights, kpts2.weights[i])
        end
    end

    GridKpoints(Kpoints(length(vectors), vectors, weights, ngrid))
end
