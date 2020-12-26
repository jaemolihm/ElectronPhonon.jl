__precompile__(true)

using Statistics

# TODO: separate to smearing.jl and kpoints.jl

# module Utils
export occ_fermion
export occ_boson
export gaussian
export generate_kvec_grid
export average_degeneracy

function occ_fermion(value, occ_type="fd")
    if occ_type == "fd"
        1.0 / (exp(value) + 1.0)
    else
        error("unknown occ_type")
    end
end

function occ_boson(value, occ_type="be")
    if occ_type == "be"
        1.0 / (exp(value) - 1.0)
    else
        error("unknown occ_type")
    end
end

const inv_sqrtpi = 1 / sqrt(pi)
function gaussian(value)
    # One without the conditional is slightly (~2.5 %) faster.
    exp(-value^2) * inv_sqrtpi
    # abs(value) < 15.0 ? exp(-value^2) * inv_sqrtpi : 0.0
end

"Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return k points for global index in the given range."
function generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer, rng)
    nk = nk1 * nk2 * nk3
    @assert rng[1] >= 1
    @assert rng[end] <= nk
    kvecs = Vector{SVector{3,Float64}}()
    for ik in rng
        # For (i, j, k), make k the fastest axis
        k = mod(ik-1, nk3)
        j = mod(div(ik-1 - (k-1), nk3), nk2)
        i = mod(div(ik-1 - (k-1) - (j-1)*nk3, nk2*nk3), nk1)
        push!(kvecs, SVector{3,Float64}(i/nk1, j/nk2, k/nk3))
    end
    kvecs
end

"Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return all k points."
function generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer)
    nk = nk1 * nk2 * nk3
    generate_kvec_grid(nk1, nk2, nk3, 1:nk)
end

# "generate grid of k points"
# function generate_kvec_grid_array(nkf)
#     kvecs = zeros(3, nkf[1] * nkf[2] * nkf[3])
#     ind = 0
#     for i in 1:nkf[1], j in 1:nkf[2], k in 1:nkf[3]
#         ind += 1
#         kvecs[:, ind] .= [(i-1) / nkf[1], (j-1) / nkf[2], (k-1) / nkf[3]]
#     end
#     nk = size(kvecs)[2]
#     kvecs, nk
# end

"Average data for degenerate states using energy"
function average_degeneracy(data, energy, degeneracy_cutoff=1.e-6)
    @assert size(data) == size(energy)
    nbnd, nk = size(data)

    data_averaged = similar(data)

    # This is fast enough, so no need for threading.
    # Threads.@threads :static for ik in 1:nk
    for ik in 1:nk
        for ib in 1:nbnd
            iblist_degen = abs.(energy[:, ik] .- energy[ib, ik]) .< degeneracy_cutoff
            data_averaged[ib, ik] = mean(data[iblist_degen, ik])
        end # ib
    end # ik
    data_averaged
end
# end
