
# Data and functions for k points

export generate_kvec_grid

struct Kpoints{T <: Real}
    n::Int                  # Number of k points
    vectors::Vector{Vec3{T}} # Fractional coordinate of k points
    weights::Vector{T}       # Weight of each k points
end

"Generate regular nk1 * nk2 * nk3 grid of k points as Vector of StaticVectors.
Return k points for global index in the given range."
function generate_kvec_grid(nk1::Integer, nk2::Integer, nk3::Integer, rng)
    # TODO: Type
    nk_grid = nk1 * nk2 * nk3
    @assert rng[1] >= 1
    @assert rng[end] <= nk_grid
    kvecs = Vector{Vec3{Float64}}()
    for ik in rng
        # For (i, j, k), make k the fastest axis
        k = mod(ik-1, nk3)
        j = mod(div(ik-1 - (k-1), nk3), nk2)
        i = mod(div(ik-1 - (k-1) - (j-1)*nk3, nk2*nk3), nk1)
        push!(kvecs, Vec3{Float64}(i/nk1, j/nk2, k/nk3))
    end
    nk = length(kvecs)
    Kpoints(nk, kvecs, fill(1/nk_grid, (nk,)))
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
