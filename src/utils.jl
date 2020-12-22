__precompile__(true)

# module Utils
export occ_fermion
export occ_boson
export gaussian
export generate_kvec_grid

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

"generate grid of k points as Vector of StaticVectors"
function generate_kvec_grid(nkf)
    kvecs = Vector{Vec3{Float64}}()
    for i in 1:nkf[1], j in 1:nkf[2], k in 1:nkf[3]
        push!(kvecs, Vec3{Float64}((i-1)/nkf[1], (j-1)/nkf[2], (k-1)/nkf[3]))
    end
    kvecs
end

"generate grid of k points"
function generate_kvec_grid_array(nkf)
    kvecs = zeros(3, nkf[1] * nkf[2] * nkf[3])
    ind = 0
    for i in 1:nkf[1], j in 1:nkf[2], k in 1:nkf[3]
        ind += 1
        kvecs[:, ind] .= [(i-1) / nkf[1], (j-1) / nkf[2], (k-1) / nkf[3]]
    end
    nk = size(kvecs)[2]
    kvecs, nk
end
# end
