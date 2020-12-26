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
