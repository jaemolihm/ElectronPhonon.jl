__precompile__(true)

using Statistics

# TODO: separate to smearing.jl and kpoints.jl

# module Utils
export occ_fermion
export occ_boson
export gaussian
export generate_kvec_grid
export average_degeneracy
export bisect

function occ_fermion(value, occ_type="fd")
    if occ_type == "fd"
        1.0 / (exp(value) + 1.0)
    else
        error("unknown occ_type")
    end
end

function occ_fermion_derivative(value)
    1 / (2 + exp(-value) + exp(value))
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

"Average data for degenerate states using energy. Non allocating version."
function average_degeneracy!(data_averaged, data, energy, degeneracy_cutoff=1.e-6)
    @assert size(data) == size(energy)
    @assert size(data) == size(data_averaged)
    nbnd, nk = size(data)

    iblist_degen = zeros(Bool, nbnd)

    # This is fast enough, so no need for threading.
    # Threads.@threads :static for ik in 1:nk
    for ik in 1:nk
        for ib in 1:nbnd
            iblist_degen .= abs.(energy[:, ik] .- energy[ib, ik]) .< degeneracy_cutoff
            data_averaged[ib, ik] = mean(data[iblist_degen, ik])
        end # ib
    end # ik
    nothing
end

"Average data for degenerate states using energy"
function average_degeneracy(data, energy, degeneracy_cutoff=1.e-6)
    data_averaged = similar(data)
    average_degeneracy!(data_averaged, data, energy, degeneracy_cutoff)
    data_averaged
end
# end

"""
    bisect(fun, xl, xu; tolf=1E-15, max_iter=500)
Solve fun(x) = 0 using bisection. Converge if |fun(x)| < tolf.
"""
function bisect(fun, xl, xu; tolf=1E-15, max_iter=500)
    if (xl > xu)
        xl, xu = xu, xl
    end
    fl, fu = fun(xl), fun(xu)
    if fl == 0.0
        return xl
    end
    if fu == 0.0
        return xu
    end
    @assert fl * fu < 0 "wrong initial condition"
    iter = 0
    while true
        iter += 1
        xr = (xu + xl) * 0.5 # bisect interval
        fr = fun(xr) # value at the midpoint
        if fr * fl < 0.0 # (fr < 0.0 && fl > 0.0) || (fr > 0.0 && fl < 0.0)
            xu, fu = xr, fr # upper --> midpoint
        else
            xl, fl = xr, fr # lower --> midpoint
        end
        if abs(fr) <= tolf # Bisect converged.
            return xr
        end
        if iter > max_iter
            error("bisect not converged")
        end
    end
end
