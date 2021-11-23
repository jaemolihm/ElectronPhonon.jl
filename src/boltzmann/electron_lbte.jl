# Construct and solve linearized Bolzmann transport equation (BTE) for electrons

using LinearAlgebra
using SparseArrays
using Interpolations
using Interpolations: WeightedArbIndex, coordslookup, value_weights, weightedindexes
using Dictionaries

"""
    occupation_to_conductivity(δf::Vector{Vec3{FT}}, el, params)
Compute electron conductivity using the occupation `δf`.
"""
function occupation_to_conductivity(δf::Vector{Vec3{FT}}, el, params) where {FT}
    @assert length(δf) == el.n
    σ = zero(Mat3{FT})
    @views for i in 1:el.n
        σ += el.k_weight[i] * (δf[i] * el.vdiag[i]')
    end
    σ * params.spin_degeneracy / params.volume
end

"""
    compute_bte_scattering_matrix(filename, params, recip_lattice)
Construct BTE scattering-in matrix: S_{ind_i <- ind_f}. [Eq. (41) of Ponce et al (2020)]
bte_scat_mat[ind_i, ind_f] = S_{ind_i <- ind_f}
S_{i<-f} = 2π * ∑_{nu} |g_{fi,p}|^2 [(n_p + 1 - f_i) * δ(e_f - e_i - ω_p) + (n_p + f_i) * δ(e_f - e_i + ω_p)]
Note that f and i are swapped compared to the calculation of inv_τ.
i: electron state in el_i (n, k)
f: electron state in el_f (m, k+q)
p: phonon state (nu, q)
TODO: Is this valid if time-reversal is broken?
"""
function compute_bte_scattering_matrix(filename, params, recip_lattice)
    # Read btedata
    fid = h5open(filename, "r")
    el_i = load_BTData(open_group(fid, "initialstate_electron"), EPW.BTStates{Float64})
    el_f = load_BTData(open_group(fid, "finalstate_electron"), EPW.BTStates{Float64})
    ph = load_BTData(open_group(fid, "phonon"), EPW.BTStates{Float64})

    # Compute chemical potential
    bte_compute_μ!(params, el_i)

    nT = length(params.Tlist)
    bte_scat_mat = [zeros(el_i.n, el_f.n) for iT in 1:nT]
    inv_τ_iscat = zeros(Float64, nT)

    # Compute scattering matrix
    group_scattering = open_group(fid, "scattering")
    mpi_isroot() && println("Original grid: Total $(length(group_scattering)) groups of scattering")
    @time for (ig, key) in enumerate(keys(group_scattering))
        mpi_isroot() && mod(ig, 100) == 0 && println("Calculating scattering for group $ig")
        g = open_group(group_scattering, key)
        scat = load_BTData(g, EPW.ElPhScatteringData{Float64})

        for (iscat, s) in enumerate(scat)
            # Swap ind_el_i and ind_el_f because we are calculating the scattering-in process
            # Before swapping, the energy change is δe = e_i - e_f - sign_ph * ω_ph.
            # To preserve δe when swapping i and f, we need to change sign_ph to -sign_ph.
            # FIXME: For tetrahedron, currently the el_i and ph are linearly interpolated.
            #        Maybe one should add an option to choose which energies are interpolated.
            s_swap = (ind_el_i=s.ind_el_f, ind_el_f=s.ind_el_i, ind_ph=s.ind_ph, sign_ph=-s.sign_ph, mel=s.mel)
            EPW._compute_lifetime_serta_single_scattering!(inv_τ_iscat, el_f, el_i, ph, params, s_swap, recip_lattice)

            # Although el_i and el_f are swapped, one should take k point weights from el_f, not el_i.
            inv_τ_iscat .*= el_f.k_weight[s.ind_el_f] / el_i.k_weight[s.ind_el_i]

            for iT in 1:nT
                bte_scat_mat[iT][s.ind_el_i, s.ind_el_f] += inv_τ_iscat[iT]
            end
        end
    end
    close(fid)
    (;bte_scat_mat, el_i, el_f, ph)
end

"""
Solve Boltzmann transport equation for electrons.
``δf_i[i] = scat_mat[i, j] * δf_f[j] / inv_τ_i[i] + δf_i_serta[i]``
scat_mat is a rectangular matrix, mapping states in `el_f` to states in `el_i`.
δf[j] is the occupations for states `el_f` and is calculated by unfolding `δf_i`.
"""
function solve_electron_bte(el_i::EPW.BTStates{FT}, el_f::EPW.BTStates{FT}, scat_mat, inv_τ_i, params, symmetry=nothing; max_iter=100, rtol=1e-10) where {FT}
    output = (σ_serta = zeros(FT, 3, 3, length(params.Tlist)),
              σ = zeros(FT, 3, 3, length(params.Tlist)),
              δf_i_serta = zeros(Vec3{FT}, el_i.n, length(params.Tlist)),
              δf_i = zeros(Vec3{FT}, el_i.n, length(params.Tlist)),
    )

    map_i_to_f = vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry)

    for iT in 1:length(params.Tlist)
        μ = params.μlist[iT]
        T = params.Tlist[iT]
        δf_i_serta = @. -EPW.occ_fermion_derivative(el_i.e - μ, T) / inv_τ_i[:, iT] * el_i.vdiag
        σ_serta = symmetrize(occupation_to_conductivity(δf_i_serta, el_i, params), symmetry)

        # Initial guess: SERTA occupations
        δf_i = copy(δf_i_serta)
        σ = σ_serta

        for iter in 1:max_iter
            σ_old = σ

            # TODO: (scat_mat[iT] * map_i_to_f) always occur together. Do this outside of loop?
            # Unfold from δf_i to δf_f
            δf_f = map_i_to_f * δf_i

            # Multiply scattering matrix, and
            δf_i .= (scat_mat[iT] * δf_f) ./ inv_τ_i[:, iT] .+ δf_i_serta
            σ = symmetrize(occupation_to_conductivity(δf_i, el_i, params), symmetry)
            if norm(σ - σ_old) / norm(σ) < rtol
                @info "iT=$iT, converged at iteration $iter"
                break
            elseif iter == max_iter
                @info "iT=$iT, convergence not reached at maximum iteration $max_iter"
            end
        end

        output.σ_serta[:, :, iT] .= σ_serta
        output.σ[:, :, iT] .= σ
        output.δf_i_serta[:, iT] .= δf_i_serta
        output.δf_i[:, iT] .= δf_i
    end
    output
end

"""
    unfold_data_map(state::EPW.BTStates{FT}, symmetry) -> (map_unfold, indmap)
Construct a sparse matrix that unfolds a vector field defined on `state` using `symmetry`.
Assume the vector field is polar (i.e. not a pseudovector) and odd under time reversal.
For the unfolded states, `indmap[(xk_int..., iband)] = i` holds.
"""
function unfold_data_map(state::EPW.BTStates{FT}, symmetry) where {FT}
    inds_i = Int[]
    inds_unfold = Int[]
    values = Mat3{FT}[]

    indmap = Dictionary{CI{4}, Int}()
    ngrid = state.ngrid
    n_unfold = 0
    ndegen_unfold = Int[]
    for i in 1:state.n
        xk = state.xks[i]
        iband = state.iband[i]
        for (S, is_tr, Scart) in zip(symmetry.S, symmetry.is_tr, symmetry.Scart)
            Sk = is_tr ? -S * xk : S * xk
            Sk_int = mod.(round.(Int, Sk .* ngrid), ngrid)
            key = CI(Sk_int.data..., iband)
            i_Sk = get(indmap, key, -1)
            if i_Sk == -1
                # New state
                n_unfold += 1
                insert!(indmap, key, n_unfold)
                push!(inds_i, i)
                push!(inds_unfold, n_unfold)
                push!(values, is_tr ? -Scart : Scart)
                push!(ndegen_unfold, 1)
            else
                # State already found
                values[i_Sk] += is_tr ? -Scart : Scart
                ndegen_unfold[i_Sk] += 1
            end
        end
    end
    values ./= ndegen_unfold
    sparse(inds_unfold, inds_i, values), indmap
end

function unfold_data_map(state::EPW.BTStates{FT}, ::Nothing) where {FT}
    # Special case where no symmetry is used. Unfolding matrix is identity.
    # One just needs to compute indmap.
    indmap = Dictionary{CI{4}, Int}()
    ngrid = state.ngrid
    for i in 1:state.n
        # FIXME: using shift in xk
        xk = state.xks[i]
        xk_int = mod.(round.(Int, xk .* ngrid), ngrid)
        iband = state.iband[i]
        key = CI(xk_int.data..., iband)
        insert!(indmap, key, i)
    end
    I(state.n), indmap
end

"""
    vector_field_unfold_and_interpolate_map(el_i, el_f, symmetry) -> map_i_to_f
Construct a sparse matrix that unfolds and interpolates a vector field defined on states
in `el_i` to states in `el_f`. For unfolding, use `symmetry`. Assume the vector field is
polar (i.e. not a pseudovector) and even under time reversal.
For vector fields ``f_i`` and ``f_f`` (defined on `el_i` and `el_f`, respectively),
``f_f = map_i_to_f * f_i`` holds.
"""
function vector_field_unfold_and_interpolate_map(el_i::EPW.BTStates{FT}, el_f, symmetry) where FT
    map_unfold, indmap_unfold = unfold_data_map(el_i, symmetry)

    ranges = map(n -> range(0, 1, length=n+1)[1:end-1], el_i.ngrid)
    inds_f = Int[]
    inds_unfold = Int[]
    weights_all = FT[]

    for i_f in 1:el_f.n
        xk = el_f.xks[i_f]
        iband = el_f.iband[i_f]

        indexes, weights = linear_interpolation_weights(ranges, xk.data)
        for (ind_k, weight) in zip(indexes, weights)
            key = CI((ind_k.-1)..., iband)
            i_unfold = get(indmap_unfold, key, -1)
            if i_unfold != -1
                if weight > 1e-14
                    push!(inds_f, i_f)
                    push!(inds_unfold, i_unfold)
                    push!(weights_all, weight)
                end
            else
                # Interpolation point for state at el_f not found in el_i.
                # This may happen when using a different window for el_i and el_f, or due to
                # slight breaking of symmetry.
                # We ignore this case, assuming that the vector field is zero for states
                # not included in el_i.
            end
        end
    end
    map_interpolate = sparse(inds_f, inds_unfold, weights_all)

    map_interpolate * map_unfold
end


# Get interpolation weights
function weightedindexes_flatten(wis::NTuple{1,WeightedArbIndex{N,FT}}) where {N, FT}
    indexes = NTuple{N^1,NTuple{1,Int}}(Base.product(map(x -> x.indexes, wis)...))
    weights = prod.(NTuple{N^1,NTuple{1,FT}}(Base.product(map(x -> x.weights, wis)...)))
    indexes, weights
end
function weightedindexes_flatten(wis::NTuple{2,WeightedArbIndex{N,FT}}) where {N, FT}
    indexes = NTuple{N^2,NTuple{2,Int}}(Base.product(map(x -> x.indexes, wis)...))
    weights = prod.(NTuple{N^2,NTuple{2,FT}}(Base.product(map(x -> x.weights, wis)...)))
    indexes, weights
end
function weightedindexes_flatten(wis::NTuple{3,WeightedArbIndex{N,FT}}) where {N, FT}
    indexes = NTuple{N^3,NTuple{3,Int}}(Base.product(map(x -> x.indexes, wis)...))
    weights = prod.(NTuple{N^3,NTuple{3,FT}}(Base.product(map(x -> x.weights, wis)...)))
    indexes, weights
end

function my_weightedindexes(ranges, x)
    @assert length(ranges) == length(x)
    # setup Interpolation parameters
    ndim = length(ranges)
    itpflag = ntuple(_ -> BSpline(Linear(Periodic(OnCell()))), ndim)
    knots = Base.OneTo.(length.(ranges))
    # scale xb from ranges to 1:n
    xs = coordslookup(itpflag, ranges, x)
    weightedindexes((value_weights,), itpflag, knots, xs)
end

"""
    linear_interpolation_weights(ranges, x) -> (indexes, weights)
Compute the indexes and weights for linear interpolation whose grid is given by `ranges`.
"""
function linear_interpolation_weights(ranges, x)
    wis = my_weightedindexes(ranges, x)
    weightedindexes_flatten(wis)
end